"""The `wf` object a workflow script's `async def main(wf, args)` receives.

This module implements `WorkflowRuntime` against the frozen public surface in
this file's docstrings (signatures are load-bearing -- other lanes, the UI,
`run_manager`, and `agent_call`, are written against them and must not see
them move).

Cross-lane handshakes worth calling out up front (see the runtime lane's
final report for the full reasoning):

- Session identity: `WorkflowRuntime.agent()` computes a deterministic
  `session_id` (`f"wf-{run_id}-{call_id}"`, or `opts["session_id"]` if the
  caller supplied one) and passes it through `opts["session_id"]` to
  `agent_call.call_agent` -- `call_agent`'s own docstring documents
  `"session_id"` as a reserved `opts` key that "overrides the default
  derived id", so `call_agent`'s implementation (lane `agent_call`) is
  expected to honor it. `session_factory` itself is forwarded unchanged;
  this runtime never constructs a `FounderOSAskSession` directly.
- Nesting guard: a module-level `ContextVar` tracks "an agent() call (or a
  script's `main`) is currently in flight in this async context". Any
  attempt to construct a second `WorkflowRuntime` while that flag is set
  raises immediately -- this is what stands in for "a workflow's agent()
  calls cannot themselves start a nested workflow" until/unless a real
  sub-workflow primitive exists.
- `log()` cannot go through `emit` (`WorkflowEvent = PhaseEvent |
  AgentCallEvent`, frozen, no log variant) -- it appends to
  `self.log_records` instead. See the module docstring on `log()` below and
  this lane's report for the open question this leaves for integration.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import contextvars
from dataclasses import dataclass, field
import os
import time
from typing import Any, TypeVar

from vibe.workflows import agent_call
from vibe.workflows.agent_call import AgentCallResult, SessionFactory
from vibe.workflows.events import (
    AgentCallEvent,
    PhaseEvent,
    WorkflowEvent,
    WorkflowMeta,
)

T = TypeVar("T")

DEFAULT_MAX_CONCURRENCY_ENV = "FOUNDEROS_WORKFLOW_MAX_CONCURRENCY"
DEFAULT_MAX_CONCURRENCY = 4

DEFAULT_MAX_CALLS_ENV = "FOUNDEROS_WORKFLOW_MAX_CALLS"
DEFAULT_MAX_CALLS = 50.0

DEFAULT_MAX_SECONDS_ENV = "FOUNDEROS_WORKFLOW_MAX_SECONDS"
DEFAULT_MAX_SECONDS = 1800.0

# Set for the duration of an in-flight `agent()` call (or a script's `main`,
# via `vibe.workflows.script`'s glue helper) so `WorkflowRuntime.__init__`
# can detect an attempt to start a second, nested workflow from inside a
# running one. Scoped with a token around the awaited span, never left set
# permanently -- see the module docstring.
_NESTING_GUARD: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_workflow_nesting_guard", default=0
)

# Tracks the call-id a pipeline step's *first* `wf.agent()` call minted, so
# any further `wf.agent()` calls issued later in that same step nest under
# it (`AgentCallEvent.parent_call_id`). A ContextVar, not an instance
# attribute, because `wf.parallel([wf.pipeline(...), wf.pipeline(...)])` is
# legal and two concurrently-running pipelines must not cross-contaminate
# each other's "current parent" state. Only ever written by `pipeline()`
# (to reset it per step) and by `agent()` (to record "this call is now the
# step's parent") -- `agent()`'s write is scoped by `pipeline()`'s own
# token/reset, so it cannot leak past the step even though `agent()` sets
# it without holding its own token. Note this "first call becomes parent"
# tracking is a plain ContextVar mutation, not per-task-copy-safe: if a
# step itself calls `wf.parallel([wf.agent(...), wf.agent(...)])`, each
# concurrent call runs in its own copied context and won't see the other's
# write, so concurrent calls within one step end up as siblings (parent
# `None`) rather than nested -- only a *sequential* run of calls within a
# step nests reliably.
_PIPELINE_PARENT_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_workflow_pipeline_parent_call_id", default=None
)

# True only while a pipeline step's `await step(value)` is executing --
# lets `agent()` tell "no parent recorded yet because we're not in a
# pipeline at all" apart from "no parent recorded yet because this is the
# first call in the current step" (both look like `_PIPELINE_PARENT_CALL_ID
# is None` otherwise).
_PIPELINE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_workflow_pipeline_active", default=False
)


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(slots=True)
class WorkflowBudget:
    """Read/spend view of the run's compute budget.

    Agent-count and elapsed-time based (NOT dollars -- `/ask` exposes no
    token usage). Both are hard ceilings: `exhausted()` is true once either
    is crossed, and stays true (there is no refund). `remaining()`'s unit is
    "calls left" -- the same unit `spend()` debits in -- since that is the
    dimension a caller can act on synchronously; the elapsed-time ceiling is
    enforced only by `exhausted()` (there's no meaningful "time remaining in
    call units" to report).

    Defaults come from `FOUNDEROS_WORKFLOW_MAX_CALLS` /
    `FOUNDEROS_WORKFLOW_MAX_SECONDS` env vars (mirroring `parallel()`'s
    `FOUNDEROS_WORKFLOW_MAX_CONCURRENCY`), read once at construction time --
    not part of the frozen `WorkflowRuntime.__init__` signature, so a
    workflow run cannot yet override its own budget except via env var. See
    this lane's report for why (the `__init__` signature is frozen and has
    no budget parameter).

    Deliberately not frozen (unlike the stub this replaces): `spend()` must
    mutate state, and a `frozen=True` dataclass cannot support that.
    """

    max_calls: float = field(
        default_factory=lambda: _read_float_env(
            DEFAULT_MAX_CALLS_ENV, DEFAULT_MAX_CALLS
        )
    )
    max_seconds: float = field(
        default_factory=lambda: _read_float_env(
            DEFAULT_MAX_SECONDS_ENV, DEFAULT_MAX_SECONDS
        )
    )
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _spent_calls: float = field(default=0.0, init=False, repr=False)
    _start: float = field(default=0.0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def _ensure_started(self) -> None:
        if not self._started:
            self._start = self.clock()
            self._started = True

    def remaining(self) -> float:
        """Calls left before the count-based ceiling is hit."""
        return max(0.0, self.max_calls - self._spent_calls)

    def spend(self, amount: float, *, label: str) -> None:
        """Debit `amount` call-units, attributed to `label` for the journal.

        `label` is accepted for the journal/log trail's sake but this stub
        surface keeps no history beyond the running total -- a caller that
        wants per-spend attribution should also `wf.log(...)` it.
        """
        self._ensure_started()
        self._spent_calls += amount

    def exhausted(self) -> bool:
        """True once the call ceiling or the elapsed-time ceiling is crossed."""
        self._ensure_started()
        if self._spent_calls >= self.max_calls:
            return True
        return (self.clock() - self._start) >= self.max_seconds


_LABEL_MAX_LEN = 60
_VALID_LOG_LEVELS = {"debug", "info", "warning", "error"}


def _default_label(prompt: str, call_id: str) -> str:
    stripped = " ".join(prompt.split())
    if not stripped:
        return call_id
    if len(stripped) <= _LABEL_MAX_LEN:
        return stripped
    return stripped[: _LABEL_MAX_LEN - 3] + "..."


class WorkflowRuntime:
    """Constructed once per run by `run_manager`, passed as `wf` to `main`."""

    def __init__(
        self,
        *,
        run_id: str,
        meta: WorkflowMeta,
        session_factory: SessionFactory,
        emit: Callable[[WorkflowEvent], None] | None = None,
        skipped_positions: frozenset[int] = frozenset(),
    ) -> None:
        """`session_factory` is forwarded to every `call_agent(...)` this
        runtime issues (directly, or via `agent()`/`parallel()`/
        `pipeline()`) -- see `vibe/workflows/agent_call.py` for why each
        call gets its own session instance.

        `emit`, if given, is called synchronously with every `PhaseEvent`
        and `AgentCallEvent` this runtime produces, in emission order, for
        the caller (a Textual Tree widget, or `run_manager`'s journal
        writer, or both via two separate `WorkflowRuntime` instances
        wrapping the same underlying calls) to render or persist. `emit`
        must not block the event loop for long -- it runs inline on the
        same task as the workflow step that produced the event.

        `skipped_positions`, if given, is the set of call-id positions
        (`call-{n}` -> `n`) that must never be issued to `agent_call`:
        `agent()` mints the id and emits `"running"` -> `"skipped"` for a
        position in this set without spending budget or opening a session.
        Added by integration to close a gap the `run_manager` lane flagged
        (an operator's `skip_run` was recorded in the journal but never
        actually honored on `resume_run`, since nothing upstream of this
        runtime could tell it "this position is permanently skipped");
        `run_manager.execute_run` populates this from the prior journal
        before constructing the runtime -- this runtime never reads the
        journal itself.

        Raises `RuntimeError` if a `WorkflowRuntime` is constructed while
        another one already has an `agent()` call (or a script's `main`, via
        `vibe.workflows.script`'s glue helper) in flight in this async
        context -- see the module docstring's nesting-guard note.
        """
        if _NESTING_GUARD.get() > 0:
            raise RuntimeError(
                "nested workflow detected: a WorkflowRuntime cannot be "
                "constructed while a workflow agent() call (or a workflow's "
                "main) is already in flight in this async context -- "
                "workflows may only nest one level deep, mirroring Claude "
                "Code's own workflow-nesting rule"
            )
        self.run_id = run_id
        self.meta = meta
        self._session_factory = session_factory
        self._emit = emit
        self._skipped_positions = skipped_positions
        self._max_concurrency = _read_int_env(
            DEFAULT_MAX_CONCURRENCY_ENV, DEFAULT_MAX_CONCURRENCY
        )
        self._budget = WorkflowBudget()
        self._next_call_index = 0
        self._next_phase_index = 0
        self._current_phase_id: str | None = None
        #: `log()` cannot go through `emit` -- `WorkflowEvent` is frozen as
        #: `PhaseEvent | AgentCallEvent` with no log variant. Records land
        #: here instead; see the module docstring and `log()` below.
        self.log_records: list[dict[str, str | None]] = []

    def _emit_event(self, event: WorkflowEvent) -> None:
        if self._emit is not None:
            self._emit(event)

    def _next_call_id(self) -> str:
        call_id = f"call-{self._next_call_index}"
        self._next_call_index += 1
        return call_id

    async def agent(
        self,
        prompt: str,
        *,
        label: str | None = None,
        opts: dict[str, Any] | None = None,
    ) -> AgentCallResult:
        """Run one `call_agent(...)` under the current phase.

        Must be called from inside an active `async with wf.phase(...):`
        block -- calling it outside any phase is a programming error in
        the workflow script (raises `RuntimeError`, not a silent no-op).
        Mints the next `call-{n}` id (see the id-minting contract in
        `vibe/workflows/events.py`), emits an `AgentCallEvent` with
        `state="running"` before the call and a terminal-state event
        (`"ok"` / `"skipped"` / `"cancelled"` / `"error"`) after, and
        returns the `AgentCallResult` unchanged.

        If `wf.budget.exhausted()`, the call is never issued: it still
        mints an id and emits `"running"` -> `"cancelled"` (so the UI shows
        the node), but `agent_call.call_agent` is not invoked and no session
        is opened. The budget check and the id mint both happen in the
        synchronous prefix of this coroutine (before any `await`), so under
        `parallel()` every concurrently-issued call sees an up-to-date
        `remaining()` -- spending happens before the `await`, not after the
        call returns, or N parallel calls would all pass the check before
        any of them recorded a spend and the ceiling would be overshot.
        """
        if self._current_phase_id is None:
            raise RuntimeError(
                "wf.agent() was called outside of an active `async with "
                "wf.phase(...):` block -- every agent call must run under a "
                "phase"
            )
        phase_id = self._current_phase_id
        call_id = self._next_call_id()
        resolved_label = label if label is not None else _default_label(prompt, call_id)
        parent_call_id = _PIPELINE_PARENT_CALL_ID.get()
        if _PIPELINE_ACTIVE.get() and parent_call_id is None:
            # First agent() call in the current pipeline step: record it as
            # the nesting parent for any later calls in this same step.
            # Scoped by pipeline()'s own token/reset -- see the ContextVar
            # docstring above.
            _PIPELINE_PARENT_CALL_ID.set(call_id)

        self._emit_event(
            AgentCallEvent(
                run_id=self.run_id,
                phase_id=phase_id,
                call_id=call_id,
                label=resolved_label,
                state="running",
                parent_call_id=parent_call_id,
            )
        )

        if self._next_call_index - 1 in self._skipped_positions:
            # An operator marked this call-id position permanently skipped
            # (`run_manager.skip_run`) before this run/resume started. Never
            # issue it, never spend budget on it -- just emit the terminal
            # "skipped" state so the UI/journal reflect the operator's
            # decision instead of silently re-running a call they rejected.
            result = AgentCallResult(
                status="skipped", text=None, reason="skipped by operator"
            )
            self._emit_event(
                AgentCallEvent(
                    run_id=self.run_id,
                    phase_id=phase_id,
                    call_id=call_id,
                    label=resolved_label,
                    state=result.status,
                    reason=result.reason,
                    parent_call_id=parent_call_id,
                )
            )
            return result

        if self._budget.exhausted():
            result = AgentCallResult(
                status="cancelled",
                text=None,
                reason="workflow budget exhausted (call-count or elapsed-time ceiling)",
            )
            self._emit_event(
                AgentCallEvent(
                    run_id=self.run_id,
                    phase_id=phase_id,
                    call_id=call_id,
                    label=resolved_label,
                    state=result.status,
                    reason=result.reason,
                    parent_call_id=parent_call_id,
                )
            )
            return result

        self._budget.spend(1.0, label=resolved_label)

        merged_opts: dict[str, Any] = dict(opts or {})
        merged_opts.setdefault("session_id", f"wf-{self.run_id}-{call_id}")

        token = _NESTING_GUARD.set(_NESTING_GUARD.get() + 1)
        try:
            result = await agent_call.call_agent(
                prompt, opts=merged_opts, session_factory=self._session_factory
            )
        finally:
            _NESTING_GUARD.reset(token)

        self._emit_event(
            AgentCallEvent(
                run_id=self.run_id,
                phase_id=phase_id,
                call_id=call_id,
                label=resolved_label,
                state=result.status,
                text=result.text,
                reason=result.reason,
                parent_call_id=parent_call_id,
            )
        )
        return result

    async def parallel(self, calls: Sequence[Awaitable[T]]) -> list[T]:
        """Run several already-constructed coroutines concurrently.

        Usage: `await wf.parallel([wf.agent(p1), wf.agent(p2)])`. Because
        `call_agent` (and therefore `agent()`) never raises for a
        server-side failure (see its docstring), this is a plain
        `asyncio.gather(*calls)` with no `return_exceptions=True` needed --
        a failed call surfaces as an `AgentCallResult(status="error", ...)`
        in the returned list, in the same order as `calls`, not as an
        exception. A workflow-level cancellation (`asyncio.CancelledError`)
        still propagates normally.

        Concurrency is capped by a semaphore sized from
        `FOUNDEROS_WORKFLOW_MAX_CONCURRENCY` (default 4), read once at
        `WorkflowRuntime` construction. Semaphore admission is FIFO, so the
        order in which each coroutine's synchronous prefix runs (and
        therefore the order call ids get minted in) matches `calls`' list
        order even when the cap is below `len(calls)`.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _bounded(call: Awaitable[T]) -> T:
            async with semaphore:
                return await call

        return await asyncio.gather(*[_bounded(call) for call in calls])

    async def pipeline(
        self, steps: Sequence[Callable[[Any], Awaitable[Any]]], *, initial: Any = None
    ) -> Any:
        """Run `steps` in sequence, each fed the previous step's return value.

        `steps[0]` is called with `initial`; `steps[i]` is called with
        whatever `steps[i - 1]` returned. Each step is free to call
        `wf.agent(...)` internally (its calls nest under the pipeline step
        via `AgentCallEvent.parent_call_id`). Returns the last step's
        return value. Stops and returns early (without running remaining
        steps) if a step raises -- the exception propagates.

        Nesting rule (this lane's documented choice -- flagged in the
        report, since the frozen docstring above specifies parent nesting
        but not exactly *which* call each step's later calls nest under):
        within one step, the *first* `wf.agent()` call it issues becomes the
        nesting parent for every subsequent `wf.agent()` call issued later
        in that same step. No synthetic id is minted for the step itself
        (call ids strictly correspond to `wf.agent()` calls, in issue order
        -- see the id-minting contract in `events.py`, which resume replay
        depends on).
        """
        value = initial
        for step in steps:
            parent_token = _PIPELINE_PARENT_CALL_ID.set(None)
            active_token = _PIPELINE_ACTIVE.set(True)
            try:
                value = await step(value)
            finally:
                _PIPELINE_ACTIVE.reset(active_token)
                _PIPELINE_PARENT_CALL_ID.reset(parent_token)
        return value

    def phase(
        self, phase_id_or_title: str, *, detail: str = ""
    ) -> AbstractAsyncContextManager[None]:
        """Enter one phase; `async with wf.phase("title"): ...`.

        NOTE ON THE ARGUMENT: callers pass the human-readable phase title
        (matching `WorkflowMeta.phases[n].title` for the Nth call in
        execution order) -- the runtime, not the caller, mints the
        `phase-{n}` id per the contract in `events.py`. Emits a
        `PhaseEvent(state="running")` on enter and
        `PhaseEvent(state="ok")` (or `"error"` if the block raised) on
        exit. Phases do not nest -- entering a second phase before the
        first exits is a programming error in the workflow script (raises
        `RuntimeError`).
        """
        return self._phase_cm(phase_id_or_title, detail=detail)

    @asynccontextmanager
    async def _phase_cm(self, title: str, *, detail: str = "") -> AsyncIterator[None]:
        if self._current_phase_id is not None:
            raise RuntimeError(
                f"wf.phase({title!r}) was entered while phase "
                f"{self._current_phase_id!r} is still active -- phases do "
                "not nest; exit the current phase's `async with` block first"
            )
        phase_id = f"phase-{self._next_phase_index}"
        self._next_phase_index += 1
        self._current_phase_id = phase_id
        self._emit_event(
            PhaseEvent(
                run_id=self.run_id,
                phase_id=phase_id,
                title=title,
                detail=detail,
                state="running",
            )
        )
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self._current_phase_id = None
            self._emit_event(
                PhaseEvent(
                    run_id=self.run_id,
                    phase_id=phase_id,
                    title=title,
                    detail=detail,
                    state="error" if failed else "ok",
                )
            )

    def log(self, message: str, *, level: str = "info") -> None:
        """Record one free-text line against the current phase.

        Synchronous and non-blocking: appends to `self.log_records` (an
        in-memory list) rather than doing journal I/O inline. Valid `level`
        values: `"debug"`, `"info"`, `"warning"`, `"error"`.

        NOTE (open question for integration, see this lane's report): the
        frozen `WorkflowEvent = PhaseEvent | AgentCallEvent` union has no
        log variant, so this cannot go through `self._emit` the way phase
        and call events do. `run_manager`'s journal writer (which does
        understand `kind: "log"` per its own schema) needs to drain
        `runtime.log_records` through some other channel, or `events.py`
        needs a `LogEvent` variant added in an integration commit.
        """
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(f"invalid log level: {level!r}")
        self.log_records.append({
            "message": message,
            "level": level,
            "phase_id": self._current_phase_id,
        })

    @property
    def budget(self) -> WorkflowBudget:
        """The run's compute budget (see `WorkflowBudget`)."""
        return self._budget


WorkflowMain = Callable[["WorkflowRuntime", dict[str, Any]], Awaitable[None]]
"""The type of the `main` coroutine function every workflow script defines."""
