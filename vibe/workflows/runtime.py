"""The `wf` object a workflow script's `async def main(wf, args)` receives.

This module defines the PUBLIC SURFACE ONLY. Every method below has a real
signature and a real docstring contract but a `raise NotImplementedError`
body -- lane (a) implements the bodies against this file without changing
any signature, so the other lanes (UI, run_manager, agent_call) can be
written today against a surface that will not move under them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, TypeVar

from vibe.workflows.agent_call import AgentCallResult, SessionFactory
from vibe.workflows.events import WorkflowEvent, WorkflowMeta

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WorkflowBudget:
    """Read/spend view of the run's compute budget.

    A stub surface for whatever compute-budget enforcement lane (a) wires
    in (token counts, dollar caps, or a call-count ceiling); the shape here
    is intentionally minimal so it can be extended without breaking the
    `WorkflowRuntime.budget` property's return type.
    """

    def remaining(self) -> float:
        """Units of budget left, in whatever unit `spend` uses."""
        raise NotImplementedError

    def spend(self, amount: float, *, label: str) -> None:
        """Debit `amount` units, attributed to `label` for the journal."""
        raise NotImplementedError

    def exhausted(self) -> bool:
        """True once `remaining() <= 0`."""
        raise NotImplementedError


class WorkflowRuntime:
    """Constructed once per run by `run_manager`, passed as `wf` to `main`."""

    def __init__(
        self,
        *,
        run_id: str,
        meta: WorkflowMeta,
        session_factory: SessionFactory,
        emit: Callable[[WorkflowEvent], None] | None = None,
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
        """
        self.run_id = run_id
        self.meta = meta
        self._session_factory = session_factory
        self._emit = emit

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
        the workflow script (implementations should raise, not silently
        assign a phase). Mints the next `call-{n}` id (see the id-minting
        contract in `vibe/workflows/events.py`), emits an `AgentCallEvent`
        with `state="running"` before the call and a terminal-state event
        (`"ok"` / `"skipped"` / `"cancelled"` / `"error"`) after, and
        returns the `AgentCallResult` unchanged.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

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
        first exits is a programming error in the workflow script.
        """
        raise NotImplementedError

    def log(self, message: str, *, level: str = "info") -> None:
        """Record one free-text line against the current phase.

        Synchronous and non-blocking: implementations should queue the
        write rather than doing journal I/O inline. Valid `level` values:
        `"debug"`, `"info"`, `"warning"`, `"error"`.
        """
        raise NotImplementedError

    @property
    def budget(self) -> WorkflowBudget:
        """The run's compute budget (see `WorkflowBudget`)."""
        raise NotImplementedError


WorkflowMain = Callable[["WorkflowRuntime", dict[str, Any]], Awaitable[None]]
"""The type of the `main` coroutine function every workflow script defines."""
