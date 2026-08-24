"""One workflow `agent()` call: send a prompt through a fresh ask session.

`call_agent` is the only place a workflow talks to an LLM. It owns exactly
one `FounderOSAskSession` instance for the lifetime of one call, so N
concurrent `call_agent` invocations (one per `session_factory()` call) never
share `_turn_active` / `_history` state and can run under `parallel()` with
no coordination beyond `asyncio.gather`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from vibe.app_server import FounderOSAskSession

AgentCallStatus = Literal["ok", "skipped", "cancelled", "error"]


@dataclass(frozen=True, slots=True)
class AgentCallResult:
    """Outcome of one `call_agent(...)` invocation.

    Exactly one of `text` / `reason` is populated: `text` for `status ==
    "ok"` (the final assistant text of the turn), `reason` for every other
    status (a short, human-readable explanation -- never a raw traceback).
    """

    status: AgentCallStatus
    text: str | None
    reason: str | None


SessionFactory = Callable[[], FounderOSAskSession]
"""Builds one fresh, not-yet-used `FounderOSAskSession` per `call_agent` call.

The caller (`WorkflowRuntime.agent`) is responsible for giving each call a
stable, deterministic `session_id` (e.g. `f"wf-{run_id}-{call_id}"`, never a
random or time-derived id) so that a resumed run reconnects to the same
logical session identity FounderOS /ask would have seen the first time.
"""


async def call_agent(
    prompt: str, *, opts: dict[str, Any], session_factory: SessionFactory
) -> AgentCallResult:
    """Run one turn through a session built by `session_factory()`.

    Contract (binding on the implementation, not just documentation --
    lane (b) implements the body against this docstring):

    - Never raises for a server-side or transport-side failure. Catches
      `FounderOSAskStreamError` (including the "awaiting_permission" /
      "awaiting_clarification" pause, since checkpoint resume is not
      exposed by this adapter -- see `_founderos_ask.py`) and
      `FounderOSAskUnavailableError`, and returns
      `AgentCallResult(status="error", text=None, reason=str(exc))` for
      both. This is what lets `WorkflowRuntime.parallel()` be a plain
      `asyncio.gather(...)` with no `return_exceptions` special-casing:
      a failed call is a normal `AgentCallResult`, not an exception.
    - May raise `asyncio.CancelledError` (propagated, not swallowed) if the
      run itself is being cancelled -- that is a cooperative-cancellation
      signal, not a call failure, and the caller distinguishes a
      `status="cancelled"` *result* (this call was skipped by workflow
      logic, e.g. a budget cutoff) from actual task cancellation.
    - Always closes the session it built via `session_factory()` in a
      `finally` block, regardless of outcome.
    - `opts` is an opaque dict forwarded from `wf.agent(..., opts=...)`;
      reserved keys any implementation should honor if present:
      `"timeout_seconds"` (float), `"pins"` (`AskPins`-shaped dict with
      `intake_model`/`worker_model`), `"session_id"` (str, overrides the
      default derived id).
    """
    raise NotImplementedError
