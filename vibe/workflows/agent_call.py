"""One workflow `agent()` call: send a prompt through a fresh ask session.

`call_agent` is the only place a workflow talks to an LLM. It owns exactly
one `FounderOSAskSession` instance for the lifetime of one call, so N
concurrent `call_agent` invocations (one per `session_factory()` call) never
share `_turn_active` / `_history` state and can run under `parallel()` with
no coordination beyond `asyncio.gather`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import json
from typing import Any, Literal

import jsonschema

from vibe.app_server import AskPins, FounderOSAskError, FounderOSAskSession
from vibe.app_server.events import HistoryEntryAdded, HistoryEntryUpdated
from vibe.app_server.models import PublicMessageEntry

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


_DECIDE_DONT_ASK_PREAMBLE = (
    "You are running unattended inside an automated workflow. There is no "
    "human available to answer clarifying questions, and this client "
    "cannot resume a paused turn. Do not ask clarifying questions and do "
    "not wait for permission -- make your best decision from the "
    "information given, state any assumptions briefly, and proceed to a "
    "final answer within this turn.\n\n"
)

_JSON_FENCE_PREFIXES = ("```json", "```")


def _strip_code_fence(text: str) -> str:
    """Best-effort unwrap of a ```json ... ``` (or bare ```) fenced block."""
    stripped = text.strip()
    for prefix in _JSON_FENCE_PREFIXES:
        if stripped.startswith(prefix):
            unwrapped = stripped[len(prefix) :]
            if unwrapped.endswith("```"):
                unwrapped = unwrapped[: -len("```")]
            return unwrapped.strip()
    return stripped


def _validate_against_schema(text: str, schema: dict[str, Any]) -> str | None:
    """Return `None` if `text` parses as JSON and matches `schema`, else a
    short human-readable validation error describing the mismatch.
    """
    decode_error: json.JSONDecodeError | None = None
    for candidate in (text, _strip_code_fence(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            decode_error = exc
            continue
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            return f"Schema validation failed: {exc.message}"
        return None
    return f"Response was not valid JSON: {decode_error}"


def _last_assistant_text(previous: str | None, event: object) -> str | None:
    """Track the running final assistant text across a stream of
    `HistoryEntryAdded` / `HistoryEntryUpdated` events. `act()` only ever
    appends/updates a single trailing assistant entry per turn, so the
    latest one seen is always the turn's answer.
    """
    entry: object | None = None
    if isinstance(event, HistoryEntryAdded):
        entry = event.entry
    elif isinstance(event, HistoryEntryUpdated):
        entry = event.entry
    if isinstance(entry, PublicMessageEntry) and entry.role == "assistant":
        return entry.text
    return previous


async def _run_turn(session: FounderOSAskSession, message: str) -> str | None:
    """Drive one full `session.act(message)` turn to completion and return
    the final assistant text seen (`act()` itself raises
    `FounderOSAskStreamError` if a turn ends with no assistant text at
    all, so `None` here is defensive, not an expected path).
    """
    text: str | None = None
    async for event in session.act(message):
        text = _last_assistant_text(text, event)
    return text


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
      (Implementation note: both are caught via their common base
      `FounderOSAskError`, which also covers a small set of client-misuse
      errors from the same fail-closed boundary -- e.g. "turn already
      active" -- that should never legitimately occur given this
      function's own session lifecycle, but are still safer folded into
      an error result than left to crash a `parallel()` phase.)
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

    Structured-output limitation: this client boundary has no tool-forced
    JSON mode. When `opts["schema"]` (a JSON Schema dict) is given, the
    turn's text is parsed as JSON (raw, then with a ```json fence
    stripped) and validated; on failure exactly ONE re-ask turn is sent on
    the SAME session with the validation error appended, asking for
    corrected JSON only. Whatever comes back after that -- valid or not --
    is accepted and returned as `status="ok"`. This is best-effort: a
    caller that needs a hard schema guarantee must re-validate
    `AgentCallResult.text` itself; `call_agent` will not retry a second
    time or fail the call over a still-invalid response.

    Timeouts: `opts["timeout_seconds"]`, if given, bounds the whole call
    (including any schema re-ask) via `asyncio.wait_for`. On expiry the
    inner turn is cancelled, which drives `FounderOSAskSession.act`'s own
    `except asyncio.CancelledError` handling (calls `session.interrupt()`,
    then re-raises) before `wait_for` converts that into `TimeoutError`
    here; this is reported back as `AgentCallResult(status="cancelled",
    reason="timed out after Ns")` -- a workflow-logic-shaped outcome, not
    a raised exception, matching the "budget cutoff" case the contract
    describes. Cancellation of the `call_agent(...)` task itself from the
    *outside* (not a timeout) is deliberately NOT caught here: it
    propagates as `asyncio.CancelledError` per the contract above, after
    `act()` has already had its chance to call `session.interrupt()` and
    this function's `finally` has closed the session.
    """
    session = session_factory()
    schema = opts.get("schema")
    timeout_seconds = opts.get("timeout_seconds")
    session_id_override = opts.get("session_id")
    pins = opts.get("pins")

    try:
        if session_id_override:
            await session.resume(session_id_override)
        if pins:
            session.pins = AskPins(**pins)

        message = _DECIDE_DONT_ASK_PREAMBLE + prompt

        async def run() -> AgentCallResult:
            text = await _run_turn(session, message)

            if schema is not None and text is not None:
                validation_error = _validate_against_schema(text, schema)
                if validation_error is not None:
                    reask = (
                        "Your previous reply did not satisfy the required "
                        "JSON schema. Validation error:\n"
                        f"{validation_error}\n\n"
                        "Reply again with corrected JSON only that "
                        "satisfies the schema -- no prose, no code fence."
                    )
                    text = await _run_turn(session, reask)

            return AgentCallResult(status="ok", text=text, reason=None)

        if timeout_seconds is not None:
            try:
                return await asyncio.wait_for(run(), timeout=timeout_seconds)
            except TimeoutError:
                return AgentCallResult(
                    status="cancelled",
                    text=None,
                    reason=f"timed out after {timeout_seconds}s",
                )
        return await run()
    except FounderOSAskError as exc:
        return AgentCallResult(status="error", text=None, reason=str(exc))
    finally:
        with contextlib.suppress(Exception):
            await session.close()
