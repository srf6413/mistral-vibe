"""Tests for `vibe.workflows.agent_call.call_agent`.

Uses the `fake_session_factory` fixture from `tests/workflows/conftest.py`
wherever a call needs exactly one `session_factory()` invocation and one
programmed turn per call (the common case). A few tests need more than one
turn on the *same* session (the schema re-ask path, the hang/interrupt
paths) or a transport shape the fixture can't express (an unavailable-
service error) -- those build a `FakeAskTransport` / `FounderOSAskSession`
directly, following the same no-network pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.workflows.conftest import FakeAskTransport
from vibe.app_server import FounderOSAskSession, FounderOSAskUnavailableError
from vibe.workflows.agent_call import AgentCallResult, call_agent

# -- happy path -------------------------------------------------------


@pytest.mark.asyncio
async def test_ok_via_final_result(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory([
        FakeAskTransport.ok("the answer")
    ])

    result = await call_agent(
        "what is the answer?", opts={}, session_factory=session_factory
    )

    assert result == AgentCallResult(status="ok", text="the answer", reason=None)
    assert transports[0].close_count == 1


@pytest.mark.asyncio
async def test_ok_via_streamed_deltas(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory([
        FakeAskTransport.ok_with_deltas("hel", "lo")
    ])

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "ok"
    assert result.text == "hello"
    assert transports[0].close_count == 1


@pytest.mark.asyncio
async def test_preamble_is_prepended_and_prompt_is_preserved(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory([FakeAskTransport.ok("ack")])

    await call_agent("do the specific task", opts={}, session_factory=session_factory)

    sent_text = transports[0].payloads[0]["text"]
    assert isinstance(sent_text, str)
    assert "do the specific task" in sent_text
    assert "Do not ask clarifying questions" in sent_text
    # preamble comes first, prompt follows
    assert sent_text.index("Do not ask clarifying questions") < sent_text.index(
        "do the specific task"
    )


# -- server/transport failures become error results, never raises -----


@pytest.mark.asyncio
async def test_awaiting_permission_becomes_error_result(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory(["awaiting_permission"])

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "error"
    assert result.text is None
    assert result.reason is not None
    assert transports[0].close_count == 1


@pytest.mark.asyncio
async def test_awaiting_clarification_becomes_error_result(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory(["awaiting_clarification"])

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "error"
    assert transports[0].close_count == 1


@pytest.mark.asyncio
async def test_error_event_becomes_error_result(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory([
        [{"type": "error", "data": {"message": "boom"}}]
    ])

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "error"
    assert result.text is None
    assert transports[0].close_count == 1


@pytest.mark.asyncio
async def test_stream_ending_without_final_result_becomes_error_result(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory([[]])

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "error"
    assert transports[0].close_count == 1


class _UnavailableTransport:
    """A transport whose stream raises `FounderOSAskUnavailableError`
    before yielding anything -- the shape `HttpFounderOSAskTransport` uses
    when the local FounderOS service cannot be reached at all.
    """

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.cancel_count = 0
        self.close_count = 0

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.payloads.append(dict(payload))

        async def generate() -> AsyncGenerator[dict[str, Any], None]:
            raise FounderOSAskUnavailableError("local FounderOS /ask unreachable")
            yield {}  # pragma: no cover - unreachable, keeps this an async gen

        return generate()

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_unavailable_service_becomes_error_result(tmp_path: Path) -> None:
    transport = _UnavailableTransport()

    def session_factory() -> FounderOSAskSession:
        return FounderOSAskSession(transport=transport, cwd=tmp_path)

    result = await call_agent("hi", opts={}, session_factory=session_factory)

    assert result.status == "error"
    assert result.text is None
    assert "unreachable" in (result.reason or "")
    assert transport.close_count == 1


# -- opts: pins / session_id ------------------------------------------


@pytest.mark.asyncio
async def test_pins_opt_is_forwarded_to_the_ask_payload(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory([FakeAskTransport.ok("ack")])

    await call_agent(
        "hi",
        opts={"pins": {"intake_model": "big-model", "worker_model": "small-model"}},
        session_factory=session_factory,
    )

    payload = transports[0].payloads[0]
    assert payload["intake_model"] == "big-model"
    assert payload["worker_model"] == "small-model"


@pytest.mark.asyncio
async def test_session_id_opt_overrides_the_default_frontend_session_id(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory([FakeAskTransport.ok("ack")])

    await call_agent(
        "hi", opts={"session_id": "wf-run-42-call-3"}, session_factory=session_factory
    )

    assert transports[0].payloads[0]["frontend_session_id"] == "wf-run-42-call-3"


@pytest.mark.asyncio
async def test_every_call_declares_chat_work_class_and_an_explicit_model(
    fake_session_factory,
) -> None:
    """Compute Budget (ask-compute-budget-dispatch-v1) treats every /ask turn
    as real dispatchable work by default -- work_class="chat" plus a real
    model on every request is what keeps a workflow review turn from coming
    back as a backgrounded engine dispatch or a "Parked /ask" stub instead of
    text. This must hold with no opts at all, not just when opted in.
    """
    session_factory, transports = fake_session_factory([FakeAskTransport.ok("ack")])

    await call_agent("hi", opts={}, session_factory=session_factory)

    payload = transports[0].payloads[0]
    assert payload["work_class"] == "chat"
    assert payload["model"]
    assert payload["model"] != "auto"


@pytest.mark.asyncio
async def test_model_opt_overrides_the_default_model_but_not_work_class(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory([FakeAskTransport.ok("ack")])

    await call_agent(
        "hi", opts={"model": "gpt-5-codex"}, session_factory=session_factory
    )

    payload = transports[0].payloads[0]
    assert payload["model"] == "gpt-5-codex"
    assert payload["work_class"] == "chat"


# -- opts: schema (best-effort structured output) ----------------------

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_schema_valid_on_first_reply_needs_no_reask(fake_session_factory) -> None:
    session_factory, transports = fake_session_factory([
        FakeAskTransport.ok('{"answer": "42"}')
    ])

    result = await call_agent(
        "give me json", opts={"schema": _SCHEMA}, session_factory=session_factory
    )

    assert result.status == "ok"
    assert result.text == '{"answer": "42"}'
    assert len(transports[0].payloads) == 1


@pytest.mark.asyncio
async def test_schema_invalid_reply_triggers_exactly_one_reask(tmp_path: Path) -> None:
    transport = FakeAskTransport([
        FakeAskTransport.ok("not json at all"),
        FakeAskTransport.ok('{"answer": "42"}'),
    ])

    def session_factory() -> FounderOSAskSession:
        return FounderOSAskSession(transport=transport, cwd=tmp_path)

    result = await call_agent(
        "give me json", opts={"schema": _SCHEMA}, session_factory=session_factory
    )

    assert result.status == "ok"
    assert result.text == '{"answer": "42"}'
    assert len(transport.payloads) == 2
    reask_text = transport.payloads[1]["text"]
    assert isinstance(reask_text, str)
    assert "did not satisfy" in reask_text or "Validation" in reask_text
    assert transport.close_count == 1


@pytest.mark.asyncio
async def test_schema_still_invalid_after_reask_is_still_status_ok(
    tmp_path: Path,
) -> None:
    """Best-effort: `call_agent` never fails the call over a schema that
    stays invalid after the one re-ask -- there is no way to force
    structured output at this client boundary (see the docstring).
    """
    transport = FakeAskTransport([
        FakeAskTransport.ok("still not json"),
        FakeAskTransport.ok("still not json either"),
    ])

    def session_factory() -> FounderOSAskSession:
        return FounderOSAskSession(transport=transport, cwd=tmp_path)

    result = await call_agent(
        "give me json", opts={"schema": _SCHEMA}, session_factory=session_factory
    )

    assert result.status == "ok"
    assert result.text == "still not json either"
    assert len(transport.payloads) == 2


# -- timeouts and cancellation -----------------------------------------


class _BlockingTransport:
    """Never delivers a `final_result` until `.release` is set -- models a
    stalled /ask stream for timeout/cancellation tests.
    """

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_count = 0
        self.close_count = 0

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.payloads.append(dict(payload))

        async def generate() -> AsyncGenerator[dict[str, Any], None]:
            self.started.set()
            await self.release.wait()
            if False:  # pragma: no cover - keeps this an async generator
                yield {}

        return generate()

    async def cancel(self) -> None:
        self.cancel_count += 1
        self.release.set()

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_timeout_seconds_becomes_cancelled_result_and_interrupts(
    tmp_path: Path,
) -> None:
    transport = _BlockingTransport()

    def session_factory() -> FounderOSAskSession:
        return FounderOSAskSession(transport=transport, cwd=tmp_path)

    result = await call_agent(
        "hi", opts={"timeout_seconds": 0.05}, session_factory=session_factory
    )

    assert result.status == "cancelled"
    assert result.text is None
    assert "timed out" in (result.reason or "")
    # act()'s own CancelledError handling calls session.interrupt() ->
    # transport.cancel() before wait_for turns this into a TimeoutError.
    assert transport.cancel_count == 1
    assert transport.close_count == 1


@pytest.mark.asyncio
async def test_external_task_cancellation_propagates_not_swallowed(
    tmp_path: Path,
) -> None:
    """Per the frozen contract: a `call_agent(...)` task cancelled from the
    outside (not a `timeout_seconds` expiry) must propagate
    `asyncio.CancelledError`, not turn into a `status="cancelled"` result
    -- that status is reserved for workflow-logic decisions (e.g. the
    timeout path above), not for actual task cancellation.
    """
    transport = _BlockingTransport()

    def session_factory() -> FounderOSAskSession:
        return FounderOSAskSession(transport=transport, cwd=tmp_path)

    task = asyncio.create_task(
        call_agent("hi", opts={}, session_factory=session_factory)
    )
    await transport.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert transport.cancel_count == 1
    assert transport.close_count == 1
