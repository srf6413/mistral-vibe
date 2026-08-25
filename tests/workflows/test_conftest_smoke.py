"""Smoke test for the `fake_session_factory` fixture itself.

This is scaffolding verification, not workflow-runtime test coverage --
each build lane writes its own tests against `WorkflowRuntime` /
`call_agent` / `run_manager` once those are implemented. This file only
proves the fixture wires a real `FounderOSAskSession` to a fake,
network-free transport correctly, so lanes can trust it.
"""

from __future__ import annotations

import pytest

from vibe.app_server.events import HistoryEntryAdded
from vibe.app_server.models import PublicMessageEntry
from vibe.workflows.agent_call import AgentCallResult


@pytest.mark.asyncio
async def test_fake_session_factory_drives_a_real_session_through_two_calls(
    fake_session_factory,
) -> None:
    session_factory, transports = fake_session_factory([
        [{"type": "final_result", "data": {"answer": "first"}}],
        "awaiting_permission",
    ])

    first_session = session_factory()
    events = [event async for event in first_session.act("hi")]
    added = [e for e in events if isinstance(e, HistoryEntryAdded)]
    assert any(
        isinstance(e.entry, PublicMessageEntry)
        and e.entry.role == "assistant"
        and e.entry.text == "first"
        for e in added
    )
    await first_session.close()

    second_session = session_factory()
    raised = False
    try:
        async for _ in second_session.act("hi again"):
            pass
    except Exception as exc:
        raised = True
        assert type(exc).__name__ == "FounderOSAskStreamError"
    assert raised
    await second_session.close()

    assert len(transports) == 2
    assert transports[0].close_count == 1  # session.close() closes the transport
    assert transports[1].payloads[0]["text"] == "hi again"


def test_agent_call_result_status_field_shape() -> None:
    # Freezes the AgentCallResult contract lane (b) implements call_agent
    # against -- if this shape ever needs to change, every lane needs to
    # know, so a broken assertion here is the tripwire.
    result = AgentCallResult(status="ok", text="hello", reason=None)
    assert result.status == "ok"
    assert result.text == "hello"
    assert result.reason is None
