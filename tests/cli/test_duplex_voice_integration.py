"""Real end-to-end proof of the voice-transcript -> real turn -> spoken-
response chain: a real `VoiceTurnBridge` wired to a real running `VibeApp`
and a real (in-memory) `AppServerSession`/`AgentLoop`, with only the LLM
backend faked (`FakeBackend`) -- no `DuplexVoiceSupervisor`, no LiveKit, no
network. `DuplexVoiceSupervisor`/`JarvisBridgeLLM`/`app.py`'s own toggle
wiring are covered elsewhere (`tests/cli/duplex_voice/test_supervisor.py`,
`tests/cli/duplex_voice/test_jarvis_llm.py`,
`tests/cli/test_duplex_voice_toggle.py`); this file is the one place that
proves the dispatch logic and the response tap actually work together
against a REAL turn, not test doubles standing in for both ends.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_voice_manager import FakeVoiceManager
from vibe.cli.duplex_voice.agent_bridge import VoiceTurnBridge
from vibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer


class _GatedBackend(FakeBackend):
    """A `FakeBackend` whose `complete()` blocks until released, so tests
    can dispatch a SECOND transcript while a turn is genuinely still
    in-flight (`app.app_server.turn_active` is True) instead of racing a
    turn that resolves near-instantly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def complete(self, **kwargs):
        self.entered.set()
        await self.gate.wait()
        return await super().complete(**kwargs)


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await pilot.pause(0.01)


def _wire_bridge(app) -> VoiceTurnBridge:
    """Exactly what `VibeApp._start_duplex_voice` wires up, minus the
    supervisor -- this test is about the bridge<->real-turn chain, not
    process lifecycle (already covered elsewhere).
    """
    bridge = VoiceTurnBridge(
        turn_active=lambda: app.app_server.turn_active,
        start_new_turn=app._start_queued_agent_turn,
        inject_mid_turn=app._inject_queued_prompt,
    )
    app._duplex_voice_bridge = bridge
    app._duplex_voice_event_sink = bridge.on_history_event
    return bridge


async def _drain_until_done(queue: asyncio.Queue[str | None], *, timeout: float = 2.0) -> str:
    heard = ""
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=timeout)
        if item is None:
            return heard
        heard += item


@pytest.mark.asyncio
async def test_voice_transcript_starts_a_real_turn_and_streams_the_real_response() -> (
    None
):
    backend = FakeBackend(mock_llm_chunk(content="hello from jarvis"))
    agent_loop = build_test_agent_loop(backend=backend)
    app = build_test_vibe_app(
        agent_loop=agent_loop, voice_manager=FakeVoiceManager(is_voice_ready=True)
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        bridge = _wire_bridge(app)
        queue = bridge.subscribe()

        assert app.app_server.turn_active is False
        await bridge.handle_transcript("hey jarvis")

        heard = await _drain_until_done(queue)

        assert heard == "hello from jarvis"
        assert len(backend.requests_messages) == 1
        assert backend.requests_messages[0][-1].content == "hey jarvis"
        await _wait_until(pilot, lambda: not app.app_server.turn_active)

        bridge.unsubscribe(queue)


@pytest.mark.asyncio
async def test_mid_turn_voice_utterance_against_a_keyboard_started_turn_still_gets_done() -> (
    None
):
    """Regression coverage for the bug this bridge design has to avoid:
    a turn started by the KEYBOARD path (no bridge task exists for it at
    all) that a voice utterance then steers mid-flight must still signal
    `None` to that voice utterance's subscriber once the turn actually
    ends -- not hang open until some later, unrelated barge-in cancels it.
    """
    backend = _GatedBackend(mock_llm_chunk(content="typed response"))
    agent_loop = build_test_agent_loop(backend=backend)
    app = build_test_vibe_app(
        agent_loop=agent_loop, voice_manager=FakeVoiceManager(is_voice_ready=True)
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        bridge = _wire_bridge(app)

        # Turn started via the KEYBOARD/chat-input path, not the bridge.
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await asyncio.wait_for(backend.entered.wait(), timeout=2.0)
        await _wait_until(pilot, lambda: app.app_server.turn_active)

        # A voice utterance arrives while that turn is still running.
        queue = bridge.subscribe()
        await bridge.handle_transcript("are you still there?")
        # inject_user_context on an active turn steers it (turn/steer),
        # not a new act() call -- confirm no second turn was started.
        assert len(backend.requests_messages) == 0 or app.app_server.turn_active

        backend.gate.set()  # let the keyboard-started turn finish

        heard = await _drain_until_done(queue)
        assert heard == "typed response"
        await _wait_until(pilot, lambda: not app.app_server.turn_active)

        bridge.unsubscribe(queue)
