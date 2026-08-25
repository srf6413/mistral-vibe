"""Deterministic, non-interactive proof of the FULL duplex-voice pipeline
this task exists to verify -- toggling `voice_mode_enabled` all the way
through to synthesized audio actually reaching the local-playback sink --
with no manual TUI interaction and no real `MISTRAL_API_KEY`.

Chains together, all for real EXCEPT where noted:

1. `app._handle_voice_settings_closed({"voice_mode_enabled": True})` (the
   exact path a real Config screen save takes) -> the real config-write ->
   `_persist_voice_settings` -> `_apply_duplex_voice_enabled` ->
   `_start_duplex_voice()` -- proving the toggle really does trigger
   duplex startup, not just that calling `_start_duplex_voice()` directly
   works (already covered by `tests/cli/test_duplex_voice_toggle.py`).
   FAKED: `DuplexVoiceSupervisor` itself (its real
   `livekit-server`/in-process-agent lifecycle is covered by
   `tests/cli/duplex_voice/test_supervisor.py`) -- everything else
   `_start_duplex_voice()` wires up (`VoiceTurnBridge`, `JarvisBridgeLLM`,
   `duplex_active`) runs for real.
2. A transcript reaching `VoiceTurnBridge.handle_transcript` -> a REAL
   `AgentLoop` turn (`FakeBackend` stands in only for the network LLM
   call) -> the real streamed response text flowing back out through the
   bridge's subscriber queue -- the same chain
   `tests/cli/test_duplex_voice_integration.py` proves, reused here as
   this test's first half.
3. That REAL response text fed into a REAL `MistralDuplexTTS.synthesize()`
   (the actual plugin `vibe.cli.duplex_voice.agent._build_session` wires
   up) -> LiveKit's real `AudioEmitter`/WAV-decode machinery -> a real
   decoded `rtc.AudioFrame`. FAKED: only `MistralTTSClient` itself (the
   Mistral HTTP call), via `FakeTTSClient` -- no `MISTRAL_API_KEY` needed.
4. That real decoded frame's PCM bytes fed into the real
   `forward_playback_audio` -> a fake `AudioSinkLike` recording what would
   have been played -- proving the plumbing this task asks for without a
   real speaker.

NOT proven here (see this task's report for the full boundary): the real
LiveKit room/network hop between the agent's published TTS track and
`PlaybackSubscriber`'s subscription to it -- that requires a real
`livekit-server` and real RTP flow, which is exactly the boundary
`scripts/duplex_voice_proof.py` (not a pytest test) exists for instead;
`tests/cli/duplex_voice/test_playback_subscriber.py` separately proves
`PlaybackSubscriber.run()`'s own orchestration (identity filtering, task
wiring) against a faked `rtc.Room`/`rtc.AudioStream`.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tts_client import FakeTTSClient
from tests.stubs.fake_voice_manager import FakeVoiceManager
from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS
from vibe.cli.duplex_voice.playback_subscriber import forward_playback_audio
from vibe.cli.tts.tts_client_port import TTSResult


class _FakeDuplexVoiceSupervisor:
    """Stands in for the real `livekit-server` + in-process agent/mic/
    playback-subscriber lifecycle (covered separately by
    `tests/cli/duplex_voice/test_supervisor.py`) -- this test is about
    everything ELSE `_start_duplex_voice()` wires up around it, which runs
    unfaked.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        pass


class _RecordingSink:
    def __init__(self) -> None:
        self.written: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.written.append(data)


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await pilot.pause(0.01)


async def _wait_until_drained(pilot, app, timeout: float = 2.0) -> None:
    # Voice settings defer their side effect to the main queue (ADR 0012;
    # see tests/cli/test_audio_config_boundary.py's own helper).
    await _wait_until(pilot, lambda: not app._queue.draining, timeout)


async def _drain_until_done(queue: asyncio.Queue[str | None], *, timeout: float = 2.0) -> str:
    heard = ""
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=timeout)
        if item is None:
            return heard
        heard += item


@pytest.mark.asyncio
async def test_toggle_to_real_turn_to_real_tts_decode_to_playback_sink() -> None:
    backend = FakeBackend(mock_llm_chunk(content="hello from jarvis, live"))
    agent_loop = build_test_agent_loop(backend=backend)
    fake_voice_manager = FakeVoiceManager(is_voice_ready=False)
    app = build_test_vibe_app(agent_loop=agent_loop, voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # -- 1. Toggle voice_mode_enabled through the REAL config path --
        with patch(
            "vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor",
            _FakeDuplexVoiceSupervisor,
        ), patch(
            "vibe.cli.textual_ui.app.check_audio_available", return_value=None
        ):
            await app._handle_voice_settings_closed({"voice_mode_enabled": True})
            await _wait_until_drained(pilot, app)

        assert app.config.voice_mode_enabled is True
        assert app._duplex_voice_supervisor is not None
        assert app._duplex_voice_supervisor.started is True
        assert app._duplex_voice_bridge is not None
        # Ctrl+R push-to-talk correctly excluded now that duplex actually
        # started -- see VoiceManagerPort.duplex_active.
        assert app._voice_manager.duplex_active is True

        # -- 2. A transcript reaching the bridge drives a REAL turn --
        bridge = app._duplex_voice_bridge
        queue = bridge.subscribe()
        assert app.app_server.turn_active is False

        await bridge.handle_transcript("hey jarvis")
        response_text = await _drain_until_done(queue)
        bridge.unsubscribe(queue)

        assert response_text == "hello from jarvis, live"
        assert backend.requests_messages[0][-1].content == "hey jarvis"

        # -- 3. That real response text through the REAL TTS plugin (only
        # the Mistral HTTP client is faked -- no MISTRAL_API_KEY needed) --
        speech = app.app_server.resources.config.current.speech
        wav_bytes = _make_wav_bytes(sample_rate=24000, num_channels=1, duration_s=0.05)
        tts_client = FakeTTSClient(result=TTSResult(audio_data=wav_bytes))
        tts_plugin = MistralDuplexTTS(
            provider=speech.provider, model=speech.model, client=tts_client
        )
        try:
            frame = await tts_plugin.synthesize(response_text).collect()
        finally:
            await tts_plugin.aclose()

        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert frame.samples_per_channel > 0

        # -- 4. That real decoded PCM reaching the playback sink -- the
        # exact plumbing `PlaybackSubscriber` feeds in production, proven
        # here against a faked sink instead of real speakers. --
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        sink = _RecordingSink()
        frame_queue.put_nowait(bytes(frame.data))
        frame_queue.put_nowait(None)

        await forward_playback_audio(frame_queue, sink)

        assert len(sink.written) == 1
        assert sink.written[0] == bytes(frame.data)
        assert len(sink.written[0]) > 0


def _make_wav_bytes(*, sample_rate: int, num_channels: int, duration_s: float) -> bytes:
    import io
    import struct
    import wave

    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        samples = [
            int(3000 * ((i % 50) / 50.0 - 0.5)) for i in range(n_samples * num_channels)
        ]
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()
