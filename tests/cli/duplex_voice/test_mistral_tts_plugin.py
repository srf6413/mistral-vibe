from __future__ import annotations

import io
import struct
import wave

from livekit.agents import APIConnectionError
from livekit.agents.types import APIConnectOptions
import pytest

from tests.stubs.fake_tts_client import FakeTTSClient
from vibe.cli.duplex_voice import announce_lock
from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS
from vibe.cli.duplex_voice.standalone_defaults import default_speech_config_view
from vibe.cli.tts.tts_client_port import TTSResult


@pytest.fixture(autouse=True)
def _isolated_announce_lock(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_run` awaits `wait_while_other_speaker()` before synthesizing (see
    `announce_lock`'s module docstring) -- point it at a private lock file
    instead of the real, Mac-wide `/tmp/.claude-announce.pidlock` so these
    tests never contend with (or wait behind) whatever is actually holding
    that lock on the machine running them.
    """
    monkeypatch.setattr(announce_lock, "_LOCK_PATH", str(tmp_path / "announce.pidlock"))


def _make_wav_bytes(*, sample_rate: int, num_channels: int, duration_s: float) -> bytes:
    """A tiny real WAV file (a short sine tone), matching the raw WAV bytes
    `MistralTTSClient.speak()` returns for the repo's configured
    `response_format="wav"`.
    """
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


@pytest.mark.asyncio
async def test_synthesize_decodes_client_wav_into_declared_format() -> None:
    speech = default_speech_config_view()
    wav_bytes = _make_wav_bytes(sample_rate=24000, num_channels=1, duration_s=0.1)
    client = FakeTTSClient(result=TTSResult(audio_data=wav_bytes))
    tts_plugin = MistralDuplexTTS(
        provider=speech.provider, model=speech.model, client=client
    )

    try:
        stream = tts_plugin.synthesize("hello world")
        frame = await stream.collect()

        assert frame.sample_rate == 24000
        assert frame.num_channels == 1
        assert frame.samples_per_channel > 0
    finally:
        await tts_plugin.aclose()


@pytest.mark.asyncio
async def test_synthesize_resamples_a_different_source_rate_to_the_declared_rate() -> (
    None
):
    """MistralTTSClient's WAV might not natively be 24kHz -- confirm LiveKit's
    AudioStreamDecoder resamples it to our declared output rate either way.
    """
    speech = default_speech_config_view()
    wav_bytes = _make_wav_bytes(sample_rate=48000, num_channels=1, duration_s=0.1)
    client = FakeTTSClient(result=TTSResult(audio_data=wav_bytes))
    tts_plugin = MistralDuplexTTS(
        provider=speech.provider, model=speech.model, client=client
    )

    try:
        frame = await tts_plugin.synthesize("hello").collect()
        assert frame.sample_rate == 24000
    finally:
        await tts_plugin.aclose()


@pytest.mark.asyncio
async def test_uses_jarvis_tts_client_speak_with_the_input_text() -> None:
    speech = default_speech_config_view()
    wav_bytes = _make_wav_bytes(sample_rate=24000, num_channels=1, duration_s=0.05)
    seen: list[str] = []

    class RecordingClient(FakeTTSClient):
        async def speak(self, text: str) -> TTSResult:
            seen.append(text)
            return await super().speak(text)

    client = RecordingClient(result=TTSResult(audio_data=wav_bytes))
    tts_plugin = MistralDuplexTTS(
        provider=speech.provider, model=speech.model, client=client
    )

    try:
        await tts_plugin.synthesize("a specific phrase").collect()
    finally:
        await tts_plugin.aclose()

    assert seen == ["a specific phrase"]


@pytest.mark.asyncio
async def test_synthesize_wraps_raw_client_exceptions_as_api_connection_error() -> None:
    """A raw exception from `speak()` (not already an APIError) must still
    surface as an APIError, or ChunkedStream's retry-on-APIError machinery
    never engages and a transient failure is treated as fatal.
    """

    class RaisingClient(FakeTTSClient):
        async def speak(self, text: str) -> TTSResult:
            raise RuntimeError("connection refused")

    speech = default_speech_config_view()
    tts_plugin = MistralDuplexTTS(
        provider=speech.provider, model=speech.model, client=RaisingClient()
    )

    try:
        with pytest.raises(APIConnectionError):
            await tts_plugin.synthesize(
                "hello", conn_options=APIConnectOptions(max_retry=0)
            ).collect()
    finally:
        await tts_plugin.aclose()
