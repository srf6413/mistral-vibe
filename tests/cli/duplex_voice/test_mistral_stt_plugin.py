from __future__ import annotations

import array
from collections.abc import AsyncIterator

from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import APIConnectOptions
import pytest

from vibe.cli.duplex_voice.duplex_config import default_transcription_config_view
from vibe.cli.duplex_voice.mistral_stt_plugin import (
    MistralDuplexSTT,
    _frame_to_mono_pcm16,
    _pcm16_rms,
)
from vibe.cli.transcribe.transcribe_client_port import (
    TranscribeDone,
    TranscribeError,
    TranscribeEvent,
    TranscribeSessionCreated,
    TranscribeTextDelta,
)


def _make_frame(
    *, loud: bool, sample_rate: int = 16000, samples_per_channel: int = 320
) -> rtc.AudioFrame:
    amplitude = 5000 if loud else 0
    samples = array.array("h", [amplitude] * samples_per_channel)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


class _DrainingFakeClient:
    """Drains the pushed byte stream fully (like the real realtime client
    does before finalizing), records what it saw, then replays canned
    events. Lets tests assert that audio frames actually reached the
    client, not just that events came back.
    """

    def __init__(self, events: list[TranscribeEvent] | None = None) -> None:
        self._events = events or [
            TranscribeSessionCreated(request_id="req-1"),
            TranscribeTextDelta(text="hello "),
            TranscribeTextDelta(text="world"),
            TranscribeDone(),
        ]
        self.segments: list[bytes] = []
        self.closed = False

    async def transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscribeEvent]:
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        self.segments.append(b"".join(chunks))
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


def test_frame_to_mono_pcm16_passes_through_mono_frames() -> None:
    frame = _make_frame(loud=True, samples_per_channel=10)
    assert _frame_to_mono_pcm16(frame) == frame.data.tobytes()


def test_frame_to_mono_pcm16_downmixes_stereo() -> None:
    left = array.array("h", [1000, 1000])
    right = array.array("h", [2000, 2000])
    interleaved = array.array("h")
    for l_sample, r_sample in zip(left, right, strict=True):
        interleaved.extend([l_sample, r_sample])
    frame = rtc.AudioFrame(
        data=interleaved.tobytes(),
        sample_rate=16000,
        num_channels=2,
        samples_per_channel=2,
    )

    mono = _frame_to_mono_pcm16(frame)

    mono_samples = array.array("h")
    mono_samples.frombytes(mono)
    assert list(mono_samples) == [1500, 1500]


def test_pcm16_rms_distinguishes_silence_from_loud_audio() -> None:
    silence = array.array("h", [0] * 100).tobytes()
    loud = array.array("h", [5000] * 100).tobytes()
    assert _pcm16_rms(silence) == 0.0
    assert _pcm16_rms(loud) > _pcm16_rms(silence)


@pytest.mark.asyncio
async def test_recognize_impl_returns_final_transcript_from_client_deltas() -> None:
    transcription = default_transcription_config_view()
    client = _DrainingFakeClient()
    stt_plugin = MistralDuplexSTT(
        provider=transcription.provider, model=transcription.model, client=client
    )

    frame = _make_frame(loud=True, samples_per_channel=160)
    event = await stt_plugin.recognize([frame])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello world"
    assert event.request_id == "req-1"
    assert len(client.segments) == 1
    assert len(client.segments[0]) > 0
    await stt_plugin.aclose()


@pytest.mark.asyncio
async def test_recognize_impl_raises_on_transcribe_error() -> None:
    transcription = default_transcription_config_view()
    client = _DrainingFakeClient(events=[TranscribeError(message="boom")])
    stt_plugin = MistralDuplexSTT(
        provider=transcription.provider, model=transcription.model, client=client
    )

    with pytest.raises(APIConnectionError):
        await stt_plugin.recognize(
            [_make_frame(loud=True, samples_per_channel=160)],
            conn_options=APIConnectOptions(max_retry=0),
        )

    await stt_plugin.aclose()


class _RaisingFakeClient:
    """Simulates the Mistral SDK raising directly (a failed websocket
    handshake, an auth rejection) instead of surfacing a TranscribeError
    event -- the shape a real connection failure takes.
    """

    async def transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscribeEvent]:
        async for _ in audio_stream:
            pass
        raise RuntimeError("connection refused")
        yield  # pragma: no cover -- makes this an async generator function

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_recognize_impl_wraps_raw_client_exceptions_as_api_connection_error() -> (
    None
):
    """A raw exception from the client (not a TranscribeError event) must
    still surface as an APIError, or RecognizeStream's/STT's retry-on-
    APIError machinery never engages and a transient failure kills the
    pump permanently.
    """
    transcription = default_transcription_config_view()
    stt_plugin = MistralDuplexSTT(
        provider=transcription.provider,
        model=transcription.model,
        client=_RaisingFakeClient(),
    )

    with pytest.raises(APIConnectionError):
        await stt_plugin.recognize(
            [_make_frame(loud=True, samples_per_channel=160)],
            conn_options=APIConnectOptions(max_retry=0),
        )

    await stt_plugin.aclose()


@pytest.mark.asyncio
async def test_recognize_stream_segments_speech_on_silence_and_emits_final_transcript() -> (
    None
):
    transcription = default_transcription_config_view()
    client = _DrainingFakeClient()
    stt_plugin = MistralDuplexSTT(
        provider=transcription.provider, model=transcription.model, client=client
    )

    stream = stt_plugin.stream()
    events: list[stt.SpeechEvent] = []
    try:
        for _ in range(5):
            stream.push_frame(_make_frame(loud=True))
        # 40 * 20ms = 800ms of trailing silence, past the 700ms threshold.
        for _ in range(40):
            stream.push_frame(_make_frame(loud=False))
        stream.end_input()

        async for ev in stream:
            events.append(ev)
    finally:
        await stream.aclose()
        await stt_plugin.aclose()

    types = [ev.type for ev in events]
    assert stt.SpeechEventType.START_OF_SPEECH in types
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in types
    assert stt.SpeechEventType.END_OF_SPEECH in types

    final = next(ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "hello world"

    # Exactly one segment was opened with the real client, and it actually
    # received the voiced frames' bytes.
    assert len(client.segments) == 1
    assert len(client.segments[0]) > 0


@pytest.mark.asyncio
async def test_recognize_stream_stays_silent_when_no_voice_is_ever_detected() -> None:
    transcription = default_transcription_config_view()
    client = _DrainingFakeClient()
    stt_plugin = MistralDuplexSTT(
        provider=transcription.provider, model=transcription.model, client=client
    )

    stream = stt_plugin.stream()
    events: list[stt.SpeechEvent] = []
    try:
        for _ in range(10):
            stream.push_frame(_make_frame(loud=False))
        stream.end_input()

        async for ev in stream:
            events.append(ev)
    finally:
        await stream.aclose()
        await stt_plugin.aclose()

    assert events == []
    assert client.segments == []
