"""LiveKit STT plugin wrapping jarvis's own :class:`MistralTranscribeClient`.

Mistral's realtime transcription API has no server-side endpointing — the
provider only finalizes a transcript when the caller stops sending audio
(closing the byte stream `MistralTranscribeClient.transcribe()` consumes).
LiveKit's default streaming STT driver (`Agent.default.stt_node`) just keeps
calling `push_frame()` for the lifetime of the call and never calls
`flush()` on its own (that's only done by `stt.StreamAdapter`, which wraps
*non*-streaming STTs — ours declares `streaming=True`). So segmenting
speech into utterances is this plugin's job, not the framework's.

We do that with a small RMS-based silence detector inside `_run()`: feed
frames into the live Mistral stream while accumulated silence stays below
threshold, and end the current byte-stream (triggering Mistral's own
finalization -> `TranscribeDone`) once enough trailing silence follows real
speech. This intentionally avoids pulling in a VAD plugin (e.g.
`livekit-plugins-silero`, which the official `livekit-plugins-mistralai` STT
hard-requires for realtime models) to keep the dependency footprint the
task asked to be mindful of.
"""

from __future__ import annotations

import array
import asyncio
from collections.abc import AsyncIterator
import logging
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from vibe.app_server.config import AudioProviderView, TranscribeModelConfigView
from vibe.cli.transcribe.mistral_transcribe_client import MistralTranscribeClient
from vibe.cli.transcribe.transcribe_client_port import (
    TranscribeDone,
    TranscribeError,
    TranscribeEvent,
    TranscribeSessionCreated,
    TranscribeTextDelta,
)

if TYPE_CHECKING:
    from vibe.cli.transcribe.transcribe_client_port import TranscribeClientPort

logger = logging.getLogger("jarvis.duplex_voice.stt")

# Silence tuning for the RMS-based segmenter. int16 PCM full scale is 32767;
# typical room/mic noise floors sit well under this threshold, and speech
# comfortably clears it.
_SILENCE_RMS_THRESHOLD = 400.0
_SILENCE_DURATION_MS = 700.0


def _frame_to_mono_pcm16(frame: rtc.AudioFrame) -> bytes:
    """Return `frame`'s samples as mono 16-bit PCM bytes.

    LiveKit's built-in resampler (wired via `RecognizeStream.__init__`'s
    `sample_rate=` kwarg) normalizes the *rate* automatically, but not the
    channel count -- that's left to the plugin, per the STT base class
    contract.
    """
    if frame.num_channels == 1:
        return frame.data.tobytes()

    samples = array.array("h")
    samples.frombytes(frame.data.tobytes())
    channels = frame.num_channels
    mono = array.array(
        "h",
        (
            sum(samples[i + c] for c in range(channels)) // channels
            for i in range(0, len(samples), channels)
        ),
    )
    return mono.tobytes()


def _pcm16_rms(data: bytes) -> float:
    if not data:
        return 0.0
    samples = array.array("h")
    samples.frombytes(data)
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class _ByteQueue:
    """Bridges push-style byte production to the pull-style async iterator
    `MistralTranscribeClient.transcribe()` expects as its `audio_stream`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def push(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


async def _drive_segment(
    client: TranscribeClientPort, byte_source: AsyncIterator[bytes], *, language: str
) -> tuple[str, str]:
    """Consume one utterance's worth of `TranscribeEvent`s to completion.

    Returns `(accumulated_text, request_id)` on a clean finish. Raises
    `APIConnectionError` on a `TranscribeError` event, or on any exception
    the client (or the Mistral SDK underneath it) raises directly -- a
    failed websocket handshake, an auth rejection, a network drop -- so the
    caller's retry machinery (`RecognizeStream._main_task` /
    `_STTPipeline._stt_pump`, both of which only retry on `APIError`; any
    other exception type "propagates and stops the pump" permanently) can
    engage. Mirrors the idiom used by the official
    `livekit-plugins-mistralai` STT.
    """
    accumulated = ""
    request_id = ""
    event: TranscribeEvent
    try:
        async for event in client.transcribe(byte_source):
            if isinstance(event, TranscribeSessionCreated):
                request_id = event.request_id
            elif isinstance(event, TranscribeTextDelta):
                accumulated += event.text
            elif isinstance(event, TranscribeDone):
                return accumulated, request_id
            elif isinstance(event, TranscribeError):
                # The realtime WS error has no HTTP-style status code; treat
                # it as a connection-class failure so the caller's retry
                # policy can engage.
                raise APIConnectionError(event.message)
    except APIError:
        raise
    except Exception as exc:
        raise APIConnectionError(str(exc)) from exc
    # The client's async generator ended without a TranscribeDone (e.g. the
    # connection dropped mid-segment without surfacing a TranscribeError).
    raise APIConnectionError("Mistral transcription stream ended without a done event")


class MistralDuplexSTT(stt.STT):
    """LiveKit STT plugin backed by jarvis's `MistralTranscribeClient`."""

    def __init__(
        self,
        provider: AudioProviderView,
        model: TranscribeModelConfigView,
        *,
        client: TranscribeClientPort | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        self._provider = provider
        self._model = model
        self._client: TranscribeClientPort = client or MistralTranscribeClient(
            provider=provider, model=model
        )

    @property
    def model(self) -> str:
        return self._model.name

    @property
    def provider(self) -> str:
        return "mistral"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """One-shot recognition, driving the same streaming client for a
        single utterance: push the whole buffer as one chunk, end the
        stream, and wait for the final transcript.
        """
        frame = rtc.combine_audio_frames(buffer) if isinstance(buffer, list) else buffer
        pcm = _frame_to_mono_pcm16(frame)

        async def _one_shot() -> AsyncIterator[bytes]:
            if pcm:
                yield pcm

        resolved_language = language if is_given(language) else self._model.language
        text, request_id = await _drive_segment(
            self._client, _one_shot(), language=resolved_language
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[stt.SpeechData(language=resolved_language, text=text)],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> MistralRecognizeStream:
        resolved_language = language if is_given(language) else self._model.language
        return MistralRecognizeStream(
            stt=self,
            client=self._client,
            language=resolved_language,
            sample_rate=self._model.sample_rate,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._client.close()


class _SegmentState:
    """Mutable state for the one utterance currently being fed to Mistral,
    if any. Split out of `MistralRecognizeStream._run` purely to keep each
    method's statement count small; the state itself is single-owner
    (never shared across streams).
    """

    __slots__ = ("byte_queue", "consumer_task", "silence_ms", "had_voice")

    def __init__(self) -> None:
        self.byte_queue: _ByteQueue | None = None
        self.consumer_task: asyncio.Task[tuple[str, str]] | None = None
        self.silence_ms = 0.0
        self.had_voice = False

    @property
    def active(self) -> bool:
        return self.byte_queue is not None


class MistralRecognizeStream(stt.RecognizeStream):
    """Segments continuous room audio into utterances via RMS silence
    detection, driving one `MistralTranscribeClient.transcribe()` call per
    utterance.
    """

    def __init__(
        self,
        *,
        stt: MistralDuplexSTT,
        client: TranscribeClientPort,
        language: str,
        sample_rate: int,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._client = client
        self._language = language

    async def _run(self) -> None:
        state = _SegmentState()
        frames_seen = 0
        try:
            async for item in self._input_ch:
                if isinstance(item, stt.RecognizeStream._FlushSentinel):
                    if state.active and state.had_voice:
                        await self._finish_segment(state)
                    continue
                frames_seen = await self._handle_frame(item, state, frames_seen)

            # Input ended (room/track closed): finalize any open segment.
            if state.active and state.had_voice:
                await self._finish_segment(state)
            elif state.active:
                self._abandon_segment(state)
        finally:
            if state.consumer_task is not None and not state.consumer_task.done():
                state.consumer_task.cancel()

    async def _handle_frame(
        self, frame: rtc.AudioFrame, state: _SegmentState, frames_seen: int
    ) -> int:
        frames_seen += 1
        if frames_seen == 1 or frames_seen % 200 == 0:
            logger.info("frames received=%d", frames_seen)

        pcm = _frame_to_mono_pcm16(frame)
        frame_duration_ms = (len(pcm) / 2) / frame.sample_rate * 1000.0
        rms = _pcm16_rms(pcm)
        is_voiced = rms >= _SILENCE_RMS_THRESHOLD

        if not state.active:
            if not is_voiced:
                # Pure silence before any speech has started: nothing to
                # open a segment for yet.
                return frames_seen
            self._start_segment(state, rms)

        assert state.byte_queue is not None
        state.byte_queue.push(pcm)

        if is_voiced:
            state.had_voice = True
            state.silence_ms = 0.0
        elif state.had_voice:
            state.silence_ms += frame_duration_ms
            if state.silence_ms >= _SILENCE_DURATION_MS:
                await self._finish_segment(state)

        return frames_seen

    def _start_segment(self, state: _SegmentState, rms: float) -> None:
        state.byte_queue = _ByteQueue()
        state.consumer_task = asyncio.create_task(
            _drive_segment(
                self._client, state.byte_queue.__aiter__(), language=self._language
            )
        )
        logger.info("segment started (voice detected, rms=%.0f)", rms)
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
        )

    def _abandon_segment(self, state: _SegmentState) -> None:
        """End a segment that was opened but never crossed the voice
        threshold (e.g. input ended mid-attack) -- no transcript to emit.
        """
        assert state.byte_queue is not None
        state.byte_queue.end()
        if state.consumer_task is not None:
            state.consumer_task.cancel()
        state.byte_queue = None
        state.consumer_task = None

    async def _finish_segment(self, state: _SegmentState) -> None:
        assert state.byte_queue is not None
        assert state.consumer_task is not None
        state.byte_queue.end()
        try:
            text, request_id = await state.consumer_task
        except Exception:
            logger.warning("segment failed", exc_info=True)
            raise
        finally:
            state.byte_queue = None
            state.consumer_task = None
            state.silence_ms = 0.0
            state.had_voice = False

        logger.info("segment finished text=%r", text)
        if text.strip():
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[stt.SpeechData(language=self._language, text=text)],
                )
            )
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
        )
