"""Subscribes to the duplex agent's synthesized-speech track and plays it
through the local machine's speakers, as a third, distinct room
participant.

Why a third participant rather than tapping the TTS plugin directly: this
is a single-machine, in-process setup, not a real multi-participant call,
so nothing in `vibe.cli.duplex_voice.agent`'s `AgentSession`/`RoomIO`
wiring renders the agent's TTS audio anywhere on its own -- it only
publishes a `LocalAudioTrack` on the agent's own room participant (see
`MistralDuplexTTS`/`AgentSession.start(room=...)`, which wires a
`RoomIO` whose `_ParticipantAudioOutput._publish_track` publishes that
track and then awaits `wait_for_subscription()`). Before this module
existed, nothing in the room ever subscribed to it, so that await parked
forever and the synthesized audio went nowhere -- captured and pushed
into the room, never rendered. `PlaybackSubscriber` is what resolves
that subscription and actually renders the audio.

It must join under its own identity, not
`vibe.cli.duplex_voice.mic_publisher`'s `human_identity`: joining under
the human's own identity (or simply subscribing to every published
track) would also receive that participant's own microphone track and
play the operator's own voice back to them through their speakers -- a
feedback loop. This module filters strictly to tracks published by
`settings.agent_identity`.

Mirrors `mic_publisher.py`'s own split: `forward_playback_audio` is the
pure, directly-testable frame-delivery logic (drain a queue into
whatever plays it, stop on a `None` sentinel), kept free of any real
`sounddevice`/`rtc.Room` wiring so it's provable without real audio
hardware or a real LiveKit connection; `PlaybackSubscriber` owns that
real wiring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import logging
from typing import Protocol

from livekit import rtc

from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings

logger = logging.getLogger("jarvis.duplex_voice.playback")

# sounddevice raises OSError on import when no audio driver is available --
# mirrors the exact guard `mic_publisher.py` uses on the capture side, so
# duplex voice mode degrades the same way (playback simply stays
# unavailable) on a machine/container with no audio hardware, rather than
# crashing the whole toggle.
_SD_IMPORT_ERROR: OSError | None = None
try:
    import sounddevice as sd
except OSError as e:  # pragma: no cover - depends on host audio driver
    logger.warning("sounddevice unavailable, duplex playback disabled: %r", e)
    _SD_IMPORT_ERROR = e
    sd = None  # type: ignore[assignment]

# Matches `MistralDuplexTTS`'s declared output format (see
# `mistral_tts_plugin.py`) -- `rtc.AudioStream` resamples whatever the
# remote track actually carries to this fixed format for us, so the local
# `sounddevice` output stream can be opened once, up front, instead of
# negotiating a format per frame.
_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1
_FRAME_SIZE_MS = 20
_QUEUE_MAXSIZE = 50  # ~1s of audio at 20ms/frame; drop oldest beyond this


class PlaybackSubscriberError(RuntimeError):
    pass


class AudioSinkLike(Protocol):
    async def write(self, data: bytes) -> None: ...


def _pcm16_from_frame(frame: rtc.AudioFrame) -> bytes:
    return bytes(frame.data)


def should_subscribe_to_track(
    *,
    participant_identity: str,
    track_kind: rtc.TrackKind.ValueType,
    agent_identity: str,
) -> bool:
    """Whether a just-subscribed remote track should be rendered locally.

    ONLY the agent's own synthesized-speech track qualifies -- never the
    mic publisher's track (`vibe.cli.duplex_voice.mic_publisher`,
    published under `settings.human_identity`), which would play the
    operator's own voice back to them (see module docstring), and never a
    non-audio track. A standalone function (rather than inline in
    `PlaybackSubscriber`'s `track_subscribed` handler) so this filtering
    decision -- the part that matters for correctness -- is directly
    testable without a real room connection.
    """
    if participant_identity != agent_identity:
        return False
    return track_kind == rtc.TrackKind.KIND_AUDIO


async def forward_playback_audio(
    frame_queue: asyncio.Queue[bytes | None],
    sink: AudioSinkLike,
) -> None:
    """Drain received PCM16 chunks from `frame_queue` into `sink`, until a
    `None` sentinel is read.

    Kept separate from `PlaybackSubscriber` (which owns the real
    `sounddevice`/`rtc.Room` wiring) so the delivery logic itself -- the
    part that matters for correctness -- is testable with a fake queue and
    a fake sink recording what would have been played, no real audio
    hardware or network involved. Mirrors
    `mic_publisher.forward_captured_audio` exactly in shape.
    """
    while True:
        data = await frame_queue.get()
        if data is None:
            return
        await sink.write(data)


class _SoundDeviceSink:
    """Wraps a single, lazily-opened `sounddevice.RawOutputStream` behind
    the `AudioSinkLike` protocol.

    `RawOutputStream.write()` blocks until the OS has consumed the buffer,
    so it's run in a worker thread (`asyncio.to_thread`) rather than
    awaited directly -- otherwise every write would stall this process's
    single asyncio event loop, including the STT/LLM/TTS pipeline sharing
    it.
    """

    def __init__(self) -> None:
        if sd is None:
            raise PlaybackSubscriberError(
                f"sounddevice unavailable: {_SD_IMPORT_ERROR!r}"
            )
        self._stream = sd.RawOutputStream(
            samplerate=_SAMPLE_RATE,
            channels=_NUM_CHANNELS,
            dtype="int16",
        )
        self._stream.start()

    async def write(self, data: bytes) -> None:
        await asyncio.to_thread(self._stream.write, data)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()


class PlaybackSubscriber:
    """Joins the duplex room as `settings.listener_identity`, subscribes
    ONLY to the track(s) published by `settings.agent_identity`, and plays
    the received audio through local speakers via `forward_playback_audio`.
    """

    def __init__(
        self,
        settings: DuplexVoiceSettings,
        *,
        sink_factory: Callable[[], AudioSinkLike] | None = None,
    ) -> None:
        self._settings = settings
        # Overridable for tests -- defaults to the real sounddevice-backed
        # sink so production callers don't need to know this exists.
        self._sink_factory = sink_factory or _SoundDeviceSink

    async def run(self) -> None:
        """Connect, subscribe, and play until cancelled.

        Cancellation (the supervisor's normal stop path -- see
        `DuplexVoiceSupervisor._stop_playback`) propagates straight through
        the `await` inside `forward_playback_audio`; the `finally` below
        only runs cleanup, it does not swallow the `CancelledError`.
        """
        # Built before connecting to the room (mirrors `MicPublisher.run()`
        # checking `sd is None` up front) so a machine with no working
        # speakers fails fast without leaving an orphaned room connection
        # behind for the `finally` below to never reach.
        sink = self._sink_factory()

        loop = asyncio.get_running_loop()
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        stream_tasks: set[asyncio.Task[None]] = set()

        def _on_track_subscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if not should_subscribe_to_track(
                participant_identity=participant.identity,
                track_kind=track.kind,
                agent_identity=self._settings.agent_identity,
            ):
                return
            logger.info(
                "subscribed to agent audio track participant=%s",
                participant.identity,
            )
            task = loop.create_task(_pump_track(track))
            stream_tasks.add(task)
            task.add_done_callback(stream_tasks.discard)

        async def _pump_track(track: rtc.Track) -> None:
            audio_stream = rtc.AudioStream(
                track=track,
                sample_rate=_SAMPLE_RATE,
                num_channels=_NUM_CHANNELS,
                frame_size_ms=_FRAME_SIZE_MS,
            )
            try:
                async for event in audio_stream:
                    _put_dropping_oldest(
                        frame_queue, _pcm16_from_frame(event.frame)
                    )
            finally:
                await audio_stream.aclose()

        room = rtc.Room()
        # Registered before connect() so a track published (by the agent)
        # before this listener joins the room still fires the handler --
        # `auto_subscribe` (the room's default) then subscribes to it as
        # soon as the room becomes aware of it.
        room.on("track_subscribed", _on_track_subscribed)

        jwt = self._settings.mint_token(
            identity=self._settings.listener_identity,
            name=self._settings.listener_identity,
        )
        logger.info(
            "playback subscriber connecting identity=%s",
            self._settings.listener_identity,
        )
        await room.connect(self._settings.livekit_url, jwt)

        try:
            await forward_playback_audio(frame_queue, sink)
        finally:
            for task in list(stream_tasks):
                task.cancel()
            for task in list(stream_tasks):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            close = getattr(sink, "close", None)
            if close is not None:
                close()
            with contextlib.suppress(Exception):
                await room.disconnect()
            logger.info("playback subscriber stopped")


def _put_dropping_oldest(queue: asyncio.Queue[bytes | None], data: bytes) -> None:
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(data)
