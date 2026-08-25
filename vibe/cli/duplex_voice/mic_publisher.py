"""Captures the operator's real microphone and publishes it into the duplex
voice room as a second, distinct participant.

Why a second participant rather than publishing from the agent's own room
connection: livekit-agents' `AgentSession` only ever transcribes REMOTE
participants' audio -- it does not (and should not) feed its own local
track back into its own STT. Something else has to be the "human" in the
room. `vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor` starts this
alongside the in-process agent task when it starts the duplex service.

Mute enforcement lives here, at the capture -> publish boundary: the mute
flag added in the mute-keybinding commit on this branch
(`VoiceManagerPort.muted`, plumbed through by
`VibeApp.action_toggle_mute`/`ctrl+t`) is read once per captured frame, and
a muted frame is dropped before it ever reaches `AudioSource.capture_frame`
-- so muted audio never reaches the room, and therefore never reaches the
STT plugin either, which only sees whatever's actually published into the
room.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import logging
from typing import Protocol

from livekit import rtc

from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings

logger = logging.getLogger("jarvis.duplex_voice.mic")

# sounddevice raises OSError on import when no audio driver is available --
# mirrors the exact guard `vibe.cli.audio_recorder.audio_recorder` uses, so
# duplex voice mode degrades the same way the existing push-to-talk voice
# manager does on a machine/container with no audio hardware, rather than
# crashing the whole toggle.
_SD_IMPORT_ERROR: OSError | None = None
try:
    import sounddevice as sd
except OSError as e:  # pragma: no cover - depends on host audio driver
    logger.warning("sounddevice unavailable, duplex mic capture disabled: %r", e)
    _SD_IMPORT_ERROR = e
    sd = None  # type: ignore[assignment]

_SAMPLE_RATE = 48_000
_CHANNELS = 1
_FRAME_MS = 20
_BLOCKSIZE = _SAMPLE_RATE * _FRAME_MS // 1000  # 960 samples/20ms @ 48kHz
_QUEUE_MAXSIZE = 50  # ~1s of audio at 20ms/frame; drop oldest beyond this


class MicPublisherError(RuntimeError):
    pass


class AudioSourceLike(Protocol):
    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...


def _frame_from_pcm16(data: bytes) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data,
        sample_rate=_SAMPLE_RATE,
        num_channels=_CHANNELS,
        samples_per_channel=len(data) // 2,
    )


async def forward_captured_audio(
    frame_queue: asyncio.Queue[bytes | None],
    source: AudioSourceLike,
    *,
    muted: Callable[[], bool],
) -> None:
    """Drain captured PCM16 chunks from `frame_queue` into `source`,
    dropping any chunk captured while `muted()` is true.

    Runs until a ``None`` sentinel is read from the queue. Kept separate
    from `MicPublisher` (which owns the real `sounddevice`/`rtc.Room`
    wiring) so the mute-suppression decision itself -- the part that
    matters for correctness -- is testable with a fake queue/source/muted
    callable and no real audio hardware or network.
    """
    while True:
        data = await frame_queue.get()
        if data is None:
            return
        if muted():
            continue
        await source.capture_frame(_frame_from_pcm16(data))


class MicPublisher:
    """Joins the duplex room as `settings.human_identity` and streams the
    real microphone into it via `forward_captured_audio`.
    """

    def __init__(
        self,
        settings: DuplexVoiceSettings,
        *,
        muted: Callable[[], bool],
    ) -> None:
        self._settings = settings
        self._muted = muted

    async def run(self) -> None:
        """Connect, publish, and capture until cancelled.

        Cancellation (the supervisor's normal stop path -- see
        `DuplexVoiceSupervisor._stop_mic`) propagates straight through the
        `await` inside `forward_captured_audio`; the `finally` below only
        runs cleanup, it does not swallow the `CancelledError`.
        """
        if sd is None:
            raise MicPublisherError(
                f"sounddevice unavailable: {_SD_IMPORT_ERROR!r}"
            )

        loop = asyncio.get_running_loop()
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )

        def _on_audio(indata: object, *_rest: object) -> None:
            data = bytes(indata)  # type: ignore[call-overload]
            loop.call_soon_threadsafe(_put_dropping_oldest, frame_queue, data)

        room = rtc.Room()
        jwt = self._settings.mint_token(
            identity=self._settings.human_identity,
            name=self._settings.human_identity,
        )
        logger.info(
            "mic publisher connecting identity=%s", self._settings.human_identity
        )
        await room.connect(self._settings.livekit_url, jwt)

        source = rtc.AudioSource(_SAMPLE_RATE, _CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("mic", source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(track, options)

        stream = sd.RawInputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="int16",
            blocksize=_BLOCKSIZE,
            callback=_on_audio,
        )
        stream.start()
        logger.info("mic capture started")

        try:
            await forward_captured_audio(frame_queue, source, muted=self._muted)
        finally:
            stream.stop()
            stream.close()
            with contextlib.suppress(Exception):
                await room.disconnect()
            logger.info("mic publisher stopped")


def _put_dropping_oldest(queue: asyncio.Queue[bytes | None], data: bytes) -> None:
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    queue.put_nowait(data)
