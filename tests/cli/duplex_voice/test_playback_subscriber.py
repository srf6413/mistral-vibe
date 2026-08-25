from __future__ import annotations

import asyncio

import pytest

from livekit import rtc

from vibe.cli.duplex_voice import playback_subscriber as playback_subscriber_module
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.playback_subscriber import (
    PlaybackSubscriber,
    PlaybackSubscriberError,
    _SoundDeviceSink,
    forward_playback_audio,
    should_subscribe_to_track,
)


class _RecordingSink:
    def __init__(self) -> None:
        self.written: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.written.append(data)


@pytest.mark.asyncio
async def test_forward_playback_audio_writes_frames_in_order() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    sink = _RecordingSink()
    for chunk in (b"\x01\x00" * 4, b"\x02\x00" * 4):
        queue.put_nowait(chunk)
    queue.put_nowait(None)

    await forward_playback_audio(queue, sink)

    assert sink.written == [b"\x01\x00" * 4, b"\x02\x00" * 4]


@pytest.mark.asyncio
async def test_forward_playback_audio_stops_on_none_sentinel() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    sink = _RecordingSink()
    queue.put_nowait(None)

    # Would hang forever if the sentinel weren't honored -- the test
    # timeout (10s, this repo's pytest default) is the real assertion here,
    # same as mic_publisher's equivalent test.
    await forward_playback_audio(queue, sink)

    assert sink.written == []


@pytest.mark.asyncio
async def test_forward_playback_audio_stops_immediately_with_no_prior_frames() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    sink = _RecordingSink()

    async def _send_after_delay() -> None:
        await asyncio.sleep(0)
        queue.put_nowait(b"\x03\x00")
        queue.put_nowait(None)

    await asyncio.gather(
        forward_playback_audio(queue, sink),
        _send_after_delay(),
    )

    assert sink.written == [b"\x03\x00"]


class TestShouldSubscribeToTrack:
    def test_accepts_the_agents_own_audio_track(self) -> None:
        assert (
            should_subscribe_to_track(
                participant_identity="jarvis-duplex-agent",
                track_kind=rtc.TrackKind.KIND_AUDIO,
                agent_identity="jarvis-duplex-agent",
            )
            is True
        )

    def test_rejects_the_human_mic_publishers_track(self) -> None:
        """The core feedback-loop-avoidance requirement: never subscribe to
        (and therefore never play back) the mic publisher's own track --
        that would render the operator's own voice back to them.
        """
        assert (
            should_subscribe_to_track(
                participant_identity="jarvis-duplex-voice-user",
                track_kind=rtc.TrackKind.KIND_AUDIO,
                agent_identity="jarvis-duplex-agent",
            )
            is False
        )

    def test_rejects_a_non_audio_track_from_the_agent(self) -> None:
        assert (
            should_subscribe_to_track(
                participant_identity="jarvis-duplex-agent",
                track_kind=rtc.TrackKind.KIND_VIDEO,
                agent_identity="jarvis-duplex-agent",
            )
            is False
        )

    def test_rejects_an_unrelated_identity_entirely(self) -> None:
        assert (
            should_subscribe_to_track(
                participant_identity="some-other-participant",
                track_kind=rtc.TrackKind.KIND_AUDIO,
                agent_identity="jarvis-duplex-agent",
            )
            is False
        )


class _FakeRawOutputStream:
    """Stands in for `sounddevice.RawOutputStream`: records construction
    kwargs and every chunk written, without touching real audio hardware.
    """

    instances: list[_FakeRawOutputStream] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.written: list[bytes] = []
        _FakeRawOutputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class _FakeSoundDeviceModule:
    RawOutputStream = _FakeRawOutputStream


@pytest.fixture
def fake_sounddevice(monkeypatch) -> type[_FakeRawOutputStream]:
    _FakeRawOutputStream.instances = []
    monkeypatch.setattr(
        playback_subscriber_module, "sd", _FakeSoundDeviceModule
    )
    return _FakeRawOutputStream


@pytest.mark.asyncio
async def test_sounddevice_sink_opens_stream_matching_the_tts_output_format(
    fake_sounddevice,
) -> None:
    sink = _SoundDeviceSink()

    assert len(fake_sounddevice.instances) == 1
    stream = fake_sounddevice.instances[0]
    assert stream.kwargs["samplerate"] == 24000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.started is True

    sink.close()
    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_sounddevice_sink_write_forwards_bytes_to_the_stream(
    fake_sounddevice,
) -> None:
    sink = _SoundDeviceSink()
    stream = fake_sounddevice.instances[0]

    await sink.write(b"\x01\x02\x03\x04")

    assert stream.written == [b"\x01\x02\x03\x04"]
    sink.close()


@pytest.mark.asyncio
async def test_full_pipeline_frame_arrival_to_faked_speaker_output(
    fake_sounddevice,
) -> None:
    """End-to-end proof of the plumbing this task asks for: frames arriving
    off the (simulated) network reach the (faked) speaker output, in
    order, via the real `forward_playback_audio`/`_SoundDeviceSink` code
    -- no real LiveKit connection or audio hardware involved.
    """
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    sink = _SoundDeviceSink()
    stream = fake_sounddevice.instances[0]

    for chunk in (b"\xaa\xbb", b"\xcc\xdd", b"\xee\xff"):
        queue.put_nowait(chunk)
    queue.put_nowait(None)

    await forward_playback_audio(queue, sink)

    assert stream.written == [b"\xaa\xbb", b"\xcc\xdd", b"\xee\xff"]
    sink.close()


def test_sounddevice_sink_raises_clearly_when_sounddevice_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(playback_subscriber_module, "sd", None)
    monkeypatch.setattr(
        playback_subscriber_module,
        "_SD_IMPORT_ERROR",
        OSError("no audio driver"),
    )

    with pytest.raises(PlaybackSubscriberError, match="sounddevice unavailable"):
        _SoundDeviceSink()


# -- `PlaybackSubscriber.run()` orchestration, against a faked `rtc.Room`/
# `rtc.AudioStream` -- ----------------------------------------------------
#
# Goes beyond the pure-function tests above: proves `run()`'s own wiring
# (identity-filtering the `track_subscribed` callback, pumping a subscribed
# track's frames into the sink, cleaning up on cancellation) actually
# works end to end, without a real LiveKit connection or audio hardware.
# A real network round trip (a real `livekit-server`, a real published
# track actually carrying RTP over the wire to a real subscriber) is
# deliberately NOT exercised here -- this repo already draws that line at
# `scripts/duplex_voice_proof.py` (explicitly not a pytest test, since
# pytest's 10s per-test timeout in this repo is too tight for a real
# server + real audio flow to settle); see also
# `tests/cli/duplex_voice/test_supervisor.py`'s module docstring for the
# same reasoning re: `livekit-server` in tests.


class _FakeAudioFrame:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _FakeAudioFrameEvent:
    def __init__(self, frame: _FakeAudioFrame) -> None:
        self.frame = frame


class _FakeTrack:
    def __init__(self, kind: rtc.TrackKind.ValueType = rtc.TrackKind.KIND_AUDIO) -> None:
        self.kind = kind
        self.frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue()


class _FakeParticipant:
    def __init__(self, identity: str) -> None:
        self.identity = identity


class _FakeAudioStream:
    """Stands in for `rtc.AudioStream`: yields whatever PCM chunks the test
    pushes onto the fake track's own queue, ending iteration on a `None`
    sentinel (a real track never "ends" this way -- this is a
    test-only convenience for determinism, instead of relying on
    cancellation timing).
    """

    def __init__(self, *, track: _FakeTrack, **_kwargs) -> None:
        self._track = track
        self.closed = False

    def __aiter__(self) -> _FakeAudioStream:
        return self

    async def __anext__(self) -> _FakeAudioFrameEvent:
        data = await self._track.frame_queue.get()
        if data is None:
            raise StopAsyncIteration
        return _FakeAudioFrameEvent(_FakeAudioFrame(data))

    async def aclose(self) -> None:
        self.closed = True


class _FakeRoom:
    def __init__(self) -> None:
        self._handlers: dict[str, object] = {}
        self.connected_url: str | None = None
        self.connected_token: str | None = None
        self.disconnected = False

    def on(self, event: str, handler) -> None:
        self._handlers[event] = handler

    async def connect(self, url: str, token: str) -> None:
        self.connected_url = url
        self.connected_token = token

    async def disconnect(self) -> None:
        self.disconnected = True

    def fire_track_subscribed(self, track, publication, participant) -> None:
        self._handlers["track_subscribed"](track, publication, participant)


@pytest.fixture
def fake_room_and_stream(monkeypatch):
    room = _FakeRoom()
    monkeypatch.setattr(playback_subscriber_module.rtc, "Room", lambda: room)
    monkeypatch.setattr(playback_subscriber_module.rtc, "AudioStream", _FakeAudioStream)
    return room


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_run_plays_frames_from_the_agents_subscribed_track(
    fake_room_and_stream, fake_sounddevice
) -> None:
    room = fake_room_and_stream
    settings = DuplexVoiceSettings(
        room="test-room", agent_identity="the-agent", human_identity="the-human"
    )
    subscriber = PlaybackSubscriber(settings)
    run_task = asyncio.create_task(subscriber.run())

    await _wait_until(lambda: room.connected_url is not None)
    assert room.connected_url == settings.livekit_url

    agent_track = _FakeTrack()
    room.fire_track_subscribed(
        agent_track, object(), _FakeParticipant("the-agent")
    )

    agent_track.frame_queue.put_nowait(b"\x11\x22")
    agent_track.frame_queue.put_nowait(b"\x33\x44")

    stream = fake_sounddevice.instances[0]
    await _wait_until(lambda: stream.written == [b"\x11\x22", b"\x33\x44"])

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert room.disconnected is True
    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_run_never_plays_the_human_mic_publishers_own_track(
    fake_room_and_stream, fake_sounddevice
) -> None:
    """Regression coverage for the feedback-loop-avoidance requirement:
    frames from the mic publisher's own track must never reach the sink,
    even though the room auto-subscribes to every published track.
    """
    room = fake_room_and_stream
    settings = DuplexVoiceSettings(
        room="test-room", agent_identity="the-agent", human_identity="the-human"
    )
    subscriber = PlaybackSubscriber(settings)
    run_task = asyncio.create_task(subscriber.run())

    await _wait_until(lambda: room.connected_url is not None)

    human_track = _FakeTrack()
    room.fire_track_subscribed(
        human_track, object(), _FakeParticipant("the-human")
    )
    human_track.frame_queue.put_nowait(b"\xde\xad")

    agent_track = _FakeTrack()
    room.fire_track_subscribed(
        agent_track, object(), _FakeParticipant("the-agent")
    )
    agent_track.frame_queue.put_nowait(b"\xbe\xef")

    stream = fake_sounddevice.instances[0]
    await _wait_until(lambda: stream.written == [b"\xbe\xef"])

    # Give the (never-registered) human track's frame every chance to have
    # wrongly arrived before asserting it didn't.
    await asyncio.sleep(0.05)
    assert stream.written == [b"\xbe\xef"]

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
