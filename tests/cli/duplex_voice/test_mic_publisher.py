from __future__ import annotations

import asyncio

import pytest

from vibe.cli.duplex_voice.mic_publisher import forward_captured_audio


class _RecordingSource:
    def __init__(self) -> None:
        self.captured: list[bytes] = []

    async def capture_frame(self, frame) -> None:
        self.captured.append(bytes(frame.data))


@pytest.mark.asyncio
async def test_forward_captured_audio_forwards_frames_when_unmuted() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    source = _RecordingSource()
    for chunk in (b"\x01\x00" * 4, b"\x02\x00" * 4):
        queue.put_nowait(chunk)
    queue.put_nowait(None)

    await forward_captured_audio(queue, source, muted=lambda: False)

    assert source.captured == [b"\x01\x00" * 4, b"\x02\x00" * 4]


@pytest.mark.asyncio
async def test_forward_captured_audio_drops_frames_while_muted() -> None:
    """The core mute-enforcement requirement: while muted, captured audio
    must never reach `capture_frame` (i.e. never reach the room, and
    therefore never reach the STT plugin).
    """
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    source = _RecordingSource()
    for chunk in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        queue.put_nowait(chunk)
    queue.put_nowait(None)

    await forward_captured_audio(queue, source, muted=lambda: True)

    assert source.captured == []


@pytest.mark.asyncio
async def test_forward_captured_audio_resumes_after_unmute() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    source = _RecordingSource()
    muted = True

    def is_muted() -> bool:
        return muted

    queue.put_nowait(b"\x01\x00")  # dropped, muted
    queue.put_nowait(b"\x02\x00")  # dropped, muted

    async def _unmute_after_two() -> None:
        nonlocal muted
        # Let the two muted frames drain first, then unmute for the rest.
        while not queue.empty():
            await asyncio.sleep(0)
        muted = False
        queue.put_nowait(b"\x03\x00")
        queue.put_nowait(None)

    await asyncio.gather(
        forward_captured_audio(queue, source, muted=is_muted),
        _unmute_after_two(),
    )

    assert source.captured == [b"\x03\x00"]


@pytest.mark.asyncio
async def test_forward_captured_audio_stops_on_none_sentinel() -> None:
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    source = _RecordingSource()
    queue.put_nowait(None)

    # Would hang forever if the sentinel weren't honored -- the test
    # timeout (10s, this repo's pytest default) is the real assertion here.
    await forward_captured_audio(queue, source, muted=lambda: False)

    assert source.captured == []
