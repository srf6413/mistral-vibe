from __future__ import annotations

import asyncio

from livekit.agents import llm
import pytest

from vibe.cli.duplex_voice.jarvis_llm import JarvisBridgeLLM


class _FakeBridge:
    """Stands in for VoiceTurnBridge: records subscribe/unsubscribe and lets
    the test control exactly what a subscriber hears and when.
    """

    def __init__(self) -> None:
        self.handled: list[str] = []
        self.subscribed = 0
        self.unsubscribed = 0
        self._queue: asyncio.Queue[str | None] | None = None

    def subscribe(self) -> asyncio.Queue[str | None]:
        self.subscribed += 1
        self._queue = asyncio.Queue()
        return self._queue

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self.unsubscribed += 1

    async def handle_transcript(self, text: str) -> None:
        self.handled.append(text)
        assert self._queue is not None
        self._queue.put_nowait("hello ")
        self._queue.put_nowait("world")
        self._queue.put_nowait(None)


def _chat_ctx(user_text: str) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content=user_text)
    return ctx


async def _collect_text(stream: llm.LLMStream) -> str:
    text = ""
    async with stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    return text


@pytest.mark.asyncio
async def test_jarvis_bridge_llm_streams_bridge_output_as_chat_chunks() -> None:
    bridge = _FakeBridge()
    llm_plugin = JarvisBridgeLLM(bridge)  # type: ignore[arg-type]

    stream = llm_plugin.chat(chat_ctx=_chat_ctx("hey jarvis"))
    text = await _collect_text(stream)

    assert text == "hello world"
    assert bridge.handled == ["hey jarvis"]
    assert bridge.subscribed == 1
    assert bridge.unsubscribed == 1


@pytest.mark.asyncio
async def test_jarvis_bridge_llm_ignores_empty_user_text() -> None:
    bridge = _FakeBridge()
    llm_plugin = JarvisBridgeLLM(bridge)  # type: ignore[arg-type]

    stream = llm_plugin.chat(chat_ctx=llm.ChatContext.empty())
    text = await _collect_text(stream)

    assert text == ""
    assert bridge.handled == []
    # Never subscribed either -- nothing to dispatch, nothing to listen for.
    assert bridge.subscribed == 0


class _HangingBridge:
    """A bridge whose subscriber queue never receives anything -- used to
    prove that cancelling the LLMStream unsubscribes cleanly rather than
    hanging or raising into the bridge.
    """

    def __init__(self) -> None:
        self.unsubscribed_queue: asyncio.Queue | None = None

    def subscribe(self) -> asyncio.Queue[str | None]:
        return asyncio.Queue()

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self.unsubscribed_queue = queue

    async def handle_transcript(self, text: str) -> None:
        return None


@pytest.mark.asyncio
async def test_cancelling_the_stream_unsubscribes_without_touching_the_bridge() -> None:
    bridge = _HangingBridge()
    llm_plugin = JarvisBridgeLLM(bridge)  # type: ignore[arg-type]

    stream = llm_plugin.chat(chat_ctx=_chat_ctx("still talking"))
    task = asyncio.ensure_future(_collect_text(stream))
    await asyncio.sleep(0.05)  # let _run() reach `await queue.get()`
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert bridge.unsubscribed_queue is not None
