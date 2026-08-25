"""LiveKit `llm.LLM` plugin that bridges duplex-voice transcripts into
jarvis's real, already-running coding-agent turn -- replacing `EchoLLM` in
the real (non-standalone) wiring.

This is the ONLY piece `vibe.cli.duplex_voice.agent`'s standalone
proof-of-concept swaps out for real TUI usage (see that module's
docstring); the STT plugin, TTS plugin, room wiring, and supervisor all
stay exactly as built in the prior stage.

See `vibe.cli.duplex_voice.agent_bridge` for why this stream never awaits
`AppServerSession.act()`'s event generator directly: an `LLMStream` gets
cancelled by the livekit-agents framework on user barge-in and on session
teardown, and letting that cancellation reach `act()`'s consumer would
call `interrupt()` on the REAL coding-agent turn. This stream only ever
reads from a `VoiceTurnBridge` subscription queue, so cancelling it just
drops the subscription -- the real turn (and the on-screen rendering /
CallbackRequested handling the keyboard path already does for it, via
`VibeApp._handle_turn_event`) is completely unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import shortuuid

if TYPE_CHECKING:
    from vibe.cli.duplex_voice.agent_bridge import VoiceTurnBridge


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            return item.text_content or ""
    return ""


class JarvisBridgeLLM(llm.LLM):
    """Feeds each finalized voice transcript into a `VoiceTurnBridge`
    instead of calling out to any model of its own.
    """

    def __init__(self, bridge: VoiceTurnBridge) -> None:
        super().__init__()
        self._bridge = bridge

    @property
    def model(self) -> str:
        return "jarvis-agent-bridge"

    @property
    def provider(self) -> str:
        return "jarvis"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> JarvisBridgeLLMStream:
        return JarvisBridgeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            bridge=self._bridge,
        )


class JarvisBridgeLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: JarvisBridgeLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        bridge: VoiceTurnBridge,
    ) -> None:
        super().__init__(
            llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._bridge = bridge

    async def _run(self) -> None:
        text = _last_user_text(self._chat_ctx)
        if not text:
            return
        queue = self._bridge.subscribe()
        try:
            await self._bridge.handle_transcript(text)
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=shortuuid("jarvis_"),
                        delta=llm.ChoiceDelta(role="assistant", content=chunk),
                    )
                )
        finally:
            # Safe on cancellation too -- see module docstring. This never
            # touches the real turn, only this stream's own subscription.
            self._bridge.unsubscribe(queue)
