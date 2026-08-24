"""A trivial placeholder LLM for this stage.

Proves the STT -> room -> TTS duplex pipeline end to end without pulling in
a real model: it just echoes the last thing the user said back to them.
The real bridge to jarvis's own agent replaces *only* this piece in a later
stage -- everything else (STT plugin, TTS plugin, room wiring, supervisor)
stays as built here.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import shortuuid


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            return item.text_content or ""
    return ""


class EchoLLM(llm.LLM):
    """Replies with a fixed acknowledgement of whatever the user said."""

    @property
    def model(self) -> str:
        return "echo"

    @property
    def provider(self) -> str:
        return "jarvis-duplex-voice"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> EchoLLMStream:
        return EchoLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class EchoLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        heard = _last_user_text(self._chat_ctx)
        reply = f"You said: {heard}" if heard else "I heard you, but caught no words."
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=shortuuid("echo_"),
                delta=llm.ChoiceDelta(role="assistant", content=reply),
            )
        )
