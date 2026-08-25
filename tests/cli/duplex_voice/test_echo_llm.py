from __future__ import annotations

from livekit.agents import llm
import pytest

from vibe.cli.duplex_voice.echo_llm import EchoLLM


async def _collect_text(stream: llm.LLMStream) -> str:
    text = ""
    async with stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    return text


@pytest.mark.asyncio
async def test_echo_llm_echoes_the_last_user_message() -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello there")

    echo = EchoLLM()
    try:
        stream = echo.chat(chat_ctx=chat_ctx)
        text = await _collect_text(stream)
    finally:
        await echo.aclose()

    assert text == "You said: hello there"


@pytest.mark.asyncio
async def test_echo_llm_ignores_assistant_messages_and_uses_the_latest_user_turn() -> (
    None
):
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="first")
    chat_ctx.add_message(role="assistant", content="You said: first")
    chat_ctx.add_message(role="user", content="second")

    echo = EchoLLM()
    try:
        text = await _collect_text(echo.chat(chat_ctx=chat_ctx))
    finally:
        await echo.aclose()

    assert text == "You said: second"


@pytest.mark.asyncio
async def test_echo_llm_handles_no_user_message() -> None:
    echo = EchoLLM()
    try:
        text = await _collect_text(echo.chat(chat_ctx=llm.ChatContext.empty()))
    finally:
        await echo.aclose()

    assert text == "I heard you, but caught no words."
