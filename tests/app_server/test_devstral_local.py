from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import cast

import httpx
import pytest

from vibe.app_server._devstral_local import (
    DEFAULT_DEVSTRAL_MODEL,
    HttpOllamaChatTransport,
    history_to_ollama_messages,
    stream_devstral_turn,
)
from vibe.app_server._founderos_ask import (
    FounderOSAskStreamError,
    FounderOSAskUnavailableError,
)
from vibe.app_server._shell import shell_effect_detail
from vibe.app_server.events import (
    HistoryEntryAdded,
    HistoryEntryUpdated,
    TurnCompleted,
    TurnStarted,
)
from vibe.app_server.models import (
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    RunningEffectState,
    TextContentBlock,
)


def _message(
    role: str, text: str, *, entry_id: str = "m", turn_id: str | None = None
) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=entry_id,
        session_id="s1",
        turn_id=turn_id,
        created_at=0,
        updated_at=0,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role=cast(str, role),
        content=[TextContentBlock(text=text)],
        source="turn_start" if role == "user" else "harness",
    )


def test_history_to_ollama_messages_preserves_order_and_role() -> None:
    history: list[PublicHistoryEntry] = [
        _message("user", "hi", entry_id="u1"),
        _message("assistant", "hello", entry_id="a1"),
        _message("user", "how are you", entry_id="u2"),
    ]

    assert history_to_ollama_messages(history) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"},
    ]


def test_history_to_ollama_messages_skips_non_message_and_empty_entries() -> None:
    effect = PublicEffectEntry(
        id="e1",
        session_id="s1",
        created_at=0,
        updated_at=0,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="shell",
        detail=shell_effect_detail("ls"),
        state=RunningEffectState(),
    )
    history: list[PublicHistoryEntry] = [
        _message("user", "", entry_id="empty"),
        cast(PublicHistoryEntry, effect),
        _message("user", "real", entry_id="u1"),
    ]

    assert history_to_ollama_messages(history) == [{"role": "user", "content": "real"}]


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_http_ollama_transport_parses_ndjson_deltas_and_stops_at_done() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            stream=_ChunkedStream([
                b'{"model":"devstral-ysf:latest","message":{"role":"assistant","content":"P"},"done":false}\n',
                b'{"model":"devstral-ysf:latest","message":{"role":"assistant","content":"ONG"},"done":false}\n',
                b'{"model":"devstral-ysf:latest","message":{"role":"assistant","content":""},'
                b'"done":true,"done_reason":"stop"}\n',
            ]),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpOllamaChatTransport(client=client)

    deltas = [delta async for delta in transport.stream([{"role": "user", "content": "hi"}])]

    assert deltas == ["P", "ONG"]
    assert len(requests) == 1
    assert requests[0].url == "http://127.0.0.1:11434/api/chat"
    await transport.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_http_ollama_transport_posts_full_history_and_model_name() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            stream=_ChunkedStream([b'{"message":{"role":"assistant","content":"ok"},"done":true}\n']),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpOllamaChatTransport(model="devstral-ysf:latest", client=client)

    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    _ = [delta async for delta in transport.stream(messages)]

    assert captured["json"] == {
        "model": "devstral-ysf:latest",
        "messages": messages,
        "stream": True,
    }
    await transport.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_http_ollama_transport_reports_unreachable_server_plainly() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpOllamaChatTransport(client=client)

    with pytest.raises(FounderOSAskUnavailableError, match="ollama serve"):
        _ = [delta async for delta in transport.stream([{"role": "user", "content": "hi"}])]

    await transport.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_http_ollama_transport_surfaces_rejected_model_as_stream_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpOllamaChatTransport(client=client)

    with pytest.raises(FounderOSAskStreamError, match="model 'nope' not found"):
        _ = [delta async for delta in transport.stream([{"role": "user", "content": "hi"}])]

    await transport.close()
    await client.aclose()


class _FakeOllamaChatTransport:
    def __init__(self, deltas: list[str] | None = None) -> None:
        self._deltas = deltas or []
        self.calls: list[list[dict[str, str]]] = []
        self.cancel_count = 0
        self.close_count = 0

    def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        self.calls.append(messages)

        async def generate() -> AsyncGenerator[str, None]:
            for delta in self._deltas:
                yield delta

        return generate()

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_stream_devstral_turn_yields_turn_started_deltas_then_completed() -> None:
    transport = _FakeOllamaChatTransport(["Hel", "lo"])
    history: list[PublicHistoryEntry] = [_message("user", "hi", entry_id="u1")]

    events = [
        event
        async for event in stream_devstral_turn(
            transport,
            history=history,
            session_id="s1",
            turn_id="turn-1",
            started_at=1000,
        )
    ]

    assert isinstance(events[0], TurnStarted)
    assert events[0].turn.id == "turn-1"
    assert isinstance(events[1], HistoryEntryAdded)
    assert isinstance(events[1].entry, PublicMessageEntry)
    assert events[1].entry.text == "Hel"
    assert isinstance(events[2], HistoryEntryUpdated)
    assert events[2].entry.text == "Hello"
    # The final delta-driven update is immediately followed by a completion
    # update (generation_status -> COMPLETED) and then TurnCompleted -- same
    # two-step finish shape as the FounderOS HTTP path's final_result.
    assert isinstance(events[3], HistoryEntryUpdated)
    assert events[3].entry.text == "Hello"
    assert events[3].entry.generation_status == PublicEntryGenerationStatus.COMPLETED
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].turn.status == "completed"
    assert transport.calls == [[{"role": "user", "content": "hi"}]]


@pytest.mark.asyncio
async def test_stream_devstral_turn_raises_on_empty_reply() -> None:
    transport = _FakeOllamaChatTransport([])
    history: list[PublicHistoryEntry] = [_message("user", "hi", entry_id="u1")]

    with pytest.raises(FounderOSAskStreamError, match="no assistant text"):
        _ = [
            event
            async for event in stream_devstral_turn(
                transport,
                history=history,
                session_id="s1",
                turn_id="turn-1",
                started_at=1000,
            )
        ]


def test_default_devstral_model_is_the_founders_finetune() -> None:
    assert DEFAULT_DEVSTRAL_MODEL == "devstral-ysf:latest"
