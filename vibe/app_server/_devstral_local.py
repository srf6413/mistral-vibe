"""A local, dependency-light Ollama chat backend for the /model devstral option.

This is a second, independent turn-execution path alongside
``HttpFounderOSAskTransport`` in ``_founderos_ask.py``: it talks to a locally
running Ollama server (``ollama serve``, native API on 127.0.0.1:11434 by
default -- NOT the OpenAI-compatible shim) instead of the FounderOS /ask HTTP
boundary, so a turn on this backend never touches the Compute Budget router
at all.

Confirmed live against a real running Ollama server (2026-08-24, Ollama
0.32.5) rather than assumed from documentation:

  POST /api/chat
  {"model": "<name>", "messages": [{"role": "user", "content": "..."}], "stream": true}

streams newline-delimited JSON (bare JSON objects per line -- NOT
"data: ..." SSE framing), each line shaped like::

    {"model": "...", "created_at": "...",
     "message": {"role": "assistant", "content": "<delta>"}, "done": false}

with a final line adding ``"done": true`` plus generation stats
(``done_reason``, ``total_duration``, ``eval_count``, ...). A rejected
request (e.g. an unknown model name) returns a single, non-streamed JSON
object -- ``{"error": "model '...' not found"}`` -- with a non-2xx status
before any streaming begins.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
import json
from typing import Any, Protocol

import httpx

from vibe.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    TurnCompleted,
    TurnStarted,
)
from vibe.app_server.models import (
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicTurn,
    PublicTurnStatus,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# The founder's personalized fine-tune, per the founder's explicit choice --
# not the base devstral:24b -- confirmed present via `ollama list` on this
# Mac alongside its base model.
DEFAULT_DEVSTRAL_MODEL = "devstral-ysf:latest"


def history_to_ollama_messages(
    history: Iterable[PublicHistoryEntry],
) -> list[dict[str, str]]:
    """Translate session history into Ollama's ``messages`` array, in order.

    Only message entries carry a role Ollama understands (system/user/
    assistant); every other history entry type (shell effects, reasoning,
    notices, ...) has no Ollama-side analog and is skipped rather than
    guessed at.
    """
    messages: list[dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, PublicMessageEntry):
            continue
        text = entry.text
        if not text:
            continue
        messages.append({"role": entry.role, "content": text})
    return messages


class OllamaChatTransport(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        """Stream assistant content deltas for exactly one Ollama chat turn."""
        ...

    async def cancel(self) -> None:
        """Stop delivery for the active request without guessing server control."""
        ...

    async def close(self) -> None: ...


class HttpOllamaChatTransport:
    """One-endpoint, no-retry client for Ollama's native ``/api/chat``.

    Mirrors ``HttpFounderOSAskTransport``'s shape (single active request,
    lock-guarded, explicit connect-vs-mid-stream failure classification) so
    the two local backends fail in a consistent, already-reviewed way.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_DEVSTRAL_MODEL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=30.0, pool=3.0),
            trust_env=False,
        )
        self._owns_client = client is None
        self._active_response: httpx.Response | None = None
        self._active_lock = asyncio.Lock()
        self._closed = False

    def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        return self._stream(messages)

    async def _stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        # Imported lazily (not at module import time) to avoid a circular
        # import: _founderos_ask imports this module at the top of the file
        # to wire up the devstral backend, so this module must not import
        # back from _founderos_ask until it actually runs.
        from vibe.app_server._founderos_ask import (
            FounderOSAskError,
            FounderOSAskStreamError,
            FounderOSAskUnavailableError,
        )

        if self._closed:
            raise FounderOSAskError("Local Ollama transport is closed")
        async with self._active_lock:
            if self._active_response is not None:
                raise FounderOSAskError("A local Ollama /api/chat request is already active")
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": True},
                ) as response:
                    self._active_response = response
                    if not response.is_success:
                        detail = await _error_detail(response)
                        raise FounderOSAskStreamError(
                            "Local Ollama server rejected the /api/chat request "
                            f"(HTTP {response.status_code}){detail}"
                        )
                    async for content in _iter_ollama_content_deltas(response):
                        yield content
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                raise FounderOSAskUnavailableError(
                    f"The local Ollama server isn't running at {self.base_url}. "
                    "Start it with `ollama serve` (or open the Ollama app), then "
                    "switch back to the devstral model."
                ) from exc
            except (
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                raise FounderOSAskUnavailableError(
                    f"Connection to the local Ollama server at {self.base_url} was lost"
                ) from exc
            finally:
                self._active_response = None

    async def cancel(self) -> None:
        response = self._active_response
        if response is not None:
            await response.aclose()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.cancel()
        if self._owns_client:
            await self._client.aclose()


async def _iter_ollama_content_deltas(
    response: httpx.Response,
) -> AsyncGenerator[str, None]:
    """Decode one ndjson-lines Ollama /api/chat response into content deltas.

    Mirrors _iter_sse_events/_decode_sse_data's split in _founderos_ask.py:
    line framing lives here, event interpretation lives in _decode_ollama_line.
    """
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        if content := _decode_ollama_line(line):
            yield content


def _decode_ollama_line(line: str) -> str:
    # Lazy for the same reason as _stream() above.
    from vibe.app_server._founderos_ask import FounderOSAskStreamError

    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise FounderOSAskStreamError("Local Ollama /api/chat sent invalid JSON") from exc
    if not isinstance(event, dict):
        raise FounderOSAskStreamError("Local Ollama /api/chat sent a non-object line")
    if error := event.get("error"):
        raise FounderOSAskStreamError(f"Local Ollama /api/chat error: {error}")
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


async def _error_detail(response: httpx.Response) -> str:
    try:
        body: Any = json.loads(await response.aread())
    except Exception:
        return ""
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return f": {body['error']}"
    return ""


async def stream_devstral_turn(
    transport: OllamaChatTransport,
    *,
    history: list[PublicHistoryEntry],
    session_id: str,
    turn_id: str,
    started_at: int,
) -> AsyncGenerator[AppServerEvent, None]:
    """Run one devstral turn, yielding the same AppServerEvent shape as the
    FounderOS /ask HTTP path (TurnStarted -> assistant deltas -> TurnCompleted)
    so every existing renderer works unchanged regardless of backend.

    ``history`` must already include the current turn's user message --
    the caller (``FounderOSAskSession.act``) owns appending it, same as it
    owns applying the HistoryEntryAdded/HistoryEntryUpdated events this
    yields back onto session history.
    """
    # Lazy for the same reason as above: these helpers live in the module
    # that imports this one.
    from vibe.app_server._founderos_ask import (
        FounderOSAskStreamError,
        _assistant_with_delta,
        _complete_assistant,
        _now_ms,
    )
    from vibe.app_server._patch import make_json_patch

    turn = PublicTurn(
        id=turn_id,
        session_id=session_id,
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=started_at,
    )
    yield TurnStarted(turn)

    messages = history_to_ollama_messages(history)
    assistant: PublicMessageEntry | None = None
    async for delta in transport.stream(messages):
        previous = assistant
        assistant = _assistant_with_delta(
            assistant,
            delta,
            session_id=session_id,
            turn_id=turn_id,
            created_at=started_at,
        )
        if previous is None:
            yield HistoryEntryAdded(assistant)
        else:
            yield HistoryEntryUpdated(
                previous=previous,
                entry=assistant,
                patch=make_json_patch(
                    previous.model_dump(mode="json", by_alias=True),
                    assistant.model_dump(mode="json", by_alias=True),
                    append_paths={"/content/0/text"},
                ),
            )

    if assistant is None:
        raise FounderOSAskStreamError(
            "Local Ollama /api/chat produced no assistant text"
        )

    previous = assistant
    assistant = _complete_assistant(
        assistant,
        assistant.text,
        session_id=session_id,
        turn_id=turn_id,
        created_at=started_at,
    )
    yield HistoryEntryUpdated(
        previous=previous,
        entry=assistant,
        patch=make_json_patch(
            previous.model_dump(mode="json", by_alias=True),
            assistant.model_dump(mode="json", by_alias=True),
            append_paths={"/content/0/text"},
        ),
    )
    completed = turn.model_copy(
        update={"status": PublicTurnStatus.COMPLETED, "completed_at": _now_ms()}
    )
    yield TurnCompleted(completed)
