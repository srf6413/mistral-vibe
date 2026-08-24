from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol, cast
from uuid import uuid4

import httpx

from vibe.agents import AgentSafety, AgentType
from vibe.app_server._patch import make_json_patch
from vibe.app_server._shell import (
    ShellController,
    shell_effect_detail,
    shell_effect_error,
    shell_effect_state,
)
from vibe.app_server.client_state import ClientBootstrap, ClientSessionState
from vibe.app_server.config import (
    AudioProviderView,
    ConfigView,
    ModelConfigView,
    SpeechConfigView,
    TranscribeModelConfigView,
    TranscriptionConfigView,
    TTSModelConfigView,
)
from vibe.app_server.connection import AppServerResourceConnection
from vibe.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    ServerWarning,
    TurnCompleted,
    TurnStarted,
)
from vibe.app_server.models import (
    AccountStatus,
    AccountView,
    AgentStatsSnapshot,
    AgentSummary,
    CallbackOutput,
    ConnectorCounts,
    IdleSessionStatus,
    ImageAttachment,
    JsonPatchOperation,
    MCPState,
    MentionStats,
    PreparedPrompt,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicError,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    PublicTurnStatus,
    RunningEffectState,
    SessionLogSummary,
    TextContentBlock,
    TokenUsage,
    UserDisplayContent,
)
from vibe.app_server.protocol import (
    AccountReadResponse,
    ConfigFieldKind,
    ConfigFieldsReadResponse,
    ConfigFieldWire,
    ConfigLayerValueWire,
    ConfigMutationResponse,
    EmptyResponse,
    FeedbackShouldShowResponse,
    IdentityReadResponse,
    LoopsListResponse,
    NarrationSummarizeResponse,
    RuntimeReadResponse,
    RuntimeSnapshot,
    ServerWarningParams,
    SessionReadyWaitResponse,
    ShellRunParams,
    ShellRunResponse,
    WorkspacePromptPrepareResponse,
    WorkspaceTrustStatusResponse,
    WorkspaceUntrustedConfigResponse,
)
from vibe.app_server.resources import AppServerResources
from vibe.app_server.session import SessionExitSummary
from vibe.core.utils.sse import iter_sse_lines
from vibe.user_content import UserResource

DEFAULT_FOUNDEROS_ASK_URL = "http://127.0.0.1:8000/ask"
DEFAULT_FOUNDEROS_WORKSPACE = "ysf"


class FounderOSAskError(RuntimeError):
    """A fail-closed error from the local FounderOS /ask boundary."""


class FounderOSAskUnavailableError(FounderOSAskError):
    """The configured local FounderOS service could not be reached."""


class FounderOSAskStreamError(FounderOSAskError):
    """The local service returned an invalid or unsuccessful stream."""


@dataclass(frozen=True, slots=True)
class AskPins:
    """Independent compute pins accepted by the FounderOS /ask router."""

    intake_model: str = "auto"
    worker_model: str = "auto"

    def __post_init__(self) -> None:
        if not self.intake_model.strip() or not self.worker_model.strip():
            raise ValueError("FounderOS compute pins cannot be empty")

    def ask_fields(self) -> dict[str, str]:
        return {"intake_model": self.intake_model, "worker_model": self.worker_model}


class AskTransport(Protocol):
    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream decoded FounderOS /ask events for exactly one request."""
        ...

    async def cancel(self) -> None:
        """Stop delivery for the active request without guessing server control."""
        ...

    async def close(self) -> None: ...


class HttpFounderOSAskTransport:
    """One-endpoint, no-retry HTTP/SSE transport for FounderOS /ask."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_FOUNDEROS_ASK_URL,
        workspace: str = DEFAULT_FOUNDEROS_WORKSPACE,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        _require_safe_endpoint(endpoint)
        self.endpoint = endpoint
        self.workspace = workspace
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=30.0, pool=3.0),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._active_response: httpx.Response | None = None
        self._active_lock = asyncio.Lock()
        self._closed = False

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        return self._stream(payload)

    async def _stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._closed:
            raise FounderOSAskError("FounderOS /ask transport is closed")
        async with self._active_lock:
            if self._active_response is not None:
                raise FounderOSAskError("A FounderOS /ask request is already active")
            headers = {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            try:
                async with self._client.stream(
                    "POST",
                    self.endpoint,
                    params={"workspace": self.workspace},
                    headers=headers,
                    json=dict(payload),
                    follow_redirects=False,
                ) as response:
                    self._active_response = response
                    if not response.is_success:
                        raise FounderOSAskStreamError(
                            "Local FounderOS /ask rejected the request "
                            f"(HTTP {response.status_code}); no provider fallback attempted"
                        )
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type.lower():
                        raise FounderOSAskStreamError(
                            "Local FounderOS /ask did not return an SSE stream; "
                            "no provider fallback attempted"
                        )
                    async for event in _iter_sse_events(response):
                        yield event
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                raise FounderOSAskUnavailableError(
                    f"Local FounderOS /ask is unavailable at {self.endpoint}; "
                    "no provider fallback attempted"
                ) from exc
            except (
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                raise FounderOSAskUnavailableError(
                    f"Connection to local FounderOS /ask at {self.endpoint} was lost; "
                    "no provider fallback attempted"
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


async def _iter_sse_events(
    response: httpx.Response,
) -> AsyncGenerator[dict[str, Any], None]:
    data_lines: list[str] = []
    async for line in iter_sse_lines(response):
        if line == "":
            if event := _decode_sse_data(data_lines):
                yield event
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    if event := _decode_sse_data(data_lines):
        yield event


def _decode_sse_data(lines: list[str]) -> dict[str, Any] | None:
    if not lines:
        return None
    payload = "\n".join(lines)
    if not payload or payload == "[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FounderOSAskStreamError(
            "Local FounderOS /ask sent invalid SSE JSON"
        ) from exc
    if not isinstance(value, dict):
        raise FounderOSAskStreamError(
            "Local FounderOS /ask sent a non-object SSE event"
        )
    return value


def _require_safe_endpoint(endpoint: str) -> None:
    try:
        url = httpx.URL(endpoint)
    except Exception as exc:
        raise ValueError(f"Invalid FounderOS /ask endpoint: {endpoint}") from exc
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("FounderOS /ask endpoint must be an HTTP(S) URL")
    loopback = url.host in {"127.0.0.1", "localhost", "::1"}
    if url.scheme == "http" and not loopback:
        raise ValueError("Non-loopback FounderOS /ask endpoints must use HTTPS")


@dataclass(frozen=True, slots=True)
class _EventsClosed:
    pass


_EVENTS_CLOSED = _EventsClosed()


class FounderOSAskSession:
    """Translate local FounderOS /ask SSE into Vibe's public session events."""

    def __init__(
        self,
        *,
        transport: AskTransport,
        cwd: Path,
        session_id: str | None = None,
        pins: AskPins | None = None,
        shell_controller: ShellController | None = None,
    ) -> None:
        self._transport = transport
        self._cwd = str(cwd.expanduser().resolve())
        self._session_id = session_id or f"vibe-{uuid4()}"
        self.pins = pins or AskPins()
        self._history: list[PublicHistoryEntry] = []
        self._pending_user_context: list[str] = []
        self._turn_active = False
        self._active_turn_id: str | None = None
        self._interrupted_turn_ids: set[str] = set()
        self._events: asyncio.Queue[AppServerEvent | _EventsClosed] = asyncio.Queue(
            maxsize=64
        )
        self._closed = False
        self.resources, self._resource_state, self._resource_client = _build_resources(
            session_id=self._session_id,
            cwd=self._cwd,
            shell_controller=shell_controller,
        )

    @classmethod
    def local(
        cls,
        *,
        cwd: Path,
        session_id: str | None = None,
        pins: AskPins | None = None,
        endpoint: str | None = None,
        workspace: str | None = None,
    ) -> FounderOSAskSession:
        selected_pins = pins or AskPins(
            intake_model=os.environ.get("FOUNDEROS_INTAKE_MODEL", "auto"),
            worker_model=os.environ.get("FOUNDEROS_WORKER_MODEL", "auto"),
        )
        return cls(
            transport=HttpFounderOSAskTransport(
                endpoint=endpoint
                or os.environ.get("FOUNDEROS_ASK_URL", DEFAULT_FOUNDEROS_ASK_URL),
                workspace=workspace
                or os.environ.get("FOUNDEROS_WORKSPACE", DEFAULT_FOUNDEROS_WORKSPACE),
                api_key=os.environ.get("FOUNDEROS_API_KEY"),
            ),
            cwd=cwd,
            session_id=session_id,
            pins=selected_pins,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def history(self) -> list[PublicHistoryEntry]:
        return self._history

    @property
    def turn_active(self) -> bool:
        return self._turn_active

    def exit_summary(self) -> SessionExitSummary:
        return SessionExitSummary(session_id=self._session_id, usage=TokenUsage())

    async def act(  # noqa: PLR0912, PLR0915 - linear protocol state machine
        self,
        message: str,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        resources: list[UserResource] | None = None,
        user_display_content: UserDisplayContent | None = None,
        mention_stats: MentionStats | None = None,
        injected: bool = False,
        require_ack: bool | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[AppServerEvent, None]:
        del auto_title, user_display_content, mention_stats, injected
        if self._closed:
            raise FounderOSAskError("FounderOS /ask session is closed")
        if self._turn_active:
            raise FounderOSAskError("A FounderOS /ask turn is already running")
        if not message.strip():
            raise FounderOSAskError("FounderOS /ask requires a non-empty message")
        if images or resources:
            raise FounderOSAskError(
                "Attachments are not yet supported by the local /ask adapter"
            )

        started_at = _now_ms()
        turn_id = f"turn-{uuid4()}"
        turn = PublicTurn(
            id=turn_id,
            session_id=self._session_id,
            status=PublicTurnStatus.IN_PROGRESS,
            started_at=started_at,
        )
        user_entry = PublicMessageEntry(
            id=client_message_id or f"user-{uuid4()}",
            session_id=self._session_id,
            turn_id=turn_id,
            created_at=started_at,
            updated_at=started_at,
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            role="user",
            content=[TextContentBlock(text=message)],
            source="turn_start",
        )
        self._history.append(user_entry)
        self._turn_active = True
        self._active_turn_id = turn_id
        assistant: PublicMessageEntry | None = None
        terminal_seen = False
        request_text = "\n\n".join([*self._pending_user_context, message])
        self._pending_user_context.clear()
        payload: dict[str, object] = {
            "text": request_text,
            "stream": True,
            "ask_mode": "chat",
            "frontend_session_id": self._session_id,
            **self.pins.ask_fields(),
        }
        # Compute Budget (decision_ref ask-compute-budget-dispatch-v1) routes every
        # /ask turn through engine selection by default. Two independent gates
        # both need clearing to reach the real in-process reasoning loop instead of
        # a backgrounded dispatch or a "Parked /ask" stub -- confirmed empirically
        # against the live dev tip, not just read from source:
        #   1) require_ack=True trips is_founder_bound_work() -> gate_action
        #      "human_surface" instead of "direct_forward". Deliberately NOT
        #      work_class="chat": that narrows the provider catalog to the single
        #      never-auto manus_delegate candidate, and when it isn't reachable
        #      decide_provider_route() resolves no provider at all, so ARC start
        #      itself raises ("requires a resolved Compute Budget provider") before
        #      the gate is ever reached. require_ack leaves the catalog alone (a
        #      real reachable engine like grok_remote still gets picked for ARC's
        #      own bookkeeping) while still avoiding direct_forward.
        #   2) An explicit model bypasses the SEPARATE frontend_session_id-driven
        #      prefer_fast->easy(haiku) auto-select in _resolve_ask_models -- that
        #      one alone (independent of gate_action) is also enough to skip the
        #      in-process loop and return a "Parked /ask" stub with no text.
        # Both are opt-in (None = today's behavior, unchanged for every other
        # caller).
        if require_ack:
            payload["require_ack"] = True
        if model:
            payload["model"] = model
        try:  # noqa: PLR1702 - keep transport teardown around the linear stream
            yield TurnStarted(turn)
            async with aclosing(self._transport.stream(payload)) as stream:
                async for event in stream:
                    event_type = event.get("type")
                    if event_type in {
                        "text_delta",
                        "content_chunk",
                        "agent_text_delta",
                    }:
                        if delta := _event_content(event):
                            previous = assistant
                            assistant = _assistant_with_delta(
                                assistant,
                                delta,
                                session_id=self._session_id,
                                turn_id=turn_id,
                                created_at=started_at,
                            )
                            if previous is None:
                                self._history.append(assistant)
                                yield HistoryEntryAdded(assistant)
                            else:
                                self._history[-1] = assistant
                                yield HistoryEntryUpdated(
                                    previous=previous,
                                    entry=assistant,
                                    patch=make_json_patch(
                                        previous.model_dump(mode="json", by_alias=True),
                                        assistant.model_dump(
                                            mode="json", by_alias=True
                                        ),
                                        append_paths={"/content/0/text"},
                                    ),
                                )
                        continue
                    if event_type == "error":
                        raise FounderOSAskStreamError(_event_error(event))
                    if event_type in {"awaiting_permission", "awaiting_clarification"}:
                        raise FounderOSAskStreamError(
                            "FounderOS /ask paused for input, but checkpoint resume "
                            "is not yet exposed through this terminal adapter"
                        )
                    if event_type != "final_result":
                        continue
                    terminal_seen = True
                    final_text = _final_text(event)
                    if assistant is None and not final_text:
                        raise FounderOSAskStreamError(
                            "FounderOS /ask final_result contained no assistant text"
                        )
                    previous = assistant
                    assistant = _complete_assistant(
                        assistant,
                        final_text,
                        session_id=self._session_id,
                        turn_id=turn_id,
                        created_at=started_at,
                    )
                    if previous is None:
                        self._history.append(assistant)
                        yield HistoryEntryAdded(assistant)
                    else:
                        self._history[-1] = assistant
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
                        update={
                            "status": PublicTurnStatus.COMPLETED,
                            "completed_at": _now_ms(),
                        }
                    )
                    yield TurnCompleted(completed)
                    break
            if not terminal_seen:
                raise FounderOSAskStreamError(
                    "Local FounderOS /ask stream ended before final_result; "
                    "the turn was not retried"
                )
        except asyncio.CancelledError:
            with suppress(Exception):
                await self.interrupt()
            raise
        finally:
            self._turn_active = False
            self._active_turn_id = None
            self._interrupted_turn_ids.discard(turn_id)

    async def events(self) -> AsyncGenerator[AppServerEvent, None]:
        while True:
            event = await self._events.get()
            if isinstance(event, _EventsClosed):
                return
            yield event

    async def inject_user_context(
        self,
        content: str,
        *,
        as_message: bool = False,
        inject_invoked_skill: bool = False,
        images: list[ImageAttachment] | None = None,
        resources: list[UserResource] | None = None,
        client_message_id: str | None = None,
        mention_stats: MentionStats | None = None,
    ) -> list[HistoryEntryAdded]:
        del inject_invoked_skill, mention_stats
        if not as_message:
            raise FounderOSAskError(
                "Non-message context injection is not supported by FounderOS /ask"
            )
        if images or resources:
            raise FounderOSAskError(
                "Attachments are not yet supported by the local /ask adapter"
            )
        if not content.strip():
            raise FounderOSAskError("Queued FounderOS /ask context cannot be empty")
        created_at = _now_ms()
        entry = PublicMessageEntry(
            id=client_message_id or f"user-{uuid4()}",
            session_id=self._session_id,
            created_at=created_at,
            updated_at=created_at,
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            role="user",
            content=[TextContentBlock(text=content)],
            source="turn_steer",
        )
        self._history.append(entry)
        self._pending_user_context.append(content)
        return [HistoryEntryAdded(entry)]

    async def interrupt(self) -> None:
        turn_id = self._active_turn_id
        if turn_id is None or turn_id in self._interrupted_turn_ids:
            return
        self._interrupted_turn_ids.add(turn_id)
        await self._transport.cancel()
        await self._events.put(
            ServerWarning(
                ServerWarningParams(
                    warning=PublicError(
                        message=(
                            "Stopped the local /ask stream. FounderOS does not yet "
                            "expose #2495 run control at the HTTP boundary, so server "
                            "work may continue."
                        )
                    )
                )
            )
        )

    async def respond_to_callback(
        self, callback_id: str, output: CallbackOutput
    ) -> None:
        del callback_id, output
        raise FounderOSAskError(
            "Callback resume is not supported by the local /ask adapter"
        )

    async def resume(self, session_id: str) -> None:
        if self._turn_active:
            raise FounderOSAskError("Cannot change sessions during an active turn")
        if not session_id:
            raise FounderOSAskError("Session ID cannot be empty")
        self._session_id = session_id
        self._history.clear()
        self._pending_user_context.clear()
        self._resource_state.state.session.id = session_id
        self._resource_state.session_log = self._resource_state.session_log.model_copy(
            update={"session_id": session_id}
        )
        self._resource_client.set_session_id(session_id)
        cast(_LocalShellResource, self.resources.shell).set_session_id(session_id)

    async def compact(self, extra_instructions: str = "") -> str:
        del extra_instructions
        raise FounderOSAskError(
            "Session compaction is owned by FounderOS /ask and has no client endpoint"
        )

    async def clear_history(self) -> None:
        raise FounderOSAskError(
            "History clearing is owned by FounderOS /ask and has no client endpoint"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.resources.telemetry.flush()
        shell = cast(_LocalShellResource, self.resources.shell)
        await shell.close()
        await self._transport.close()
        with suppress(asyncio.QueueFull):
            self._events.put_nowait(_EVENTS_CLOSED)


def _assistant_with_delta(
    entry: PublicMessageEntry | None,
    delta: str,
    *,
    session_id: str,
    turn_id: str,
    created_at: int,
) -> PublicMessageEntry:
    if entry is None:
        return PublicMessageEntry(
            id=f"assistant-{uuid4()}",
            session_id=session_id,
            turn_id=turn_id,
            created_at=created_at,
            updated_at=_now_ms(),
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            role="assistant",
            content=[TextContentBlock(text=delta)],
            source="harness",
        )
    return entry.model_copy(
        update={
            "content": [TextContentBlock(text=entry.text + delta)],
            "updated_at": _now_ms(),
        }
    )


def _complete_assistant(
    entry: PublicMessageEntry | None,
    final_text: str,
    *,
    session_id: str,
    turn_id: str,
    created_at: int,
) -> PublicMessageEntry:
    if entry is None:
        return PublicMessageEntry(
            id=f"assistant-{uuid4()}",
            session_id=session_id,
            turn_id=turn_id,
            created_at=created_at,
            updated_at=_now_ms(),
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            role="assistant",
            content=[TextContentBlock(text=final_text)],
            source="harness",
        )
    text = entry.text
    if final_text and (not text or final_text.startswith(text)):
        text = final_text
    return entry.model_copy(
        update={
            "content": [TextContentBlock(text=text)],
            "updated_at": _now_ms(),
            "generation_status": PublicEntryGenerationStatus.COMPLETED,
        }
    )


def _event_content(event: Mapping[str, Any]) -> str:
    data = event.get("data")
    if isinstance(data, Mapping):
        value = data.get("content")
        if isinstance(value, str):
            return value
    value = event.get("content")
    return value if isinstance(value, str) else ""


def _event_error(event: Mapping[str, Any]) -> str:
    for value in (event.get("error"), event.get("message")):
        if isinstance(value, str) and value:
            return value
    data = event.get("data")
    if isinstance(data, Mapping):
        for value in (data.get("error"), data.get("message")):
            if isinstance(value, str) and value:
                return value
    return "FounderOS /ask stream failed"


def _final_text(event: Mapping[str, Any]) -> str:
    data = event.get("data")
    value: object = data if isinstance(data, Mapping) else event
    for _ in range(3):
        if not isinstance(value, Mapping):
            break
        for key in ("answer", "user_feedback", "agenda_text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        result = value.get("result")
        if isinstance(result, str):
            return result
        if not isinstance(result, Mapping):
            break
        value = result
    return ""


class _StaticResourceConnection:
    def __init__(self, client: _StaticResourceClient) -> None:
        self._client = client

    async def connect(self) -> _StaticResourceClient:
        return self._client

    def mark_session_attached(self) -> None:
        return None


class _StaticResourceClient:
    """Static TUI support only; it is deliberately not a chat coordinator."""

    def __init__(self, runtime: RuntimeReadResponse) -> None:
        self._runtime = runtime

    def set_session_id(self, session_id: str) -> None:
        self._runtime = self._runtime.model_copy(
            update={
                "session_log": self._runtime.session_log.model_copy(
                    update={"session_id": session_id}
                )
            }
        )

    async def request(
        self,
        method: str,
        params: object | None = None,
        *,
        wait_for_incoming: bool = False,
    ) -> dict[str, Any]:
        del wait_for_incoming
        if method == "runtime/read":
            response: object = self._runtime
        elif method == "config/fields/read":
            alias = self._runtime.runtime.config.active_model.alias
            response = ConfigFieldsReadResponse(
                fields=[
                    ConfigFieldWire(
                        name="active_model",
                        kind=ConfigFieldKind.ENUM,
                        description="The model used by the attached FounderOS /ask session.",
                        value=alias,
                        path="active_model",
                        enum_choices=[alias],
                        layer_values=[ConfigLayerValueWire(layer="admin", value=alias)],
                    )
                ],
                targets=[],
            )
        elif method == "session/ready/wait":
            response = SessionReadyWaitResponse(ready=True, init_duration_ms=0)
        elif method == "workspace/prompt/prepare":
            message = getattr(params, "message", "")
            response = WorkspacePromptPrepareResponse(
                prompt=PreparedPrompt(display_text=message, prompt_text=message)
            )
        elif method == "workspace/trust/status":
            response = WorkspaceTrustStatusResponse(status="session")
        elif method == "workspace/trust/untrustedConfig":
            response = WorkspaceUntrustedConfigResponse()
        elif method == "account/read":
            response = AccountReadResponse(
                account=AccountView(status=AccountStatus.UNAVAILABLE)
            )
        elif method == "identity/read":
            response = IdentityReadResponse(identity=None)
        elif method == "feedback/shouldShow":
            response = FeedbackShouldShowResponse(show=False)
        elif method in {"feedback/record", "telemetry/record"}:
            response = EmptyResponse()
        elif method == "narration/summarize":
            response = NarrationSummarizeResponse(summary=None)
        elif method == "loops/list":
            response = LoopsListResponse(loops=[])
        elif method == "config/reload":
            response = ConfigMutationResponse(
                runtime=self._runtime.runtime, stripped_history_images=0
            )
        else:
            raise FounderOSAskError(
                f"{method} is not available in the attached local /ask session"
            )
        return cast(Any, response).model_dump(mode="json", by_alias=True)


class _LocalShellResource:
    def __init__(
        self, *, controller: ShellController, session_id: str, cwd: str
    ) -> None:
        self._controller = controller
        self._session_id = session_id
        self._cwd = cwd

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    async def run(
        self, command: str, *, timeout_seconds: float = 30.0
    ) -> AsyncGenerator[HistoryEntryAdded | HistoryEntryUpdated, None]:
        if not command.strip():
            raise FounderOSAskError("Shell command cannot be empty")
        operation_id = str(uuid4())
        created_at = _now_ms()
        started = time.monotonic()
        entry = PublicEffectEntry(
            id=operation_id,
            session_id=self._session_id,
            created_at=created_at,
            updated_at=created_at,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            title="shell",
            detail=shell_effect_detail(command),
            state=RunningEffectState(),
        )
        yield HistoryEntryAdded(entry)

        output: list[str] = []
        chunks: asyncio.Queue[str | None] = asyncio.Queue()

        async def observe(chunk: str) -> None:
            output.append(chunk)
            await chunks.put(chunk)

        async def execute() -> ShellRunResponse:
            try:
                return await self._controller.run(
                    ShellRunParams(
                        session_id=self._session_id,
                        operation_id=operation_id,
                        command=command,
                        timeout_seconds=timeout_seconds,
                        cwd=self._cwd,
                    ),
                    observe,
                )
            finally:
                await chunks.put(None)

        task = asyncio.create_task(execute())
        try:
            while (chunk := await chunks.get()) is not None:
                previous = entry
                entry = entry.model_copy(
                    update={
                        "updated_at": _now_ms(),
                        "state": RunningEffectState(output_text="".join(output)),
                    }
                )
                yield HistoryEntryUpdated(
                    previous=previous,
                    entry=entry,
                    patch=[
                        JsonPatchOperation(
                            op="append", path="/state/outputText", value=chunk
                        )
                    ],
                )
            result = await task
            state = shell_effect_state(
                result,
                output_text="".join(output),
                duration_ms=(time.monotonic() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state = shell_effect_error(
                exc,
                output_text="".join(output),
                duration_ms=(time.monotonic() - started) * 1000,
            )
        finally:
            if not task.done():
                await self._controller.interrupt(operation_id)
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        previous = entry
        entry = entry.model_copy(
            update={
                "updated_at": _now_ms(),
                "generation_status": PublicEntryGenerationStatus.COMPLETED,
                "state": state,
            }
        )
        yield HistoryEntryUpdated(
            previous=previous,
            entry=entry,
            patch=make_json_patch(
                previous.model_dump(mode="json", by_alias=True),
                entry.model_dump(mode="json", by_alias=True),
            ),
        )

    async def close(self) -> None:
        await self._controller.close()


def _build_resources(
    *, session_id: str, cwd: str, shell_controller: ShellController | None
) -> tuple[AppServerResources, ClientSessionState, _StaticResourceClient]:
    model = ModelConfigView(
        name="founderos-ask",
        alias="founderos-ask",
        thinking="off",
        supports_images=False,
        display_name="FounderOS /ask",
    )
    audio_provider = AudioProviderView(
        api_base="", api_key_env_var="", client="mistral"
    )
    config = ConfigView(
        active_model=model,
        active_model_pinned=True,
        default_model_alias=model.alias,
        theme="auto",
        log_level="WARNING",
        disable_welcome_banner_animation=False,
        show_greeting=False,
        autocopy_to_clipboard=False,
        file_watcher_for_autocomplete=True,
        ask_confirmation_on_exit=False,
        voice_mode_enabled=False,
        narrator_enabled=False,
        show_thinking_nodes=False,
        enable_update_checks=False,
        enable_notifications=False,
        vibe_code_enabled=False,
        models=[model],
        transcribe_models=[],
        tts_models=[],
        transcription=TranscriptionConfigView(
            model=TranscribeModelConfigView(
                name="auto",
                sample_rate=16_000,
                encoding="pcm_s16le",
                language="en",
                target_streaming_delay_ms=320,
            ),
            provider=audio_provider,
        ),
        speech=SpeechConfigView(
            model=TTSModelConfigView(name="auto", voice="auto", response_format="pcm"),
            provider=audio_provider,
        ),
        validation_warnings=[],
    )
    agent = AgentSummary(
        name="founderos-ask",
        display_name="FounderOS /ask",
        description="Local FounderOS router",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.AGENT,
    )
    runtime = RuntimeReadResponse(
        runtime=RuntimeSnapshot(
            config=config,
            active_agent=agent,
            agents=[agent],
            skills=[],
            tools=[],
            stats=AgentStatsSnapshot(),
            context_window=0,
            issues=[],
            hooks_count=0,
            connectors=ConnectorCounts(),
            mcp=MCPState(),
        ),
        session_log=SessionLogSummary(
            enabled=True, session_id=session_id, persisted=True, title="FounderOS /ask"
        ),
        ready=True,
    )
    state = PublicSessionState(
        event_id=0,
        session=PublicSession(
            id=session_id,
            status=IdleSessionStatus(),
            created_at=_now_ms(),
            updated_at=_now_ms(),
            cwd=cwd,
            model=model.alias,
            agent=agent,
            token_usage=TokenUsage(),
        ),
        history=[],
        turns=[],
    )
    client_state = ClientSessionState(ClientBootstrap(state=state, runtime=runtime))
    static_client = _StaticResourceClient(runtime)
    connection = cast(
        AppServerResourceConnection, _StaticResourceConnection(static_client)
    )
    resources = AppServerResources(connection, client_state)
    resources.shell = cast(
        Any,
        _LocalShellResource(
            controller=shell_controller or ShellController(Path(cwd)),
            session_id=session_id,
            cwd=cwd,
        ),
    )
    return resources, client_state, static_client


def _now_ms() -> int:
    return int(time.time() * 1000)
