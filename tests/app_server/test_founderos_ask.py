from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator, Mapping
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from vibe.app_server import (
    AskPins,
    FounderOSAskSession,
    FounderOSAskUnavailableError,
    HttpFounderOSAskTransport,
)
from vibe.app_server.events import (
    HistoryEntryAdded,
    HistoryEntryUpdated,
    ServerWarning,
    TurnCompleted,
)
from vibe.app_server.models import PublicEffectEntry, PublicMessageEntry, TokenUsage
from vibe.app_server.protocol import ShellRunParams, ShellRunResponse
from vibe.app_server.session import AppServerSessionClient, SessionExitSummary
from vibe.cli import cli as cli_mod
from vibe.cli.textual_ui.app import (
    StartupOptions,
    VibeApp,
    _noop_narrator_manager,
    _noop_voice_manager,
)


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_http_transport_posts_once_and_parses_fragmented_sse() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            stream=_ChunkedStream([
                b'data: {"type":"text_del',
                b'ta","content":"hel"}\r\n\r\n',
                b'data: {"type":"final_result",\n',
                b'data: "data":{"answer":"hello"}}\n\n',
            ]),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpFounderOSAskTransport(
        endpoint="http://127.0.0.1:8000/ask", workspace="ysf", client=client
    )
    payload = {
        "text": "hi",
        "stream": True,
        "ask_mode": "chat",
        "frontend_session_id": "stable-session",
        "intake_model": "auto",
        "worker_model": "auto",
    }

    events = [event async for event in transport.stream(payload)]

    assert events == [
        {"type": "text_delta", "content": "hel"},
        {"type": "final_result", "data": {"answer": "hello"}},
    ]
    assert len(requests) == 1
    assert requests[0].url == "http://127.0.0.1:8000/ask?workspace=ysf"
    assert json.loads(requests[0].content) == payload
    assert requests[0].headers["accept"] == "text/event-stream"
    await transport.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_http_transport_allows_explicit_https_dev_endpoint_only() -> None:
    explicit = HttpFounderOSAskTransport(endpoint="https://intel.example.com/ask")
    assert explicit.endpoint == "https://intel.example.com/ask"
    await explicit.close()

    with pytest.raises(ValueError, match="must use HTTPS"):
        HttpFounderOSAskTransport(endpoint="http://intel.example.com/ask")


@pytest.mark.asyncio
async def test_local_factory_defaults_to_m2_loopback_and_independent_compute_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOUNDEROS_INTAKE_MODEL", "intake-pinned")
    monkeypatch.setenv("FOUNDEROS_WORKER_MODEL", "worker-pinned")
    session = FounderOSAskSession.local(cwd=tmp_path)
    transport = cast(HttpFounderOSAskTransport, session._transport)

    assert transport.endpoint == "http://127.0.0.1:8000/ask"
    assert session.pins == AskPins(
        intake_model="intake-pinned", worker_model="worker-pinned"
    )
    await session.close()


@pytest.mark.asyncio
async def test_http_transport_reports_local_service_unavailable_without_fallback() -> (
    None
):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpFounderOSAskTransport(client=client)

    with pytest.raises(
        FounderOSAskUnavailableError,
        match="unavailable.*no provider fallback attempted",
    ):
        _ = [event async for event in transport.stream({"text": "hi"})]

    assert calls == 1
    await transport.close()
    await client.aclose()


class _FakeAskTransport:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = events or []
        self.payloads: list[dict[str, object]] = []
        self.cancel_count = 0
        self.close_count = 0

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.payloads.append(dict(payload))

        async def generate() -> AsyncGenerator[dict[str, Any], None]:
            for event in self.events:
                yield event

        return generate()

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_session_uses_stable_identity_and_existing_stream_event_path(
    tmp_path: Path,
) -> None:
    transport = _FakeAskTransport([
        {"type": "thinking_step", "data": {"message": "routing"}},
        {"type": "text_delta", "content": "Hello"},
        {"type": "content_chunk", "data": {"content": " there"}},
        {"type": "final_result", "data": {"answer": "Hello there"}},
    ])
    pins = AskPins(intake_model="router-local", worker_model="worker-local")
    session = FounderOSAskSession(
        transport=transport,
        cwd=tmp_path / ".." / tmp_path.name,
        session_id="stable-session",
        pins=pins,
    )

    events = [event async for event in session.act("hi")]

    assert transport.payloads == [
        {
            "text": "hi",
            "stream": True,
            "ask_mode": "chat",
            "frontend_session_id": "stable-session",
            "intake_model": "router-local",
            "worker_model": "worker-local",
        }
    ]
    assert "cwd" not in transport.payloads[0]
    assert session.cwd == str(tmp_path.resolve())
    assert session.session_id == "stable-session"
    assert sum(isinstance(event, HistoryEntryAdded) for event in events) == 1
    assert sum(isinstance(event, HistoryEntryUpdated) for event in events) == 2
    assert isinstance(events[-1], TurnCompleted)
    assistant = cast(PublicMessageEntry, session.history[-1])
    assert assistant.text == "Hello there"
    assert session.exit_summary().session_id == "stable-session"


@pytest.mark.asyncio
async def test_work_class_and_model_are_omitted_unless_explicitly_passed(
    tmp_path: Path,
) -> None:
    """Both are opt-in kwargs added for vibe/workflows -- every other caller
    (plain chat, /btw, etc.) must see byte-identical payloads to before.
    """
    transport = _FakeAskTransport([
        {"type": "final_result", "data": {"answer": "ok"}},
    ])
    session = FounderOSAskSession(
        transport=transport, cwd=tmp_path, session_id="s1"
    )
    [_ async for _ in session.act("hi")]
    assert "work_class" not in transport.payloads[0]
    assert "model" not in transport.payloads[0]

    transport2 = _FakeAskTransport([
        {"type": "final_result", "data": {"answer": "ok"}},
    ])
    session2 = FounderOSAskSession(
        transport=transport2, cwd=tmp_path, session_id="s2"
    )
    [_ async for _ in session2.act("hi", work_class="chat", model="gpt-5-mini")]
    assert transport2.payloads[0]["work_class"] == "chat"
    assert transport2.payloads[0]["model"] == "gpt-5-mini"


@pytest.mark.asyncio
async def test_session_renders_final_only_response(tmp_path: Path) -> None:
    transport = _FakeAskTransport([
        {"type": "final_result", "data": {"result": {"answer": "done"}}}
    ])
    session = FounderOSAskSession(transport=transport, cwd=tmp_path)

    events = [event async for event in session.act("go")]

    added = next(event for event in events if isinstance(event, HistoryEntryAdded))
    assert isinstance(added.entry, PublicMessageEntry)
    assert added.entry.text == "done"


@pytest.mark.asyncio
async def test_queued_user_messages_join_the_next_single_ask_turn(
    tmp_path: Path,
) -> None:
    transport = _FakeAskTransport([
        {"type": "final_result", "data": {"answer": "done"}}
    ])
    session = FounderOSAskSession(transport=transport, cwd=tmp_path)

    injected = await session.inject_user_context("first", as_message=True)
    _ = [event async for event in session.act("second")]

    assert len(injected) == 1
    assert transport.payloads[0]["text"] == "first\n\nsecond"
    assert len(transport.payloads) == 1


class _BlockingAskTransport(_FakeAskTransport):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.payloads.append(dict(payload))

        async def generate() -> AsyncGenerator[dict[str, Any], None]:
            self.started.set()
            await self.release.wait()
            if False:  # pragma: no cover - async-generator shape
                yield {}

        return generate()


@pytest.mark.asyncio
async def test_session_interrupt_cancels_delivery_once_and_warns_about_2495(
    tmp_path: Path,
) -> None:
    transport = _BlockingAskTransport()
    session = FounderOSAskSession(transport=transport, cwd=tmp_path)

    async def consume() -> None:
        _ = [event async for event in session.act("wait")]

    turn = asyncio.create_task(consume())
    await transport.started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    warning = await anext(session.events())
    assert isinstance(warning, ServerWarning)
    assert "#2495" in warning.params.warning.message
    assert "server work may continue" in warning.params.warning.message
    assert transport.cancel_count == 1
    await session.interrupt()
    assert transport.cancel_count == 1


class _FakeShellController:
    def __init__(self) -> None:
        self.params: list[ShellRunParams] = []
        self.closed = False

    async def run(
        self, params: ShellRunParams, observe_output: Any
    ) -> ShellRunResponse:
        self.params.append(params)
        await observe_output("hello\n")
        return ShellRunResponse(
            operation_id=params.operation_id,
            command=params.command,
            cwd=params.cwd or "",
            stdout="hello\n",
            exit_code=0,
        )

    async def interrupt(self, operation_id: str) -> bool:
        del operation_id
        return True

    async def close(self) -> None:
        self.closed = True


class _BlockingShellController(_FakeShellController):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.interrupt_ids: list[str] = []

    async def run(
        self, params: ShellRunParams, observe_output: Any
    ) -> ShellRunResponse:
        del observe_output
        self.params.append(params)
        self.started.set()
        await self.release.wait()
        return ShellRunResponse(
            operation_id=params.operation_id,
            command=params.command,
            cwd=params.cwd or "",
            exit_code=0,
        )

    async def interrupt(self, operation_id: str) -> bool:
        self.interrupt_ids.append(operation_id)
        self.release.set()
        return True


@pytest.mark.asyncio
async def test_attached_session_preserves_local_bang_shell_without_agent_loop(
    tmp_path: Path,
) -> None:
    controller = _FakeShellController()
    session = FounderOSAskSession(
        transport=_FakeAskTransport(),
        cwd=tmp_path,
        shell_controller=cast(Any, controller),
    )

    events = [event async for event in session.resources.shell.run("printf hello")]

    assert len(controller.params) == 1
    assert controller.params[0].cwd == str(tmp_path.resolve())
    assert isinstance(events[0], HistoryEntryAdded)
    assert isinstance(events[0].entry, PublicEffectEntry)
    assert isinstance(events[-1], HistoryEntryUpdated)
    completed = cast(PublicEffectEntry, events[-1].entry)
    assert completed.state.status == "completed"
    await session.close()
    assert controller.closed is True


@pytest.mark.asyncio
async def test_closing_attached_bang_shell_interrupts_its_local_controller(
    tmp_path: Path,
) -> None:
    controller = _BlockingShellController()
    session = FounderOSAskSession(
        transport=_FakeAskTransport(),
        cwd=tmp_path,
        shell_controller=cast(Any, controller),
    )

    async def consume() -> None:
        _ = [event async for event in session.resources.shell.run("wait")]

    command = asyncio.create_task(consume())
    await controller.started.wait()
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command

    assert controller.interrupt_ids == [controller.params[0].operation_id]
    await session.close()


@pytest.mark.asyncio
async def test_attached_resources_mount_the_existing_textual_ui(tmp_path: Path) -> None:
    session = FounderOSAskSession(transport=_FakeAskTransport(), cwd=tmp_path)
    app = VibeApp(
        history_file=tmp_path / "history",
        app_server=session,
        startup=StartupOptions(prompt_for_workspace_trust=False),
        voice_manager=_noop_voice_manager(),
        narrator_manager=_noop_narrator_manager(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await asyncio.wait_for(app._session_ready.wait(), timeout=1)
        await pilot.pause(0.05)

        assert app._fatal_init_error is False
        assert app.config.active_model.alias == "founderos-ask"


@pytest.mark.asyncio
async def test_attached_config_fields_report_admin_model_and_pin(
    tmp_path: Path,
) -> None:
    session = FounderOSAskSession(transport=_FakeAskTransport(), cwd=tmp_path)

    response = await session.resources.config.read_fields()

    field = next(field for field in response.fields if field.name == "active_model")
    assert field.origin == "admin"
    assert field.value == "founderos-ask"
    assert response.targets == []
    assert session.resources.config.current.active_model_pinned is True
    await session.close()


def test_interactive_launcher_attaches_founderos_session_without_local_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_session = cast(AppServerSessionClient, object())
    captured: dict[str, object] = {}

    def local(**kwargs: object) -> AppServerSessionClient:
        captured["local"] = kwargs
        return fake_session

    def run_textual_ui(**kwargs: object) -> SessionExitSummary:
        captured["tui"] = kwargs
        starter = cast(Any, kwargs["start_app_server"])
        assert asyncio.run(starter()) is fake_session
        return SessionExitSummary(session_id="stable", usage=TokenUsage())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vibe.app_server.FounderOSAskSession.local", local)
    monkeypatch.setattr("vibe.cli.textual_ui.app.run_textual_ui", run_textual_ui)
    monkeypatch.setattr(cli_mod, "print_session_resume_message", lambda summary: None)
    args = argparse.Namespace(
        resume="stable", continue_session=False, initial_prompt=None, teleport=False
    )

    cli_mod._run_interactive_mode(
        args=args, stdin_prompt=None, update_cache_repository=cast(Any, object())
    )

    assert captured["local"] == {"cwd": tmp_path, "session_id": "stable"}
    startup = cast(dict[str, Any], captured["tui"])["startup"]
    assert startup.prompt_for_workspace_trust is False
    assert startup.show_resume_picker is False
