from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import MagicMock

import pytest

from vibe.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    TurnCompleted,
)
from vibe.app_server.models import (
    CallbackOutput,
    ImageAttachment,
    JsonPatchOperation,
    MentionStats,
    PreparedPrompt,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicTurn,
    PublicTurnStatus,
    TextContentBlock,
    TokenUsage,
    UserDisplayContent,
)
from vibe.app_server.resources import AppServerResources
from vibe.app_server.session import AppServerSessionClient, SessionExitSummary
from vibe.cli.textual_ui.app import VibeApp, _split_app_server_source
from vibe.cli.textual_ui.handlers.event_handler import EventHandler
from vibe.cli.textual_ui.widgets.messages import AssistantMessage
from vibe.user_content import UserResource


def _stream_events(session_id: str) -> list[AppServerEvent]:
    initial = PublicMessageEntry(
        id="assistant-1",
        session_id=session_id,
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        role="assistant",
        content=[TextContentBlock(text="Hello")],
    )
    completed = initial.model_copy(
        update={
            "content": [TextContentBlock(text="Hello from /ask")],
            "updated_at": 2,
            "generation_status": PublicEntryGenerationStatus.COMPLETED,
        }
    )
    return [
        HistoryEntryAdded(initial),
        HistoryEntryUpdated(
            previous=initial,
            entry=completed,
            patch=[
                JsonPatchOperation(
                    op="append", path="/content/0/text", value=" from /ask"
                )
            ],
        ),
        TurnCompleted(
            PublicTurn(
                id="turn-1",
                session_id=session_id,
                status=PublicTurnStatus.COMPLETED,
                started_at=1,
                completed_at=2,
            )
        ),
    ]


class FakeAskSession:
    """Test-only external session; it never constructs a Vibe harness or model."""

    resources = cast(AppServerResources, object())

    def __init__(self, *, block_turn: bool = False) -> None:
        self._session_id = "ask-session-1"
        self._history: list[PublicHistoryEntry] = []
        self._turn_active = False
        self._block_turn = block_turn
        self._release_turn = asyncio.Event()
        self.turn_started = asyncio.Event()
        self.submissions: list[str] = []
        self.interrupt_count = 0
        self.resume_ids: list[str] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cwd(self) -> str:
        return "/workspace"

    @property
    def history(self) -> list[PublicHistoryEntry]:
        return self._history

    @property
    def turn_active(self) -> bool:
        return self._turn_active

    def exit_summary(self) -> SessionExitSummary:
        return SessionExitSummary(session_id=self.session_id, usage=TokenUsage())

    async def act(
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
    ) -> AsyncGenerator[AppServerEvent, None]:
        del (
            client_message_id,
            auto_title,
            images,
            resources,
            user_display_content,
            mention_stats,
            injected,
        )
        self.submissions.append(message)
        self._turn_active = True
        self.turn_started.set()
        try:
            if self._block_turn:
                await self._release_turn.wait()
                return
            for event in _stream_events(self.session_id):
                if isinstance(event, HistoryEntryAdded):
                    self._history.append(event.entry)
                elif isinstance(event, HistoryEntryUpdated):
                    self._history[-1] = event.entry
                yield event
        except asyncio.CancelledError:
            await self.interrupt()
            raise
        finally:
            self._turn_active = False

    async def events(self) -> AsyncGenerator[AppServerEvent, None]:
        while True:
            await asyncio.sleep(3600)
            if False:  # pragma: no cover - establishes the async-generator shape
                yield cast(AppServerEvent, None)

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
        del (
            content,
            as_message,
            inject_invoked_skill,
            images,
            resources,
            client_message_id,
            mention_stats,
        )
        return []

    async def interrupt(self) -> None:
        self.interrupt_count += 1
        self._release_turn.set()

    async def respond_to_callback(
        self, callback_id: str, output: CallbackOutput
    ) -> None:
        del callback_id, output

    async def resume(self, session_id: str) -> None:
        self.resume_ids.append(session_id)
        self._session_id = session_id

    async def compact(self, extra_instructions: str = "") -> str:
        del extra_instructions
        return ""

    async def clear_history(self) -> None:
        self._history.clear()

    async def close(self) -> None:
        return None


class _LoadingArea:
    async def remove_children(self) -> None:
        return None


class _TurnHarness:
    def __init__(self, session: AppServerSessionClient) -> None:
        self.app_server = session
        self.event_handler = EventHandler(
            self._mount_and_scroll, get_tools_collapsed=lambda: True
        )
        self.mounted: list[object] = []
        self._loading_widget = None
        self._narrator_manager = MagicMock()
        self._fatal_init_error = False
        self._interrupt_requested = False
        self._agent_task: asyncio.Task[None] | None = None
        self._active_callback = None
        self._pending_callbacks: deque[object] = deque()
        self._pending_local_question = None
        self._loading_area = _LoadingArea()
        self._cached_chat = None

    async def _mount_and_scroll(self, widget: object, **_: object) -> None:
        if isinstance(widget, AssistantMessage):
            # Textual normally initializes the Markdown child while mounting.
            # This focused harness keeps the widget detached, so drive compose
            # once before the existing streaming handler appends its next delta.
            list(widget.compose())
        self.mounted.append(widget)

    async def _remove_loading_widget(self) -> None:
        self._loading_widget = None

    async def _ensure_runtime_ready(self) -> None:
        return None

    async def _ensure_loading_widget(self, *_: object, **__: object) -> None:
        return None

    async def _handle_turn_events(
        self, events: AsyncGenerator[AppServerEvent, None]
    ) -> None:
        await VibeApp._handle_turn_events(cast(VibeApp, self), events)

    async def _handle_turn_event(self, event: AppServerEvent) -> None:
        await VibeApp._handle_turn_event(cast(VibeApp, self), event)

    def _track_narrator_event(self, event: AppServerEvent) -> None:
        VibeApp._track_narrator_event(cast(VibeApp, self), event)

    async def _show_callback(self, _: object) -> None:
        return None

    def _update_context_progress(self, _: object) -> None:
        return None

    def notify(self, *_: object, **__: object) -> None:
        return None

    async def _handle_turn_error(self, *, cancelled: bool = False) -> None:
        del cancelled

    async def _mount_turn_error(self, error: Exception, message: str) -> None:
        raise AssertionError(f"unexpected turn error: {message}") from error

    def _resolve_turn_error_message(self, error: Exception) -> str:
        return str(error)

    async def _finalize_turn_ui(self, *, resume_queue: bool = True) -> None:
        del resume_queue
        await self.event_handler.finalize_streaming()
        self._agent_task = None

    def _agent_job_active(self) -> bool:
        return self._agent_task is not None and not self._agent_task.done()


@pytest.mark.asyncio
async def test_external_session_submits_once_and_streams_through_existing_events() -> (
    None
):
    fake = FakeAskSession()
    session: AppServerSessionClient = fake
    selected, starter = _split_app_server_source(session)
    assert selected is fake
    assert starter is None

    harness = _TurnHarness(session)
    await VibeApp._handle_turn(
        cast(VibeApp, harness),
        "hello",
        prepared_prompt=PreparedPrompt(display_text="hello", prompt_text="hello"),
    )

    assert fake.submissions == ["hello"]
    assistant = next(
        widget for widget in harness.mounted if isinstance(widget, AssistantMessage)
    )
    assert assistant.get_content() == "Hello from /ask"


@pytest.mark.asyncio
async def test_external_session_interrupt_uses_session_control() -> None:
    fake = FakeAskSession(block_turn=True)
    harness = _TurnHarness(fake)
    turn = asyncio.create_task(
        VibeApp._handle_turn(
            cast(VibeApp, harness),
            "wait",
            prepared_prompt=PreparedPrompt(display_text="wait", prompt_text="wait"),
        )
    )
    harness._agent_task = turn
    await fake.turn_started.wait()

    await VibeApp._interrupt_turn(cast(VibeApp, harness))

    assert fake.interrupt_count == 1
    assert turn.cancelled()


@pytest.mark.asyncio
async def test_external_session_keeps_resume_identity_available() -> None:
    fake = FakeAskSession()

    assert fake.session_id == "ask-session-1"
    await fake.resume("ask-session-resumed")

    assert fake.resume_ids == ["ask-session-resumed"]
    assert fake.session_id == "ask-session-resumed"
    assert fake.exit_summary().session_id == "ask-session-resumed"
