from __future__ import annotations

import pytest

from vibe.app_server.events import HistoryEntryAdded, HistoryEntryUpdated
from vibe.app_server.models import (
    JsonPatchOperation,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
)
from vibe.cli.duplex_voice.agent_bridge import VoiceTurnBridge, _extract_assistant_delta


def _assistant_entry(text: str = "") -> PublicMessageEntry:
    return PublicMessageEntry(
        id="entry-1",
        session_id="session-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        role="assistant",
        content=[{"type": "text", "text": text}],
    )


def _append_event(value: str, *, path: str = "/content/0/text") -> HistoryEntryUpdated:
    entry = _assistant_entry(value)
    return HistoryEntryUpdated(
        previous=_assistant_entry(""),
        entry=entry,
        patch=[JsonPatchOperation(op="append", path=path, value=value)],
    )


def test_extract_assistant_delta_ignores_non_assistant_and_non_append() -> None:
    # A user-role message (echoing back what the human said) is never spoken.
    user_entry = PublicMessageEntry(
        id="e",
        session_id="s",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="user",
        content=[{"type": "text", "text": "hello"}],
    )
    assert (
        _extract_assistant_delta(
            HistoryEntryUpdated(
                previous=user_entry,
                entry=user_entry,
                patch=[
                    JsonPatchOperation(op="append", path="/content/0/text", value="hi")
                ],
            )
        )
        is None
    )
    # A non-append op (e.g. a full replace) on an assistant entry is not a delta.
    entry = _assistant_entry("whole thing")
    assert (
        _extract_assistant_delta(
            HistoryEntryUpdated(
                previous=_assistant_entry(""),
                entry=entry,
                patch=[
                    JsonPatchOperation(
                        op="replace", path="/content/0/text", value="whole thing"
                    )
                ],
            )
        )
        is None
    )
    # Unrelated event types are ignored entirely (fall through both
    # isinstance branches).
    assert _extract_assistant_delta(object()) is None  # type: ignore[arg-type]


def test_extract_assistant_delta_reads_append_ops() -> None:
    assert _extract_assistant_delta(_append_event("hello ")) == "hello "


def test_extract_assistant_delta_reads_one_shot_added_entry() -> None:
    entry = _assistant_entry("all at once")
    event = HistoryEntryAdded(entry=entry)
    assert _extract_assistant_delta(event) == "all at once"


def test_extract_assistant_delta_ignores_empty_one_shot_added_entry() -> None:
    # The common case: HistoryEntryAdded fires with empty text before any
    # streaming deltas arrive. Nothing to speak yet.
    entry = _assistant_entry("")
    assert _extract_assistant_delta(HistoryEntryAdded(entry=entry)) is None


@pytest.mark.asyncio
async def test_handle_transcript_starts_a_new_turn_when_none_active() -> None:
    started_with: list[str] = []

    def start_new_turn(text: str):
        started_with.append(text)
        return None

    async def inject_mid_turn(text: str) -> None:
        raise AssertionError("should not inject mid-turn when no turn is active")

    bridge = VoiceTurnBridge(
        turn_active=lambda: False,
        start_new_turn=start_new_turn,
        inject_mid_turn=inject_mid_turn,
    )

    await bridge.handle_transcript("hello jarvis")

    assert started_with == ["hello jarvis"]


@pytest.mark.asyncio
async def test_handle_transcript_injects_mid_turn_when_active() -> None:
    injected_with: list[str] = []

    def start_new_turn(text: str):
        raise AssertionError("should not start a new turn while one is active")

    async def inject_mid_turn(text: str) -> None:
        injected_with.append(text)

    bridge = VoiceTurnBridge(
        turn_active=lambda: True,
        start_new_turn=start_new_turn,
        inject_mid_turn=inject_mid_turn,
    )

    await bridge.handle_transcript("keep going")

    assert injected_with == ["keep going"]


@pytest.mark.asyncio
async def test_subscriber_receives_broadcast_text_and_done_sentinel_on_turn_finished() -> (
    None
):
    bridge = VoiceTurnBridge(
        turn_active=lambda: False,
        start_new_turn=lambda text: None,
        inject_mid_turn=lambda text: _never_called(),
    )

    queue = bridge.subscribe()
    await bridge.handle_transcript("hi")

    bridge.on_history_event(_append_event("hel"))
    bridge.on_history_event(_append_event("lo"))
    assert await queue.get() == "hel"
    assert await queue.get() == "lo"

    assert queue.empty()
    bridge.on_turn_finished()
    assert await queue.get() is None


@pytest.mark.asyncio
async def test_unsubscribe_stops_further_broadcasts() -> None:
    bridge = VoiceTurnBridge(
        turn_active=lambda: False,
        start_new_turn=lambda text: None,
        inject_mid_turn=lambda text: _never_called(),
    )
    queue = bridge.subscribe()
    bridge.unsubscribe(queue)

    bridge.on_history_event(_append_event("should not arrive"))

    assert queue.empty()


@pytest.mark.asyncio
async def test_mid_turn_subscriber_also_gets_the_done_signal() -> None:
    """The bug this regression-tests: a voice utterance that arrives
    MID-TURN against a turn the keyboard path started (no bridge task to
    hang a done-callback off of) must still see its stream close when that
    turn ends -- `on_turn_finished()` is the app's single, turn-origin-
    agnostic signal for that (see `VibeApp._handle_turn`'s `finally`),
    not something tied to whichever task `start_new_turn` happened to
    return.
    """
    bridge = VoiceTurnBridge(
        turn_active=lambda: False,
        start_new_turn=lambda text: None,
        inject_mid_turn=lambda text: _never_called(),
    )
    await bridge.handle_transcript("start")

    # Now simulate a second utterance arriving mid-turn.
    bridge._turn_active = lambda: True
    late_queue = bridge.subscribe()

    bridge.on_turn_finished()
    assert await late_queue.get() is None


async def _never_called() -> None:
    raise AssertionError("should not be called")
