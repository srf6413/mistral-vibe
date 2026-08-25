"""Routes finalized voice transcripts into jarvis's REAL, already-running
coding-agent turn, and fans the agent's streaming response text back out to
whichever voice utterance(s) are currently waiting to speak it.

Design note -- why turn-dispatch and response-fan-out are decoupled from
any one LLMStream's lifecycle:

A livekit-agents `llm.LLMStream` gets cancelled by the framework on user
barge-in (the human starts speaking again while the agent is still
"talking") and on session teardown (voice mode toggled off). If a stream
directly awaited `AppServerSession.act()`'s event-generator, that
cancellation would propagate into the generator's `except
asyncio.CancelledError` branch, which calls `self.interrupt()` on the turn
(see `vibe/app_server/session.py`) -- i.e. cancelling a *voice* stream would
interrupt the REAL coding-agent turn out from under it, even mid tool-call.
`VoiceTurnBridge` instead:

- dispatches the turn once, via a callback the caller supplies (mirroring
  exactly what the keyboard path already does -- see
  `VibeApp._start_queued_agent_turn` / `VibeApp._inject_queued_prompt`) and
  lets it run to completion independently of who's currently listening;
- lets any number of `LLMStream`s "listen in" via `subscribe()`/
  `unsubscribe()`, which is safe to drop at any point (including from
  inside `asyncio.CancelledError` handling) without touching the
  underlying turn.

Turn-completion signaling (:meth:`on_turn_finished`) deliberately does NOT
hang off the task `start_new_turn` returns: a voice utterance can also
arrive mid-turn against a turn the KEYBOARD path started (no bridge task to
attach to), so the one place that reliably sees every turn -- voice- or
keyboard-initiated -- through to its actual end is `VibeApp._handle_turn`'s
own `finally`, which calls `on_turn_finished()` there instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging

from vibe.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
)
from vibe.app_server.models import PublicMessageEntry

logger = logging.getLogger("jarvis.duplex_voice.bridge")

# The wire path an assistant message's plain-text content streams over (see
# `vibe.app_server.events._STREAMING_TEXT_PATHS`, which additionally covers
# reasoning-entry and tool-state paths this module deliberately does NOT
# want to speak -- defined locally rather than importing that private
# constant across a module boundary).
_ASSISTANT_TEXT_PATH = "/content/0/text"


def _extract_assistant_delta(event: AppServerEvent) -> str | None:
    """Return newly-produced, speakable assistant text from a history
    event, or ``None`` if this event doesn't carry any.

    Only text appended to an assistant *message* entry's first text block
    counts -- reasoning/tool-call/callback entries are never spoken.
    """
    if isinstance(event, HistoryEntryUpdated):
        entry = event.entry
        if not (isinstance(entry, PublicMessageEntry) and entry.role == "assistant"):
            return None
        delta = "".join(
            op.value
            for op in event.patch
            if op.op == "append"
            and op.path == _ASSISTANT_TEXT_PATH
            and isinstance(op.value, str)
        )
        return delta or None
    if isinstance(event, HistoryEntryAdded):
        entry = event.entry
        if (
            isinstance(entry, PublicMessageEntry)
            and entry.role == "assistant"
            and entry.text
        ):
            # Rare: a fully-formed assistant message arrives in one shot
            # (no HistoryEntryUpdated appends will follow it). Speak it
            # whole rather than waiting for deltas that never come.
            return entry.text
    return None


class VoiceTurnBridge:
    """Bridges duplex-voice transcripts to jarvis's real `AppServerSession`
    turn machinery, via callbacks supplied by the caller (the running
    `VibeApp`) rather than by holding a reference to the app or its
    `AppServerSession` directly -- keeps this package free of any TUI
    dependency.
    """

    def __init__(
        self,
        *,
        turn_active: Callable[[], bool],
        start_new_turn: Callable[[str], asyncio.Task],
        inject_mid_turn: Callable[[str], Awaitable[None]],
    ) -> None:
        self._turn_active = turn_active
        self._start_new_turn = start_new_turn
        self._inject_mid_turn = inject_mid_turn
        self._subscribers: set[asyncio.Queue[str | None]] = set()

    def subscribe(self) -> asyncio.Queue[str | None]:
        """Register interest in the current/next real turn's response text.

        Yields text chunks as they're produced, then a single ``None``
        sentinel once the turn that's driving them finishes (however it
        finishes -- success, error, or cancellation). Always pair with
        :meth:`unsubscribe`, including on cancellation of whatever is
        reading the queue -- it never touches the underlying turn.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self._subscribers.discard(queue)

    def on_history_event(self, event: AppServerEvent) -> None:
        """Feed one `AppServerEvent` to the bridge.

        Intended to be called for every event the app already renders on
        screen (keyboard- and voice-triggered turns alike -- see
        `VibeApp._handle_turn_event`), so a voice utterance that arrives
        mid-turn against a turn the KEYBOARD path started still hears the
        response. (There has to be an active voice subscriber for any of
        this to go anywhere -- a purely keyboard-driven turn with voice
        mode on but nobody currently listening broadcasts to an empty set
        and is a no-op, same as for anything that isn't streaming
        assistant text.)
        """
        delta = _extract_assistant_delta(event)
        if delta:
            self._broadcast_text(delta)

    def on_turn_finished(self) -> None:
        """Call once a real turn -- voice- or keyboard-initiated -- has
        fully ended (see `VibeApp._handle_turn`'s `finally`), so every
        current subscriber's stream closes cleanly instead of hanging open
        until the next barge-in happens to cancel it.
        """
        self._broadcast_done()

    def _broadcast_text(self, text: str) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(text)

    def _broadcast_done(self) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(None)

    async def handle_transcript(self, text: str) -> None:
        """Dispatch one finalized voice transcript the same way the
        keyboard path would: start a new turn if none is running,
        otherwise steer the one already in flight.

        No ``await`` separates the `turn_active` check from dispatch, so
        this is race-free against anything else on the app's single
        asyncio loop -- nothing can flip `turn_active` in that gap.
        """
        if self._turn_active():
            logger.info("voice transcript mid-turn -> inject_user_context")
            await self._inject_mid_turn(text)
            return
        logger.info("voice transcript starts a new turn")
        self._start_new_turn(text)
