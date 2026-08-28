"""Cross-process coordination with the Mac's shared TTS announce queue.

``~/.claude/hooks/speak.sh`` is a Mac-wide, provider-agnostic serialized
speech channel: ANY process on this machine that wants to talk over the
Mac's speakers is expected to take a turn through its symlink PID lock at
``/tmp/.claude-announce.pidlock`` (see that script's own docstring --
"ANY process/provider on this Mac can call this directly to join the same
queue ... nothing here is Claude-specific"). The duplex voice pipeline's
own TTS (``MistralDuplexTTS``, wired in ``mistral_tts_plugin.py``) speaks
through a completely separate path (a LiveKit audio track, not ``say``), so
without this module the two could talk over each other -- e.g. a background
Claude Code session's task-completion announcement firing mid-sentence over
a live voice-mode reply, or vice versa.

This module makes coordination bidirectional, using the identical
atomic-symlink protocol ``speak.sh`` implements in bash (see that script for
the full correctness rationale): a lock is only ever reclaimed once its
recorded holder PID is confirmed dead via a ``kill(pid, 0)`` probe, never by
a time-based guess. Reusing the SAME lock file means:

- ``speak.sh`` needs NO changes at all: its existing wait loop already
  blocks until whatever PID holds ``/tmp/.claude-announce.pidlock``
  releases it or dies. `SpeakingLockCoordinator` (below) makes the duplex
  agent's "speaking" state hold that lock, so once it does, the Mac
  announce path automatically queues behind it.
- Symmetrically, `wait_while_other_speaker` is awaited from
  `mistral_tts_plugin.MistralChunkedStream._run` right before synthesis
  starts, so the duplex agent itself waits out a `speak.sh` announcement
  already in progress rather than talking over it.

Deliberately best-effort, never a hard dependency: on a non-macOS host, or
one where ``/tmp`` isn't writable, lock acquisition degrades to "speak
anyway" rather than silencing (or crashing) the voice pipeline --
coordination is a nicety layered on top of a working duplex pipeline, never
a gate that can block it outright. Both waits are also capped, for the same
reason -- a wedged lock file must never permanently mute jarvis's voice
output -- but with deliberately different bounds (see each constant below):
`SpeakingLockCoordinator` may hold the lock for a whole multi-sentence
reply, so its acquire can wait out a long Mac announcement; the synth-side
wait guards a live, turn-taking conversation, so it stays short even if
that means occasionally starting to speak over the tail of a long
announcement instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents.voice.events import AgentStateChangedEvent

logger = logging.getLogger("jarvis.duplex_voice.announce_lock")

# Same path `~/.claude/hooks/speak.sh` locks -- deliberately not
# configurable, since the whole point is that both sides agree on one file
# without a shared config.
_LOCK_PATH = "/tmp/.claude-announce.pidlock"
_POLL_INTERVAL_S = 0.2
# speak.sh's own longest possible utterance is ~350 chars, which at its
# default `say -r 175` (words/minute) is roughly 70 words -> ~24s of actual
# speech. 30s gives that headroom to finish naturally before
# `SpeakingLockCoordinator` gives up and speaks anyway -- see module
# docstring on why this can never become an indefinite block regardless.
_MAX_WAIT_S = 30.0
# Deliberately much shorter than `_MAX_WAIT_S`: this bounds
# `wait_while_other_speaker`, awaited on the live conversational turn-taking
# path before the duplex agent starts speaking. A car conversation should
# not stall for up to 30s behind a long Mac announcement -- occasionally
# starting to speak over its last few seconds is the accepted tradeoff.
_SYNTH_WAIT_MAX_S = 3.0


def _holder_alive(holder: str) -> bool:
    try:
        pid = int(holder)
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        # Covers both "no such process" and "exists but not ours to
        # signal" -- matches speak.sh's own `kill -0` check, which treats
        # any failure as "stale, reclaim it".
        return False
    return True


async def _acquire(pid: int, *, timeout: float = _MAX_WAIT_S) -> bool:
    """Best-effort acquire of the shared announce lock for `pid`.

    Returns True once acquired, or False if the wait timed out or the lock
    file couldn't be touched at all (non-macOS host, unwritable /tmp,
    etc.) -- either way, the caller should proceed to speak rather than
    stay silent.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.symlink(str(pid), _LOCK_PATH)
            return True
        except FileExistsError:
            holder = ""
            with contextlib.suppress(OSError):
                holder = os.readlink(_LOCK_PATH)
            if holder and _holder_alive(holder):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "announce lock held by pid=%s past %.0fs wait; "
                        "speaking anyway rather than staying silent",
                        holder,
                        timeout,
                    )
                    return False
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue
            # Stale: recorded holder is dead (or unreadable). Reclaim and
            # retry the symlink on the next loop iteration.
            with contextlib.suppress(FileNotFoundError, OSError):
                os.remove(_LOCK_PATH)
        except OSError as exc:
            logger.debug("announce lock unavailable, skipping coordination: %r", exc)
            return False


def _release(pid: int) -> None:
    with contextlib.suppress(OSError):
        if os.readlink(_LOCK_PATH) == str(pid):
            os.remove(_LOCK_PATH)


async def wait_while_other_speaker(*, timeout: float = _SYNTH_WAIT_MAX_S) -> None:
    """Best-effort wait for a DIFFERENT process's `speak.sh` announcement to
    finish before the duplex agent starts synthesizing its own reply.

    Never acquires the lock itself (no matching release call needed) --
    just polls until it's free, held by a now-dead pid, or `timeout`
    elapses, then returns either way. Deliberately exempts this process's
    own pid: `SpeakingLockCoordinator` legitimately holds the lock across a
    whole multi-sentence turn, and this function is awaited once per
    sentence-level synthesis call within that same turn (see
    `mistral_tts_plugin.MistralChunkedStream._run`) -- without the
    self-exemption, sentence 2+ of a reply would wait out its own turn's
    lock until timeout, every time.
    """
    my_pid = os.getpid()
    deadline = time.monotonic() + timeout
    while True:
        holder = ""
        with contextlib.suppress(OSError):
            holder = os.readlink(_LOCK_PATH)
        if not holder or holder == str(my_pid) or not _holder_alive(holder):
            return
        if time.monotonic() >= deadline:
            logger.warning(
                "announce lock held by pid=%s past %.0fs wait before synth; "
                "speaking anyway",
                holder,
                timeout,
            )
            return
        await asyncio.sleep(_POLL_INTERVAL_S)


class SpeakingLockCoordinator:
    """Holds the shared Mac announce lock for as long as the duplex agent's
    `AgentSession.agent_state` is ``"speaking"``.

    Wired as a sync ``AgentSession.on("agent_state_changed", ...)``
    listener (see ``vibe.cli.duplex_voice.agent.run_duplex_agent``) because
    livekit-agents' ``EventEmitter.emit()`` calls listeners synchronously
    and never awaits a coroutine result -- this class owns its own asyncio
    task for the actual (awaitable) lock acquire rather than trying to be a
    coroutine itself.

    Every state transition -- including two "speaking" spans back to back
    with nothing in between -- cancels any acquire still in flight and
    releases the lock if held, before deciding whether to start a new
    acquire. That makes every call idempotent and race-free against rapid
    state changes on the same single-threaded event loop: cancellation of
    an in-flight acquire can only take effect at that coroutine's own
    ``await`` points, never between this method's own (non-awaiting) lines.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._task: asyncio.Task[None] | None = None
        self._held = False

    def on_state_changed(self, event: AgentStateChangedEvent) -> None:
        self._reset()
        if event.new_state == "speaking":
            self._task = asyncio.create_task(
                self._acquire_for_speaking(), name="announce-lock-acquire"
            )

    async def _acquire_for_speaking(self) -> None:
        self._held = await _acquire(self._pid)

    def _reset(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        if self._held:
            self._held = False
            _release(self._pid)

    def close(self) -> None:
        """Cancel any pending acquire and release the lock if held.

        Call on session teardown so a voice-mode toggle-off (or any other
        early exit) mid-speech never leaves the shared lock stuck held.
        """
        self._reset()
