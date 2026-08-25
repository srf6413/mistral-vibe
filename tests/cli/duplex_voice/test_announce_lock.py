from __future__ import annotations

import asyncio
import os
import time

from livekit.agents.voice.events import AgentState, AgentStateChangedEvent
import pytest

from vibe.cli.duplex_voice import announce_lock as lock_module
from vibe.cli.duplex_voice.announce_lock import (
    SpeakingLockCoordinator,
    _acquire,
    _release,
    wait_while_other_speaker,
)


def _state_event(
    new_state: AgentState, *, old_state: AgentState = "idle"
) -> AgentStateChangedEvent:
    return AgentStateChangedEvent(old_state=old_state, new_state=new_state)


@pytest.fixture(autouse=True)
def _fast_lock_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every test gets its own lock file (never the real
    `/tmp/.claude-announce.pidlock`) and a short poll interval so a
    contended-lock test doesn't sit for the real (multi-second) production
    timeouts -- each test still passes its own explicit `timeout=`.
    """
    monkeypatch.setattr(lock_module, "_LOCK_PATH", str(tmp_path / "announce.pidlock"))
    monkeypatch.setattr(lock_module, "_POLL_INTERVAL_S", 0.01)


@pytest.mark.asyncio
async def test_acquire_when_lock_is_free_succeeds_immediately() -> None:
    acquired = await _acquire(os.getpid(), timeout=1.0)

    assert acquired is True
    assert os.readlink(lock_module._LOCK_PATH) == str(os.getpid())


@pytest.mark.asyncio
async def test_acquire_reclaims_a_lock_whose_holder_pid_is_dead() -> None:
    # A pid essentially guaranteed not to exist, standing in for a process
    # that crashed without cleaning up its lock -- same scenario speak.sh's
    # own stale-lock reclaim branch handles.
    dead_pid = 2**30
    os.symlink(str(dead_pid), lock_module._LOCK_PATH)

    acquired = await _acquire(os.getpid(), timeout=1.0)

    assert acquired is True
    assert os.readlink(lock_module._LOCK_PATH) == str(os.getpid())


@pytest.mark.asyncio
async def test_acquire_waits_for_a_live_holder_then_succeeds_once_released() -> None:
    holder_pid = os.getpid()  # our own pid is guaranteed alive
    os.symlink(str(holder_pid), lock_module._LOCK_PATH)

    async def _release_shortly() -> None:
        await asyncio.sleep(0.03)
        _release(holder_pid)

    release_task = asyncio.create_task(_release_shortly())
    acquired = await _acquire(os.getpid(), timeout=1.0)
    await release_task

    assert acquired is True


@pytest.mark.asyncio
async def test_acquire_times_out_rather_than_blocking_forever_on_a_live_holder() -> (
    None
):
    os.symlink(str(os.getpid()), lock_module._LOCK_PATH)  # our own pid: stays "alive"

    acquired = await _acquire(os.getpid(), timeout=0.05)

    assert acquired is False


@pytest.mark.asyncio
async def test_acquire_is_best_effort_when_the_lock_path_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lock_module, "_LOCK_PATH", "/nonexistent-dir/announce.pidlock")

    acquired = await _acquire(os.getpid(), timeout=1.0)

    assert acquired is False


def test_release_only_removes_a_lock_this_pid_actually_holds() -> None:
    os.symlink("999999", lock_module._LOCK_PATH)  # someone else's lock

    _release(os.getpid())

    assert os.path.islink(lock_module._LOCK_PATH)  # untouched


@pytest.mark.asyncio
async def test_coordinator_acquires_lock_on_entering_speaking_state() -> None:
    coordinator = SpeakingLockCoordinator()

    coordinator.on_state_changed(_state_event("speaking"))
    await asyncio.sleep(0)  # let the acquire task run

    assert os.readlink(lock_module._LOCK_PATH) == str(os.getpid())

    coordinator.close()


@pytest.mark.asyncio
async def test_coordinator_releases_lock_on_leaving_speaking_state() -> None:
    coordinator = SpeakingLockCoordinator()
    coordinator.on_state_changed(_state_event("speaking"))
    await asyncio.sleep(0)

    coordinator.on_state_changed(_state_event("listening"))
    await asyncio.sleep(0)

    assert not os.path.exists(lock_module._LOCK_PATH)


@pytest.mark.asyncio
async def test_coordinator_never_double_speaks_without_releasing_between() -> None:
    """Two "speaking" transitions back to back (no intervening state) must
    not leak: the second transition releases the first acquire before
    starting its own, and the lock file always reflects this coordinator's
    own current pid -- never orphaned.
    """
    coordinator = SpeakingLockCoordinator()
    coordinator.on_state_changed(_state_event("speaking"))
    await asyncio.sleep(0)

    coordinator.on_state_changed(_state_event("speaking"))
    await asyncio.sleep(0)

    assert os.readlink(lock_module._LOCK_PATH) == str(os.getpid())
    coordinator.close()
    assert not os.path.exists(lock_module._LOCK_PATH)


@pytest.mark.asyncio
async def test_coordinator_close_cancels_an_in_flight_acquire_and_leaves_no_lock() -> (
    None
):
    # Someone else (our own pid, guaranteed alive) holds the lock, so the
    # coordinator's acquire is left pending inside its poll loop.
    os.symlink(str(os.getpid()), lock_module._LOCK_PATH)
    other_holder_marker = os.readlink(lock_module._LOCK_PATH)

    coordinator = SpeakingLockCoordinator()
    coordinator.on_state_changed(_state_event("speaking"))
    await asyncio.sleep(0)  # acquire task starts, blocks in its poll loop

    coordinator.close()
    await asyncio.sleep(0.02)  # let the cancellation actually propagate

    # The lock this test pre-seeded is untouched (never stolen mid-wait),
    # and the coordinator never marked itself as holding it.
    assert os.readlink(lock_module._LOCK_PATH) == other_holder_marker
    assert coordinator._held is False


@pytest.mark.asyncio
async def test_wait_while_other_speaker_returns_immediately_when_lock_is_free() -> None:
    started = time.monotonic()

    await wait_while_other_speaker(timeout=1.0)

    assert time.monotonic() - started < 0.1
    assert not os.path.exists(lock_module._LOCK_PATH)  # never acquires


@pytest.mark.asyncio
async def test_wait_while_other_speaker_is_exempt_from_its_own_pid() -> None:
    """A `SpeakingLockCoordinator` holding the lock across a multi-sentence
    turn must not make sentence 2+ of that SAME turn wait out its own lock.
    """
    os.symlink(str(os.getpid()), lock_module._LOCK_PATH)
    started = time.monotonic()

    await wait_while_other_speaker(timeout=1.0)

    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_wait_while_other_speaker_waits_for_a_different_live_pid_then_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_pid = os.getpid()  # this test process's real pid: guaranteed alive
    os.symlink(str(other_pid), lock_module._LOCK_PATH)

    async def _release_shortly() -> None:
        await asyncio.sleep(0.03)
        _release(other_pid)

    release_task = asyncio.create_task(_release_shortly())

    # Report a fake "self" pid different from the lock's real holder, so
    # `wait_while_other_speaker` treats this as a genuinely different,
    # still-alive process rather than exempting it via the self-pid check.
    monkeypatch.setattr(lock_module.os, "getpid", lambda: other_pid + 1)
    await wait_while_other_speaker(timeout=1.0)
    await release_task

    assert not os.path.exists(lock_module._LOCK_PATH)


@pytest.mark.asyncio
async def test_wait_while_other_speaker_times_out_rather_than_blocking_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_pid = os.getpid()
    os.symlink(str(other_pid), lock_module._LOCK_PATH)
    monkeypatch.setattr(lock_module.os, "getpid", lambda: other_pid + 1)
    started = time.monotonic()

    await wait_while_other_speaker(timeout=0.05)

    assert time.monotonic() - started >= 0.05
    assert os.path.islink(lock_module._LOCK_PATH)  # left untouched, not stolen
