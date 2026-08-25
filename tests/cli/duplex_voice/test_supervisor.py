from __future__ import annotations

import asyncio
import os

import pytest

from vibe.cli.duplex_voice import supervisor as supervisor_module
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.standalone_defaults import (
    default_speech_config_view,
    default_transcription_config_view,
)
from vibe.cli.duplex_voice.supervisor import (
    DuplexVoiceSupervisor,
    DuplexVoiceSupervisorError,
)

# `DuplexVoiceSupervisor` requires real transcription/speech config (see its
# docstring for why it can't compute defaults itself); these tests don't
# care what's in it, so one shared dummy pair, reused everywhere below.
_CONFIG = {
    "transcription": default_transcription_config_view(),
    "speech": default_speech_config_view(),
}

# `livekit-server --dev` always binds the same fixed ports (HTTP 7880 *and*
# a UDP media port livekit picks from the host's addresses -- not something
# `DuplexVoiceSupervisor` configures or probes). Running more than one real
# instance per test session is flaky purely from OS socket-release timing
# between tests, independent of anything this branch changed, so only ONE
# test below (`test_start_and_stop_manage_a_real_server_process_and_inprocess_agent_task`)
# spins up a real server; every other test fakes `_start_server`/
# `_stop_server` via `_patch_fake_server` below and exercises the REAL
# in-process agent/mic asyncio.Task lifecycle against that fake.


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class _FakeServerProcess:
    """Stands in for `asyncio.subprocess.Process` for tests that don't care
    about the real `livekit-server` subprocess itself.
    """

    _next_pid = 900_000

    def __init__(self) -> None:
        self.pid = _FakeServerProcess._next_pid
        _FakeServerProcess._next_pid += 1
        self.returncode: int | None = None


def _patch_fake_server(monkeypatch: pytest.MonkeyPatch, sup: DuplexVoiceSupervisor) -> None:
    """Replace `sup`'s real livekit-server subprocess management with an
    in-memory fake, so `start()`/`stop()` exercise the REAL agent/mic
    asyncio.Task lifecycle logic without needing a real, port-binding
    subprocess (see module docstring above for why more than one real
    server per test run is unreliable). Also fakes the module-level
    `_port_is_bound` probe `start()`/`stop()` call directly (not just
    `sup`'s own methods) -- otherwise these tests would still read the
    REAL port 7880's state, which is a real, shared, global resource that
    genuinely can be bound by the one real-server test running
    concurrently on another xdist worker.
    """

    async def _fake_start_server() -> None:
        sup._server_proc = _FakeServerProcess()

    async def _fake_stop_server() -> None:
        proc = sup._server_proc
        if proc is not None:
            proc.returncode = 0
        sup._server_proc = None

    async def _fake_port_is_bound(host: str, port: int, *, timeout: float = 0.5) -> bool:
        return False

    monkeypatch.setattr(sup, "_start_server", _fake_start_server)
    monkeypatch.setattr(sup, "_stop_server", _fake_stop_server)
    monkeypatch.setattr(supervisor_module, "_port_is_bound", _fake_port_is_bound)


async def _fake_run_duplex_agent(
    settings,
    *,
    llm_plugin=None,
    stop_event: asyncio.Event | None = None,
    **_kwargs,  # transcription/speech -- unused by this fake
) -> None:
    """Stands in for the real STT/LLM/TTS/room pipeline: a real in-process
    asyncio task that does nothing but wait to be told to stop, so these
    tests exercise the REAL task-lifecycle logic (creation, liveness
    grace-check, cancellation/stop_event teardown) without needing real
    Mistral credentials or a real LiveKit `AgentSession` round trip.
    """
    assert stop_event is not None
    await stop_event.wait()


@pytest.mark.asyncio
async def test_start_refuses_to_run_when_port_already_bound(monkeypatch) -> None:
    async def _fake_port_is_bound(
        host: str, port: int, *, timeout: float = 0.5
    ) -> bool:
        return True

    monkeypatch.setattr(supervisor_module, "_port_is_bound", _fake_port_is_bound)

    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(), **_CONFIG)

    with pytest.raises(DuplexVoiceSupervisorError, match="already bound"):
        await sup.start()

    assert not sup.is_running


def test_status_reports_not_running_before_start() -> None:
    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(), **_CONFIG)

    status = sup.status()

    assert status.server_running is False
    assert status.agent_running is False
    assert status.server_pid is None
    assert status.agent_pid is None


@pytest.mark.asyncio
async def test_start_and_stop_manage_a_real_server_process_and_inprocess_agent_task(
    monkeypatch,
) -> None:
    """Real `livekit-server` subprocess lifecycle + real in-process agent
    asyncio.Task lifecycle, proving: start() brings both up, is_running/
    status() reflect that, and stop() cleanly tears both down with no
    process left running and the port freed -- toggling voice mode on/off
    with no orphaned processes, per this task's requirement (a). The one
    real-subprocess test in this module; see the module docstring for why
    the others fake the server.
    """
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    sup = DuplexVoiceSupervisor(
        settings=DuplexVoiceSettings(room="test-room-lifecycle"), **_CONFIG
    )
    await sup.start()
    server_pid: int | None = None
    try:
        assert sup.is_running

        status = sup.status()
        assert status.server_running is True
        assert status.server_pid is not None
        assert status.agent_running is True
        assert status.agent_pid is None  # in-process task, not an OS process
        assert status.agent_log_path is None

        server_pid = status.server_pid
        assert _pid_alive(server_pid)
    finally:
        await sup.stop()

    assert not sup.is_running
    assert sup.status().server_running is False
    assert sup.status().agent_running is False
    assert server_pid is not None
    assert not _pid_alive(server_pid)
    assert not await supervisor_module._port_is_bound("127.0.0.1", 7880)


@pytest.mark.asyncio
async def test_start_is_idempotent_when_already_running(monkeypatch) -> None:
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)
    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(room="test-room-idempotent"), **_CONFIG)
    _patch_fake_server(monkeypatch, sup)
    await sup.start()
    try:
        first_pid = sup.status().server_pid
        await sup.start()  # should be a no-op, not a second server
        assert sup.status().server_pid == first_pid
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_agent_task_stopping_gracefully_sets_the_stop_event(monkeypatch) -> None:
    stop_event_seen: list[asyncio.Event] = []

    async def _capture_stop_event(settings, *, llm_plugin=None, stop_event=None, **_kw) -> None:
        assert stop_event is not None
        stop_event_seen.append(stop_event)
        await stop_event.wait()

    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _capture_stop_event)
    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(room="test-room-graceful"), **_CONFIG)
    _patch_fake_server(monkeypatch, sup)
    await sup.start()
    await sup.stop()

    assert len(stop_event_seen) == 1
    assert stop_event_seen[0].is_set()


@pytest.mark.asyncio
async def test_agent_task_that_wont_stop_gets_cancelled(monkeypatch) -> None:
    """An agent task that ignores its stop_event (hung STT/TTS call, e.g.)
    must still be torn down -- stop() falls back to cancellation rather
    than hanging forever or leaving the task running.
    """

    async def _ignores_stop_event(settings, *, llm_plugin=None, stop_event=None, **_kw) -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _ignores_stop_event)
    monkeypatch.setattr(supervisor_module, "_GRACEFUL_STOP_TIMEOUT_S", 0.2)

    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(room="test-room-hung"), **_CONFIG)
    _patch_fake_server(monkeypatch, sup)
    await sup.start()
    task = sup._agent_task
    assert task is not None

    await sup.stop()

    assert task.cancelled()
    assert not sup.is_running


@pytest.mark.asyncio
async def test_start_raises_and_cleans_up_when_agent_fails_immediately(
    monkeypatch,
) -> None:
    async def _fails_immediately(settings, *, llm_plugin=None, stop_event=None, **_kw) -> None:
        raise RuntimeError("boom: bad token")

    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fails_immediately)
    monkeypatch.setattr(supervisor_module, "_AGENT_START_GRACE_S", 0.05)

    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings(room="test-room-failfast"), **_CONFIG)
    _patch_fake_server(monkeypatch, sup)
    with pytest.raises(DuplexVoiceSupervisorError, match="failed to start"):
        await sup.start()

    assert not sup.is_running
    # The (fake) server it brought up before the agent failed must have
    # been torn down too, not left dangling.
    assert sup._server_proc is None


@pytest.mark.asyncio
async def test_mic_publisher_failure_is_non_fatal_to_start(monkeypatch) -> None:
    """This sandbox (and plenty of real deployment targets) has no working
    microphone. `enable_mic=True` must not turn that into a failed toggle
    -- the agent bridge and TTS-out are still useful without it.
    """
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    class _FailingMic:
        def __init__(self, settings, *, muted) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("sounddevice unavailable: no audio driver")

    monkeypatch.setattr(supervisor_module, "MicPublisher", _FailingMic)

    sup = DuplexVoiceSupervisor(
        settings=DuplexVoiceSettings(room="test-room-mic-fail"),
        enable_mic=True,
        **_CONFIG,
    )
    _patch_fake_server(monkeypatch, sup)
    try:
        await sup.start()  # must not raise
        assert sup.is_running
        assert sup._mic_task is None
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_playback_subscriber_failure_is_non_fatal_to_start(monkeypatch) -> None:
    """Mirrors `test_mic_publisher_failure_is_non_fatal_to_start`: a machine
    with no working speakers (this sandbox included) must not turn
    `enable_playback=True` into a failed toggle -- the agent bridge and
    mic-in are still useful without local playback.
    """
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    class _FailingPlayback:
        def __init__(self, settings) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("sounddevice unavailable: no audio driver")

    monkeypatch.setattr(supervisor_module, "PlaybackSubscriber", _FailingPlayback)

    sup = DuplexVoiceSupervisor(
        settings=DuplexVoiceSettings(room="test-room-playback-fail"),
        enable_playback=True,
        **_CONFIG,
    )
    _patch_fake_server(monkeypatch, sup)
    try:
        await sup.start()  # must not raise
        assert sup.is_running
        assert sup._playback_task is None
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_playback_subscriber_starts_and_stops_with_the_supervisor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    constructed: list = []

    class _RecordingPlayback:
        def __init__(self, settings) -> None:
            constructed.append(settings)
            self.stopped = False

        async def run(self) -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.stopped = True
                raise

    monkeypatch.setattr(supervisor_module, "PlaybackSubscriber", _RecordingPlayback)

    settings = DuplexVoiceSettings(room="test-room-playback-lifecycle")
    sup = DuplexVoiceSupervisor(settings=settings, enable_playback=True, **_CONFIG)
    _patch_fake_server(monkeypatch, sup)

    await sup.start()
    assert len(constructed) == 1
    assert constructed[0] is settings
    assert sup._playback_task is not None

    await sup.stop()
    assert sup._playback_task is None


@pytest.mark.asyncio
async def test_playback_subscriber_not_started_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    class _AssertNeverConstructed:
        def __init__(self, settings) -> None:
            raise AssertionError("PlaybackSubscriber must not be constructed")

    monkeypatch.setattr(supervisor_module, "PlaybackSubscriber", _AssertNeverConstructed)

    sup = DuplexVoiceSupervisor(
        settings=DuplexVoiceSettings(room="test-room-playback-disabled"), **_CONFIG
    )
    _patch_fake_server(monkeypatch, sup)
    try:
        await sup.start()
        assert sup._playback_task is None
    finally:
        await sup.stop()


@pytest.mark.asyncio
async def test_mic_publisher_muted_callable_is_threaded_through(monkeypatch) -> None:
    monkeypatch.setattr(supervisor_module, "run_duplex_agent", _fake_run_duplex_agent)

    received_muted: list = []

    class _RecordingMic:
        def __init__(self, settings, *, muted) -> None:
            received_muted.append(muted)

        async def run(self) -> None:
            await asyncio.sleep(3600)

    monkeypatch.setattr(supervisor_module, "MicPublisher", _RecordingMic)

    is_muted = True
    sup = DuplexVoiceSupervisor(
        settings=DuplexVoiceSettings(room="test-room-mic-muted"),
        enable_mic=True,
        muted=lambda: is_muted,
        **_CONFIG,
    )
    _patch_fake_server(monkeypatch, sup)
    try:
        await sup.start()
        assert len(received_muted) == 1
        assert received_muted[0]() is True
        is_muted = False
        assert received_muted[0]() is False
    finally:
        await sup.stop()
