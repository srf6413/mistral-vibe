"""Tests for wiring the duplex voice service into the real running TUI app:
toggling starts/stops `DuplexVoiceSupervisor`, the response-text tap feeds
`VoiceTurnBridge`, the mute flag is threaded through, and shutdown tears
the service down. `DuplexVoiceSupervisor` itself is faked here (its own
lifecycle -- real subprocess/asyncio.Task management -- is covered by
`tests/cli/duplex_voice/test_supervisor.py`); this module is about the
APP'S wiring to it.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import build_test_vibe_app
from tests.stubs.fake_voice_manager import FakeVoiceManager
from vibe.app_server.events import HistoryEntryUpdated
from vibe.app_server.models import (
    JsonPatchOperation,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
)

# Force `livekit.agents` (and its transitive `psutil` import) to finish
# loading now, outside of any live Textual pilot session -- importing it
# for the FIRST time from inside `app._start_duplex_voice()`'s lazy import
# while `app.run_test()`'s pilot/driver threads are active hits a real
# `psutil` import race on this machine (`cannot import name '_psutil_linux'
# from partially initialized module 'psutil'`), unrelated to this branch's
# own logic. `vibe/cli/textual_ui/app.py` still only imports
# `vibe.cli.duplex_voice.*` lazily -- this eager import is test-only.
import vibe.cli.duplex_voice.supervisor  # noqa: F401


def _append_event(text: str) -> HistoryEntryUpdated:
    entry = PublicMessageEntry(
        id="entry-1",
        session_id="session-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        role="assistant",
        content=[{"type": "text", "text": text}],
    )
    return HistoryEntryUpdated(
        previous=entry,
        entry=entry,
        patch=[JsonPatchOperation(op="append", path="/content/0/text", value=text)],
    )


class _FakeDuplexVoiceSupervisor:
    """Records what it was constructed with and whether start()/stop()
    were called, without touching any real process or network resource.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def fake_supervisors(monkeypatch) -> list[_FakeDuplexVoiceSupervisor]:
    created: list[_FakeDuplexVoiceSupervisor] = []

    def _factory(**kwargs) -> _FakeDuplexVoiceSupervisor:
        sup = _FakeDuplexVoiceSupervisor(**kwargs)
        created.append(sup)
        return sup

    monkeypatch.setattr(
        "vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor", _factory
    )
    return created


@pytest.mark.asyncio
async def test_start_duplex_voice_constructs_and_starts_the_supervisor(
    fake_supervisors,
) -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        await app._start_duplex_voice()

        assert len(fake_supervisors) == 1
        sup = fake_supervisors[0]
        assert sup.started is True
        assert sup.kwargs["enable_mic"] is True
        assert sup.kwargs["enable_playback"] is True
        assert app._duplex_voice_supervisor is sup
        assert app._duplex_voice_bridge is not None
        assert app._duplex_voice_event_sink == app._duplex_voice_bridge.on_history_event
        # Ctrl+R push-to-talk must be excluded once duplex is actually
        # running -- see `VoiceManagerPort.duplex_active`.
        assert app._voice_manager.duplex_active is True

        # Calling it again while already "on" must not construct a second
        # supervisor.
        await app._start_duplex_voice()
        assert len(fake_supervisors) == 1

        await app._stop_duplex_voice()
        assert app._voice_manager.duplex_active is False


@pytest.mark.asyncio
async def test_bridge_is_wired_to_the_apps_real_turn_dispatch_methods(
    fake_supervisors,
) -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._start_duplex_voice()

        bridge = app._duplex_voice_bridge
        assert bridge is not None
        # Reused verbatim -- not re-implemented -- exactly mirroring the
        # keyboard/queue path's own act()/inject_user_context() dispatch.
        assert bridge._start_new_turn == app._start_queued_agent_turn
        assert bridge._inject_mid_turn == app._inject_queued_prompt
        assert bridge._turn_active() == app.app_server.turn_active

        await app._stop_duplex_voice()


@pytest.mark.asyncio
async def test_muted_flag_flows_from_voice_manager_to_the_supervisor(
    fake_supervisors,
) -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._start_duplex_voice()

        muted = fake_supervisors[0].kwargs["muted"]
        assert muted() is False

        fake_voice_manager.muted = True
        assert muted() is True

        fake_voice_manager.muted = False
        assert muted() is False

        await app._stop_duplex_voice()


@pytest.mark.asyncio
async def test_handle_turn_event_taps_into_the_bridge_when_voice_is_on(
    fake_supervisors,
) -> None:
    """The mechanism `JarvisBridgeLLM`'s TTS-out relies on: every event the
    app already renders (`_handle_turn_event`) reaches the bridge's
    subscribers while voice mode is on.
    """
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._start_duplex_voice()

        bridge = app._duplex_voice_bridge
        assert bridge is not None
        queue = bridge.subscribe()
        try:
            await app._handle_turn_event(_append_event("hi there"))
            heard = await asyncio.wait_for(queue.get(), timeout=1.0)
            assert heard == "hi there"
        finally:
            bridge.unsubscribe(queue)

        await app._stop_duplex_voice()


@pytest.mark.asyncio
async def test_handle_turn_event_is_a_noop_sink_when_voice_is_off() -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app._duplex_voice_event_sink is None
        # Must not raise just because voice mode was never turned on.
        await app._handle_turn_event(_append_event("nobody is listening"))


@pytest.mark.asyncio
async def test_duplex_active_stays_false_when_supervisor_start_fails(
    monkeypatch,
) -> None:
    """`duplex_active` gates Ctrl+R (see `VoiceManagerPort.duplex_active`);
    if duplex never actually came up, Ctrl+R push-to-talk must keep
    working as the fallback -- so a failed `supervisor.start()` must never
    flip it on.
    """

    class _FailingSupervisor:
        def __init__(self, **kwargs) -> None:
            pass

        async def start(self) -> None:
            raise RuntimeError("boom: port already bound")

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(
        "vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor", _FailingSupervisor
    )
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        await app._start_duplex_voice()

        assert app._duplex_voice_supervisor is None
        assert app._voice_manager.duplex_active is False


@pytest.mark.asyncio
async def test_stop_duplex_voice_clears_state_and_stops_the_supervisor(
    fake_supervisors,
) -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._start_duplex_voice()
        sup = fake_supervisors[0]

        await app._stop_duplex_voice()

        assert sup.stopped is True
        assert app._duplex_voice_supervisor is None
        assert app._duplex_voice_bridge is None
        assert app._duplex_voice_event_sink is None

        # Stopping again (voice already off) must be a harmless no-op.
        await app._stop_duplex_voice()
        assert sup.stopped is True


@pytest.mark.asyncio
async def test_apply_duplex_voice_enabled_runs_in_the_background(
    fake_supervisors,
) -> None:
    """`_apply_duplex_voice_enabled` must not block its caller -- it's
    invoked from inside a queued command's payload, and a slow
    `supervisor.start()` must not stall queue drain / keyboard input.
    """
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        app._apply_duplex_voice_enabled(True)
        # Returns immediately -- the worker hasn't necessarily run yet.
        await pilot.pause(0.2)

        assert len(fake_supervisors) == 1
        assert fake_supervisors[0].started is True

        app._apply_duplex_voice_enabled(False)
        await pilot.pause(0.2)

        assert fake_supervisors[0].stopped is True
        assert app._duplex_voice_supervisor is None


@pytest.mark.asyncio
async def test_shutdown_cleanup_stops_an_active_duplex_voice_service(
    fake_supervisors,
) -> None:
    # `app.prepare()` rather than `app.run_test()`: calling
    # `shutdown_cleanup()` -> `app_server.close()` while a live pilot's own
    # `_listen_app_server_events` worker is still consuming that same
    # stream races the worker's teardown against pytest's -- an existing,
    # unrelated hazard of that combination (see the same
    # `app.prepare()`-not-`run_test()` pattern already used by
    # `tests/cli/textual_ui/test_quit_confirmation.py`'s own
    # `shutdown_cleanup` tests). `prepare()` alone is enough: it's what
    # wires `_voice_manager`/`app_server`, which is all these assertions need.
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)
    await app.prepare()

    await app._start_duplex_voice()
    sup = fake_supervisors[0]

    await app.shutdown_cleanup()

    assert sup.stopped is True
    assert app._duplex_voice_supervisor is None


@pytest.mark.asyncio
async def test_shutdown_cleanup_is_harmless_when_voice_was_never_used() -> None:
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)
    await app.prepare()

    # Must not raise, and must not import/touch duplex_voice at all.
    await app.shutdown_cleanup()
    assert app._duplex_voice_supervisor is None
