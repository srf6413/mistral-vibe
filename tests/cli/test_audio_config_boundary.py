from __future__ import annotations

from collections.abc import Callable
import time
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from tests.stubs.fake_voice_manager import FakeVoiceManager


@pytest.fixture(autouse=True)
def _no_real_duplex_voice_service():
    """This file is about the simple push-to-talk voice manager / config
    persistence boundary, not the duplex (LiveKit) voice service --
    `voice_mode_enabled=True` now also starts that (see
    `VibeApp._apply_duplex_voice_enabled`), which would otherwise try to
    boot a REAL `livekit-server` in the background on any machine that
    happens to have the binary installed. Faked here so these tests stay
    scoped to what they're actually about; the real thing is covered by
    `tests/cli/duplex_voice/test_supervisor.py` and
    `tests/cli/test_duplex_voice_toggle.py`.
    """

    class _FakeDuplexVoiceSupervisor:
        def __init__(self, **_kwargs) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    with patch(
        "vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor",
        _FakeDuplexVoiceSupervisor,
    ):
        yield


async def _wait_until(
    pilot, predicate: Callable[[], bool], timeout: float = 2.0
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


async def _wait_until_drained(pilot, app, timeout: float = 2.0) -> bool:
    # Voice settings defer their side effect to the main queue, so callers
    # must wait for the drain before asserting on the applied state.
    # See ADR 0012.
    return await _wait_until(pilot, lambda: not app._queue.draining, timeout)


@pytest.mark.asyncio
async def test_voice_mode_is_persisted_before_local_application() -> None:
    voice = FakeVoiceManager(is_voice_ready=False)
    app = build_test_vibe_app(
        config=build_test_vibe_config(voice_mode_enabled=False), voice_manager=voice
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        config_resource = app.app_server.resources.config
        update = config_resource.update

        async def accepted_update(changes: dict[str, object]) -> None:
            assert voice.is_enabled is False
            await update(changes)

        with (
            patch("vibe.cli.textual_ui.app.check_audio_available", return_value=None),
            patch.object(
                config_resource, "update", new=AsyncMock(side_effect=accepted_update)
            ) as update_config,
        ):
            await app._handle_voice_settings_closed({"voice_mode_enabled": True})
            # Persistence is deferred to the queue; wait for it to drain so
            # config is accepted and voice applied locally before we assert.
            assert await _wait_until_drained(pilot, app)

        update_config.assert_awaited_once_with({"voice_mode_enabled": True})
        assert app.config.voice_mode_enabled is True
        assert voice.is_enabled is True


@pytest.mark.asyncio
async def test_local_voice_failure_does_not_undo_accepted_config() -> None:
    voice = FakeVoiceManager(is_voice_ready=False)
    app = build_test_vibe_app(
        config=build_test_vibe_config(voice_mode_enabled=False), voice_manager=voice
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        with (
            patch("vibe.cli.textual_ui.app.check_audio_available", return_value=None),
            patch.object(
                voice, "apply_enabled", side_effect=RuntimeError("no microphone")
            ),
            patch.object(app, "notify") as notify,
        ):
            await app._handle_voice_settings_closed({"voice_mode_enabled": True})
            assert await _wait_until_drained(pilot, app)

        assert app.config.voice_mode_enabled is True
        notify.assert_called_once()
        assert notify.call_args.kwargs["severity"] == "warning"
