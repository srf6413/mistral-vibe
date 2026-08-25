from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.conftest import build_test_vibe_app
from tests.stubs.fake_voice_manager import FakeVoiceManager
from vibe.utils.audio import RecordingMode


@pytest.mark.asyncio
async def test_ctrl_t_toggles_mute_when_voice_enabled() -> None:
    """ctrl+t flips the voice manager's muted flag and plays a cue when voice mode is on."""
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    with patch("vibe.cli.textual_ui.app.play_mute_cue") as mock_play_cue:
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert fake_voice_manager.muted is False

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert fake_voice_manager.muted is True
            mock_play_cue.assert_called_once_with(True)

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert fake_voice_manager.muted is False
            mock_play_cue.assert_called_with(False)
            assert mock_play_cue.call_count == 2


@pytest.mark.asyncio
async def test_ctrl_t_is_noop_when_voice_disabled() -> None:
    """ctrl+t must not toggle the flag, play a sound, or error when voice mode is off."""
    fake_voice_manager = FakeVoiceManager(is_voice_ready=False)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    with patch("vibe.cli.textual_ui.app.play_mute_cue") as mock_play_cue:
        async with app.run_test() as pilot:
            await pilot.pause(0.1)

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)

            assert fake_voice_manager.muted is False
            mock_play_cue.assert_not_called()


@pytest.mark.asyncio
async def test_ctrl_t_toggles_mute_during_active_recording() -> None:
    """ctrl+t must still reach the App (priority binding) while a recording is in flight,
    since ChatTextArea swallows every other key in that state.
    """
    fake_voice_manager = FakeVoiceManager(is_voice_ready=True)
    app = build_test_vibe_app(voice_manager=fake_voice_manager)

    with patch("vibe.cli.textual_ui.app.play_mute_cue") as mock_play_cue:
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            fake_voice_manager.start_recording(RecordingMode.STREAM)
            await pilot.pause(0.1)

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)

            assert fake_voice_manager.muted is True
            mock_play_cue.assert_called_once_with(True)
