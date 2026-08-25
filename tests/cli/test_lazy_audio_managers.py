from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.stubs.app_config import build_test_app_config
from tests.stubs.fake_voice_manager import FakeVoiceManager
from vibe.cli.lazy_audio_managers import LazyVoiceManager
from vibe.cli.voice_manager.voice_manager_port import TranscribeState


def test_importing_tui_app_does_not_import_optional_audio_modules() -> None:
    code = """
import sys
import vibe.cli.textual_ui.app

blocked = [
    "sounddevice",
    "vibe.cli.voice_manager.voice_manager",
    "vibe.cli.audio_player.audio_player",
    "vibe.cli.audio_recorder.audio_recorder",
    "vibe.cli.transcribe.factory",
    "vibe.cli.tts.factory",
]
loaded = [name for name in blocked if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected optional modules loaded: {loaded}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.asyncio
async def test_default_tui_app_does_not_materialize_disabled_optional_managers() -> (
    None
):
    config = build_test_vibe_config(voice_mode_enabled=False)

    with patch(
        "vibe.cli.lazy_audio_managers._create_real_voice_manager",
        side_effect=AssertionError("voice should stay lazy"),
    ):
        app = build_test_vibe_app(config=config, voice_manager=None)
        await app.prepare()

    assert app._voice_manager.is_enabled is False
    assert app._voice_manager.transcribe_state == TranscribeState.IDLE


def test_lazy_voice_manager_materializes_when_used() -> None:
    config = build_test_app_config(voice_mode_enabled=False)
    factory = MagicMock(return_value=FakeVoiceManager(is_voice_ready=True))
    manager = LazyVoiceManager(lambda: config, factory)

    assert manager.is_enabled is False
    assert manager.transcribe_state == TranscribeState.IDLE
    factory.assert_not_called()

    manager.start_recording()

    factory.assert_called_once()
    assert manager.transcribe_state == TranscribeState.RECORDING


def test_lazy_voice_manager_duplex_active_defaults_false_and_stays_lazy() -> None:
    config = build_test_app_config(voice_mode_enabled=False)
    factory = MagicMock(return_value=FakeVoiceManager(is_voice_ready=True))
    manager = LazyVoiceManager(lambda: config, factory)

    assert manager.duplex_active is False
    factory.assert_not_called()

    manager.duplex_active = False  # setting False must not force materialization
    factory.assert_not_called()


def test_lazy_voice_manager_duplex_active_forwards_to_real_manager_once_set() -> None:
    config = build_test_app_config(voice_mode_enabled=False)
    real = FakeVoiceManager(is_voice_ready=True)
    factory = MagicMock(return_value=real)
    manager = LazyVoiceManager(lambda: config, factory)

    manager.duplex_active = True

    factory.assert_called_once()
    assert real.duplex_active is True
    assert manager.duplex_active is True

    manager.duplex_active = False
    assert real.duplex_active is False
    assert manager.duplex_active is False
