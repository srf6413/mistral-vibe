from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from vibe.cli.audio_player.mute_sound_cue import play_mute_cue


def test_play_mute_cue_muted_launches_distinct_sound_non_blocking() -> None:
    with patch("vibe.cli.audio_player.mute_sound_cue.subprocess.Popen") as mock_popen:
        play_mute_cue(True)

    mock_popen.assert_called_once_with(
        ["afplay", "/System/Library/Sounds/Pop.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_play_mute_cue_unmuted_launches_a_different_sound() -> None:
    with patch("vibe.cli.audio_player.mute_sound_cue.subprocess.Popen") as mock_popen:
        play_mute_cue(False)

    mock_popen.assert_called_once_with(
        ["afplay", "/System/Library/Sounds/Tink.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_play_mute_cue_swallows_exceptions() -> None:
    with patch(
        "vibe.cli.audio_player.mute_sound_cue.subprocess.Popen",
        side_effect=FileNotFoundError("afplay not found"),
    ):
        play_mute_cue(True)  # must not raise


def test_play_mute_cue_does_not_block(monkeypatch) -> None:
    """play_mute_cue must never wait on the subprocess (fire-and-forget)."""
    process = MagicMock()
    with patch(
        "vibe.cli.audio_player.mute_sound_cue.subprocess.Popen", return_value=process
    ):
        play_mute_cue(True)

    process.wait.assert_not_called()
    process.communicate.assert_not_called()
