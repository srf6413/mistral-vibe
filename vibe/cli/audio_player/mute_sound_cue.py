from __future__ import annotations

import subprocess

from vibe.observability.logging import logger

# Two distinct stock macOS system sounds so mute-on and mute-off are audibly
# different cues, not just the same blip played twice.
_MUTE_ON_SOUND = "/System/Library/Sounds/Pop.aiff"
_MUTE_OFF_SOUND = "/System/Library/Sounds/Tink.aiff"


def play_mute_cue(muted: bool) -> None:
    """Play a short, non-blocking sound cue for a mic mute-state change.

    Launches ``afplay`` as a fire-and-forget subprocess (never awaited) so the
    UI thread is never blocked on playback. This is a best-effort cue, not a
    hard requirement: any failure — afplay missing, non-macOS host, no audio
    device, etc. — is swallowed silently rather than surfaced to the user.
    """
    sound_path = _MUTE_ON_SOUND if muted else _MUTE_OFF_SOUND
    try:
        subprocess.Popen(
            ["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        logger.debug("Failed to play mute sound cue", exc_info=True)
