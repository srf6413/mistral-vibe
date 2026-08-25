from __future__ import annotations

from enum import StrEnum, auto
from typing import Protocol

from vibe.utils.audio import RecordingMode


class TranscribeState(StrEnum):
    IDLE = auto()
    RECORDING = auto()
    FLUSHING = auto()


class RecordingStartError(Exception):
    pass


class VoiceManagerListener:
    def on_transcribe_state_change(self, state: TranscribeState) -> None:
        pass

    def on_voice_mode_change(self, enabled: bool) -> None:
        pass

    def on_transcribe_text(self, text: str) -> None:
        pass

    def on_transcribe_error(self, message: str) -> None:
        pass

    def on_transcribe_notice(self, message: str) -> None:
        pass


class VoiceManagerPort(Protocol):
    @property
    def is_enabled(self) -> bool: ...

    @property
    def transcribe_state(self) -> TranscribeState: ...

    @property
    def peak(self) -> float: ...

    # Single source of truth for whether mic capture should be suppressed
    # (input side only; playback is unaffected). Owned here so other
    # subsystems (e.g. a future duplex-voice audio pipeline) can check it
    # without depending on the TUI.
    @property
    def muted(self) -> bool: ...
    @muted.setter
    def muted(self, value: bool) -> None: ...

    # True whenever `vibe.cli.duplex_voice`'s always-listening pipeline is
    # actually running (set by `VibeApp._start_duplex_voice` only after its
    # `DuplexVoiceSupervisor.start()` succeeds, cleared on stop/failure --
    # see `app.py`). Both `MicPublisher` (duplex mode's continuous capture)
    # and this port's own `start_recording()` (Ctrl+R push-to-talk) open
    # their own `sounddevice` input stream against the same default
    # microphone; with both flagged "enabled" by the very same
    # `voice_mode_enabled` config toggle, nothing previously stopped a
    # Ctrl+R press from opening a second, redundant capture stream
    # alongside duplex's already-running one -- and, worse, transcribing
    # the same utterance through TWO independent pipelines (duplex
    # dispatching it as a real turn immediately; push-to-talk inserting it
    # into the input box for a later, easy-to-miss accidental Enter).
    # `VoiceManager.start_recording()` (the concrete implementation) checks
    # this and refuses with `RecordingStartError` while it's true --
    # `text_area.py`'s existing `_handle_voice_key` already catches that
    # error and surfaces it via `notify()`, same as every other
    # `start_recording()` precondition failure, so no separate check is
    # needed at the keybinding layer.
    @property
    def duplex_active(self) -> bool: ...
    @duplex_active.setter
    def duplex_active(self, value: bool) -> None: ...

    def apply_enabled(self, enabled: bool) -> None: ...

    def start_recording(self, mode: RecordingMode = RecordingMode.STREAM) -> None: ...

    async def stop_recording(self) -> None: ...

    def cancel_recording(self) -> None: ...

    def add_listener(self, listener: VoiceManagerListener) -> None: ...

    def remove_listener(self, listener: VoiceManagerListener) -> None: ...

    async def close(self) -> None: ...
