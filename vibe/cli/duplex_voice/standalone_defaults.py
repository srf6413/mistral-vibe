"""Transcription/speech config defaults for STANDALONE duplex-voice usage
only (`vibe.cli.duplex_voice.agent_main`'s ``python -m`` entrypoint and
`scripts/duplex_voice_proof.py`).

Deliberately its own file, not part of `duplex_config.py`/`agent.py`: it
imports `vibe.core.config`, and NOTHING that `vibe.cli.textual_ui.app`
reaches (directly or transitively, including this package's own
`supervisor.py`/`agent.py`) may import `vibe.core` -- see
`tests/cli/textual_ui/test_app_server_boundary.py`. The real, TUI-wired
path doesn't need this module at all: `VibeApp._start_duplex_voice` passes
the current session's own `self.app_server.resources.config.current
.transcription`/`.speech` instead, which is more correct anyway (it
respects whatever the user actually has configured, rather than always
defaulting to provider/model index 0).
"""

from __future__ import annotations

from vibe.app_server.config import (
    AudioProviderView,
    SpeechConfigView,
    TranscribeModelConfigView,
    TranscriptionConfigView,
    TTSModelConfigView,
)
from vibe.core.config import (
    DEFAULT_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_PROVIDERS,
    DEFAULT_TTS_MODELS,
    DEFAULT_TTS_PROVIDERS,
)


def default_transcription_config_view() -> TranscriptionConfigView:
    """Build the transcription config view from jarvis's own default provider/model.

    Mirrors ``vibe.app_server._projection._project_transcribe_model`` /
    ``_project_transcribe_provider`` without importing that private module.
    """
    provider = DEFAULT_TRANSCRIBE_PROVIDERS[0]
    model = DEFAULT_TRANSCRIBE_MODELS[0]
    return TranscriptionConfigView(
        model=TranscribeModelConfigView(
            name=model.name,
            sample_rate=model.sample_rate,
            encoding=model.encoding,
            language=model.language,
            target_streaming_delay_ms=model.target_streaming_delay_ms,
        ),
        provider=AudioProviderView(
            api_base=provider.api_base,
            api_key_env_var=provider.api_key_env_var,
            client="mistral",
        ),
    )


def default_speech_config_view() -> SpeechConfigView:
    """Build the speech (TTS) config view from jarvis's own default provider/model."""
    provider = DEFAULT_TTS_PROVIDERS[0]
    model = DEFAULT_TTS_MODELS[0]
    return SpeechConfigView(
        model=TTSModelConfigView(
            name=model.name, voice=model.voice, response_format=model.response_format
        ),
        provider=AudioProviderView(
            api_base=provider.api_base,
            api_key_env_var=provider.api_key_env_var,
            client="mistral",
        ),
    )
