from __future__ import annotations

from vibe.cli.duplex_voice.standalone_defaults import (
    default_speech_config_view,
    default_transcription_config_view,
)


def test_default_transcription_config_view_matches_jarvis_defaults() -> None:
    view = default_transcription_config_view()

    assert view.model.name == "voxtral-mini-transcribe-realtime-2602"
    assert view.model.sample_rate == 16000
    assert view.model.encoding == "pcm_s16le"
    assert view.model.language == "en"
    assert view.model.target_streaming_delay_ms == 500
    assert view.provider.api_key_env_var == "MISTRAL_API_KEY"
    assert view.provider.client == "mistral"


def test_default_speech_config_view_matches_jarvis_defaults() -> None:
    view = default_speech_config_view()

    assert view.model.name == "voxtral-mini-tts-latest"
    assert view.model.voice == "gb_jane_neutral"
    assert view.model.response_format == "wav"
    assert view.provider.api_key_env_var == "MISTRAL_API_KEY"
    assert view.provider.client == "mistral"
