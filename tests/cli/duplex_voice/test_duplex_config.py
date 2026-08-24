from __future__ import annotations

from vibe.cli.duplex_voice.duplex_config import (
    DuplexVoiceSettings,
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


def test_settings_default_to_livekit_server_dev_placeholders(monkeypatch) -> None:
    for key in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_ROOM",
        "DUPLEX_IDENTITY",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = DuplexVoiceSettings()

    assert settings.livekit_url == "ws://127.0.0.1:7880"
    assert settings.api_key == "devkey"
    assert settings.api_secret == "secret"
    assert settings.as_env()["LIVEKIT_URL"] == settings.livekit_url


def test_settings_respect_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://example.test:9999")
    monkeypatch.setenv("LIVEKIT_API_KEY", "custom-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "custom-secret")
    monkeypatch.setenv("LIVEKIT_ROOM", "custom-room")
    monkeypatch.setenv("DUPLEX_IDENTITY", "custom-identity")

    settings = DuplexVoiceSettings()

    assert settings.livekit_url == "ws://example.test:9999"
    assert settings.room == "custom-room"
    assert settings.agent_identity == "custom-identity"


def test_mint_token_produces_a_jwt_scoped_to_the_room() -> None:
    settings = DuplexVoiceSettings(
        livekit_url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secret",
        room="a-room",
        agent_identity="an-agent",
    )

    token = settings.mint_token()

    # A JWT is three dot-separated base64url segments; no network call is
    # involved in minting one.
    assert token.count(".") == 2
    assert settings.mint_token(identity="someone-else") != token
