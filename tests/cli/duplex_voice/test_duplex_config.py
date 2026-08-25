from __future__ import annotations

from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings

# `default_transcription_config_view`/`default_speech_config_view` moved to
# `vibe.cli.duplex_voice.standalone_defaults` (see
# `tests/cli/duplex_voice/test_standalone_defaults.py`) -- this module
# (`duplex_config.py`) must not import `vibe.core` any more, since it's
# reachable from `vibe.cli.textual_ui.app` now. See `duplex_config.py`'s
# module docstring.


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
