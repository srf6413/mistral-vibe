from __future__ import annotations

from dataclasses import dataclass, field
import os

from livekit import api

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

# Local dev defaults for `livekit-server --dev`: it always mints these
# placeholder credentials when no config/flags are given (confirmed in a
# prior smoke test). Override via env for anything else.
_DEV_LIVEKIT_URL = "ws://127.0.0.1:7880"
_DEV_LIVEKIT_API_KEY = "devkey"
_DEV_LIVEKIT_API_SECRET = "secret"
_DEV_ROOM = "jarvis-duplex-voice"
_DEV_AGENT_IDENTITY = "jarvis-duplex-agent"


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


@dataclass(frozen=True, slots=True)
class DuplexVoiceSettings:
    """Connection settings for the duplex voice room + agent process.

    Every field reads from an env var first (matching the naming used by the
    dormant hybrid-spike script this package's agent entrypoint is adapted
    from), falling back to `livekit-server --dev`'s well-known local
    defaults. This lets the supervisor and the agent process agree on the
    same room without a shared config file.
    """

    livekit_url: str = field(
        default_factory=lambda: os.environ.get("LIVEKIT_URL", _DEV_LIVEKIT_URL)
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("LIVEKIT_API_KEY", _DEV_LIVEKIT_API_KEY)
    )
    api_secret: str = field(
        default_factory=lambda: os.environ.get(
            "LIVEKIT_API_SECRET", _DEV_LIVEKIT_API_SECRET
        )
    )
    room: str = field(default_factory=lambda: os.environ.get("LIVEKIT_ROOM", _DEV_ROOM))
    agent_identity: str = field(
        default_factory=lambda: os.environ.get("DUPLEX_IDENTITY", _DEV_AGENT_IDENTITY)
    )

    def as_env(self) -> dict[str, str]:
        """Serialize to the env vars a subprocess agent reads on startup."""
        return {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.api_key,
            "LIVEKIT_API_SECRET": self.api_secret,
            "LIVEKIT_ROOM": self.room,
            "DUPLEX_IDENTITY": self.agent_identity,
        }

    def mint_token(
        self, *, identity: str | None = None, name: str | None = None
    ) -> str:
        """Mint a room-join JWT for `identity` (defaults to `agent_identity`)."""
        resolved_identity = identity or self.agent_identity
        return (
            api
            .AccessToken(self.api_key, self.api_secret)
            .with_identity(resolved_identity)
            .with_name(name or resolved_identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True, room=self.room, can_publish=True, can_subscribe=True
                )
            )
            .to_jwt()
        )
