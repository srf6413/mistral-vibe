from __future__ import annotations

from dataclasses import dataclass, field
import os

from livekit import api

# Deliberately NO `vibe.core` import anywhere in this file (nor in
# `agent.py`/`supervisor.py`, which this module feeds): this package is
# reachable from `vibe.cli.textual_ui.app` as of the TUI-wiring stage, and
# `tests/cli/textual_ui/test_app_server_boundary.py` statically enforces
# that nothing reachable from `textual_ui` transitively imports
# `vibe.core`. The transcription/speech config views this package used to
# compute internally (from `vibe.core.config`'s default provider/model
# list) now come from the caller instead -- `vibe.cli.textual_ui.app`
# passes the REAL current session's `self.app_server.resources.config
# .current.transcription`/`.speech` (a `vibe.app_server`-only, already-
# public type), and the standalone entrypoint
# (`vibe.cli.duplex_voice.agent_main`, which is NOT imported by
# `supervisor.py`/`agent.py` and so isn't part of that reachable set) gets
# its own defaults from `vibe.cli.duplex_voice.standalone_defaults`.

# Local dev defaults for `livekit-server --dev`: it always mints these
# placeholder credentials when no config/flags are given (confirmed in a
# prior smoke test). Override via env for anything else.
_DEV_LIVEKIT_URL = "ws://127.0.0.1:7880"
_DEV_LIVEKIT_API_KEY = "devkey"
_DEV_LIVEKIT_API_SECRET = "secret"
_DEV_ROOM = "jarvis-duplex-voice"
_DEV_AGENT_IDENTITY = "jarvis-duplex-agent"
_DEV_HUMAN_IDENTITY = "jarvis-duplex-voice-user"


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
    # A second, distinct room identity for whichever participant publishes
    # the human's microphone (see `vibe.cli.duplex_voice.mic_publisher`).
    # Needs its own identity because the agent's `AgentSession` only ever
    # transcribes REMOTE participants' audio, never its own local track --
    # the mic publisher must join as someone else.
    human_identity: str = field(
        default_factory=lambda: os.environ.get(
            "DUPLEX_HUMAN_IDENTITY", _DEV_HUMAN_IDENTITY
        )
    )

    def as_env(self) -> dict[str, str]:
        """Serialize to the env vars a subprocess agent reads on startup.

        Only used by the standalone ``python -m vibe.cli.duplex_voice.agent_main``
        entrypoint now (see that module's docstring) -- the supervisor's
        real start()/stop() path runs the agent in-process and passes
        `DuplexVoiceSettings` directly, no subprocess/env handoff involved.
        """
        return {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.api_key,
            "LIVEKIT_API_SECRET": self.api_secret,
            "LIVEKIT_ROOM": self.room,
            "DUPLEX_IDENTITY": self.agent_identity,
            "DUPLEX_HUMAN_IDENTITY": self.human_identity,
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
