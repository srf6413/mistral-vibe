"""Long-running duplex voice agent: room-join + ``AgentSession`` lifecycle.

Adapted from the dormant hybrid-spike script
(``artifacts/voice_livekit_hybrid_spike/agent/duplex_session.py``) for the
room-join / token-minting / ``AgentSession`` wiring pattern, but:

- uses jarvis's own Mistral STT/TTS plugins instead of the spike's
  throwaway ``openai.STT()``/``openai.TTS()``;
- takes its LLM, and the transcription/speech config the STT/TTS plugins
  need, as parameters (see :func:`run_duplex_agent`) instead of hardcoding
  or computing them internally. The real wiring in
  `vibe.cli.duplex_voice.supervisor` (started/stopped from the TUI's
  voice-mode toggle) passes ``JarvisBridgeLLM`` and the current session's
  own transcription/speech config; the standalone entrypoint
  (`vibe.cli.duplex_voice.agent_main`) passes `EchoLLM` and this package's
  hardcoded defaults instead. This module deliberately has NO knowledge of
  either -- see `vibe.cli.duplex_voice.duplex_config`'s module docstring
  for why (the short version: this module is reachable from
  `vibe.cli.textual_ui.app` now, which may not transitively import
  `vibe.core`, and the default-computing helpers do);
- does not self-terminate after a fixed sleep. It runs until told to stop
  (`stop_event` set, or the room disconnecting from the server side).
  There is no LiveKit-agents ``JobContext``/dispatch worker involved here
  (this process makes its own manual ``rtc.Room.connect()``, same as the
  spike), so "run until no longer needed" is implemented directly rather
  than via a framework lifecycle hook.

Runnable standalone for manual verification (uses `EchoLLM`, joins its own
`livekit-server --dev`) via `vibe.cli.duplex_voice.agent_main`::

    python -m vibe.cli.duplex_voice.agent_main

For real TUI usage, `vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor`
runs :func:`run_duplex_agent` as an in-process asyncio task (not a
subprocess -- it shares the TUI's own `AppServerSession` directly, which a
separate process could not do without adding IPC) and stops it by setting
its `stop_event`, not by sending it a signal. Because this module is now
imported from inside the (possibly already-running) TUI process, it must
NOT reconfigure logging as an import side effect -- that only happens in
`agent_main`'s own `if __name__ == "__main__"` block.
"""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm

from vibe.app_server.config import SpeechConfigView, TranscriptionConfigView
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.mistral_stt_plugin import MistralDuplexSTT
from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS

logger = logging.getLogger("jarvis.duplex_voice.agent")


class DuplexVoiceAgent(Agent):
    """The room-facing persona for the duplex voice pipeline.

    Kept minimal on purpose: the actual conversational behavior comes from
    whichever `llm.LLM` is wired into the session (see `_build_session`),
    not from anything here.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are jarvis's duplex voice pipeline. Respond with "
                "whatever the configured LLM produces; this persona only "
                "supplies the room-facing identity, not the behavior."
            )
        )


def _build_session(
    llm_plugin: llm.LLM,
    *,
    transcription: TranscriptionConfigView,
    speech: SpeechConfigView,
) -> AgentSession:
    """Build the STT -> LLM -> TTS session from caller-supplied config."""
    return AgentSession(
        stt=MistralDuplexSTT(
            provider=transcription.provider, model=transcription.model
        ),
        llm=llm_plugin,
        tts=MistralDuplexTTS(provider=speech.provider, model=speech.model),
    )


async def run_duplex_agent(
    settings: DuplexVoiceSettings,
    *,
    llm_plugin: llm.LLM,
    transcription: TranscriptionConfigView,
    speech: SpeechConfigView,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Join `settings.room` and run the agent session until `stop_event` is
    set (by the caller) or the room disconnects from the server side.

    Shared by the standalone entrypoint (`vibe.cli.duplex_voice.agent_main`)
    and `DuplexVoiceSupervisor`'s in-process agent task -- the only things
    that differ between those two callers are which `llm_plugin` and which
    transcription/speech config they pass.
    """
    stop_event = stop_event or asyncio.Event()

    def _request_stop(*_args: object) -> None:
        if not stop_event.is_set():
            logger.info("stop requested")
            stop_event.set()

    jwt = settings.mint_token()
    room = rtc.Room()
    room.on("disconnected", _request_stop)

    logger.info(
        "connecting url=%s room=%s identity=%s",
        settings.livekit_url,
        settings.room,
        settings.agent_identity,
    )
    await room.connect(settings.livekit_url, jwt)
    logger.info(
        "connected remotes=%s", [p.identity for p in room.remote_participants.values()]
    )

    session = _build_session(llm_plugin, transcription=transcription, speech=speech)
    try:
        await session.start(agent=DuplexVoiceAgent(), room=room)
        logger.info("session started -- duplex voice agent running")
        await stop_event.wait()
    finally:
        logger.info("tearing down session")
        await session.aclose()
        if room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await room.disconnect()
        logger.info("agent stopped")
