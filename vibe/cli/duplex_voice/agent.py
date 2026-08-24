"""Long-running duplex voice agent process.

Adapted from the dormant hybrid-spike script
(``artifacts/voice_livekit_hybrid_spike/agent/duplex_session.py``) for the
room-join / token-minting / ``AgentSession`` wiring pattern, but:

- uses jarvis's own Mistral STT/TTS plugins instead of the spike's
  throwaway ``openai.STT()``/``openai.TTS()``, and an :class:`EchoLLM`
  placeholder instead of the spike's separate ``openai.LLM("gpt-4o-mini")``
  test persona (the real LLM bridge lands in a later stage);
- does not self-terminate after a fixed sleep. It runs until told to stop:
  SIGTERM/SIGINT, or the room disconnecting from the server side. There is
  no LiveKit-agents ``JobContext``/dispatch worker involved here (this
  process makes its own manual ``rtc.Room.connect()``, same as the spike),
  so "run until no longer needed" is implemented directly rather than via
  a framework lifecycle hook.

Runnable standalone for manual verification::

    python -m vibe.cli.duplex_voice.agent

The supervisor (`vibe.cli.duplex_voice.supervisor`) launches this as a
subprocess and stops it by sending SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from livekit import rtc
from livekit.agents import Agent, AgentSession

from vibe.cli.duplex_voice.duplex_config import (
    DuplexVoiceSettings,
    default_speech_config_view,
    default_transcription_config_view,
)
from vibe.cli.duplex_voice.echo_llm import EchoLLM
from vibe.cli.duplex_voice.mistral_stt_plugin import MistralDuplexSTT
from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis.duplex_voice.agent")


class DuplexVoiceAgent(Agent):
    """The room-facing persona for this stage's plumbing proof.

    Kept intentionally minimal: this stage proves STT -> room -> TTS works,
    not conversational quality. A later stage swaps the LLM (and likely
    this instructions string) for jarvis's real agent bridge.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are jarvis's duplex voice pipeline running in test mode. "
                "You only echo back what the user said; this proves the audio "
                "pipeline works end to end."
            )
        )


def _build_session() -> AgentSession:
    transcription = default_transcription_config_view()
    speech = default_speech_config_view()
    return AgentSession(
        stt=MistralDuplexSTT(
            provider=transcription.provider, model=transcription.model
        ),
        llm=EchoLLM(),
        tts=MistralDuplexTTS(provider=speech.provider, model=speech.model),
    )


async def main() -> None:
    settings = DuplexVoiceSettings()
    stop_event = asyncio.Event()

    def _request_stop(*_args: object) -> None:
        if not stop_event.is_set():
            logger.info("stop requested")
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Signal handlers aren't available on every platform/loop combo
            # (e.g. Windows' default proactor loop); the process can still
            # be stopped by killing it outright in that case.
            pass

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

    session = _build_session()
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


if __name__ == "__main__":
    asyncio.run(main())
