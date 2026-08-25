"""Standalone entrypoint for manual verification of the duplex voice
pipeline, using `EchoLLM` and this package's hardcoded default
transcription/speech config -- joins its own `livekit-server --dev`,
echoes back whatever it transcribes.

Runnable directly::

    python -m vibe.cli.duplex_voice.agent_main

This is intentionally a SEPARATE module from `agent.py` (which
`vibe.cli.duplex_voice.supervisor` imports for the real, TUI-wired path):
it's the only place in this package that imports
`vibe.cli.duplex_voice.standalone_defaults` (and therefore `vibe.core`),
and `agent.py`/`supervisor.py` must not -- see
`vibe.cli.duplex_voice.duplex_config`'s module docstring.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from vibe.cli.duplex_voice.agent import run_duplex_agent
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.echo_llm import EchoLLM
from vibe.cli.duplex_voice.standalone_defaults import (
    default_speech_config_view,
    default_transcription_config_view,
)

logger = logging.getLogger("jarvis.duplex_voice.agent")


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

    await run_duplex_agent(
        settings,
        llm_plugin=EchoLLM(),
        transcription=default_transcription_config_view(),
        speech=default_speech_config_view(),
        stop_event=stop_event,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
