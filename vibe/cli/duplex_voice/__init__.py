"""LiveKit-based duplex voice service layer.

Standalone service layer proving STT -> room -> TTS works over a real
LiveKit room, wrapping jarvis's own Mistral clients. Not wired into the
TUI/app.py yet -- see `vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor`
for the start()/stop() pair a later stage will call.

Nothing in this package is imported by the app's startup path, so importing
it (and therefore `livekit`) has no effect on normal CLI/TUI startup cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
    from vibe.cli.duplex_voice.echo_llm import EchoLLM
    from vibe.cli.duplex_voice.mistral_stt_plugin import MistralDuplexSTT
    from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS
    from vibe.cli.duplex_voice.supervisor import DuplexVoiceSupervisor

__all__ = [
    "DuplexVoiceSettings",
    "DuplexVoiceSupervisor",
    "EchoLLM",
    "MistralDuplexSTT",
    "MistralDuplexTTS",
]

_LAZY_ATTRS = {
    "DuplexVoiceSettings": "vibe.cli.duplex_voice.duplex_config",
    "DuplexVoiceSupervisor": "vibe.cli.duplex_voice.supervisor",
    "EchoLLM": "vibe.cli.duplex_voice.echo_llm",
    "MistralDuplexSTT": "vibe.cli.duplex_voice.mistral_stt_plugin",
    "MistralDuplexTTS": "vibe.cli.duplex_voice.mistral_tts_plugin",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
