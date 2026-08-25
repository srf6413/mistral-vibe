"""LiveKit-based duplex voice service layer.

Wired into the TUI (`vibe/cli/textual_ui/app.py`) via
`vibe.cli.duplex_voice.supervisor.DuplexVoiceSupervisor`'s start()/stop()
pair, `vibe.cli.duplex_voice.agent_bridge.VoiceTurnBridge` (routes
transcripts into the app's real, already-running `AppServerSession`), and
`vibe.cli.duplex_voice.jarvis_llm.JarvisBridgeLLM` (the LLM plugin that
connects the two).

Nothing in this package is imported by the app's startup path -- the app
only imports it lazily, from inside the voice-mode toggle handler --
so importing it (and therefore `livekit`) has no effect on normal
CLI/TUI startup cost for users who never turn voice mode on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.cli.duplex_voice.agent_bridge import VoiceTurnBridge
    from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
    from vibe.cli.duplex_voice.echo_llm import EchoLLM
    from vibe.cli.duplex_voice.jarvis_llm import JarvisBridgeLLM
    from vibe.cli.duplex_voice.mic_publisher import MicPublisher
    from vibe.cli.duplex_voice.mistral_stt_plugin import MistralDuplexSTT
    from vibe.cli.duplex_voice.mistral_tts_plugin import MistralDuplexTTS
    from vibe.cli.duplex_voice.supervisor import DuplexVoiceSupervisor

__all__ = [
    "DuplexVoiceSettings",
    "DuplexVoiceSupervisor",
    "EchoLLM",
    "JarvisBridgeLLM",
    "MicPublisher",
    "MistralDuplexSTT",
    "MistralDuplexTTS",
    "VoiceTurnBridge",
]

_LAZY_ATTRS = {
    "DuplexVoiceSettings": "vibe.cli.duplex_voice.duplex_config",
    "DuplexVoiceSupervisor": "vibe.cli.duplex_voice.supervisor",
    "EchoLLM": "vibe.cli.duplex_voice.echo_llm",
    "JarvisBridgeLLM": "vibe.cli.duplex_voice.jarvis_llm",
    "MicPublisher": "vibe.cli.duplex_voice.mic_publisher",
    "MistralDuplexSTT": "vibe.cli.duplex_voice.mistral_stt_plugin",
    "MistralDuplexTTS": "vibe.cli.duplex_voice.mistral_tts_plugin",
    "VoiceTurnBridge": "vibe.cli.duplex_voice.agent_bridge",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(name)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
