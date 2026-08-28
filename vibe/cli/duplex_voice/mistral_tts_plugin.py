"""LiveKit TTS plugin wrapping jarvis's own :class:`MistralTTSClient`.

`MistralTTSClient.speak()` is a one-shot (non-streamed) call, matching
`tts.ChunkedStream`'s own docstring: "used by the non-streamed synthesize
API". LiveKit's `AudioEmitter`/`AudioStreamDecoder` handle WAV decoding and
resampling to our declared output rate automatically, so no manual audio
parsing is needed here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit.agents import APIConnectionError, APIError, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from vibe.app_server.config import AudioProviderView, TTSModelConfigView
from vibe.cli.duplex_voice.announce_lock import wait_while_other_speaker
from vibe.cli.tts.mistral_tts_client import MistralTTSClient

if TYPE_CHECKING:
    from vibe.cli.tts.tts_client_port import TTSClientPort

logger = logging.getLogger("jarvis.duplex_voice.tts")

# Same model ("voxtral-mini-tts-latest") the official livekit-plugins-mistralai
# TTS plugin targets; its declared output format is 24kHz mono, and LiveKit's
# AudioStreamDecoder resamples the actual WAV payload to whatever we declare
# here regardless, so this is the safe/confirmed choice rather than a guess.
_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1


class MistralDuplexTTS(tts.TTS):
    """LiveKit TTS plugin backed by jarvis's `MistralTTSClient`."""

    def __init__(
        self,
        provider: AudioProviderView,
        model: TTSModelConfigView,
        *,
        client: TTSClientPort | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._model = model
        self._client: TTSClientPort = client or MistralTTSClient(
            provider=provider, model=model
        )

    @property
    def model(self) -> str:
        return self._model.name

    @property
    def provider(self) -> str:
        return "mistral"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> MistralChunkedStream:
        return MistralChunkedStream(
            tts=self, client=self._client, input_text=text, conn_options=conn_options
        )

    async def aclose(self) -> None:
        await self._client.close()


class MistralChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: MistralDuplexTTS,
        client: TTSClientPort,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Wait out a `speak.sh`-driven Mac announcement already in progress
        # before this utterance starts speaking -- the synth-side half of
        # the shared announce lock (see `announce_lock`'s module
        # docstring); `SpeakingLockCoordinator` holds the other half.
        await wait_while_other_speaker()
        logger.info("synthesizing %d chars", len(self._input_text))
        try:
            result = await self._client.speak(self._input_text)
        except APIError:
            raise
        except Exception as exc:
            # MistralTTSClient (or the SDK/HTTP layer underneath it) can
            # raise directly rather than an APIError -- without wrapping it,
            # ChunkedStream._main_task's retry loop (which only retries on
            # APIError) would treat a single transient failure as fatal.
            raise APIConnectionError(str(exc)) from exc
        logger.info("synthesized %d bytes of audio", len(result.audio_data))
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/wav",
            stream=False,
        )
        output_emitter.push(result.audio_data)
