"""Standalone, non-pytest proof that the duplex voice service layer works
end to end, WITHOUT the TUI.

Not a pytest test (pytest's global per-test timeout in this repo is 10s;
starting a real livekit-server + agent process and letting audio flow
comfortably needs longer). Run it directly::

    uv run python scripts/duplex_voice_proof.py

What it does, for real, no mocks:

1. Starts the `DuplexVoiceSupervisor` (real `livekit-server --dev` +
   real `vibe.cli.duplex_voice.agent` subprocess).
2. Joins the same room as a second participant ("the user").
3. Publishes a synthetic tone as that participant's mic track.
4. Waits, then tears everything down and reports:
   - whether the server/agent came up and the agent joined the room,
   - whether the agent's STT plugin actually received audio frames,
   - whether it opened a Mistral transcription segment,
   - and how far it got: a real transcript (if MISTRAL_API_KEY is set and
     valid) or a clean, observable failure (if not).

If MISTRAL_API_KEY isn't available in this environment, that is reported
explicitly rather than silently mocked around -- the pipeline plumbing
(server up, agent joins, frames delivered, segment opened, connection
attempted) is still proven; only the live Mistral round trip is not.
"""

from __future__ import annotations

import array
import asyncio
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import rtc

from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.supervisor import (
    DuplexVoiceSupervisor,
    DuplexVoiceSupervisorError,
)
from vibe.utils.api_keys import resolve_api_key

TONE_SAMPLE_RATE = 48000
TONE_DURATION_S = 2.0
TONE_FREQUENCY_HZ = 440.0
TONE_AMPLITUDE = 12000  # loud relative to the plugin's 400.0 RMS threshold
SETTLE_AFTER_PUBLISH_S = 3.0


def _make_tone_frame() -> rtc.AudioFrame:
    n_samples = int(TONE_SAMPLE_RATE * TONE_DURATION_S)
    samples = array.array("h")
    for i in range(n_samples):
        value = TONE_AMPLITUDE * math.sin(
            2 * math.pi * TONE_FREQUENCY_HZ * i / TONE_SAMPLE_RATE
        )
        samples.append(int(value))
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=TONE_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=n_samples,
    )


async def _publish_tone(room: rtc.Room) -> None:
    source = rtc.AudioSource(TONE_SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("proof-tone", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)

    frame = _make_tone_frame()
    chunk_samples = TONE_SAMPLE_RATE // 50  # 20ms chunks
    total_samples = frame.samples_per_channel
    data = frame.data
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = rtc.AudioFrame(
            data=data[start:end].tobytes(),
            sample_rate=TONE_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=end - start,
        )
        await source.capture_frame(chunk)
        await asyncio.sleep(0.02)


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _read_log(path: Path) -> str:
    if not path.exists():
        return "(no log file)"
    return path.read_text(errors="replace")


def _report_mistral_key_status() -> None:
    api_key = resolve_api_key("MISTRAL_API_KEY")
    if api_key:
        print("MISTRAL_API_KEY: present -- expecting a full Mistral round trip.")
    else:
        print(
            "MISTRAL_API_KEY: NOT FOUND in environment or keyring.\n"
            "  Proceeding anyway: this proves the server/room/agent/audio-frame\n"
            "  plumbing, but the Mistral connection attempt itself will fail\n"
            "  (reported below), not be silently mocked around."
        )


async def _start_supervisor_or_report(supervisor: DuplexVoiceSupervisor) -> bool:
    _print_header("1. Starting supervisor (livekit-server --dev + agent)")
    try:
        await supervisor.start()
    except DuplexVoiceSupervisorError as exc:
        print(f"FAILED to start supervisor: {exc}")
        return False

    status = supervisor.status()
    print(f"server running: {status.server_running} (pid={status.server_pid})")
    print(f"agent running:  {status.agent_running} (pid={status.agent_pid})")
    print(f"server log: {status.server_log_path}")
    print(f"agent log:  {status.agent_log_path}")
    if not (status.server_running and status.agent_running):
        print("FAILED: server and/or agent did not come up.")
        return False
    return True


async def _join_and_publish_tone(settings: DuplexVoiceSettings) -> None:
    _print_header("2. Joining as a second participant and publishing a tone")
    user_room = rtc.Room()
    user_jwt = settings.mint_token(identity="duplex-voice-proof-user")
    await user_room.connect(settings.livekit_url, user_jwt)
    print(
        f"test participant connected: identity={user_room.local_participant.identity}"
    )

    await _publish_tone(user_room)
    print(f"published {TONE_DURATION_S:.1f}s tone; letting the pipeline settle...")
    await asyncio.sleep(SETTLE_AFTER_PUBLISH_S)

    await user_room.disconnect()
    print("test participant disconnected")


def _assess_agent_log(agent_log: str) -> None:
    _print_header("4. Assessment")
    joined = "connected remotes=" in agent_log
    got_frames = "frames received=" in agent_log
    opened_segment = "segment started" in agent_log
    got_transcript = "segment finished text=" in agent_log
    segment_failed = "segment failed" in agent_log
    print(f"agent joined the room:               {joined}")
    print(f"STT plugin received audio frames:    {got_frames}")
    print(f"STT plugin opened a Mistral segment:  {opened_segment}")
    if got_transcript:
        print("Mistral transcription round trip:    SUCCEEDED (real transcript)")
    elif segment_failed and "HTTP 401" in agent_log:
        print(
            "Mistral transcription round trip:    FAILED with HTTP 401 Unauthorized -- "
            "reached the real Mistral realtime endpoint and was rejected for lacking a "
            "valid MISTRAL_API_KEY. This is the expected outcome without one."
        )
    elif segment_failed and "CERTIFICATE_VERIFY_FAILED" in agent_log:
        print(
            "Mistral transcription round trip:    FAILED at the TLS handshake -- local "
            "Python SSL trust store issue (common on python.org macOS builds), not a "
            "plugin defect. Set SSL_CERT_FILE to certifi's bundle to get past this and "
            "see the real auth-level result."
        )
    elif segment_failed:
        print(
            "Mistral transcription round trip:    FAILED (see agent log above -- "
            "expected when MISTRAL_API_KEY is unavailable)"
        )
    else:
        print(
            "Mistral transcription round trip:    inconclusive (no segment outcome logged)"
        )


async def main() -> int:
    _print_header("Duplex voice service layer -- standalone end-to-end proof")
    _report_mistral_key_status()

    supervisor = DuplexVoiceSupervisor()
    try:
        if not await _start_supervisor_or_report(supervisor):
            return 1

        await _join_and_publish_tone(supervisor.settings)

        _print_header("3. Agent log evidence")
        agent_log = _read_log(supervisor.agent_log_path)
        print(agent_log)
        _assess_agent_log(agent_log)
    finally:
        _print_header("5. Tearing down")
        await supervisor.stop()
        port_freed = not await _port_still_bound()
        print(f"supervisor stopped; port 7880 freed: {port_freed}")

    return 0


async def _port_still_bound() -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 7880), timeout=1.0
        )
    except (TimeoutError, OSError):
        return False
    writer.close()
    return True


if __name__ == "__main__":
    start = time.monotonic()
    exit_code = asyncio.run(main())
    print(f"\nTotal wall time: {time.monotonic() - start:.1f}s")
    raise SystemExit(exit_code)
