"""Process/task supervisor for the duplex voice service layer.

Owns:

1. ``livekit-server --dev`` -- a local, config-free LiveKit server
   subprocess that binds ``ws://127.0.0.1:7880`` and auto-generates the
   placeholder ``devkey``/``secret`` API credentials.
2. the long-running duplex voice agent (`vibe.cli.duplex_voice.agent`),
   which joins the room the server hosts and runs the STT -> LLM -> TTS
   pipeline until told to stop -- run as an IN-PROCESS asyncio task, not a
   subprocess (see "In-process, not subprocess" below).
3. optionally (`enable_mic=True`), a mic publisher
   (`vibe.cli.duplex_voice.mic_publisher.MicPublisher`) that captures the
   real microphone and publishes it into the room as a second participant
   -- also an in-process task.

This is intentionally *not* a system-level daemon (no launchd unit, no
supervisor process that outlives its parent) -- it is an importable
start()/stop() pair. `VibeApp` calls `start()` when voice mode toggles on
and `stop()` when it toggles off (and unconditionally on app shutdown), all
in `vibe/cli/textual_ui/app.py`.

In-process, not subprocess
---------------------------
The originally proposed shape for this supervisor spawned the agent as a
*subprocess* (`python -m vibe.cli.duplex_voice.agent`), talking to the real
`AppServerSession` in the TUI process over some form of IPC. That's no
longer how this works: the agent (this module's `_start_agent`) now runs as
an `asyncio.Task` *inside* the same process as the TUI, and its LLM plugin
(`JarvisBridgeLLM`, via `llm_factory` below) calls straight into that
process's own `AppServerSession` methods -- the exact same object the
keyboard input path already drives. No IPC of any kind is needed for that;
only `livekit-server` itself remains a separate OS process, because it's a
real network service other participants (this process's own mic publisher,
and in principle any other LiveKit client) need to reach over a socket.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import TYPE_CHECKING

from vibe.cli.duplex_voice.agent import run_duplex_agent
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.echo_llm import EchoLLM
from vibe.cli.duplex_voice.mic_publisher import MicPublisher
from vibe.cli.duplex_voice.playback_subscriber import PlaybackSubscriber
from vibe.utils.platform import is_windows

if TYPE_CHECKING:
    from collections.abc import Callable

    from livekit.agents import llm

    from vibe.app_server.config import SpeechConfigView, TranscriptionConfigView

logger = logging.getLogger("jarvis.duplex_voice.supervisor")

_DEFAULT_LIVEKIT_SERVER_BINARY = "/usr/local/bin/livekit-server"
_HTTP_HOST = "127.0.0.1"
_HTTP_PORT = 7880
_GRACEFUL_STOP_TIMEOUT_S = 5.0
_AGENT_START_GRACE_S = 0.5
_MIC_START_GRACE_S = 0.2
_PLAYBACK_START_GRACE_S = 0.2


class DuplexVoiceSupervisorError(RuntimeError):
    """Raised when the supervisor can't reach a required precondition
    (e.g. the port it needs is already occupied by something else, or a
    child process/task failed to become ready in time).
    """


async def _http_probe(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """Return True if a plain HTTP GET to (host, port) gets any response.

    Mirrors the manual smoke test this task description references:
    `livekit-server --dev` answers 200 OK on a bare GET to its HTTP port.
    We only check that *something* answered (not the exact status), since
    the goal here is "is the port bound and serving", not response
    validation.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (TimeoutError, OSError):
        return False
    try:
        writer.write(
            f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
        )
        await writer.drain()
        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return status_line.startswith(b"HTTP/")
    except (TimeoutError, OSError):
        return False
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def _kill_subprocess(proc: asyncio.subprocess.Process) -> None:
    """Force-terminate `proc` (SIGKILL its process group on Unix; `terminate()`
    on Windows) and wait for it to actually exit.

    A small, local equivalent of
    `vibe.core.utils.async_subprocess.kill_async_subprocess` -- not reused
    directly because this module is reachable from `vibe.cli.textual_ui.app`
    now, which may not transitively import `vibe.core` (see
    `tests/cli/textual_ui/test_app_server_boundary.py` and
    `vibe.cli.duplex_voice.duplex_config`'s module docstring). Only ever
    called on `self._server_proc`, which this supervisor always starts with
    `start_new_session=True`, so it's always safe to signal its whole
    process group.
    """
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if is_windows():
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        await proc.wait()


async def _port_is_bound(host: str, port: int, *, timeout: float = 0.5) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (TimeoutError, OSError):
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


@dataclass
class SupervisorStatus:
    server_running: bool
    agent_running: bool
    server_pid: int | None
    # The agent now runs as an in-process asyncio task, not a separate OS
    # process -- there is no PID or dedicated log file for it any more.
    # Kept on this dataclass (always None) rather than removed, so any
    # existing caller matching on this shape doesn't need to change.
    agent_pid: None
    server_log_path: Path | None
    agent_log_path: None


class DuplexVoiceSupervisor:
    """Starts/stops `livekit-server --dev` + the in-process duplex voice
    agent (+ optionally the mic publisher) together.
    """

    def __init__(  # noqa: PLR0913
        self,
        settings: DuplexVoiceSettings | None = None,
        *,
        transcription: TranscriptionConfigView,
        speech: SpeechConfigView,
        livekit_server_binary: str | None = None,
        http_host: str = _HTTP_HOST,
        http_port: int = _HTTP_PORT,
        startup_timeout: float = 15.0,
        log_dir: Path | None = None,
        llm_factory: Callable[[], llm.LLM] | None = None,
        enable_mic: bool = False,
        enable_playback: bool = False,
        muted: Callable[[], bool] | None = None,
    ) -> None:
        """`transcription`/`speech` are required, caller-supplied config for
        the STT/TTS plugins -- this module deliberately does not compute
        defaults for them itself (that would need `vibe.core.config`; see
        `vibe.cli.duplex_voice.duplex_config`'s module docstring for why
        that's off limits here). Real TUI wiring
        (`vibe/cli/textual_ui/app.py`) passes the current session's own
        `self.app_server.resources.config.current.transcription`/`.speech`;
        standalone callers (`scripts/duplex_voice_proof.py`) pass
        `vibe.cli.duplex_voice.standalone_defaults`'s hardcoded defaults.

        `llm_factory` builds the LLM plugin the in-process agent uses;
        defaults to `EchoLLM` (this supervisor's original, standalone-
        provable behavior -- see `scripts/duplex_voice_proof.py`). Real TUI
        wiring passes ``lambda: JarvisBridgeLLM(bridge)`` instead.

        `enable_mic` additionally starts a `MicPublisher` capturing the
        real microphone; `muted` (defaults to "never muted") is threaded
        through to it to gate mic forwarding. Off by default so the
        existing proof script's own fake-participant/synthetic-tone
        publishing keeps working unchanged.

        `enable_playback` additionally starts a `PlaybackSubscriber` that
        joins the room and renders the agent's synthesized speech through
        local speakers -- see `vibe.cli.duplex_voice.playback_subscriber`
        for why that's a separate, real room participant rather than
        something wired straight into the TTS plugin. Off by default for
        the same reason `enable_mic` is: keeps
        `scripts/duplex_voice_proof.py` (which asserts on log lines from a
        room with exactly two participants) unchanged.
        """
        self._settings = settings or DuplexVoiceSettings()
        self._transcription = transcription
        self._speech = speech
        self._binary = (
            livekit_server_binary
            or shutil.which("livekit-server")
            or _DEFAULT_LIVEKIT_SERVER_BINARY
        )
        self._http_host = http_host
        self._http_port = http_port
        self._startup_timeout = startup_timeout
        self._log_dir = log_dir or Path(tempfile.mkdtemp(prefix="jarvis-duplex-voice-"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._server_log_path = self._log_dir / "livekit-server.log"

        self._llm_factory = llm_factory
        self._enable_mic = enable_mic
        self._enable_playback = enable_playback
        self._muted = muted

        self._server_proc: asyncio.subprocess.Process | None = None
        self._server_log_fh: object | None = None

        self._agent_task: asyncio.Task[None] | None = None
        self._agent_stop_event: asyncio.Event | None = None

        self._mic_task: asyncio.Task[None] | None = None
        self._playback_task: asyncio.Task[None] | None = None

    @property
    def settings(self) -> DuplexVoiceSettings:
        return self._settings

    @property
    def server_log_path(self) -> Path:
        return self._server_log_path

    @property
    def is_running(self) -> bool:
        return (
            self._server_proc is not None
            and self._server_proc.returncode is None
            and self._agent_task is not None
            and not self._agent_task.done()
        )

    def status(self) -> SupervisorStatus:
        return SupervisorStatus(
            server_running=self._server_proc is not None
            and self._server_proc.returncode is None,
            agent_running=self._agent_task is not None and not self._agent_task.done(),
            server_pid=self._server_proc.pid if self._server_proc is not None else None,
            agent_pid=None,
            server_log_path=self._server_log_path
            if self._server_proc is not None
            else None,
            agent_log_path=None,
        )

    async def start(self) -> None:
        if self.is_running:
            return

        if await _port_is_bound(self._http_host, self._http_port):
            raise DuplexVoiceSupervisorError(
                f"{self._http_host}:{self._http_port} is already bound by something else -- "
                "refusing to start a second livekit-server against it, and refusing to adopt "
                "(and later tear down) a process this supervisor didn't start."
            )

        await self._start_server()
        try:
            await self._start_agent()
        except DuplexVoiceSupervisorError:
            await self._stop_server()
            raise

        if self._enable_mic:
            # Deliberately non-fatal: a machine/container with no working
            # microphone (this dev sandbox included) should still get a
            # working agent bridge + TTS-out, not a hard failure on the
            # whole toggle. `_start_mic` logs and leaves `_mic_task` unset
            # on failure.
            await self._start_mic()

        if self._enable_playback:
            # Same non-fatal contract as `_start_mic`: a machine/container
            # with no working speakers should still get a working agent
            # bridge (text still round-trips through the turn), not a hard
            # failure on the whole toggle. `_start_playback` logs and
            # leaves `_playback_task` unset on failure.
            await self._start_playback()

    async def _start_server(self) -> None:
        self._server_log_fh = open(self._server_log_path, "wb")
        logger.info("starting livekit-server binary=%s", self._binary)
        self._server_proc = await asyncio.create_subprocess_exec(
            self._binary,
            "--dev",
            stdout=self._server_log_fh,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._server_proc.returncode is not None:
                raise DuplexVoiceSupervisorError(
                    f"livekit-server exited early (code={self._server_proc.returncode}); "
                    f"see {self._server_log_path}"
                )
            if await _http_probe(self._http_host, self._http_port):
                logger.info(
                    "livekit-server listening on %s:%d pid=%d",
                    self._http_host,
                    self._http_port,
                    self._server_proc.pid,
                )
                return
            await asyncio.sleep(0.2)

        await self._stop_server()
        raise DuplexVoiceSupervisorError(
            f"livekit-server did not become ready within {self._startup_timeout}s; "
            f"see {self._server_log_path}"
        )

    async def _start_agent(self) -> None:
        llm_plugin = (self._llm_factory or EchoLLM)()
        stop_event = asyncio.Event()
        self._agent_stop_event = stop_event
        logger.info(
            "starting in-process duplex voice agent room=%s identity=%s",
            self._settings.room,
            self._settings.agent_identity,
        )
        self._agent_task = asyncio.create_task(
            run_duplex_agent(
                self._settings,
                llm_plugin=llm_plugin,
                transcription=self._transcription,
                speech=self._speech,
                stop_event=stop_event,
            ),
            name="jarvis-duplex-voice-agent",
        )

        # Confirm the task didn't die immediately (bad token, can't reach
        # the server, import error, etc.) -- cheap, and catches the most
        # common misconfigurations before callers assume the pipeline is
        # up.
        await asyncio.sleep(_AGENT_START_GRACE_S)
        if self._agent_task.done():
            exc = self._agent_task.exception()
            self._agent_task = None
            self._agent_stop_event = None
            raise DuplexVoiceSupervisorError(
                f"duplex voice agent failed to start: {exc!r}"
            )
        logger.info("duplex voice agent running (in-process task)")

    async def _start_mic(self) -> None:
        mic = MicPublisher(self._settings, muted=self._muted or (lambda: False))
        task = asyncio.create_task(mic.run(), name="jarvis-duplex-voice-mic")
        await asyncio.sleep(_MIC_START_GRACE_S)
        if task.done():
            exc = task.exception()
            logger.warning(
                "duplex mic publisher failed to start; continuing without a mic "
                "(agent bridge and TTS-out are unaffected): %r",
                exc,
            )
            return
        self._mic_task = task
        logger.info("duplex mic publisher running (in-process task)")

    async def _start_playback(self) -> None:
        playback = PlaybackSubscriber(self._settings)
        task = asyncio.create_task(
            playback.run(), name="jarvis-duplex-voice-playback"
        )
        await asyncio.sleep(_PLAYBACK_START_GRACE_S)
        if task.done():
            exc = task.exception()
            logger.warning(
                "duplex playback subscriber failed to start; continuing without "
                "local speaker output (agent bridge and mic-in are unaffected): %r",
                exc,
            )
            return
        self._playback_task = task
        logger.info("duplex playback subscriber running (in-process task)")

    async def stop(self) -> None:
        await self._stop_playback()
        await self._stop_mic()
        await self._stop_agent()
        await self._stop_server()
        self._close_log_handles()

        if await _port_is_bound(self._http_host, self._http_port):
            logger.warning(
                "port %s:%d still bound after teardown",
                self._http_host,
                self._http_port,
            )

    async def _stop_agent(self) -> None:
        task = self._agent_task
        stop_event = self._agent_stop_event
        self._agent_task = None
        self._agent_stop_event = None
        if task is None:
            return
        if stop_event is not None:
            stop_event.set()
        if task.done():
            return
        logger.info("stopping duplex voice agent")
        try:
            await asyncio.wait_for(task, timeout=_GRACEFUL_STOP_TIMEOUT_S)
        except TimeoutError:
            logger.warning("duplex voice agent did not stop gracefully; cancelling")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except Exception:
            logger.exception("duplex voice agent raised while stopping")

    async def _stop_mic(self) -> None:
        task = self._mic_task
        self._mic_task = None
        if task is None or task.done():
            return
        logger.info("stopping duplex mic publisher")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _stop_playback(self) -> None:
        task = self._playback_task
        self._playback_task = None
        if task is None or task.done():
            return
        logger.info("stopping duplex playback subscriber")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _stop_server(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        if proc is None or proc.returncode is not None:
            return
        logger.info("stopping livekit-server pid=%d", proc.pid)
        await _kill_subprocess(proc)

    def _close_log_handles(self) -> None:
        if self._server_log_fh is not None:
            try:
                self._server_log_fh.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        self._server_log_fh = None

    async def __aenter__(self) -> DuplexVoiceSupervisor:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()
