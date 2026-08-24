"""Process supervisor for the duplex voice service layer.

Owns two child processes:

1. ``livekit-server --dev`` -- a local, config-free LiveKit server that
   binds ``ws://127.0.0.1:7880`` and auto-generates the placeholder
   ``devkey``/``secret`` API credentials.
2. the long-running duplex voice agent (`vibe.cli.duplex_voice.agent`),
   which joins the room the server hosts and runs the STT -> room -> TTS
   pipeline until told to stop.

This is intentionally *not* a system-level daemon (no launchd unit, no
supervisor process that outlives its parent) -- it is an importable
start()/stop() pair. A later stage calls `start()` when voice mode toggles
on and `stop()` when it toggles off.
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
import sys
import tempfile
import time

from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.core.utils.async_subprocess import kill_async_subprocess

logger = logging.getLogger("jarvis.duplex_voice.supervisor")

_DEFAULT_LIVEKIT_SERVER_BINARY = "/usr/local/bin/livekit-server"
_HTTP_HOST = "127.0.0.1"
_HTTP_PORT = 7880
_GRACEFUL_STOP_TIMEOUT_S = 5.0


class DuplexVoiceSupervisorError(RuntimeError):
    """Raised when the supervisor can't reach a required precondition
    (e.g. the port it needs is already occupied by something else, or a
    child process failed to become ready in time).
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
    agent_pid: int | None
    server_log_path: Path | None
    agent_log_path: Path | None


class DuplexVoiceSupervisor:
    """Starts/stops `livekit-server --dev` + the duplex voice agent together."""

    def __init__(
        self,
        settings: DuplexVoiceSettings | None = None,
        *,
        livekit_server_binary: str | None = None,
        http_host: str = _HTTP_HOST,
        http_port: int = _HTTP_PORT,
        startup_timeout: float = 15.0,
        log_dir: Path | None = None,
    ) -> None:
        self._settings = settings or DuplexVoiceSettings()
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
        self._agent_log_path = self._log_dir / "duplex-agent.log"

        self._server_proc: asyncio.subprocess.Process | None = None
        self._agent_proc: asyncio.subprocess.Process | None = None
        self._server_log_fh: object | None = None
        self._agent_log_fh: object | None = None

    @property
    def settings(self) -> DuplexVoiceSettings:
        return self._settings

    @property
    def server_log_path(self) -> Path:
        return self._server_log_path

    @property
    def agent_log_path(self) -> Path:
        return self._agent_log_path

    @property
    def is_running(self) -> bool:
        return (
            self._server_proc is not None
            and self._server_proc.returncode is None
            and self._agent_proc is not None
            and self._agent_proc.returncode is None
        )

    def status(self) -> SupervisorStatus:
        return SupervisorStatus(
            server_running=self._server_proc is not None
            and self._server_proc.returncode is None,
            agent_running=self._agent_proc is not None
            and self._agent_proc.returncode is None,
            server_pid=self._server_proc.pid if self._server_proc is not None else None,
            agent_pid=self._agent_proc.pid if self._agent_proc is not None else None,
            server_log_path=self._server_log_path
            if self._server_proc is not None
            else None,
            agent_log_path=self._agent_log_path
            if self._agent_proc is not None
            else None,
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
        await self._start_agent()

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
        self._agent_log_fh = open(self._agent_log_path, "wb")
        env = {**os.environ, **self._settings.as_env()}
        logger.info(
            "starting duplex voice agent room=%s identity=%s",
            self._settings.room,
            self._settings.agent_identity,
        )
        self._agent_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "vibe.cli.duplex_voice.agent",
            env=env,
            stdout=self._agent_log_fh,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        # Confirm the process is actually alive after a short grace period
        # rather than instantly exiting (bad token, can't reach the server,
        # import error, etc.) -- cheap, and catches the most common
        # misconfigurations before callers assume the pipeline is up.
        await asyncio.sleep(1.0)
        if self._agent_proc.returncode is not None:
            await self._stop_server()
            raise DuplexVoiceSupervisorError(
                f"duplex voice agent exited early (code={self._agent_proc.returncode}); "
                f"see {self._agent_log_path}"
            )
        logger.info("duplex voice agent running pid=%d", self._agent_proc.pid)

    async def stop(self) -> None:
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
        proc = self._agent_proc
        self._agent_proc = None
        if proc is None or proc.returncode is not None:
            return
        logger.info("stopping duplex voice agent pid=%d", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=_GRACEFUL_STOP_TIMEOUT_S)
        except (TimeoutError, ProcessLookupError):
            await kill_async_subprocess(proc)

    async def _stop_server(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        if proc is None or proc.returncode is not None:
            return
        logger.info("stopping livekit-server pid=%d", proc.pid)
        await kill_async_subprocess(proc)

    def _close_log_handles(self) -> None:
        for fh in (self._server_log_fh, self._agent_log_fh):
            if fh is not None:
                try:
                    fh.close()  # type: ignore[attr-defined]
                except OSError:
                    pass
        self._server_log_fh = None
        self._agent_log_fh = None

    async def __aenter__(self) -> DuplexVoiceSupervisor:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()
