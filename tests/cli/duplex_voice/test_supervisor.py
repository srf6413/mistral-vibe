from __future__ import annotations

import pytest

from vibe.cli.duplex_voice import supervisor as supervisor_module
from vibe.cli.duplex_voice.duplex_config import DuplexVoiceSettings
from vibe.cli.duplex_voice.supervisor import (
    DuplexVoiceSupervisor,
    DuplexVoiceSupervisorError,
)


@pytest.mark.asyncio
async def test_start_refuses_to_run_when_port_already_bound(monkeypatch) -> None:
    async def _fake_port_is_bound(
        host: str, port: int, *, timeout: float = 0.5
    ) -> bool:
        return True

    monkeypatch.setattr(supervisor_module, "_port_is_bound", _fake_port_is_bound)

    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings())

    with pytest.raises(DuplexVoiceSupervisorError, match="already bound"):
        await sup.start()

    assert not sup.is_running


def test_status_reports_not_running_before_start() -> None:
    sup = DuplexVoiceSupervisor(settings=DuplexVoiceSettings())

    status = sup.status()

    assert status.server_running is False
    assert status.agent_running is False
    assert status.server_pid is None
    assert status.agent_pid is None
