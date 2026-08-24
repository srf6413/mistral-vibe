"""Tests for `vibe.workflows.run_manager`.

`runtime.py` and `agent_call.py` still have `NotImplementedError` bodies in
this worktree (other lanes own them), so these tests exercise run_manager's
own layers directly -- journal writing/reading, the liveness oracle,
create_run's persistence, list/show/skip/cancel, and the resume-cache
transport -- rather than driving a full `execute_run()` through a real
`WorkflowRuntime`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vibe.workflows.events import AgentCallEvent, PhaseEvent
from vibe.workflows.run_manager import (
    HEARTBEAT_STALE_AFTER_S,
    JournalWriter,
    RunPaths,
    _CachedAskTransport,
    _compute_status,
    _pid_alive,
    _read_journal_records,
    _resume_cache_by_position,
    cancel_run,
    create_run,
    list_runs,
    show_run,
    skip_run,
)
from vibe.workflows.script import WorkflowScriptError

HELLO_SCRIPT = """
meta = {
    "name": "hello",
    "description": "demo",
    "phases": [{"title": "Greet", "detail": ""}],
}


async def main(wf, args):
    async with wf.phase("Greet"):
        await wf.agent("hi", label="greet")
"""


def _write_script(tmp_path: Path, source: str = HELLO_SCRIPT) -> Path:
    path = tmp_path / "hello.py"
    path.write_text(source, encoding="utf-8")
    return path


# -- create_run ---------------------------------------------------------------


def test_create_run_persists_script_meta_and_returns_paths(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)

    paths = create_run(
        "wf-hello-abc123",
        script_path,
        {"name": "Ada"},
        runs_root=runs_root,
        cwd=tmp_path,
    )

    assert paths.run_id == "wf-hello-abc123"
    assert paths.script_path.read_text(encoding="utf-8") == HELLO_SCRIPT
    meta_doc = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    assert meta_doc["name"] == "hello"
    assert meta_doc["args"] == {"name": "Ada"}
    assert meta_doc["phases"] == [{"title": "Greet", "detail": ""}]
    assert not paths.journal_path.exists()  # nothing executed yet


def test_create_run_rejects_invalid_script(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("meta = {}\n", encoding="utf-8")
    with pytest.raises(WorkflowScriptError):
        create_run("wf-bad-1", bad, {}, runs_root=tmp_path / "runs")


def test_create_run_raises_on_id_collision(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run("wf-dup-1", script_path, {}, runs_root=runs_root)
    with pytest.raises(FileExistsError):
        create_run("wf-dup-1", script_path, {}, runs_root=runs_root)


# -- JournalWriter --------------------------------------------------------


def test_journal_writer_assigns_monotonic_seq_and_stamps_run_id(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")
    writer.log("first")
    writer.log("second")

    records = _read_journal_records(journal_path)
    assert [r["seq"] for r in records] == [0, 1]
    assert all(r["run_id"] == "wf-x" for r in records)
    assert all(r["kind"] == "log" for r in records)
    assert [r["message"] for r in records] == ["first", "second"]


def test_journal_writer_continues_seq_across_instances(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    JournalWriter(journal_path, run_id="wf-x").log("first")
    JournalWriter(journal_path, run_id="wf-x").log("second")

    records = _read_journal_records(journal_path)
    assert [r["seq"] for r in records] == [0, 1]


def test_journal_writer_write_event_phase_and_call(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")

    writer.write_event(
        PhaseEvent(
            run_id="wf-x",
            phase_id="phase-0",
            title="Greet",
            detail="d",
            state="running",
        )
    )
    writer.write_event(
        AgentCallEvent(
            run_id="wf-x",
            phase_id="phase-0",
            call_id="call-0",
            label="greet",
            state="ok",
            text="hello there",
        )
    )

    records = _read_journal_records(journal_path)
    assert records[0]["kind"] == "phase"
    assert records[0]["state"] == "running"
    assert records[0]["label"] == "Greet"
    assert records[0]["detail"] == "d"

    assert records[1]["kind"] == "call"
    assert records[1]["call_id"] == "call-0"
    assert records[1]["state"] == "ok"
    assert records[1]["text"] == "hello there"


def test_journal_writer_rewrites_ok_to_cached_for_cached_position(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")

    writer.write_event(
        AgentCallEvent(
            run_id="wf-x",
            phase_id="phase-0",
            call_id="call-3",
            label="x",
            state="ok",
            text="cached answer",
        ),
        cached_positions={3},
    )
    writer.write_event(
        AgentCallEvent(
            run_id="wf-x",
            phase_id="phase-0",
            call_id="call-4",
            label="y",
            state="ok",
            text="live answer",
        ),
        cached_positions={3},
    )

    records = _read_journal_records(journal_path)
    assert records[0]["state"] == "cached"
    assert records[1]["state"] == "ok"


def test_journal_writer_heartbeat_and_run_status_carry_pid(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")
    writer.heartbeat()
    writer.write_run_status("completed")

    records = _read_journal_records(journal_path)
    assert records[0]["kind"] == "heartbeat"
    assert records[0]["pid"] == os.getpid()
    assert records[1]["kind"] == "run_status"
    assert records[1]["run_status"] == "completed"
    assert records[1]["pid"] == os.getpid()


# -- liveness oracle ------------------------------------------------------


def test_pid_alive_true_for_self_and_false_for_finished_process() -> None:
    assert _pid_alive(os.getpid()) is True

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert _pid_alive(proc.pid) is False


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_compute_status_running_when_heartbeat_fresh_and_pid_alive() -> None:
    now = datetime.now(UTC)
    records = [
        {
            "kind": "run_status",
            "run_status": "running",
            "pid": os.getpid(),
            "ts": _iso(now),
        },
        {"kind": "heartbeat", "pid": os.getpid(), "ts": _iso(now)},
    ]
    status, last_hb, pid = _compute_status(records, created_at=_iso(now))
    assert status == "running"
    assert pid == os.getpid()
    assert last_hb == _iso(now)


def test_compute_status_lost_when_heartbeat_stale() -> None:
    stale = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_STALE_AFTER_S + 5)
    records = [
        {
            "kind": "run_status",
            "run_status": "running",
            "pid": os.getpid(),
            "ts": _iso(stale),
        },
        {"kind": "heartbeat", "pid": os.getpid(), "ts": _iso(stale)},
    ]
    status, _last_hb, _pid = _compute_status(records, created_at=_iso(stale))
    assert status == "lost"


def test_compute_status_lost_when_pid_dead() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    now = datetime.now(UTC)
    records = [
        {
            "kind": "run_status",
            "run_status": "running",
            "pid": proc.pid,
            "ts": _iso(now),
        },
        {"kind": "heartbeat", "pid": proc.pid, "ts": _iso(now)},
    ]
    status, _last_hb, _pid = _compute_status(records, created_at=_iso(now))
    assert status == "lost"


def test_compute_status_terminal_status_wins_even_if_heartbeat_stale() -> None:
    stale = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_STALE_AFTER_S + 100)
    records = [
        {
            "kind": "run_status",
            "run_status": "running",
            "pid": os.getpid(),
            "ts": _iso(stale),
        },
        {"kind": "heartbeat", "pid": os.getpid(), "ts": _iso(stale)},
        {
            "kind": "run_status",
            "run_status": "completed",
            "pid": os.getpid(),
            "ts": _iso(stale),
        },
    ]
    status, _last_hb, _pid = _compute_status(records, created_at=_iso(stale))
    assert status == "completed"


def test_compute_status_running_before_first_heartbeat_uses_created_at() -> None:
    now = datetime.now(UTC)
    records = [
        {
            "kind": "run_status",
            "run_status": "running",
            "pid": os.getpid(),
            "ts": _iso(now),
        }
    ]
    status, last_hb, _pid = _compute_status(records, created_at=_iso(now))
    assert status == "running"
    assert last_hb is None


# -- list_runs / show_run --------------------------------------------------


def test_list_runs_empty_when_root_missing(tmp_path: Path) -> None:
    assert list_runs(runs_root=tmp_path / "does-not-exist") == []


def test_list_runs_and_show_run_reflect_created_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run(
        "wf-hello-1", script_path, {"name": "Ada"}, runs_root=runs_root, cwd=tmp_path
    )

    summaries = list_runs(runs_root=runs_root)
    assert len(summaries) == 1
    assert summaries[0].run_id == "wf-hello-1"
    assert summaries[0].name == "hello"
    # No run_status record at all yet and created_at is "now" -> not stale.
    assert summaries[0].status == "running"

    detail = show_run("wf-hello-1", runs_root=runs_root)
    assert detail.meta.name == "hello"
    assert detail.args == {"name": "Ada"}
    assert detail.status == "running"


def test_show_run_raises_for_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        show_run("wf-does-not-exist", runs_root=tmp_path / "runs")


def test_list_runs_orders_most_recently_created_first(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run("wf-a", script_path, {}, runs_root=runs_root)
    # Force distinct created_at ordering deterministically (test infra, not
    # workflow script code -- fine to touch wall clock/files here).
    paths_a = RunPaths.for_run("wf-a", runs_root=runs_root)
    meta_a = json.loads(paths_a.meta_path.read_text(encoding="utf-8"))
    meta_a["created_at"] = "2020-01-01T00:00:00+00:00"
    paths_a.meta_path.write_text(json.dumps(meta_a), encoding="utf-8")

    create_run("wf-b", script_path, {}, runs_root=runs_root)
    paths_b = RunPaths.for_run("wf-b", runs_root=runs_root)
    meta_b = json.loads(paths_b.meta_path.read_text(encoding="utf-8"))
    meta_b["created_at"] = "2030-01-01T00:00:00+00:00"
    paths_b.meta_path.write_text(json.dumps(meta_b), encoding="utf-8")

    summaries = list_runs(runs_root=runs_root)
    assert [s.run_id for s in summaries] == ["wf-b", "wf-a"]


# -- skip_run / cancel_run --------------------------------------------------


@pytest.mark.asyncio
async def test_skip_run_appends_skipped_call_record(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run("wf-skip-1", script_path, {}, runs_root=runs_root)
    paths = RunPaths.for_run("wf-skip-1", runs_root=runs_root)
    JournalWriter(paths.journal_path, run_id="wf-skip-1").write_event(
        AgentCallEvent(
            run_id="wf-skip-1",
            phase_id="phase-0",
            call_id="call-0",
            label="greet",
            state="running",
        )
    )

    await skip_run("wf-skip-1", "call-0", runs_root=runs_root)

    records = _read_journal_records(paths.journal_path)
    last = records[-1]
    assert last["kind"] == "call"
    assert last["call_id"] == "call-0"
    assert last["state"] == "skipped"
    assert last["phase_id"] == "phase-0"  # recovered from the earlier record
    assert last["reason"]


@pytest.mark.asyncio
async def test_skip_run_raises_for_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await skip_run("wf-does-not-exist", "call-0", runs_root=tmp_path / "runs")


@pytest.mark.asyncio
async def test_cancel_run_on_lost_run_only_appends_terminal_record(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run("wf-cancel-1", script_path, {}, runs_root=runs_root)
    paths = RunPaths.for_run("wf-cancel-1", runs_root=runs_root)

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    stale = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_STALE_AFTER_S + 5)
    writer = JournalWriter(paths.journal_path, run_id="wf-cancel-1")
    writer._append({  # noqa: SLF001 - test seeds a synthetic lost-run journal
        "kind": "run_status",
        "phase_id": None,
        "call_id": None,
        "parent_call_id": None,
        "state": None,
        "label": None,
        "detail": None,
        "text": None,
        "reason": None,
        "message": None,
        "level": None,
        "pid": proc.pid,
        "run_status": "running",
    })

    await cancel_run("wf-cancel-1", runs_root=runs_root)  # must not raise

    detail = show_run("wf-cancel-1", runs_root=runs_root)
    assert detail.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_run_never_signals_current_process(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path)
    create_run("wf-cancel-2", script_path, {}, runs_root=runs_root)
    paths = RunPaths.for_run("wf-cancel-2", runs_root=runs_root)
    JournalWriter(paths.journal_path, run_id="wf-cancel-2").write_run_status("running")

    # If this ever sent SIGTERM to os.getpid(), the test process would die.
    await cancel_run("wf-cancel-2", runs_root=runs_root)

    detail = show_run("wf-cancel-2", runs_root=runs_root)
    assert detail.status == "cancelled"


# -- resume cache -----------------------------------------------------------


def test_resume_cache_by_position_only_includes_terminal_ok_or_cached(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")
    writer.write_call_state(
        phase_id="phase-0", call_id="call-0", state="ok", text="answer 0"
    )
    writer.write_call_state(
        phase_id="phase-0", call_id="call-1", state="error", reason="boom"
    )
    writer.write_call_state(phase_id="phase-0", call_id="call-2", state="running")

    cache = _resume_cache_by_position(journal_path)
    assert cache == {0: "answer 0"}


def test_resume_cache_by_position_uses_latest_state_per_call_id(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    writer = JournalWriter(journal_path, run_id="wf-x")
    writer.write_call_state(phase_id="phase-0", call_id="call-0", state="running")
    writer.write_call_state(
        phase_id="phase-0", call_id="call-0", state="ok", text="final answer"
    )

    cache = _resume_cache_by_position(journal_path)
    assert cache == {0: "final answer"}


@pytest.mark.asyncio
async def test_cached_ask_transport_yields_one_final_result_with_no_network() -> None:
    transport = _CachedAskTransport("the cached reply")
    events = [event async for event in transport.stream({"text": "ignored"})]
    assert events == [{"type": "final_result", "data": {"answer": "the cached reply"}}]
    await transport.cancel()
    await transport.close()


@pytest.mark.asyncio
async def test_cached_ask_transport_drives_a_real_session_without_network(
    tmp_path: Path,
) -> None:
    from vibe.app_server._founderos_ask import FounderOSAskSession
    from vibe.app_server.events import HistoryEntryAdded
    from vibe.app_server.models import PublicMessageEntry

    transport = _CachedAskTransport("cached hello")
    session = FounderOSAskSession(transport=transport, cwd=tmp_path, session_id="s-1")
    events = [event async for event in session.act("hi")]
    added = [e for e in events if isinstance(e, HistoryEntryAdded)]
    assert any(
        isinstance(e.entry, PublicMessageEntry)
        and e.entry.role == "assistant"
        and e.entry.text == "cached hello"
        for e in added
    )
    await session.close()
