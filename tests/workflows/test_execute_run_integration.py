"""Integration-lane tests for `execute_run`'s wiring across `runtime.py`,
`agent_call.py`, and `run_manager.py` together -- as opposed to
`test_run_manager.py` (which predates all three lanes landing and tests
`run_manager`'s own layers in isolation) or `test_runtime.py`/
`test_agent_call.py` (which mock across their own lane boundary).

These exercise the REAL `agent_call.call_agent` and the REAL
`WorkflowRuntime`, with only `HttpFounderOSAskTransport` swapped out (via
`run_manager.HttpFounderOSAskTransport`) for the network-free
`FakeAskTransport` from `tests/workflows/conftest.py` -- so a bug in how
`execute_run` wires the two together (not just in either lane's own code)
would actually be caught here. Covers three integration gaps found and
fixed while merging the lane branches:

- `wf.log(...)` (which cannot go through the frozen `WorkflowEvent`
  union's `emit` channel) is drained into the journal and `extra_log`.
- `extra_emit` (the live-fan-out hook `/workflows run` wires to a
  `WorkflowProgressScreen`) sees the same `"ok"` -> `"cached"` rewrite the
  journal record gets, so a live viewer and the journal agree.
- `skip_run` is actually honored on the next `execute_run` (not just
  recorded for display), AND without corrupting the resume cache's call
  position for any call issued *after* the skipped one -- a regression
  test for a real bug this integration found: the original
  `itertools.count()`-based position tracking in `execute_run`'s
  `session_factory` only advanced on an actual `session_factory()`
  invocation, so a skipped call (which never reaches `session_factory`)
  silently shifted every later call's resume-cache lookup by one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.workflows.conftest import FakeAskTransport
from vibe.workflows import run_manager
from vibe.workflows.events import AgentCallEvent
from vibe.workflows.run_manager import (
    RunPaths,
    _read_journal_records,
    create_run,
    skip_run,
)

ONE_CALL_SCRIPT = """
meta = {
    "name": "hello",
    "description": "demo",
    "phases": [{"title": "Greet", "detail": ""}],
}


async def main(wf, args):
    async with wf.phase("Greet"):
        await wf.agent("hi", label="greet")
"""

LOG_SCRIPT = """
meta = {
    "name": "logger",
    "description": "demo",
    "phases": [{"title": "Work", "detail": ""}],
}


async def main(wf, args):
    async with wf.phase("Work"):
        wf.log("before call", level="info")
        await wf.agent("hi", label="greet")
        wf.log("after call", level="warning")
"""

THREE_CALL_SCRIPT = """
meta = {
    "name": "three-calls",
    "description": "demo",
    "phases": [{"title": "Work", "detail": ""}],
}


async def main(wf, args):
    async with wf.phase("Work"):
        await wf.agent("first", label="first")
        await wf.agent("second", label="second")
        await wf.agent("third", label="third")
"""


def _write_script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "script.py"
    path.write_text(source, encoding="utf-8")
    return path


def _latest_call_records(journal_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in _read_journal_records(journal_path):
        if record.get("kind") == "call" and record.get("call_id"):
            latest[record["call_id"]] = record
    return latest


@pytest.mark.asyncio
async def test_execute_run_drains_wf_log_into_journal_and_extra_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path, LOG_SCRIPT)
    create_run("wf-log-1", script_path, {}, runs_root=runs_root, cwd=tmp_path)

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok("hello back")])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )

    seen_logs: list[tuple[str, str]] = []
    await run_manager.execute_run(
        "wf-log-1",
        runs_root=runs_root,
        extra_log=lambda level, message: seen_logs.append((level, message)),
    )

    paths = RunPaths.for_run("wf-log-1", runs_root=runs_root)
    log_records = [
        r for r in _read_journal_records(paths.journal_path) if r["kind"] == "log"
    ]
    assert [(r["level"], r["message"]) for r in log_records] == [
        ("info", "before call"),
        ("warning", "after call"),
    ]
    assert seen_logs == [("info", "before call"), ("warning", "after call")]


@pytest.mark.asyncio
async def test_execute_run_extra_emit_sees_cached_rewrite_matching_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path, ONE_CALL_SCRIPT)
    create_run("wf-emit-1", script_path, {}, runs_root=runs_root, cwd=tmp_path)

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok("hi back")])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )
    await run_manager.execute_run("wf-emit-1", runs_root=runs_root)

    def forbidden_transport_factory(**_kwargs: object) -> FakeAskTransport:
        raise AssertionError(
            "the only call in this run was already 'ok' -- the resumed "
            "execute_run must serve it from the resume cache, never build "
            "a real transport"
        )

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", forbidden_transport_factory
    )

    emitted: list[AgentCallEvent] = []
    await run_manager.execute_run(
        "wf-emit-1",
        runs_root=runs_root,
        extra_emit=lambda event: (
            emitted.append(event) if isinstance(event, AgentCallEvent) else None
        ),
    )

    terminal = [e for e in emitted if e.state != "running"]
    assert len(terminal) == 1
    assert terminal[0].state == "cached"
    assert terminal[0].text == "hi back"

    paths = RunPaths.for_run("wf-emit-1", runs_root=runs_root)
    latest = _latest_call_records(paths.journal_path)
    assert latest["call-0"]["state"] == "cached"
    assert latest["call-0"]["text"] == "hi back"


@pytest.mark.asyncio
async def test_execute_run_honors_skip_without_shifting_later_call_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: skip a middle call, then resume -- the trailing
    call must still resolve to ITS OWN cached answer (position 2's "C"),
    not whatever the (now-misaligned) Nth `session_factory()` invocation
    would have pointed at under the old invocation-count approach. See
    this module's docstring.
    """
    runs_root = tmp_path / "runs"
    script_path = _write_script(tmp_path, THREE_CALL_SCRIPT)
    create_run("wf-skip-align-1", script_path, {}, runs_root=runs_root, cwd=tmp_path)

    answers = iter(["A", "B", "C"])

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok(next(answers))])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )
    await run_manager.execute_run("wf-skip-align-1", runs_root=runs_root)

    paths = RunPaths.for_run("wf-skip-align-1", runs_root=runs_root)
    latest = _latest_call_records(paths.journal_path)
    assert [latest[f"call-{i}"]["text"] for i in range(3)] == ["A", "B", "C"]

    await skip_run("wf-skip-align-1", "call-1", runs_root=runs_root)

    def forbidden_transport_factory(**_kwargs: object) -> FakeAskTransport:
        raise AssertionError(
            "call-0 and call-2 are already 'ok' in the journal and call-1 "
            "is permanently skipped -- no call in this resumed run should "
            "ever need a real transport"
        )

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", forbidden_transport_factory
    )
    await run_manager.execute_run("wf-skip-align-1", runs_root=runs_root)

    latest = _latest_call_records(paths.journal_path)
    assert latest["call-0"]["state"] == "cached"
    assert latest["call-0"]["text"] == "A"
    assert latest["call-1"]["state"] == "skipped"
    assert latest["call-2"]["state"] == "cached"
    assert latest["call-2"]["text"] == "C"
