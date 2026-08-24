"""Tests for `vibe.workflows.background_runner`.

Unit-level coverage of argument parsing and of `run_to_completion`'s calls
into `run_manager` (mocked -- never a real network transport), plus one
real, non-mocked-`execute_run` integration test that empirically proves the
race-safety finding this module's docstring documents: after a `/background`
handoff, the run's journal shows a terminal `"cancelled"` `run_status` (the
local task's own cancellation handling wrote it), so `resume_run` would
refuse to continue it (`ValueError`, status not in `{"lost", "failed"}`) --
`run_to_completion` must go through `execute_run` directly instead, which
has no such precondition and completes the run anyway.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.workflows.conftest import FakeAskTransport
from vibe.workflows import background_runner, run_manager

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


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "hello.py"
    path.write_text(HELLO_SCRIPT, encoding="utf-8")
    return path


# -- argument parsing ---------------------------------------------------------


def test_build_arg_parser_requires_run_id() -> None:
    parser = background_runner.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_arg_parser_parses_run_id_only() -> None:
    parser = background_runner.build_arg_parser()
    args = parser.parse_args(["--run-id", "wf-abc123"])
    assert args.run_id == "wf-abc123"
    assert args.runs_root is None
    assert args.endpoint is None
    assert args.workspace is None


def test_build_arg_parser_parses_all_optional_args(tmp_path: Path) -> None:
    parser = background_runner.build_arg_parser()
    args = parser.parse_args([
        "--run-id",
        "wf-abc123",
        "--runs-root",
        str(tmp_path),
        "--endpoint",
        "https://ask.example.test",
        "--workspace",
        "ws-1",
    ])
    assert args.run_id == "wf-abc123"
    assert args.runs_root == tmp_path
    assert args.endpoint == "https://ask.example.test"
    assert args.workspace == "ws-1"


# -- run_to_completion: calls the right run_manager function -----------------


def test_run_to_completion_calls_execute_run_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_to_completion` must call `run_manager.execute_run` (never
    `resume_run` -- see the module docstring for why) with the run_id and
    every optional override forwarded unchanged.
    """
    calls: list[dict[str, Any]] = []

    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        calls.append({"run_id": run_id, **kwargs})

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    exit_code = asyncio.run(
        background_runner.run_to_completion(
            "wf-xyz",
            runs_root=tmp_path,
            endpoint="https://ask.example.test",
            workspace="ws-9",
        )
    )

    assert exit_code == 0
    assert calls == [
        {
            "run_id": "wf-xyz",
            "runs_root": tmp_path,
            "endpoint": "https://ask.example.test",
            "workspace": "ws-9",
        }
    ]


def test_run_to_completion_never_calls_resume_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        del run_id, kwargs

    async def unexpected_resume_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "background_runner must call execute_run, not resume_run -- "
            "resume_run's status gate would reject a just-cancelled run"
        )

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)
    monkeypatch.setattr(run_manager, "resume_run", unexpected_resume_run)

    exit_code = asyncio.run(
        background_runner.run_to_completion("wf-xyz", runs_root=tmp_path)
    )
    assert exit_code == 0


# -- run_to_completion: error handling ----------------------------------------


def test_run_to_completion_returns_1_and_logs_to_stderr_when_run_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        del kwargs
        raise FileNotFoundError(f"no run directory for {run_id!r} under {tmp_path}")

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    exit_code = asyncio.run(
        background_runner.run_to_completion("wf-missing", runs_root=tmp_path)
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "wf-missing" in captured.err
    assert "no run directory" in captured.err


def test_run_to_completion_returns_1_and_logs_traceback_on_other_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        del run_id, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    exit_code = asyncio.run(
        background_runner.run_to_completion("wf-broken", runs_root=tmp_path)
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "wf-broken" in captured.err
    assert "boom" in captured.err
    assert "RuntimeError" in captured.err  # traceback.print_exc() landed too


def test_run_to_completion_propagates_cancelled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine kill signal (SIGTERM -> asyncio cancellation) must not be
    swallowed as an ordinary failure -- `execute_run`'s own cancellation
    handling has already written the terminal journal record by the time
    this propagates (see the module docstring).
    """

    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        del run_id, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            background_runner.run_to_completion("wf-killed", runs_root=tmp_path)
        )


# -- main(): argv -> exit code -------------------------------------------------


def test_main_returns_0_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        del run_id, kwargs

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    exit_code = background_runner.main([
        "--run-id",
        "wf-main-1",
        "--runs-root",
        str(tmp_path),
    ])
    assert exit_code == 0


def test_main_defaults_runs_root_to_run_manager_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_manager, "DEFAULT_RUNS_ROOT", tmp_path)
    seen: dict[str, Any] = {}

    async def fake_execute_run(run_id: str, **kwargs: Any) -> None:
        seen["run_id"] = run_id
        seen["runs_root"] = kwargs["runs_root"]

    monkeypatch.setattr(run_manager, "execute_run", fake_execute_run)

    exit_code = background_runner.main(["--run-id", "wf-main-2"])
    assert exit_code == 0
    assert seen == {"run_id": "wf-main-2", "runs_root": tmp_path}


# -- real integration: proves the race-safety / execute_run-not-resume_run --
# -- finding empirically, not just by mocking ---------------------------------


@pytest.mark.asyncio
async def test_background_runner_completes_a_run_that_resume_run_would_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates exactly the post-handoff journal state: a run whose local
    owner was cancelled (so its journal's latest `run_status` is
    `"cancelled"`, written by `execute_run`'s own `except
    asyncio.CancelledError` handler -- see `run_manager.py`). `resume_run`
    refuses to touch such a run; `run_to_completion` (via `execute_run`
    directly) must still drive it to `"completed"`.
    """
    runs_root = tmp_path / "runs"
    run_id = "wf-handoff-1"
    script_path = _write_script(tmp_path)
    run_manager.create_run(run_id, script_path, {}, runs_root=runs_root, cwd=tmp_path)

    paths = run_manager.RunPaths.for_run(run_id, runs_root=runs_root)
    journal = run_manager.JournalWriter(paths.journal_path, run_id=run_id)
    journal.write_run_status("running")
    journal.write_run_status("cancelled")

    detail = run_manager.show_run(run_id, runs_root=runs_root)
    assert detail.status == "cancelled"

    with pytest.raises(ValueError, match="only a 'lost' or 'failed' run"):
        await run_manager.resume_run(run_id, runs_root=runs_root)

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok("hi back")])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )

    exit_code = await background_runner.run_to_completion(run_id, runs_root=runs_root)

    assert exit_code == 0
    final = run_manager.show_run(run_id, runs_root=runs_root)
    assert final.status == "completed"
