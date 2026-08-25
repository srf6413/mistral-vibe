"""Tests for `/background` (`_background_command` /
`_handoff_workflow_to_background` in `vibe/cli/textual_ui/app.py`).

Covers the three "no local task" error paths (never existed, already
finished, already running outside this session) plus the two "no run_id
given" paths (none running, ambiguous multiple), all via `ErrorMessage`
inspection -- and one happy-path test that drives a REAL
`subprocess.Popen` (via `_background_runner_argv`'s test seam, see its
docstring) against a trivial fixture script that writes a sentinel file
and exits, so it proves actual OS-level detachment (start_new_session,
log-file redirection under the run's own `RunPaths` directory, and the
post-spawn PID-alive verification) rather than merely that the right
function was called with the right arguments.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
import sys
from unittest.mock import AsyncMock

import pytest

from tests.conftest import build_test_vibe_app
import vibe.cli.textual_ui.app as app_module
from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.workflows import run_manager

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


async def _never_completes(*_args: object, **_kwargs: object) -> None:
    """Stand-in `execute_run`/`resume_run`: hangs until cancelled, so the
    task stays `_workflow_tasks`-resident (not `.done()`) until the test
    explicitly cancels it.
    """
    await asyncio.Event().wait()


def _last_error_text(vibe_app: VibeApp) -> str:
    mounted = vibe_app._mount_and_scroll.call_args_list  # type: ignore[attr-defined]
    errors = [
        args.args[0]._error
        for args in mounted
        if isinstance(args.args[0], ErrorMessage)
    ]
    assert errors, "expected at least one ErrorMessage to be mounted"
    return str(errors[-1])


def _last_notice_text(vibe_app: VibeApp) -> str:
    mounted = vibe_app._mount_and_scroll.call_args_list  # type: ignore[attr-defined]
    notices = [
        args.args[0]._content
        for args in mounted
        if isinstance(args.args[0], UserCommandMessage)
    ]
    assert notices, "expected at least one UserCommandMessage to be mounted"
    return str(notices[-1])


@pytest.fixture
def vibe_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> VibeApp:
    monkeypatch.setattr(app_module, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    return build_test_vibe_app()


# -- no run_id given -----------------------------------------------------------


@pytest.mark.asyncio
async def test_background_no_arg_zero_running_is_an_error(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        await vibe_app._background_command("")

        assert "no running workflow" in _last_error_text(vibe_app).lower()


@pytest.mark.asyncio
async def test_background_no_arg_multiple_running_lists_them_and_asks_to_specify(
    vibe_app: VibeApp,
) -> None:
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()
        task_a = asyncio.create_task(_never_completes())
        task_b = asyncio.create_task(_never_completes())
        vibe_app._workflow_tasks["wf-a"] = task_a
        vibe_app._workflow_tasks["wf-b"] = task_b
        try:
            await vibe_app._background_command("")

            text = _last_error_text(vibe_app)
            assert "wf-a" in text
            assert "wf-b" in text
            assert "/background <run_id>" in text
        finally:
            for task in (task_a, task_b):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


# -- explicit run_id, no local task: three distinct messages ------------------


@pytest.mark.asyncio
async def test_background_unknown_run_id_says_never_existed(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        await vibe_app._background_command("wf-does-not-exist")

        text = _last_error_text(vibe_app).lower()
        assert "no workflow run" in text
        assert "never" in text


@pytest.mark.asyncio
async def test_background_already_finished_run_id_says_so(
    vibe_app: VibeApp, tmp_path: Path
) -> None:
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        runs_root = app_module.DEFAULT_RUNS_ROOT
        script_path = _write_script(tmp_path)
        run_manager.create_run(
            "wf-done-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
        )
        journal = run_manager.JournalWriter(
            run_manager.RunPaths.for_run("wf-done-1", runs_root=runs_root).journal_path,
            run_id="wf-done-1",
        )
        journal.write_run_status("completed")

        await vibe_app._background_command("wf-done-1")

        text = _last_error_text(vibe_app).lower()
        assert "already finished" in text
        assert "completed" in text


@pytest.mark.asyncio
async def test_background_running_elsewhere_run_id_says_so(
    vibe_app: VibeApp, tmp_path: Path
) -> None:
    """A run whose journal shows `"running"` with a live pid and a fresh
    heartbeat, but which has no entry in `self._workflow_tasks` -- i.e. a
    run genuinely executing outside this process (a prior `/background`,
    or another session entirely).
    """
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        runs_root = app_module.DEFAULT_RUNS_ROOT
        script_path = _write_script(tmp_path)
        run_manager.create_run(
            "wf-elsewhere-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
        )
        journal = run_manager.JournalWriter(
            run_manager.RunPaths.for_run(
                "wf-elsewhere-1", runs_root=runs_root
            ).journal_path,
            run_id="wf-elsewhere-1",
        )
        journal.write_run_status("running")
        journal.heartbeat()

        await vibe_app._background_command("wf-elsewhere-1")

        text = _last_error_text(vibe_app).lower()
        assert "already running" in text
        assert "workflows show wf-elsewhere-1" in text


# -- happy path: real subprocess.Popen, real detachment mechanics -------------

SENTINEL_RUNNER_SCRIPT = """
import sys
import time
from pathlib import Path

args = sys.argv[1:]
run_id = args[args.index("--run-id") + 1]
runs_root = Path(args[args.index("--runs-root") + 1])

sentinel = runs_root / run_id / "sentinel.txt"
sentinel.write_text("background runner ran\\n", encoding="utf-8")
time.sleep(0.05)
"""


@pytest.mark.asyncio
async def test_background_happy_path_spawns_a_real_detached_process(
    vibe_app: VibeApp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture exits (code 0) well inside `_BACKGROUND_SPAWN_CHECK_DELAY_S`
    -- this exercises the `exit_code == 0` branch of the post-spawn check:
    a fast-finishing child is a SUCCESSFUL handoff, not a crash, and the
    notice must say so without claiming the (already-dead) process is
    still running -- that would be exactly the kind of fake liveness claim
    this codebase's liveness oracle exists to prevent. See
    `test_background_still_running_after_check_delay` for the
    `exit_code is None` ("actually still running") branch.

    The check-delay constant is patched generously larger here (real
    production value stays `0.3s`, unaffected -- this only patches the
    module attribute this one test observes). The fixture's own 0.05s
    sleep plus interpreter startup is normally well inside 0.3s, but under
    a heavily parallel full-suite run (many xdist workers contending for
    CPU) that margin is not reliable -- process fork/exec/import can itself
    take longer than 0.3s under load, which is a test-timing concern, not
    a defect in `_handoff_workflow_to_background`'s own logic. Widening the
    margin here removes the flake without weakening what the test proves:
    it is still a real, unmocked `subprocess.Popen` that the code under
    test polls for real.
    """
    monkeypatch.setattr(app_module, "_BACKGROUND_SPAWN_CHECK_DELAY_S", 3.0)
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        runs_root = app_module.DEFAULT_RUNS_ROOT
        script_path = _write_script(tmp_path)
        run_manager.create_run(
            "wf-live-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
        )

        fixture_script = tmp_path / "sentinel_runner.py"
        fixture_script.write_text(SENTINEL_RUNNER_SCRIPT, encoding="utf-8")

        vibe_app._background_runner_argv = lambda run_id: [  # type: ignore[method-assign]
            sys.executable,
            str(fixture_script),
            "--run-id",
            run_id,
            "--runs-root",
            str(runs_root),
        ]

        task = asyncio.create_task(_never_completes())
        vibe_app._workflow_tasks["wf-live-1"] = task
        vibe_app._workflow_buses["wf-live-1"] = app_module._WorkflowRunBus()

        await vibe_app._background_command("wf-live-1")

        # Give the real detached process a moment to run and exit (it
        # sleeps 0.05s itself); the sentinel file proves it actually ran
        # as an independent OS process our own event loop never awaited.
        run_dir = run_manager.RunPaths.for_run("wf-live-1", runs_root=runs_root).root
        sentinel = run_dir / "sentinel.txt"
        for _ in range(50):
            if sentinel.exists():
                break
            await asyncio.sleep(0.05)
        assert sentinel.exists(), "background_runner subprocess never ran"
        assert "background runner ran" in sentinel.read_text(encoding="utf-8")

        # The local task was cancelled and removed, never left dangling.
        assert "wf-live-1" not in vibe_app._workflow_tasks
        assert "wf-live-1" not in vibe_app._workflow_buses
        assert task.cancelled() or task.done()

        text = _last_notice_text(vibe_app)
        assert "wf-live-1" in text
        assert "handed off to background pid" in text
        # The fixture has already exited (code 0) by the time the
        # post-spawn check runs -- the notice must say so, and must NOT
        # claim the process is "still running" (fake liveness).
        assert "already finished" in text
        assert "still running" not in text
        assert "/workflows show wf-live-1" in text

        log_path = run_dir / "background.log"
        assert log_path.exists()


STILL_RUNNING_RUNNER_SCRIPT = """
import sys
import time
from pathlib import Path

args = sys.argv[1:]
run_id = args[args.index("--run-id") + 1]
runs_root = Path(args[args.index("--runs-root") + 1])

sentinel = runs_root / run_id / "sentinel.txt"
sentinel.write_text("background runner ran\\n", encoding="utf-8")
time.sleep(2.0)
"""


@pytest.mark.asyncio
async def test_background_still_running_after_check_delay(
    vibe_app: VibeApp, tmp_path: Path
) -> None:
    """Fixture is still alive when the post-spawn check fires (`poll()` ->
    `None`) -- the primary real-world case (a workflow run genuinely takes
    longer than `_BACKGROUND_SPAWN_CHECK_DELAY_S`). The notice must claim
    it is still running, since that claim is actually true here.
    """
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        runs_root = app_module.DEFAULT_RUNS_ROOT
        script_path = _write_script(tmp_path)
        run_manager.create_run(
            "wf-still-running-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
        )

        fixture_script = tmp_path / "still_running_runner.py"
        fixture_script.write_text(STILL_RUNNING_RUNNER_SCRIPT, encoding="utf-8")

        vibe_app._background_runner_argv = lambda run_id: [  # type: ignore[method-assign]
            sys.executable,
            str(fixture_script),
            "--run-id",
            run_id,
            "--runs-root",
            str(runs_root),
        ]

        task = asyncio.create_task(_never_completes())
        vibe_app._workflow_tasks["wf-still-running-1"] = task

        await vibe_app._background_command("wf-still-running-1")

        text = _last_notice_text(vibe_app)
        assert "wf-still-running-1" in text
        assert "handed off to background pid" in text
        assert "still running after you close this terminal" in text
        assert "already finished" not in text
        assert "/workflows show wf-still-running-1" in text

        # Let the (still-alive) fixture process finish naturally so the
        # test doesn't leak a lingering child.
        run_dir = run_manager.RunPaths.for_run(
            "wf-still-running-1", runs_root=runs_root
        ).root
        sentinel = run_dir / "sentinel.txt"
        for _ in range(50):
            if sentinel.exists():
                break
            await asyncio.sleep(0.05)
        assert sentinel.exists(), "background_runner subprocess never ran"


@pytest.mark.asyncio
async def test_background_reports_immediate_child_crash_plainly(
    vibe_app: VibeApp, tmp_path: Path
) -> None:
    """The child process exiting immediately (nonzero) must be reported as
    a failure, with the log tail included -- never claimed as a success.
    """
    async with vibe_app.run_test():
        vibe_app._mount_and_scroll = AsyncMock()

        runs_root = app_module.DEFAULT_RUNS_ROOT
        script_path = _write_script(tmp_path)
        run_manager.create_run(
            "wf-crash-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
        )

        crashing_script = tmp_path / "crash_runner.py"
        crashing_script.write_text(
            "import sys\n"
            "print('boom: deliberately crashing', file=sys.stderr)\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )

        vibe_app._background_runner_argv = lambda run_id: [  # type: ignore[method-assign]
            sys.executable,
            str(crashing_script),
        ]

        task = asyncio.create_task(_never_completes())
        vibe_app._workflow_tasks["wf-crash-1"] = task

        await vibe_app._background_command("wf-crash-1")

        text = _last_error_text(vibe_app)
        assert "wf-crash-1" in text
        assert "exited immediately" in text
        assert "boom: deliberately crashing" in text
