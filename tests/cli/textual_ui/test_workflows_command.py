"""Regression coverage for the exact bug class this integration branch
fixed: `_push_workflow_screen` used to import from a module
(`vibe.cli.textual_ui.widgets.workflow_screen`) that never existed, so the
`try/except ImportError: return` around it silently no-op'd on every
`/workflows run` -- the workflow ran fine in the background, but the live
progress screen the founder was told to expect never appeared, with no
error anywhere. `_push_workflow_screen` now imports the real
`WorkflowProgressScreen` (`vibe.cli.textual_ui.screens.workflow.workflow_screen`);
this test drives `/workflows run` through a real `VibeApp` pilot and
asserts the screen is actually the one mounted -- a return-type assertion
here would NOT have caught the original bug (the guarded import just
returns `None` either way), so this checks `app.screen` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_app
from tests.workflows.conftest import FakeAskTransport
import vibe.cli.textual_ui.app as app_module
from vibe.cli.textual_ui.screens.workflow.workflow_screen import WorkflowProgressScreen
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


@pytest.mark.asyncio
async def test_workflows_run_mounts_the_real_workflow_progress_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(app_module, "DEFAULT_RUNS_ROOT", runs_root)

    script_path = tmp_path / "hello.py"
    script_path.write_text(HELLO_SCRIPT, encoding="utf-8")

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok("hi back")])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )

    vibe_app = build_test_vibe_app()
    async with vibe_app.run_test() as pilot:
        await pilot.pause(0.1)
        await vibe_app._workflows_run(str(script_path))
        await pilot.pause(0.2)

        assert isinstance(vibe_app.screen, WorkflowProgressScreen), (
            f"expected WorkflowProgressScreen, got {type(vibe_app.screen)!r} -- "
            "this is the exact silent-no-op bug class this test guards against"
        )

        run_id = vibe_app.screen._run_id
        assert run_id in vibe_app._workflow_buses

        task = vibe_app._workflow_tasks.get(run_id)
        if task is not None:
            await task  # let the background run finish before the app exits

        # The tree pre-renders one pending node per WorkflowMeta.phases
        # entry and the run actually executed its one agent() call -- both
        # should have landed on the real, mounted tree by now.
        await pilot.pause(0.1)
        detail = run_manager.show_run(run_id, runs_root=runs_root)
        assert detail.status == "completed"


@pytest.mark.asyncio
async def test_workflows_show_reattaches_the_screen_with_replayed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Esc-detach -> `/workflows show <run_id>` reattach path: a run
    this session did NOT launch (no live bus) must still get a screen,
    populated from the journal via `replay_events`.
    """
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(app_module, "DEFAULT_RUNS_ROOT", runs_root)

    script_path = tmp_path / "hello.py"
    script_path.write_text(HELLO_SCRIPT, encoding="utf-8")
    run_manager.create_run(
        "wf-reattach-1", script_path, {}, runs_root=runs_root, cwd=tmp_path
    )

    def fake_transport_factory(**_kwargs: object) -> FakeAskTransport:
        return FakeAskTransport([FakeAskTransport.ok("hi back")])

    monkeypatch.setattr(
        run_manager, "HttpFounderOSAskTransport", fake_transport_factory
    )
    await run_manager.execute_run("wf-reattach-1", runs_root=runs_root)

    vibe_app = build_test_vibe_app()
    async with vibe_app.run_test() as pilot:
        await pilot.pause(0.1)
        # Nothing in this session's process launched this run -- no bus.
        assert "wf-reattach-1" not in vibe_app._workflow_buses

        await vibe_app._workflows_show("wf-reattach-1")
        await pilot.pause(0.2)

        assert isinstance(vibe_app.screen, WorkflowProgressScreen)
        assert vibe_app.screen._run_id == "wf-reattach-1"
        assert len(vibe_app.screen._initial_events) > 0
