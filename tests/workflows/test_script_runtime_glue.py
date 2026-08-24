"""Tests for `vibe.workflows.script.run_workflow_script` -- the glue that
loads/lints/compiles a workflow script file and runs its `main(wf, args)`
against a real `WorkflowRuntime`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibe.workflows import agent_call
from vibe.workflows.agent_call import AgentCallResult
from vibe.workflows.events import WorkflowMeta
from vibe.workflows.runtime import WorkflowRuntime
from vibe.workflows.script import WorkflowScriptError, run_workflow_script

GOOD_SCRIPT = """
meta = {
    "name": "greet",
    "description": "says hi",
    "phases": [{"title": "Greet"}],
}


async def main(wf, args):
    async with wf.phase("Greet"):
        result = await wf.agent(f"say hi to {args['name']}")
        wf.log(f"got: {result.text}")
"""


def _session_factory():
    raise AssertionError("not used in this test")


@pytest.mark.asyncio
async def test_run_workflow_script_executes_main_against_a_real_runtime(
    tmp_path: Path, monkeypatch
):
    async def fake_call_agent(prompt, *, opts, session_factory):
        return AgentCallResult(status="ok", text="hi there", reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)

    script_path = tmp_path / "greet.py"
    script_path.write_text(GOOD_SCRIPT)

    wf = WorkflowRuntime(
        run_id="run-glue",
        meta=WorkflowMeta(name="greet", description="says hi", phases=[]),
        session_factory=_session_factory,
    )

    await run_workflow_script(script_path, wf, {"name": "world"})

    assert wf.log_records[-1]["message"] == "got: hi there"


@pytest.mark.asyncio
async def test_run_workflow_script_raises_workflow_script_error_for_bad_script(
    tmp_path: Path,
):
    script_path = tmp_path / "bad.py"
    script_path.write_text("import time\nasync def main(wf, args):\n    pass\n")

    wf = WorkflowRuntime(
        run_id="run-bad",
        meta=WorkflowMeta(name="bad", description="", phases=[]),
        session_factory=_session_factory,
    )

    with pytest.raises(WorkflowScriptError):
        await run_workflow_script(script_path, wf, {})
