"""Tests for `vibe.workflows.runtime.WorkflowRuntime`.

These tests mock `agent_call.call_agent` directly (module-attribute
monkeypatch, per the runtime lane's contract note) rather than driving a
real/fake `FounderOSAskSession` -- the `agent_call` lane's real
implementation may not exist yet, and this module's job is to test
orchestration (phases, ids, concurrency, budget, nesting), not the
LLM-call path itself (that's `tests/workflows/test_agent_call.py`'s job).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vibe.workflows import agent_call
from vibe.workflows.agent_call import AgentCallResult
from vibe.workflows.events import AgentCallEvent, PhaseEvent, WorkflowMeta
from vibe.workflows.runtime import WorkflowBudget, WorkflowRuntime


def _session_factory():
    raise AssertionError("session_factory should never be called directly by tests")


def _meta(*titles: str) -> WorkflowMeta:
    from vibe.workflows.events import PhaseSpec

    return WorkflowMeta(
        name="t", description="t", phases=[PhaseSpec(title=t) for t in titles]
    )


def _make_runtime(
    *, meta: WorkflowMeta | None = None, emit=None, run_id: str = "run-1"
) -> WorkflowRuntime:
    return WorkflowRuntime(
        run_id=run_id,
        meta=meta or _meta("Phase A"),
        session_factory=_session_factory,
        emit=emit,
    )


class _RecordingCallAgent:
    """Programmable stand-in for `agent_call.call_agent`.

    Records every `(prompt, opts, session_factory)` it was called with (in
    call order) and returns results from a queue (or a single canned result
    for every call if only one was programmed).
    """

    def __init__(self, results: list[AgentCallResult] | AgentCallResult) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, prompt: str, *, opts: dict[str, Any], session_factory
    ) -> AgentCallResult:
        self.calls.append({
            "prompt": prompt,
            "opts": opts,
            "session_factory": session_factory,
        })
        if isinstance(self._results, list):
            return self._results.pop(0)
        return self._results


@pytest.mark.asyncio
async def test_agent_outside_phase_raises(monkeypatch):
    monkeypatch.setattr(
        agent_call,
        "call_agent",
        _RecordingCallAgent(AgentCallResult(status="ok", text="x", reason=None)),
    )
    wf = _make_runtime()
    with pytest.raises(RuntimeError, match="phase"):
        await wf.agent("hello")


@pytest.mark.asyncio
async def test_phase_nesting_raises():
    wf = _make_runtime()
    with pytest.raises(RuntimeError, match="do not nest"):
        async with wf.phase("Phase A"):
            async with wf.phase("Phase A"):
                pass


@pytest.mark.asyncio
async def test_agent_mints_call_ids_and_emits_running_then_terminal(monkeypatch):
    fake = _RecordingCallAgent(AgentCallResult(status="ok", text="hi", reason=None))
    monkeypatch.setattr(agent_call, "call_agent", fake)

    events: list[Any] = []
    wf = _make_runtime(emit=events.append)

    async with wf.phase("Phase A"):
        result = await wf.agent("do the thing", label="mylabel")

    assert result.status == "ok"
    phase_events = [e for e in events if isinstance(e, PhaseEvent)]
    call_events = [e for e in events if isinstance(e, AgentCallEvent)]

    assert [e.state for e in phase_events] == ["running", "ok"]
    assert phase_events[0].phase_id == "phase-0"

    assert [e.state for e in call_events] == ["running", "ok"]
    assert call_events[0].call_id == "call-0"
    assert call_events[0].phase_id == "phase-0"
    assert call_events[0].label == "mylabel"
    assert call_events[0].text is None
    assert call_events[1].call_id == "call-0"
    assert call_events[1].text == "hi"
    assert call_events[1].parent_call_id is None


@pytest.mark.asyncio
async def test_call_id_counter_is_run_scoped_not_reset_per_phase(monkeypatch):
    fake = _RecordingCallAgent(AgentCallResult(status="ok", text="hi", reason=None))
    monkeypatch.setattr(agent_call, "call_agent", fake)

    events: list[Any] = []
    wf = _make_runtime(meta=_meta("A", "B"), emit=events.append)

    async with wf.phase("A"):
        await wf.agent("p1")
    async with wf.phase("B"):
        await wf.agent("p2")
        await wf.agent("p3")

    call_events = [e for e in events if isinstance(e, AgentCallEvent)]
    running = [e for e in call_events if e.state == "running"]
    assert [e.call_id for e in running] == ["call-0", "call-1", "call-2"]
    assert [e.phase_id for e in running] == ["phase-0", "phase-1", "phase-1"]


@pytest.mark.asyncio
async def test_session_id_default_and_override(monkeypatch):
    fake = _RecordingCallAgent([
        AgentCallResult(status="ok", text="a", reason=None),
        AgentCallResult(status="ok", text="b", reason=None),
    ])
    monkeypatch.setattr(agent_call, "call_agent", fake)

    wf = _make_runtime(run_id="run-xyz")
    async with wf.phase("Phase A"):
        await wf.agent("p1")
        await wf.agent("p2", opts={"session_id": "custom-id"})

    assert fake.calls[0]["opts"]["session_id"] == "wf-run-xyz-call-0"
    assert fake.calls[1]["opts"]["session_id"] == "custom-id"
    # session_factory is forwarded unchanged, never wrapped.
    assert fake.calls[0]["session_factory"] is _session_factory


@pytest.mark.asyncio
async def test_parallel_runs_concurrently_and_preserves_result_order(monkeypatch):
    started = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def fake_call_agent(prompt, *, opts, session_factory):
        nonlocal started, max_in_flight
        started += 1
        max_in_flight = max(max_in_flight, started)
        if started >= 2:
            release.set()
        await release.wait()
        started -= 1
        return AgentCallResult(status="ok", text=prompt, reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)
    wf = _make_runtime()

    async with wf.phase("Phase A"):
        results = await wf.parallel([wf.agent("p1"), wf.agent("p2")])

    assert max_in_flight >= 2  # true overlap, not serialized
    assert [r.text for r in results] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_parallel_concurrency_cap_holds(monkeypatch):
    max_in_flight = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def fake_call_agent(prompt, *, opts, session_factory):
        nonlocal max_in_flight, in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)
        async with lock:
            in_flight -= 1
        return AgentCallResult(status="ok", text=prompt, reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)
    monkeypatch.setenv("FOUNDEROS_WORKFLOW_MAX_CONCURRENCY", "2")
    wf = _make_runtime()

    async with wf.phase("Phase A"):
        await wf.parallel([wf.agent(f"p{i}") for i in range(6)])

    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_parallel_call_ids_mint_in_list_order(monkeypatch):
    order: list[str] = []

    async def fake_call_agent(prompt, *, opts, session_factory):
        order.append(opts["session_id"])
        return AgentCallResult(status="ok", text=prompt, reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)
    wf = _make_runtime(run_id="r")

    async with wf.phase("Phase A"):
        await wf.parallel([wf.agent(f"p{i}") for i in range(5)])

    assert order == [f"wf-r-call-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_budget_exhausted_short_circuits_without_calling_call_agent(monkeypatch):
    fake = _RecordingCallAgent(AgentCallResult(status="ok", text="x", reason=None))
    monkeypatch.setattr(agent_call, "call_agent", fake)
    monkeypatch.setenv("FOUNDEROS_WORKFLOW_MAX_CALLS", "1")

    events: list[Any] = []
    wf = _make_runtime(emit=events.append)

    async with wf.phase("Phase A"):
        first = await wf.agent("p1")
        second = await wf.agent("p2")

    assert first.status == "ok"
    assert second.status == "cancelled"
    assert second.reason is not None
    assert len(fake.calls) == 1  # call_agent never invoked for the 2nd call

    call_events = [e for e in events if isinstance(e, AgentCallEvent)]
    second_call_events = [e for e in call_events if e.call_id == "call-1"]
    assert [e.state for e in second_call_events] == ["running", "cancelled"]


@pytest.mark.asyncio
async def test_budget_elapsed_time_ceiling_short_circuits_agent(monkeypatch):
    fake = _RecordingCallAgent(AgentCallResult(status="ok", text="x", reason=None))
    monkeypatch.setattr(agent_call, "call_agent", fake)

    wf = _make_runtime()
    # Swap in a budget whose clock reports the ceiling already crossed on
    # the very first exhausted() check, independent of call count. Two
    # ticks: one for `_ensure_started()`'s start-stamp, one for the elapsed
    # check inside the same `exhausted()` call.
    ticks = iter([0.0, 10.0])
    wf._budget = WorkflowBudget(
        max_calls=999.0, max_seconds=1.0, clock=lambda: next(ticks)
    )

    async with wf.phase("Phase A"):
        result = await wf.agent("p1")

    assert result.status == "cancelled"
    assert fake.calls == []  # call_agent never invoked


@pytest.mark.asyncio
async def test_budget_does_not_overshoot_under_parallel(monkeypatch):
    async def fake_call_agent(prompt, *, opts, session_factory):
        await asyncio.sleep(0.01)
        return AgentCallResult(status="ok", text=prompt, reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)
    monkeypatch.setenv("FOUNDEROS_WORKFLOW_MAX_CALLS", "3")
    wf = _make_runtime()

    async with wf.phase("Phase A"):
        results = await wf.parallel([wf.agent(f"p{i}") for i in range(10)])

    ok_count = sum(1 for r in results if r.status == "ok")
    assert ok_count == 3


@pytest.mark.asyncio
async def test_nesting_guard_raises_during_in_flight_agent_call(monkeypatch):
    outer = _make_runtime(run_id="outer")

    async def fake_call_agent(prompt, *, opts, session_factory):
        with pytest.raises(RuntimeError, match="nested workflow"):
            _make_runtime(run_id="inner")
        return AgentCallResult(status="ok", text="x", reason=None)

    monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)

    async with outer.phase("Phase A"):
        result = await outer.agent("p1")
    assert result.status == "ok"


def test_nesting_guard_clears_after_agent_call_completes(monkeypatch):
    # After a completed agent() call, constructing a new WorkflowRuntime in
    # the same context must succeed -- the guard must not leak.
    async def fake_call_agent(prompt, *, opts, session_factory):
        return AgentCallResult(status="ok", text="x", reason=None)

    async def run():
        monkeypatch.setattr(agent_call, "call_agent", fake_call_agent)
        wf = _make_runtime(run_id="a")
        async with wf.phase("Phase A"):
            await wf.agent("p1")
        # Should not raise.
        _make_runtime(run_id="b")

    asyncio.run(run())


@pytest.mark.asyncio
async def test_pipeline_sequential_fold_and_stage_chaining(monkeypatch):
    async def double(x: int) -> int:
        return x * 2

    async def plus_one(x: int) -> int:
        return x + 1

    wf = _make_runtime()
    result = await wf.pipeline([double, plus_one], initial=5)
    assert result == 11


@pytest.mark.asyncio
async def test_pipeline_stops_early_on_raise():
    calls: list[str] = []

    async def ok_step(x):
        calls.append("ok_step")
        return x

    async def raising_step(x):
        calls.append("raising_step")
        raise ValueError("boom")

    async def never_reached(x):
        calls.append("never_reached")
        return x

    wf = _make_runtime()
    with pytest.raises(ValueError, match="boom"):
        await wf.pipeline([ok_step, raising_step, never_reached], initial=1)

    assert calls == ["ok_step", "raising_step"]


@pytest.mark.asyncio
async def test_pipeline_step_agent_calls_nest_under_first_call_in_step(monkeypatch):
    fake = _RecordingCallAgent([
        AgentCallResult(status="ok", text="a", reason=None),
        AgentCallResult(status="ok", text="b", reason=None),
        AgentCallResult(status="ok", text="c", reason=None),
    ])
    monkeypatch.setattr(agent_call, "call_agent", fake)

    events: list[Any] = []
    wf = _make_runtime(meta=_meta("Phase A", "Phase B"), emit=events.append)

    async def step_a(_):
        async with wf.phase("Phase A"):
            r1 = await wf.agent("first")
            r2 = await wf.agent("second")
        return (r1, r2)

    async def step_b(prev):
        async with wf.phase("Phase B"):
            r3 = await wf.agent("third")
        return prev + (r3,)

    await wf.pipeline([step_a, step_b], initial=None)

    call_events = [
        e for e in events if isinstance(e, AgentCallEvent) and e.state == "running"
    ]
    assert len(call_events) == 3
    first, second, third = call_events
    assert first.parent_call_id is None
    assert second.parent_call_id == first.call_id  # nested under step A's first call
    assert third.parent_call_id is None  # new step -> no carried-over parent


@pytest.mark.asyncio
async def test_log_records_message_and_level_against_current_phase():
    wf = _make_runtime()
    wf.log("before any phase")
    async with wf.phase("Phase A"):
        wf.log("inside phase", level="warning")

    assert wf.log_records[0]["message"] == "before any phase"
    assert wf.log_records[0]["level"] == "info"
    assert wf.log_records[0]["phase_id"] is None
    assert wf.log_records[1]["message"] == "inside phase"
    assert wf.log_records[1]["level"] == "warning"
    assert wf.log_records[1]["phase_id"] == "phase-0"


def test_log_rejects_invalid_level():
    wf = _make_runtime()
    with pytest.raises(ValueError):
        wf.log("x", level="bogus")


def test_budget_elapsed_time_ceiling_is_enforced_independent_of_call_count():
    # `_ensure_started()` consumes one clock tick to stamp `_start`; each
    # `exhausted()` call after that consumes one more to compute elapsed
    # time -- three ticks for two `exhausted()` calls.
    ticks = iter([0.0, 10.0, 100.0])
    budget = WorkflowBudget(
        max_calls=999.0, max_seconds=50.0, clock=lambda: next(ticks)
    )
    assert budget.exhausted() is False  # elapsed 10s < 50s ceiling
    assert (
        budget.exhausted() is True
    )  # elapsed 100s >= 50s ceiling, call count untouched


def test_budget_property_exposes_workflow_budget():
    wf = _make_runtime()
    assert wf.budget.remaining() > 0
    assert wf.budget.exhausted() is False
    wf.budget.spend(wf.budget.remaining(), label="drain")
    assert wf.budget.exhausted() is True
