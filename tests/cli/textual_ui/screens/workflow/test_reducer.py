from __future__ import annotations

from vibe.cli.textual_ui.screens.workflow.reducer import WorkflowProgressState
from vibe.workflows.events import AgentCallEvent, PhaseEvent, PhaseSpec, WorkflowMeta


def _meta() -> WorkflowMeta:
    return WorkflowMeta(
        name="demo",
        description="demo workflow",
        phases=[PhaseSpec(title="Research"), PhaseSpec(title="Write")],
    )


def test_pre_renders_one_pending_slot_per_declared_phase() -> None:
    state = WorkflowProgressState(_meta())

    assert state.phase_order == ["phase-0", "phase-1"]
    assert state.phases["phase-0"].title == "Research"
    assert state.phases["phase-0"].state == "pending"
    assert state.phases["phase-1"].title == "Write"
    assert state.phases["phase-1"].state == "pending"
    assert state.calls == {}


def test_phase_event_updates_the_pre_rendered_slot_in_place() -> None:
    state = WorkflowProgressState(_meta())

    state.apply(
        PhaseEvent(
            run_id="wf-1",
            phase_id="phase-0",
            title="Research",
            detail="",
            state="running",
        )
    )

    assert state.phases["phase-0"].state == "running"
    # phase-1 slot is untouched
    assert state.phases["phase-1"].state == "pending"
    assert state.phase_order == ["phase-0", "phase-1"]


def test_call_events_key_by_node_key_and_preserve_issue_order() -> None:
    state = WorkflowProgressState(_meta())

    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Draft A",
            state="running",
        )
    )
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="Draft B",
            state="running",
        )
    )
    # call-1 finishes before call-0 (parallel() issue order != completion order)
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="Draft B",
            state="ok",
            text="done B",
        )
    )

    assert state.call_order["phase-0"] == ["call-0", "call-1"]
    assert state.calls[("phase-0", "call-0")].state == "running"
    assert state.calls[("phase-0", "call-1")].state == "ok"
    assert state.calls[("phase-0", "call-1")].text == "done B"


def test_last_event_for_a_node_key_wins_without_touching_other_nodes() -> None:
    state = WorkflowProgressState(_meta())
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Draft A",
            state="running",
        )
    )
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Draft A",
            state="error",
            reason="boom",
        )
    )

    assert state.calls[("phase-0", "call-0")].state == "error"
    assert state.calls[("phase-0", "call-0")].reason == "boom"
    assert len(state.calls) == 1


def test_parent_call_id_is_preserved_for_nesting() -> None:
    state = WorkflowProgressState(_meta())
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Pipeline",
            state="running",
        )
    )
    state.apply(
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="Step 1",
            state="running",
            parent_call_id="call-0",
        )
    )

    assert state.calls[("phase-0", "call-1")].parent_call_id == "call-0"


def test_replaying_buffered_history_then_live_events_is_equivalent_to_one_stream() -> (
    None
):
    events = [
        PhaseEvent(
            run_id="wf-1",
            phase_id="phase-0",
            title="Research",
            detail="",
            state="running",
        ),
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Draft A",
            state="running",
        ),
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Draft A",
            state="ok",
            text="done",
        ),
    ]

    straight_through = WorkflowProgressState(_meta())
    for event in events:
        straight_through.apply(event)

    replayed = WorkflowProgressState(_meta())
    for event in events[:2]:  # "buffered history" on reattach
        replayed.apply(event)
    for event in events[2:]:  # then live events continue seamlessly
        replayed.apply(event)

    assert replayed.phases == straight_through.phases
    assert replayed.calls == straight_through.calls
    assert replayed.call_order == straight_through.call_order
