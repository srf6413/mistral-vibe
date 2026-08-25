"""Pure, framework-independent state reducer for workflow progress events.

This module holds no Textual imports on purpose: it turns a stream of
`vibe.workflows.events.WorkflowEvent` into a plain dict of node states keyed
by the SAME `node_key` the events already define (`(phase_id, None)` for a
phase, `(phase_id, call_id)` for a call -- see `vibe/workflows/events.py`).
`WorkflowTree` (in `tree_view.py`) is the only thing that turns this state
into Textual `TreeNode` labels; keeping the two separate makes the reducer
testable with plain dataclasses and no `App.run_test()` pilot.

Deliberately NOT a "last message wins" reducer: every node keeps its own
slot forever (by node_key), so a Tree-shaped UI can render every phase and
every call at once, not just the most recent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vibe.workflows.events import (
    AgentCallEvent,
    CallState,
    PhaseEvent,
    PhaseState,
    WorkflowEvent,
    WorkflowMeta,
)

NodeKey = tuple[str, str | None]
"""`(phase_id, None)` for a phase node, `(phase_id, call_id)` for a call
node -- identical in shape to `PhaseEvent.node_key` / `AgentCallEvent.node_key`."""


@dataclass(frozen=True, slots=True)
class PhaseNode:
    """Current state of one phase, as rendered by the tree."""

    phase_id: str
    title: str
    detail: str = ""
    state: PhaseState = "pending"


@dataclass(frozen=True, slots=True)
class CallNode:
    """Current state of one agent call, as rendered by the tree."""

    phase_id: str
    call_id: str
    label: str
    state: CallState = "pending"
    text: str | None = None
    reason: str | None = None
    parent_call_id: str | None = None

    @property
    def node_key(self) -> NodeKey:
        return (self.phase_id, self.call_id)


@dataclass(slots=True)
class WorkflowProgressState:
    """Mutable, in-memory projection of one run's phase/call tree.

    Construct from the run's `WorkflowMeta` so every declared phase has a
    `"pending"` slot before any event arrives -- the UI contract in
    `events.py` requires the tree to pre-render one pending node per
    `WorkflowMeta.phases` entry, keyed by the same `phase-{n}` id the
    runtime will later use for that phase's events.

    `apply()` is idempotent per node: replaying a run's full buffered
    journal history (on reattach/resume) and then continuing with live
    events uses this exact same method for both, in emission order, with
    no special "seed" mode needed -- the Nth apply() for a given node_key
    always reflects the most recent event for that key.
    """

    meta: WorkflowMeta
    phases: dict[str, PhaseNode] = field(init=False)
    phase_order: list[str] = field(init=False)
    calls: dict[NodeKey, CallNode] = field(init=False)
    call_order: dict[str, list[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.phases = {
            f"phase-{n}": PhaseNode(
                phase_id=f"phase-{n}", title=spec.title, detail=spec.detail
            )
            for n, spec in enumerate(self.meta.phases)
        }
        self.phase_order = list(self.phases)
        self.calls = {}
        self.call_order = {phase_id: [] for phase_id in self.phases}

    def apply(self, event: WorkflowEvent) -> None:
        if isinstance(event, PhaseEvent):
            self._apply_phase(event)
        elif isinstance(event, AgentCallEvent):
            self._apply_call(event)
        else:
            raise TypeError(f"unknown workflow event type: {type(event)!r}")

    def _apply_phase(self, event: PhaseEvent) -> None:
        if event.phase_id not in self.phases:
            # Defensive: an event for a phase index beyond WorkflowMeta.phases
            # should never happen per the id-minting contract, but a reducer
            # must not crash the UI over a runtime bug -- render it anyway.
            self.phase_order.append(event.phase_id)
            self.call_order.setdefault(event.phase_id, [])
        self.phases[event.phase_id] = PhaseNode(
            phase_id=event.phase_id,
            title=event.title,
            detail=event.detail,
            state=event.state,
        )

    def _apply_call(self, event: AgentCallEvent) -> None:
        key = event.node_key
        is_new = key not in self.calls
        self.calls[key] = CallNode(
            phase_id=event.phase_id,
            call_id=event.call_id,
            label=event.label,
            state=event.state,
            text=event.text,
            reason=event.reason,
            parent_call_id=event.parent_call_id,
        )
        if is_new:
            self.call_order.setdefault(event.phase_id, []).append(event.call_id)
