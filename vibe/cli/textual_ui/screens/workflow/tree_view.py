"""`WorkflowTree`: a Textual `Tree` incrementally updated from `WorkflowEvent`s.

Node labels are built from `rich.text.Text` directly (never
`Text.from_markup(...)`, which `Tree.process_label` applies to any plain
`str` label) -- workflow phase titles / call labels / reasons come from
agent- or script-authored strings and must never be interpreted as Rich
markup, the same defensive posture `NoMarkupStatic` takes elsewhere in this
UI.
"""

from __future__ import annotations

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from vibe.cli.textual_ui.screens.workflow.reducer import (
    CallNode,
    NodeKey,
    PhaseNode,
    WorkflowProgressState,
)
from vibe.cli.textual_ui.widgets.spinner import Spinner, SpinnerType, create_spinner
from vibe.workflows.events import (
    AgentCallEvent,
    PhaseEvent,
    WorkflowEvent,
    WorkflowMeta,
)

_GLYPH_PENDING = "○"
_GLYPH_OK = "✓"
_GLYPH_ERROR = "✕"
_GLYPH_SKIPPED = "⏭"
_GLYPH_CANCELLED = "⊘"
_GLYPH_CACHED = "↻"
_GLYPH_LOST = "☠"

_STYLE_PENDING = "dim"
_STYLE_RUNNING = "bold cyan"
_STYLE_OK = "bold green"
_STYLE_ERROR = "bold red"
_STYLE_SKIPPED = "yellow"
_STYLE_CANCELLED = "yellow"
_STYLE_CACHED = "cyan"
_STYLE_LOST = "bold red"

_TERMINAL_GLYPHS: dict[str, tuple[str, str]] = {
    "pending": (_GLYPH_PENDING, _STYLE_PENDING),
    "ok": (_GLYPH_OK, _STYLE_OK),
    "skipped": (_GLYPH_SKIPPED, _STYLE_SKIPPED),
    "cancelled": (_GLYPH_CANCELLED, _STYLE_CANCELLED),
    "error": (_GLYPH_ERROR, _STYLE_ERROR),
    "cached": (_GLYPH_CACHED, _STYLE_CACHED),
}

SPINNER_INTERVAL_S = 0.1


def _glyph_for(state: str, *, lost: bool, spinner_frame: str) -> tuple[str, str]:
    if state == "running":
        if lost:
            return _GLYPH_LOST, _STYLE_LOST
        return spinner_frame, _STYLE_RUNNING
    return _TERMINAL_GLYPHS.get(state, (_GLYPH_PENDING, _STYLE_PENDING))


class WorkflowTree(Tree[NodeKey]):
    """A `Tree` of phase > agent-call nodes, keyed by `(phase_id, call_id)`.

    `handle_event` has exactly the shape `Callable[[WorkflowEvent], None]`
    that `WorkflowRuntime(emit=...)` expects (see the module docstring in
    `vibe/workflows/runtime.py`) -- it can be passed directly as that
    callback, or invoked once per buffered journal record when reattaching
    to an already-running background run. Feeding a full history through it
    and then continuing with live events is safe and requires no separate
    "seed" mode: every call simply updates that node's current state.
    """

    def __init__(
        self, meta: WorkflowMeta, *, run_id: str, id: str | None = None
    ) -> None:
        super().__init__(meta.name or run_id, id=id)
        self.show_root = False
        self.guide_depth = 3
        self._state = WorkflowProgressState(meta)
        self._tree_nodes: dict[NodeKey, TreeNode[NodeKey]] = {}
        self._spinner: Spinner = create_spinner(SpinnerType.BRAILLE)
        self._spinner_timer: Timer | None = None
        self._run_lost = False
        self._render_all_phases()

    # -- lifecycle ---------------------------------------------------------

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(SPINNER_INTERVAL_S, self._tick_spinner)
        if self.cursor_line < 0 and self.root.children:
            self.cursor_line = 0

    def on_unmount(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    # -- public API ----------------------------------------------------

    def handle_event(self, event: WorkflowEvent) -> None:
        """Apply one event to state and update the matching tree node."""
        self._state.apply(event)
        if isinstance(event, PhaseEvent):
            self._render_phase(event.phase_id)
        elif isinstance(event, AgentCallEvent):
            self._render_call(event.node_key)

    def set_run_lost(self, lost: bool) -> None:
        """Flip the run-level liveness oracle.

        `lost` is computed elsewhere (from the run's heartbeat staleness /
        owning-pid check -- see `HEARTBEAT_STALE_AFTER_S` in
        `run_manager.py`) and pushed in here; the reducer has no
        wall-clock or pid access of its own, on purpose, so it stays free
        of nondeterminism. Any node still `"running"` is re-rendered
        immediately so a crashed run can never keep showing a spinner
        (the project's "no fake liveness" rule).
        """
        if lost == self._run_lost:
            return
        self._run_lost = lost
        for key, call in self._state.calls.items():
            if call.state == "running":
                self._render_call(key)
        for phase_id, phase in self._state.phases.items():
            if phase.state == "running":
                self._render_phase(phase_id)

    def call_state(self, phase_id: str, call_id: str) -> CallNode | None:
        return self._state.calls.get((phase_id, call_id))

    def focused_node_key(self) -> NodeKey | None:
        node = self.cursor_node
        if node is None:
            return None
        return node.data

    # -- rendering -----------------------------------------------------

    def _render_all_phases(self) -> None:
        for phase_id in self._state.phase_order:
            key: NodeKey = (phase_id, None)
            node = self.root.add(
                self._phase_label(self._state.phases[phase_id]), data=key, expand=True
            )
            self._tree_nodes[key] = node

    def _render_phase(self, phase_id: str) -> None:
        phase = self._state.phases[phase_id]
        key: NodeKey = (phase_id, None)
        node = self._tree_nodes.get(key)
        label = self._phase_label(phase)
        if node is None:
            node = self.root.add(label, data=key, expand=True)
            self._tree_nodes[key] = node
        else:
            node.set_label(label)

    def _render_call(self, key: NodeKey) -> None:
        call = self._state.calls[key]
        label = self._call_label(call)
        node = self._tree_nodes.get(key)
        if node is None:
            parent_key: NodeKey = (
                (call.phase_id, call.parent_call_id)
                if call.parent_call_id is not None
                else (call.phase_id, None)
            )
            parent_node = self._tree_nodes.get(parent_key, self.root)
            node = parent_node.add(label, data=key, expand=True)
            self._tree_nodes[key] = node
        else:
            node.set_label(label)

    def _tick_spinner(self) -> None:
        if self._run_lost:
            return
        has_running = any(
            call.state == "running" for call in self._state.calls.values()
        ) or any(phase.state == "running" for phase in self._state.phases.values())
        if not has_running:
            return
        frame = self._spinner.next_frame()
        for key, call in self._state.calls.items():
            if call.state == "running":
                self._tree_nodes[key].set_label(
                    self._call_label(call, spinner_frame=frame)
                )
        for phase_id, phase in self._state.phases.items():
            if phase.state == "running":
                self._tree_nodes[(phase_id, None)].set_label(
                    self._phase_label(phase, spinner_frame=frame)
                )

    def _phase_label(
        self, phase: PhaseNode, *, spinner_frame: str | None = None
    ) -> Text:
        glyph, style = _glyph_for(
            phase.state,
            lost=self._run_lost,
            spinner_frame=spinner_frame or self._spinner.current_frame(),
        )
        label = Text()
        label.append(f"{glyph} ", style=style)
        label.append(phase.title, style="bold")
        if phase.detail:
            label.append(f"  {phase.detail}", style="dim")
        if self._run_lost and phase.state == "running":
            label.append("  (lost -- heartbeat stale)", style=_STYLE_LOST)
        return label

    def _call_label(self, call: CallNode, *, spinner_frame: str | None = None) -> Text:
        glyph, style = _glyph_for(
            call.state,
            lost=self._run_lost,
            spinner_frame=spinner_frame or self._spinner.current_frame(),
        )
        label = Text()
        label.append(f"{glyph} ", style=style)
        label.append(call.label)
        if self._run_lost and call.state == "running":
            label.append("  (lost -- heartbeat stale)", style=_STYLE_LOST)
        elif call.reason and call.state in {"skipped", "cancelled", "error"}:
            label.append(f"  ({call.reason})", style="dim")
        return label
