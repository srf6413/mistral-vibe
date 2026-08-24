"""Pilot-driven tests for `WorkflowProgressScreen` / `WorkflowTree`.

This is the first `App.run_test()` / `Pilot` test in the repo exercising a
`Tree`-shaped `ModalScreen` (see the lane task): rather than building the
full `VibeApp`, it wraps the screen in a minimal `textual.app.App`
subclass that just pushes it on mount -- the cheapest harness that still
runs the real Textual event loop, so keystrokes, timers, and the actual
mount/compose lifecycle are exercised for real, matching the pattern
`ConfigScreen`-adjacent snapshot tests use but without needing the full
agent-loop/config-orchestrator machinery those pull in.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual.app import App, ComposeResult

from vibe.cli.textual_ui.screens.workflow.tree_view import WorkflowTree
from vibe.cli.textual_ui.screens.workflow.workflow_screen import WorkflowProgressScreen
from vibe.workflows.events import (
    AgentCallEvent,
    PhaseEvent,
    PhaseSpec,
    WorkflowEvent,
    WorkflowMeta,
)
from vibe.workflows.run_manager import RunStatus


class _EventBus:
    """Minimal `Callable[[handler], unsubscribe]` pub/sub for tests."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[WorkflowEvent], None]] = []

    def subscribe(self, handler: Callable[[WorkflowEvent], None]) -> Callable[[], None]:
        self._listeners.append(handler)

        def unsubscribe() -> None:
            if handler in self._listeners:
                self._listeners.remove(handler)

        return unsubscribe

    def emit(self, event: WorkflowEvent) -> None:
        for listener in list(self._listeners):
            listener(event)


class _HarnessApp(App[None]):
    """Pushes one `WorkflowProgressScreen` on mount -- nothing else."""

    def __init__(self, screen: WorkflowProgressScreen) -> None:
        super().__init__()
        self._screen_to_push = screen

    def compose(self) -> ComposeResult:
        return iter(())

    def on_mount(self) -> None:
        self.push_screen(self._screen_to_push)


def _meta(*titles: str) -> WorkflowMeta:
    return WorkflowMeta(
        name="demo workflow",
        description="d",
        phases=[PhaseSpec(title=title) for title in titles],
    )


def _node(tree: WorkflowTree, phase_id: str, call_id: str | None = None):
    return tree._tree_nodes[(phase_id, call_id)]


@pytest.mark.asyncio
async def test_tree_prerenders_one_pending_node_per_declared_phase() -> None:
    screen = WorkflowProgressScreen(run_id="wf-1", meta=_meta("Research", "Write"))
    app = _HarnessApp(screen)
    async with app.run_test():
        tree = screen.query_one(WorkflowTree)
        research = _node(tree, "phase-0")
        write = _node(tree, "phase-1")
        assert "Research" in research.label.plain
        assert "○" in research.label.plain  # pending glyph
        assert "Write" in write.label.plain
        assert "○" in write.label.plain


@pytest.mark.asyncio
async def test_buffered_history_replays_then_live_events_extend_the_tree() -> None:
    bus = _EventBus()
    history = [
        PhaseEvent(
            run_id="wf-1",
            phase_id="phase-0",
            title="Research",
            detail="",
            state="running",
        )
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1",
        meta=_meta("Research"),
        events=history,
        subscribe_events=bus.subscribe,
    )
    app = _HarnessApp(screen)
    async with app.run_test() as pilot:
        tree = screen.query_one(WorkflowTree)
        phase_node = _node(tree, "phase-0")
        assert "○" not in phase_node.label.plain  # no longer pending

        bus.emit(
            AgentCallEvent(
                run_id="wf-1",
                phase_id="phase-0",
                call_id="call-0",
                label="Draft outline",
                state="running",
            )
        )
        await pilot.pause()
        call_node = _node(tree, "phase-0", "call-0")
        assert "Draft outline" in call_node.label.plain
        assert tree.call_state("phase-0", "call-0").state == "running"

        bus.emit(
            AgentCallEvent(
                run_id="wf-1",
                phase_id="phase-0",
                call_id="call-0",
                label="Draft outline",
                state="ok",
                text="done",
            )
        )
        await pilot.pause()
        assert tree.call_state("phase-0", "call-0").state == "ok"
        assert "✓" in call_node.label.plain


@pytest.mark.asyncio
async def test_parallel_calls_render_as_siblings_keyed_by_call_id() -> None:
    events: list[WorkflowEvent] = [
        PhaseEvent(
            run_id="wf-1",
            phase_id="phase-0",
            title="Research",
            detail="",
            state="running",
        ),
        # issued in order but call-1 finishes first, like parallel() would
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="A",
            state="running",
        ),
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="B",
            state="running",
        ),
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="B",
            state="ok",
            text="b done",
        ),
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), events=events
    )
    app = _HarnessApp(screen)
    async with app.run_test():
        tree = screen.query_one(WorkflowTree)
        assert tree.call_state("phase-0", "call-0").state == "running"
        assert tree.call_state("phase-0", "call-1").state == "ok"
        assert "✓" in _node(tree, "phase-0", "call-1").label.plain


@pytest.mark.asyncio
async def test_cached_call_renders_with_the_distinct_cycle_glyph() -> None:
    """A call resumed from journal replay as already-`"cached"` must look
    different from a fresh `"ok"` -- the task calls for "a cycle glyph".
    """
    events: list[WorkflowEvent] = [
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Resumed from journal",
            state="cached",
            text="cached answer",
        )
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), events=events
    )
    app = _HarnessApp(screen)
    async with app.run_test():
        tree = screen.query_one(WorkflowTree)
        label = _node(tree, "phase-0", "call-0").label.plain
        assert "↻" in label
        assert "✓" not in label  # distinct from a fresh "ok", not aliased to it


@pytest.mark.asyncio
async def test_nested_pipeline_step_attaches_under_its_parent_call_node() -> None:
    """`parent_call_id` nests a call under another call WITHIN the same
    phase (e.g. a `pipeline()` step) -- the child TreeNode's real Textual
    parent must be the parent call's node, not the phase node, so the Tree
    widget actually renders it indented one level deeper.
    """
    events: list[WorkflowEvent] = [
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Pipeline",
            state="running",
        ),
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-1",
            label="Step 1",
            state="running",
            parent_call_id="call-0",
        ),
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), events=events
    )
    app = _HarnessApp(screen)
    async with app.run_test():
        tree = screen.query_one(WorkflowTree)
        parent_node = _node(tree, "phase-0", "call-0")
        child_node = _node(tree, "phase-0", "call-1")
        assert child_node.parent is parent_node
        assert child_node in parent_node.children
        phase_node = _node(tree, "phase-0")
        assert child_node not in phase_node.children


@pytest.mark.asyncio
async def test_skip_binding_only_fires_for_the_focused_running_call() -> None:
    skips: list[tuple[str, str]] = []

    async def on_skip(phase_id: str, call_id: str) -> None:
        skips.append((phase_id, call_id))

    events: list[WorkflowEvent] = [
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
            label="Running call",
            state="running",
        ),
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), events=events, on_skip=on_skip
    )
    app = _HarnessApp(screen)
    async with app.run_test() as pilot:
        tree = screen.query_one(WorkflowTree)
        # Cursor starts on the phase node (not running->skippable): 's' is a no-op.
        assert tree.focused_node_key() == ("phase-0", None)
        await pilot.press("s")
        await pilot.pause()
        assert skips == []

        # Move onto the running call node and skip it.
        tree.cursor_line = 1
        await pilot.pause()
        assert tree.focused_node_key() == ("phase-0", "call-0")
        await pilot.press("s")
        await pilot.pause()
        assert skips == [("phase-0", "call-0")]


@pytest.mark.asyncio
async def test_cancel_binding_calls_on_cancel_regardless_of_focus() -> None:
    cancelled = 0

    async def on_cancel() -> None:
        nonlocal cancelled
        cancelled += 1

    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), on_cancel=on_cancel
    )
    app = _HarnessApp(screen)
    async with app.run_test() as pilot:
        await pilot.press("c")
        await pilot.pause()
        assert cancelled == 1


@pytest.mark.asyncio
async def test_log_toggle_binding_shows_and_hides_the_narrator_panel() -> None:
    screen = WorkflowProgressScreen(
        run_id="wf-1", meta=_meta("Research"), log_lines=[("info", "starting up")]
    )
    app = _HarnessApp(screen)
    async with app.run_test() as pilot:
        log_panel = screen.query_one("#workflow-screen-log")
        assert log_panel.display is False

        await pilot.press("l")
        await pilot.pause()
        assert log_panel.display is True

        await pilot.press("l")
        await pilot.pause()
        assert log_panel.display is False


@pytest.mark.asyncio
async def test_escape_detaches_without_invoking_skip_or_cancel() -> None:
    skips: list[tuple[str, str]] = []
    cancels = 0

    async def on_skip(phase_id: str, call_id: str) -> None:
        skips.append((phase_id, call_id))

    async def on_cancel() -> None:
        nonlocal cancels
        cancels += 1

    events: list[WorkflowEvent] = [
        AgentCallEvent(
            run_id="wf-1",
            phase_id="phase-0",
            call_id="call-0",
            label="Running call",
            state="running",
        )
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1",
        meta=_meta("Research"),
        events=events,
        on_skip=on_skip,
        on_cancel=on_cancel,
    )
    app = _HarnessApp(screen)
    async with app.run_test() as pilot:
        assert app.screen is screen
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not screen
        assert skips == []
        assert cancels == 0


@pytest.mark.asyncio
async def test_lost_run_status_flips_a_still_running_node_and_freezes_the_spinner() -> (
    None
):
    status: RunStatus = "running"

    def get_run_status() -> RunStatus:
        return status

    events: list[WorkflowEvent] = [
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
            label="Stuck call",
            state="running",
        ),
    ]
    screen = WorkflowProgressScreen(
        run_id="wf-1",
        meta=_meta("Research"),
        events=events,
        get_run_status=get_run_status,
    )
    app = _HarnessApp(screen)
    async with app.run_test():
        tree = screen.query_one(WorkflowTree)
        call_node = _node(tree, "phase-0", "call-0")
        # Still nominally "running" in the reducer -- glyph is a spinner
        # frame, not the lost marker, and never the raw state string.
        assert "lost" not in call_node.label.plain
        assert tree.call_state("phase-0", "call-0").state == "running"

        status = "lost"
        screen.refresh_run_status()

        # The node's reducer-level state is still "running" (no journal
        # event ever changed it) but it must never be *rendered* as still
        # running once the run is known lost -- the project's "no fake
        # liveness" rule.
        assert tree.call_state("phase-0", "call-0").state == "running"
        assert "lost" in call_node.label.plain
        assert "☠" in call_node.label.plain
