"""`WorkflowProgressScreen`: full-screen live progress viewer for one run.

Modeled on `vibe.cli.textual_ui.screens.config.config_screen.ConfigScreen`
(the closest existing full-screen `ModalScreen` in this codebase) --
same `SCOPED_CSS = False` + sibling `.tcss` file, same
`Vertical(id="...-content")` + bordered-panel + help-bar-at-the-bottom
composition, same non-printable `Binding(..., show=False)` style so single
letter keys never collide with a search-to-filter box (this screen has
none, but the convention is kept for consistency across screens).

Ownership boundary (frozen by the parent task's "no fake liveness" /
"background run" requirements): this screen NEVER owns the run. It is a
pure viewer over externally-supplied state:

- `events` / `log_lines`: buffered history to replay on mount (e.g. from a
  journal tail on reattach via `/workflows show <run_id>`).
- `subscribe_events` / `subscribe_logs`: optional live feeds, each shaped
  `Callable[[handler], unsubscribe]` -- the same "register a listener, get
  an unsubscribe-back" shape regardless of whether the source is a
  `WorkflowRuntime(emit=...)` still running in this process or a journal
  tail owned elsewhere. Not supplying them still renders `events` /
  `log_lines` correctly; the screen just won't update further.
- `get_run_status`: optional liveness oracle, polled every
  `RUN_STATUS_POLL_INTERVAL_S`. `vibe/workflows/run_manager.py`'s
  `RunStatus`/heartbeat contract is the intended source, but that lane's
  bodies are still `NotImplementedError` as of this screen -- see this
  package's `README`-equivalent note in the lane report for the open
  question this leaves for integration.
- `on_skip` / `on_cancel`: optional async callbacks wired to
  `vibe.workflows.run_manager.skip_run` / `cancel_run` by whoever
  constructs this screen; a `None` callback disables the corresponding key.

Pressing Esc always just calls `dismiss()` -- it never touches any of the
above, so the run (whatever owns it) keeps going after the screen closes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import RichLog

from vibe.cli.textual_ui.screens.workflow.reducer import NodeKey
from vibe.cli.textual_ui.screens.workflow.tree_view import WorkflowTree
from vibe.cli.textual_ui.shortcut_hints import shortcut, shortcut_hint
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.workflows.events import WorkflowEvent, WorkflowMeta
from vibe.workflows.run_manager import RunStatus

WORKFLOW_SCREEN_ID = "workflow-screen"

RUN_STATUS_POLL_INTERVAL_S = 2.0
"""Well under `HEARTBEAT_STALE_AFTER_S` (15s) so a lost run is caught
promptly -- see `vibe/workflows/run_manager.py`."""

EventSubscribe = Callable[[Callable[[WorkflowEvent], None]], Callable[[], None]]
LogSubscribe = Callable[[Callable[[str, str], None]], Callable[[], None]]


class WorkflowProgressScreen(ModalScreen[None]):
    """Live phase/agent-call progress tree for one workflow run."""

    SCOPED_CSS = False
    CSS_PATH = "workflow_screen.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "detach", "Detach", show=False),
        Binding("s", "skip_focused", "Skip", show=False),
        Binding("c", "cancel_run", "Cancel run", show=False),
        Binding("l", "toggle_log", "Toggle log", show=False),
    ]

    def __init__(
        self,
        *,
        run_id: str,
        meta: WorkflowMeta,
        events: Sequence[WorkflowEvent] = (),
        log_lines: Sequence[tuple[str, str]] = (),
        subscribe_events: EventSubscribe | None = None,
        subscribe_logs: LogSubscribe | None = None,
        get_run_status: Callable[[], RunStatus] | None = None,
        on_skip: Callable[[str, str], Awaitable[None]] | None = None,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(id=WORKFLOW_SCREEN_ID)
        self._run_id = run_id
        self._meta = meta
        self._initial_events = list(events)
        self._initial_log_lines = list(log_lines)
        self._subscribe_events = subscribe_events
        self._subscribe_logs = subscribe_logs
        self._get_run_status = get_run_status
        self._on_skip = on_skip
        self._on_cancel = on_cancel
        self._unsubscribe_events: Callable[[], None] | None = None
        self._unsubscribe_logs: Callable[[], None] | None = None
        self._log_visible = False
        self._status_poll_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="workflow-screen-content"):
            yield NoMarkupStatic(
                "", id="workflow-screen-status", classes="workflow-screen-status"
            )
            yield WorkflowTree(
                self._meta, run_id=self._run_id, id="workflow-screen-tree"
            )
            yield RichLog(
                id="workflow-screen-log",
                classes="workflow-screen-log",
                wrap=True,
                markup=False,
                highlight=False,
            )
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('↑↓')} Navigate  {shortcut('s')} Skip  "
                    f"{shortcut('c')} Cancel run  {shortcut('l')} Log  "
                    f"{shortcut('Esc')} Detach"
                ),
                classes="workflow-screen-help",
            )

    def on_mount(self) -> None:
        content = self.query_one("#workflow-screen-content")
        content.border_title = self._meta.name or self._run_id
        self.query_one("#workflow-screen-log").display = False

        tree = self.query_one(WorkflowTree)
        for event in self._initial_events:
            tree.handle_event(event)
        log = self.query_one(RichLog)
        for level, message in self._initial_log_lines:
            log.write(f"[{level}] {message}")

        if self._subscribe_events is not None:
            self._unsubscribe_events = self._subscribe_events(self._on_live_event)
        if self._subscribe_logs is not None:
            self._unsubscribe_logs = self._subscribe_logs(self._on_live_log)
        if self._get_run_status is not None:
            self.refresh_run_status()
            self._status_poll_timer = self.set_interval(
                RUN_STATUS_POLL_INTERVAL_S, self.refresh_run_status
            )

        tree.focus()

    def on_unmount(self) -> None:
        if self._unsubscribe_events is not None:
            self._unsubscribe_events()
            self._unsubscribe_events = None
        if self._unsubscribe_logs is not None:
            self._unsubscribe_logs()
            self._unsubscribe_logs = None
        if self._status_poll_timer is not None:
            self._status_poll_timer.stop()
            self._status_poll_timer = None

    # -- live feeds ------------------------------------------------------

    def _on_live_event(self, event: WorkflowEvent) -> None:
        self.query_one(WorkflowTree).handle_event(event)

    def _on_live_log(self, level: str, message: str) -> None:
        self.query_one(RichLog).write(f"[{level}] {message}")

    def refresh_run_status(self) -> None:
        """Re-poll `get_run_status()` and update the tree/status line.

        Called on a timer while the screen is mounted; also safe to call
        directly (e.g. from a test) for a synchronous, deterministic check
        of the liveness oracle without waiting on real wall-clock time.
        """
        assert self._get_run_status is not None
        status = self._get_run_status()
        self.query_one(WorkflowTree).set_run_lost(status == "lost")
        self.query_one("#workflow-screen-status", NoMarkupStatic).update(
            f"{self._run_id}  --  {status}"
        )

    # -- bindings ----------------------------------------------------

    def action_detach(self) -> None:
        """Close the screen without touching the run -- it keeps going."""
        self.dismiss(None)

    def action_toggle_log(self) -> None:
        self._log_visible = not self._log_visible
        self.query_one("#workflow-screen-log").display = self._log_visible

    def action_skip_focused(self) -> None:
        if self._on_skip is None:
            return
        key = self._focused_running_call_key()
        if key is None:
            return
        phase_id, call_id = key
        assert call_id is not None
        self._dispatch_skip(phase_id, call_id)

    def action_cancel_run(self) -> None:
        if self._on_cancel is None:
            return
        self._dispatch_cancel()

    def _focused_running_call_key(self) -> NodeKey | None:
        tree = self.query_one(WorkflowTree)
        key = tree.focused_node_key()
        if key is None:
            return None
        phase_id, call_id = key
        if call_id is None:
            return None
        call = tree.call_state(phase_id, call_id)
        if call is None or call.state != "running":
            return None
        return key

    @work
    async def _dispatch_skip(self, phase_id: str, call_id: str) -> None:
        assert self._on_skip is not None
        await self._on_skip(phase_id, call_id)

    @work
    async def _dispatch_cancel(self) -> None:
        assert self._on_cancel is not None
        await self._on_cancel()
