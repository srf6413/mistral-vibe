"""Event and metadata shapes shared by the workflow runtime, run manager, and UI.

These are plain, immutable dataclasses (not pydantic models) because they are
produced at high frequency inside a hot async loop (one event per agent-call
state transition) and are appended verbatim as journal.jsonl records -- see
`vibe/workflows/run_manager.py` for the on-disk journal schema that these
shapes are flattened into.

Node identity contract (frozen -- do not change without updating all lanes):

- `phase_id` values are minted by the runtime as `f"phase-{n}"` where `n` is
  the 0-based index of the `wf.phase(...)` call in *execution* order. The
  Nth `wf.phase(...)` call corresponds to `WorkflowMeta.phases[n]` -- the UI
  pre-renders one pending tree node per `WorkflowMeta.phases` entry keyed by
  that same `phase-{n}` id before any events arrive, then updates it in
  place as `PhaseEvent`s land.
- `call_id` values are minted by the runtime as `f"call-{n}"` where `n` is a
  run-scoped monotonic counter (not reset per phase), in the order
  `wf.agent(...)` calls are *issued* (not necessarily the order they
  complete -- `parallel()` issues several before any of them finish).
- A Tree-shaped progress UI keys nodes by the pair `(phase_id, call_id)`:
  that pair is unique for the lifetime of a run. `parent_call_id` lets a
  call nest under another call within the same phase (e.g. a step inside
  `pipeline()`) without changing the `(phase_id, call_id)` key.
- Nothing in this module may call `time.time()`, `datetime.now()`,
  `random.*`, or `uuid.*` -- id minting is a pure function of call order so
  that replaying journal.jsonl deterministically reproduces the same ids on
  resume. Wall-clock timestamps are stamped onto journal records by
  `run_manager` at write time, never by these dataclasses or by workflow
  script code (see the AST lint in `vibe/workflows/script.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# -- meta ---------------------------------------------------------------

CallState = Literal[
    "pending", "running", "ok", "skipped", "cancelled", "error", "cached"
]
"""State of one `AgentCallEvent` node."""

PhaseState = Literal["pending", "running", "ok", "error"]
"""State of one `PhaseEvent` node."""


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One declared phase in `WorkflowMeta.phases`, in execution order."""

    title: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowMeta:
    """The `meta = {...}` literal at the top of a workflow script.

    Parsed via `ast.literal_eval` by `vibe/workflows/script.py` -- every
    field here must be reconstructible from a pure Python literal (str,
    list, dict of str/list/dict), never from a call or a name reference.
    """

    name: str
    description: str
    phases: list[PhaseSpec] = field(default_factory=list)


# -- events ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseEvent:
    """Emitted on entry to and exit from `wf.phase(phase_id, ...)`."""

    run_id: str
    phase_id: str
    title: str
    detail: str
    state: PhaseState

    @property
    def node_key(self) -> tuple[str, None]:
        """Tree key for the phase node itself (no call under it yet)."""
        return (self.phase_id, None)


@dataclass(frozen=True, slots=True)
class AgentCallEvent:
    """One state transition of one `wf.agent(...)` call.

    `text` carries streamed/final assistant text for state in
    {"running", "ok", "cached"}; `reason` carries a human-readable
    explanation for state in {"skipped", "cancelled", "error"}. Exactly one
    of `text` / `reason` is meaningful for any given state -- the other is
    `None`.
    """

    run_id: str
    phase_id: str
    call_id: str
    label: str
    state: CallState
    text: str | None = None
    reason: str | None = None
    parent_call_id: str | None = None

    @property
    def node_key(self) -> tuple[str, str]:
        """Unique key a Tree-shaped progress UI uses for this node."""
        return (self.phase_id, self.call_id)


WorkflowEvent = PhaseEvent | AgentCallEvent
"""Everything `WorkflowRuntime` can emit through its `emit` callback."""
