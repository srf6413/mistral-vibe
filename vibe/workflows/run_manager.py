"""On-disk run directory layout, journal schema, and run lifecycle operations.

Run directory layout (frozen):

    ~/.jarvis/workflows/runs/<run_id>/
        script.py       # verbatim copy of the workflow script this run executes
        meta.json        # WorkflowMeta + run-level bookkeeping, written once at creation
        journal.jsonl    # append-only, one JSON object per line -- see JOURNAL SCHEMA below

`<run_id>` is minted once by whatever creates the run (interactive command,
resume, etc.) -- NOT by any code in this package -- and is thereafter just a
directory name; nothing under `vibe/workflows/` mints it, so the
nondeterminism ban on this package's own code is not in tension with a
run needing a unique id. `<run_id>` should look like
`wf-<script-stem>-<short-opaque-token>` but that format is a convention,
not a parsed contract.

JOURNAL SCHEMA (one JSON object per line in journal.jsonl, append-only):

    {
      "seq": 0,                     // int: monotonic per-run counter,
                                     //   assigned by the journal WRITER
                                     //   (run_manager), never by script code
      "ts": "2026-08-24T10:15:30.123456+00:00",
                                     // str: ISO-8601 UTC, stamped by the
                                     //   writer at append time -- this is
                                     //   the one place wall-clock time is
                                     //   allowed, because it is
                                     //   infrastructure recording history,
                                     //   not workflow script logic
                                     //   deciding what to do next
      "run_id": "wf-...",
      "kind": "phase" | "call" | "log" | "heartbeat" | "run_status",
      "phase_id": "phase-0" | null,
      "call_id": "call-0" | null,
      "parent_call_id": "call-0" | null,
      "state": "pending"|"running"|"ok"|"skipped"|"cancelled"|"error"|"cached"|null,
                                     // populated for kind in {"phase","call"};
                                     //   "phase" only ever uses the subset
                                     //   pending/running/ok/error
      "label": "..." | null,        // kind == "call": AgentCallEvent.label
                                     // kind == "phase": PhaseEvent.title
      "detail": "..." | null,       // kind == "phase": PhaseEvent.detail
      "text": "..." | null,         // kind == "call", state ok/cached
      "reason": "..." | null,       // kind == "call", state skipped/cancelled/error
      "message": "..." | null,      // kind == "log": the log line
      "level": "info" | null,       // kind == "log": debug/info/warning/error
      "pid": 12345 | null,          // kind in {"heartbeat","run_status"}: the
                                     //   owning process's pid, for the
                                     //   liveness oracle below
      "run_status": "running"|"completed"|"failed"|"cancelled" | null
                                     // kind == "run_status" only
    }

LIVENESS ORACLE (frozen -- "lost" is never itself written to the journal):

A run's displayed status is computed by the READER, not looked up from a
field, specifically so a crashed run cannot keep showing "running" forever
(see the "No fake liveness" constraint this package is built under). The
owning process writes a `kind: "heartbeat"` record every `HEARTBEAT_INTERVAL_S`
seconds while the run is active. A reader computes:

    lost = (now - last_heartbeat_ts > HEARTBEAT_STALE_AFTER_S) or (pid is dead)

and reports `"lost"` instead of `"running"` whenever the last known
`run_status` was `"running"` (no terminal `run_status` record exists) and
`lost` is true. A run that ended normally always has a terminal
`kind: "run_status"` record (`"completed"` / `"failed"` / `"cancelled"`)
appended before the process exits, so `lost` only ever applies to runs that
never got the chance to write one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vibe.workflows.events import WorkflowMeta

HEARTBEAT_INTERVAL_S = 5.0
"""How often the owning process appends a `kind: "heartbeat"` record."""

HEARTBEAT_STALE_AFTER_S = 15.0
"""Three missed heartbeats (at `HEARTBEAT_INTERVAL_S`) before a reader may
treat a run with no terminal `run_status` as `"lost"`."""

DEFAULT_RUNS_ROOT = Path.home() / ".jarvis" / "workflows" / "runs"

RunStatus = Literal["running", "completed", "failed", "cancelled", "lost"]
"""`"lost"` is a READER-computed status (see the liveness oracle above) --
it is never the value of a `run_status` journal field."""


@dataclass(frozen=True, slots=True)
class RunPaths:
    """The on-disk layout for one run, rooted at `<runs_root>/<run_id>/`."""

    run_id: str
    root: Path

    @classmethod
    def for_run(cls, run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> RunPaths:
        return cls(run_id=run_id, root=runs_root / run_id)

    @property
    def script_path(self) -> Path:
        return self.root / "script.py"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One row of `list_runs()` output."""

    run_id: str
    name: str
    status: RunStatus
    created_at: str
    last_heartbeat_at: str | None


@dataclass(frozen=True, slots=True)
class RunDetail:
    """Full state of one run, as `show_run()` returns it."""

    run_id: str
    paths: RunPaths
    meta: WorkflowMeta
    status: RunStatus
    args: dict[str, object]


def list_runs(*, runs_root: Path = DEFAULT_RUNS_ROOT) -> list[RunSummary]:
    """List every run under `runs_root`, most recently created first.

    Computes `status` per run via the liveness oracle above -- never
    trusts a cached "running" without checking the heartbeat/pid.
    """
    raise NotImplementedError


def show_run(run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> RunDetail:
    """Load one run's full detail, replaying its journal for current state."""
    raise NotImplementedError


async def resume_run(run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> None:
    """Re-attach to `run_id` and continue it from its last journal record.

    Only valid when `show_run(run_id).status in {"lost", "failed"}` (a
    `"completed"`/`"cancelled"` run has nothing to resume, and a genuinely
    `"running"` run already has an owning process). Replays `call-{n}` /
    `phase-{n}` ids from the journal so already-`"ok"` calls are not
    re-issued -- this is the reason `vibe/workflows/events.py` freezes
    deterministic id-minting from call order alone.
    """
    raise NotImplementedError


async def skip_run(
    run_id: str, call_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT
) -> None:
    """Mark one pending/failed call as permanently `"skipped"` on resume.

    Used when a human decides a stuck or errored call should not be
    retried; appends a `kind: "call"`, `state: "skipped"` record so the
    next `resume_run` treats it as already terminal.
    """
    raise NotImplementedError


async def cancel_run(run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> None:
    """Signal the owning process (if alive) to stop, then write a terminal
    `kind: "run_status"`, `run_status: "cancelled"` record.

    Must be safe to call on a `"lost"` run (no live owner to signal) --
    in that case it only appends the terminal record.
    """
    raise NotImplementedError
