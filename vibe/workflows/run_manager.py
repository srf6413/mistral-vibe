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

RUN CREATION / EXECUTION (not part of the frozen stub list -- added here per
the contract's note that run creation and the execution driver belong to
whichever lane implements this module's bodies):

- `create_run(...)` persists a new run directory (script + meta.json) but
  does not execute anything.
- `execute_run(...)` drives one run to completion (or its first crash):
  compiles the persisted script, constructs a real `WorkflowRuntime` wired
  to a journal-writing `emit` callback and a caching `session_factory`, runs
  `main(wf, args)`, and writes heartbeat + terminal `run_status` records
  throughout. `resume_run` is a thin wrapper around this that first checks
  the run is actually resumable and then re-runs the same driver -- the
  driver itself is what makes resume "skip already-`ok` calls": its
  `session_factory` hands out a no-network transport that replays a cached
  answer for any call *position* (`call-{n}`) the prior journal already
  recorded as `ok`/`cached`, and a real transport otherwise. See
  `_CachedAskTransport` and `_resume_cache_by_position` below, and the
  module-level "RESUME CACHING" note for why this keys by call position
  rather than by a content hash of (prompt, opts).

RESUME CACHING -- why by position, not by a (prompt, opts) content hash:

The task this module was speced against describes keying the resume cache
by "a stable hash of (prompt, canonical-sorted opts, occurrence-index)".
That is the *semantic* goal, but the frozen JOURNAL SCHEMA above has no
`prompt` or `opts` field on a `kind: "call"` record -- only `call_id`,
`label`, `state`, `text`, `reason`. There is therefore no way to recover a
prior call's prompt/opts from the journal to hash against. What the schema
*does* give us is `call_id`, which is itself already a deterministic,
content-independent position key (`call-{n}`, minted by the runtime as a
monotonic counter in issuance order -- see `events.py`). Keying the resume
cache by that position is equivalent to a content hash for the common case
this feature exists for (re-attaching to a crashed run of the *same*
script), and it is the only thing actually derivable from the frozen
journal. If a script is edited between the original run and a resume, a
position match no longer implies a prompt match -- this is a known,
documented gap, not a silent bug; see the module's `README`-style comment
on `_resume_cache_by_position` and this lane's integration report.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
import contextlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
from typing import Any, Literal

from vibe.app_server._founderos_ask import (
    DEFAULT_FOUNDEROS_ASK_URL,
    DEFAULT_FOUNDEROS_WORKSPACE,
    AskTransport,
    FounderOSAskSession,
    HttpFounderOSAskTransport,
)
from vibe.workflows.events import (
    AgentCallEvent,
    CallState,
    PhaseEvent,
    PhaseSpec,
    WorkflowEvent,
    WorkflowMeta,
)
from vibe.workflows.runtime import WorkflowRuntime
from vibe.workflows.script import (
    build_restricted_globals,
    compile_workflow_main,
    load_workflow_script,
)

HEARTBEAT_INTERVAL_S = 5.0
"""How often the owning process appends a `kind: "heartbeat"` record."""

HEARTBEAT_STALE_AFTER_S = 15.0
"""Three missed heartbeats (at `HEARTBEAT_INTERVAL_S`) before a reader may
treat a run with no terminal `run_status` as `"lost"`."""

DEFAULT_RUNS_ROOT = Path.home() / ".jarvis" / "workflows" / "runs"

RunStatus = Literal["running", "completed", "failed", "cancelled", "lost"]
"""`"lost"` is a READER-computed status (see the liveness oracle above) --
it is never the value of a `run_status` journal field."""

TerminalRunStatus = Literal["completed", "failed", "cancelled"]


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


# -- journal I/O ------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_journal_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _next_seq(path: Path) -> int:
    records = _read_journal_records(path)
    if not records:
        return 0
    return max((r.get("seq", -1) for r in records), default=-1) + 1


def _position_of(call_id: str) -> int:
    """`"call-7"` -> `7`. Matches the `events.py` id-minting contract."""
    return int(call_id.rsplit("-", 1)[-1])


class JournalWriter:
    """Append-only writer for one run's `journal.jsonl`.

    Reads the file once at construction to continue its `seq` counter
    monotonically across process restarts (a fresh `JournalWriter` is built
    on every `execute_run`/`resume_run`/`skip_run`/`cancel_run` call, and
    `seq` must never reset or collide).
    """

    def __init__(self, path: Path, *, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = _next_seq(self._path)

    def _append(self, record: dict[str, Any]) -> None:
        full = {
            "seq": self._seq,
            "ts": _utc_now_iso(),
            "run_id": self._run_id,
            **record,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(full, sort_keys=True))
            fh.write("\n")
        self._seq += 1

    def write_event(
        self,
        event: WorkflowEvent,
        *,
        cached_positions: frozenset[int] | set[int] = frozenset(),
    ) -> None:
        """Flatten one `PhaseEvent`/`AgentCallEvent` into a journal record.

        For an `AgentCallEvent` whose call position (see `_position_of`) is
        in `cached_positions`, a terminal `"ok"` state is rewritten to
        `"cached"` before writing -- `call_agent`/`WorkflowRuntime` can only
        ever report `"ok"` for such a call (see `AgentCallStatus`, which has
        no `"cached"` member); the cache/resume bookkeeping that knows a
        call was actually served from disk lives entirely in this module.
        """
        if isinstance(event, PhaseEvent):
            self._append({
                "kind": "phase",
                "phase_id": event.phase_id,
                "call_id": None,
                "parent_call_id": None,
                "state": event.state,
                "label": event.title,
                "detail": event.detail,
                "text": None,
                "reason": None,
                "message": None,
                "level": None,
                "pid": None,
                "run_status": None,
            })
        elif isinstance(event, AgentCallEvent):
            state: CallState = event.state
            if state == "ok" and _position_of(event.call_id) in cached_positions:
                state = "cached"
            self.write_call_state(
                phase_id=event.phase_id,
                call_id=event.call_id,
                state=state,
                label=event.label,
                text=event.text,
                reason=event.reason,
                parent_call_id=event.parent_call_id,
            )
        else:  # pragma: no cover - WorkflowEvent is a closed union
            raise TypeError(f"unknown WorkflowEvent type: {type(event)!r}")

    def write_call_state(
        self,
        *,
        phase_id: str | None,
        call_id: str,
        state: CallState,
        label: str | None = None,
        text: str | None = None,
        reason: str | None = None,
        parent_call_id: str | None = None,
    ) -> None:
        """Append one `kind: "call"` record directly (used by `skip_run`,
        which has no `AgentCallEvent` instance to hand `write_event`).
        """
        self._append({
            "kind": "call",
            "phase_id": phase_id,
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "state": state,
            "label": label,
            "detail": None,
            "text": text,
            "reason": reason,
            "message": None,
            "level": None,
            "pid": None,
            "run_status": None,
        })

    def log(self, message: str, *, level: str = "info") -> None:
        self._append({
            "kind": "log",
            "phase_id": None,
            "call_id": None,
            "parent_call_id": None,
            "state": None,
            "label": None,
            "detail": None,
            "text": None,
            "reason": None,
            "message": message,
            "level": level,
            "pid": None,
            "run_status": None,
        })

    def heartbeat(self) -> None:
        self._append({
            "kind": "heartbeat",
            "phase_id": None,
            "call_id": None,
            "parent_call_id": None,
            "state": None,
            "label": None,
            "detail": None,
            "text": None,
            "reason": None,
            "message": None,
            "level": None,
            "pid": os.getpid(),
            "run_status": None,
        })

    def write_run_status(self, status: TerminalRunStatus | Literal["running"]) -> None:
        self._append({
            "kind": "run_status",
            "phase_id": None,
            "call_id": None,
            "parent_call_id": None,
            "state": None,
            "label": None,
            "detail": None,
            "text": None,
            "reason": None,
            "message": None,
            "level": None,
            "pid": os.getpid(),
            "run_status": status,
        })


# -- liveness oracle ----------------------------------------------------------


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists, we just can't signal it -- still alive.
        return True
    except OSError:
        return False
    return True


def _is_stale(ts_iso: str) -> bool:
    try:
        ts = datetime.fromisoformat(ts_iso)
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - ts).total_seconds()
    return age > HEARTBEAT_STALE_AFTER_S


def _compute_status(
    records: list[dict[str, Any]], *, created_at: str
) -> tuple[RunStatus, str | None, int | None]:
    """Fold a run's journal records into `(status, last_heartbeat_at, pid)`.

    Never trusts a cached "running" value -- always recomputes liveness from
    the last heartbeat age and pid liveness when no terminal `run_status`
    record exists, per the frozen liveness oracle.
    """
    last_run_status_record: dict[str, Any] | None = None
    last_heartbeat_ts: str | None = None
    pid: int | None = None
    for record in records:
        if record.get("kind") == "run_status":
            last_run_status_record = record
            if record.get("pid") is not None:
                pid = record["pid"]
        elif record.get("kind") == "heartbeat":
            last_heartbeat_ts = record["ts"]
            if record.get("pid") is not None:
                pid = record["pid"]

    if (
        last_run_status_record is not None
        and last_run_status_record["run_status"] != "running"
    ):
        status: RunStatus = last_run_status_record["run_status"]
        return status, last_heartbeat_ts, pid

    if last_run_status_record is None:
        # The run directory exists (create_run happened) but no process has
        # ever appended a "running" record -- give it one heartbeat
        # interval's grace from creation before treating it as "lost" (no
        # pid was ever recorded, so pid-liveness can't be checked here).
        lost = _is_stale(created_at)
        return ("lost" if lost else "running"), last_heartbeat_ts, pid

    reference_ts = last_heartbeat_ts or last_run_status_record["ts"]
    lost = _is_stale(reference_ts) or not _pid_alive(pid)
    return ("lost" if lost else "running"), last_heartbeat_ts, pid


def _phase_id_for_call(journal_path: Path, call_id: str) -> str | None:
    phase_id: str | None = None
    for record in _read_journal_records(journal_path):
        if record.get("kind") == "call" and record.get("call_id") == call_id:
            phase_id = record.get("phase_id")
    return phase_id


def _skipped_positions_from_journal(journal_path: Path) -> frozenset[int]:
    """Fold a prior journal into the set of call positions permanently
    skipped by an operator (`skip_run`), for `execute_run` to hand
    `WorkflowRuntime(skipped_positions=...)` so a resume actually honors a
    skip instead of re-issuing the call -- see `skip_run`'s docstring for
    the gap this closes.
    """
    latest_by_call: dict[str, dict[str, Any]] = {}
    for record in _read_journal_records(journal_path):
        if record.get("kind") == "call" and record.get("call_id"):
            latest_by_call[record["call_id"]] = record
    return frozenset(
        _position_of(call_id)
        for call_id, record in latest_by_call.items()
        if record.get("state") == "skipped"
    )


def _resume_cache_by_position(journal_path: Path) -> dict[int, str]:
    """Fold a prior journal into `{call position: cached final text}` for
    every call whose latest recorded state is `"ok"`/`"cached"`. See the
    module-level "RESUME CACHING" note for why this keys by position.
    """
    latest_by_call: dict[str, dict[str, Any]] = {}
    for record in _read_journal_records(journal_path):
        if record.get("kind") == "call" and record.get("call_id"):
            latest_by_call[record["call_id"]] = record

    cache: dict[int, str] = {}
    for call_id, record in latest_by_call.items():
        if record.get("state") in {"ok", "cached"} and record.get("text") is not None:
            cache[_position_of(call_id)] = record["text"]
    return cache


class _CachedAskTransport:
    """A no-network `AskTransport` that replays one cached final answer.

    Handed out by `execute_run`'s `session_factory` for any call position a
    prior journal already recorded as `"ok"`/`"cached"`, so resuming a run
    never re-issues (and never re-bills) an agent call whose result is
    already on disk.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        del payload
        return self._generate()

    async def _generate(self) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "final_result", "data": {"answer": self._text}}

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _RunExecutionHooks:
    """The `session_factory`/`emit` pair `execute_run` wires into one
    `WorkflowRuntime`, plus the runtime-log-draining glue -- split into its
    own class only to keep `execute_run` itself under this repo's ruff
    statement/local-variable caps; behavior is unchanged from having these
    as closures defined inline (see git history for that version).

    `SessionFactory` is zero-arg (frozen by `agent_call.py`'s contract), so
    it has no direct way to know which call position it is being invoked
    for. A plain invocation-count `itertools.count()` (this class's
    original approach, before this refactor) is WRONG once any call in the
    run can be skipped (see `WorkflowRuntime.agent()`'s `skipped_positions`
    check) or budget-cut: those calls mint a call-id and emit `"running"`
    but never call `session_factory`, so a bare invocation counter drifts
    out of sync with the true call position for every call after the first
    gap. Instead, `emit` -- which fires an `AgentCallEvent(state="running")`
    synchronously, in call-id order, for EVERY call including ones about to
    be skipped/cut, strictly before `WorkflowRuntime.agent()` can reach
    `session_factory` for that same call and with no `await` in between --
    stashes the position on `self` for `session_factory` to read. A stale
    leftover from a skipped/cut call is harmless: it is always overwritten
    by the next call's own `"running"` emission before that call's own
    (if any) `session_factory` invocation.
    """

    def __init__(
        self,
        *,
        run_id: str,
        cwd: Path,
        journal: JournalWriter,
        resume_texts: dict[int, str],
        resolved_endpoint: str | None,
        resolved_workspace: str | None,
        extra_emit: Callable[[WorkflowEvent], None] | None,
        extra_log: Callable[[str, str], None] | None,
    ) -> None:
        self._run_id = run_id
        self._cwd = cwd
        self._journal = journal
        self._resume_texts = resume_texts
        self._resolved_endpoint = resolved_endpoint
        self._resolved_workspace = resolved_workspace
        self._extra_emit = extra_emit
        self._extra_log = extra_log
        self.cached_positions: set[int] = set()
        self._pending_call_position: int | None = None
        # Bound after `WorkflowRuntime.__init__` returns, since the runtime
        # needs `self.emit` before it exists -- see `bind_runtime`.
        self._runtime: WorkflowRuntime | None = None
        self._drained_log_count = 0

    def bind_runtime(self, runtime: WorkflowRuntime) -> None:
        self._runtime = runtime

    def session_factory(self) -> FounderOSAskSession:
        n = self._pending_call_position
        if n is None:  # pragma: no cover - defensive; see class docstring
            raise RuntimeError(
                "session_factory invoked with no known call position -- "
                "expected an AgentCallEvent(state='running') to have fired "
                "first"
            )
        cached_text = self._resume_texts.get(n)
        transport: AskTransport
        if cached_text is not None:
            self.cached_positions.add(n)
            transport = _CachedAskTransport(cached_text)
        else:
            transport = HttpFounderOSAskTransport(
                endpoint=self._resolved_endpoint
                or os.environ.get("FOUNDEROS_ASK_URL", DEFAULT_FOUNDEROS_ASK_URL),
                workspace=self._resolved_workspace
                or os.environ.get("FOUNDEROS_WORKSPACE", DEFAULT_FOUNDEROS_WORKSPACE),
                api_key=os.environ.get("FOUNDEROS_API_KEY"),
            )
        return FounderOSAskSession(
            transport=transport, cwd=self._cwd, session_id=f"wf-{self._run_id}-call-{n}"
        )

    def drain_logs(self) -> None:
        """Flush any `wf.log(...)` records not yet persisted into the
        journal (and `extra_log`, if given). Called after every
        `PhaseEvent`/`AgentCallEvent` (via `emit`) and once more after
        `main()` returns/raises, so no trailing `wf.log()` call issued
        after the last event is ever lost -- see `WorkflowRuntime.log()`'s
        docstring for why `wf.log()` cannot go through `emit` itself.
        """
        if self._runtime is None:
            return
        records = self._runtime.log_records
        while self._drained_log_count < len(records):
            record = records[self._drained_log_count]
            level = record.get("level") or "info"
            message = record.get("message") or ""
            self._journal.log(message, level=level)
            if self._extra_log is not None:
                self._extra_log(level, message)
            self._drained_log_count += 1

    def emit(self, event: WorkflowEvent) -> None:
        """Wired as `WorkflowRuntime(emit=...)`. Writes the journal record,
        fans the event out to `extra_emit` (if given) -- with the SAME
        `"cached"` rewrite the journal record gets, so a live viewer and a
        journal-replay viewer show identical state for a resumed call --
        and drains any log lines the workflow issued since the last event.
        """
        if isinstance(event, AgentCallEvent) and event.state == "running":
            self._pending_call_position = _position_of(event.call_id)
        effective_event = event
        if (
            isinstance(event, AgentCallEvent)
            and event.state == "ok"
            and _position_of(event.call_id) in self.cached_positions
        ):
            effective_event = replace(event, state="cached")
        self._journal.write_event(event, cached_positions=self.cached_positions)
        if self._extra_emit is not None:
            self._extra_emit(effective_event)
        self.drain_logs()


# -- run creation and execution ----------------------------------------------


def create_run(
    run_id: str,
    script_path: Path,
    args: dict[str, Any],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    cwd: Path | None = None,
    endpoint: str | None = None,
    workspace: str | None = None,
) -> RunPaths:
    """Create a new run directory for `run_id` executing `script_path`.

    Loads and lints the script (raises `WorkflowScriptError` if invalid),
    copies its source verbatim into the run dir, and writes `meta.json`
    once. Does not execute the script -- see `execute_run` for that.
    `run_id` is minted by the caller (see the module docstring); this
    function never mints one itself, and raises `FileExistsError` if the
    run directory already exists (an id collision, which should not happen
    with proper minting).
    """
    loaded = load_workflow_script(script_path)
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.script_path.write_text(loaded.source, encoding="utf-8")
    meta_doc = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "name": loaded.meta.name,
        "description": loaded.meta.description,
        "phases": [{"title": p.title, "detail": p.detail} for p in loaded.meta.phases],
        "args": args,
        "cwd": str((cwd or Path.cwd()).expanduser().resolve()),
        "endpoint": endpoint,
        "workspace": workspace,
    }
    paths.meta_path.write_text(
        json.dumps(meta_doc, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


async def _heartbeat_loop(journal: JournalWriter) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        journal.heartbeat()


async def execute_run(
    run_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    endpoint: str | None = None,
    workspace: str | None = None,
    extra_emit: Callable[[WorkflowEvent], None] | None = None,
    extra_log: Callable[[str, str], None] | None = None,
) -> None:
    """Execute (or resume) `run_id`'s workflow to completion.

    Loads the run's persisted `script.py`/`meta.json`, builds a real
    `WorkflowRuntime` wired to a journal-writing `emit` and a
    resume-caching `session_factory` (see `_CachedAskTransport` /
    `_resume_cache_by_position`), runs `main(wf, args)`, and writes
    heartbeat + terminal `run_status` records throughout -- including on
    failure or cancellation, so a run's owning process always leaves a
    terminal record unless it is killed outright (the case the liveness
    oracle's `"lost"` status exists for).

    `extra_emit`/`extra_log`, if given, are called with every
    `PhaseEvent`/`AgentCallEvent` and every `wf.log(...)` line this run
    produces, in addition to (never instead of) the journal write -- this
    is the hook a same-process caller (the CLI's `/workflows run`, wiring a
    live `WorkflowProgressScreen`) uses to fan events out live, since the
    journal itself is not something a UI widget should poll/tail. An
    `AgentCallEvent` handed to `extra_emit` reflects the SAME `"cached"`
    rewrite the journal record gets (see `JournalWriter.write_event`) so a
    live viewer and a journal-replay viewer show identical state for a
    resumed call.

    `runtime.log_records` (see `vibe/workflows/runtime.py`'s `WorkflowRuntime.log()`
    docstring -- `wf.log()` cannot go through `emit`, since the frozen
    `WorkflowEvent` union has no log variant) is drained into the journal
    -- and `extra_log`, if given -- every time a `PhaseEvent`/`AgentCallEvent`
    fires, plus once more after `main()` returns/raises, so no trailing
    `wf.log()` call issued after the last event is ever lost.

    Not part of the frozen stub surface -- see the module's "RUN CREATION /
    EXECUTION" note for why this lives here. `resume_run` is a thin
    wrapper around this function.
    """
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"no run directory for {run_id!r} under {runs_root}")

    loaded = load_workflow_script(paths.script_path)
    meta_doc = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    args: dict[str, Any] = meta_doc.get("args", {})
    cwd = Path(meta_doc.get("cwd") or str(Path.cwd()))
    resolved_endpoint = endpoint or meta_doc.get("endpoint")
    resolved_workspace = workspace or meta_doc.get("workspace")

    journal = JournalWriter(paths.journal_path, run_id=run_id)
    hooks = _RunExecutionHooks(
        run_id=run_id,
        cwd=cwd,
        journal=journal,
        resume_texts=_resume_cache_by_position(paths.journal_path),
        resolved_endpoint=resolved_endpoint,
        resolved_workspace=resolved_workspace,
        extra_emit=extra_emit,
        extra_log=extra_log,
    )
    skipped_positions = _skipped_positions_from_journal(paths.journal_path)

    # A terminal "running" record (and the heartbeat loop) must exist before
    # anything else that can fail -- compiling the script, constructing the
    # runtime -- runs, so even a failure to *start* leaves the run showing
    # "failed" rather than sitting with no run_status record at all (see
    # `_compute_status`'s no-record grace-then-"lost" branch).
    journal.write_run_status("running")
    heartbeat_task = asyncio.create_task(_heartbeat_loop(journal))
    try:
        globals_dict = build_restricted_globals()
        main = compile_workflow_main(loaded, globals_dict)
        runtime = WorkflowRuntime(
            run_id=run_id,
            meta=loaded.meta,
            session_factory=hooks.session_factory,
            emit=hooks.emit,
            skipped_positions=skipped_positions,
        )
        hooks.bind_runtime(runtime)
        await main(runtime, args)
    except asyncio.CancelledError:
        hooks.drain_logs()
        journal.write_run_status("cancelled")
        raise
    except Exception as exc:
        hooks.drain_logs()
        journal.log(f"workflow run failed: {exc}", level="error")
        if extra_log is not None:
            extra_log("error", f"workflow run failed: {exc}")
        journal.write_run_status("failed")
        raise
    else:
        hooks.drain_logs()
        journal.write_run_status("completed")
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task


# -- lifecycle stubs ----------------------------------------------------------


def list_runs(*, runs_root: Path = DEFAULT_RUNS_ROOT) -> list[RunSummary]:
    """List every run under `runs_root`, most recently created first.

    Computes `status` per run via the liveness oracle above -- never
    trusts a cached "running" without checking the heartbeat/pid.
    """
    if not runs_root.exists():
        return []
    summaries: list[RunSummary] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        journal_path = entry / "journal.jsonl"
        status, last_heartbeat_at, _pid = _compute_status(
            _read_journal_records(journal_path),
            created_at=meta_doc.get("created_at", ""),
        )
        summaries.append(
            RunSummary(
                run_id=entry.name,
                name=meta_doc.get("name", entry.name),
                status=status,
                created_at=meta_doc.get("created_at", ""),
                last_heartbeat_at=last_heartbeat_at,
            )
        )
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def show_run(run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> RunDetail:
    """Load one run's full detail, replaying its journal for current state."""
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"no run directory for {run_id!r} under {runs_root}")
    meta_doc = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    meta = WorkflowMeta(
        name=meta_doc.get("name", run_id),
        description=meta_doc.get("description", ""),
        phases=[
            PhaseSpec(title=p["title"], detail=p.get("detail", ""))
            for p in meta_doc.get("phases", [])
        ],
    )
    status, _last_heartbeat_at, _pid = _compute_status(
        _read_journal_records(paths.journal_path),
        created_at=meta_doc.get("created_at", ""),
    )
    return RunDetail(
        run_id=run_id,
        paths=paths,
        meta=meta,
        status=status,
        args=meta_doc.get("args", {}),
    )


def replay_events(
    run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT
) -> tuple[list[WorkflowEvent], list[tuple[str, str]]]:
    """Convert one run's journal into `(events, log_lines)` for a freshly
    mounted `WorkflowProgressScreen`'s buffered-history args -- the
    `/workflows show <run_id>` reattach path, or attaching to a run this
    process did not itself launch.

    Records are replayed in journal (`seq`) order, the same order a live
    `WorkflowRuntime(emit=...)` would have produced them in --
    `WorkflowProgressState.apply()` (the screen's reducer) is documented as
    idempotent per node ("last write wins"), so replaying the full history
    and then switching to live events reproduces identical final state
    either way.

    `skip_run` writes a `kind: "call"` record with `label=None` (it has no
    `AgentCallEvent` to carry a label). `AgentCallEvent.label` is a
    required `str` the tree renders directly, so a `None` label here would
    either crash the dataclass or blank out a node that previously had a
    real label -- this falls back to the call's most recently seen label,
    or the bare `call_id` if none was ever recorded.
    """
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    records = _read_journal_records(paths.journal_path)
    events: list[WorkflowEvent] = []
    log_lines: list[tuple[str, str]] = []
    last_label_by_call: dict[str, str] = {}
    for record in records:
        kind = record.get("kind")
        if kind == "phase":
            events.append(
                PhaseEvent(
                    run_id=run_id,
                    phase_id=record["phase_id"],
                    title=record.get("label") or record["phase_id"],
                    detail=record.get("detail") or "",
                    state=record["state"],
                )
            )
        elif kind == "call":
            call_id = record["call_id"]
            label = record.get("label")
            if label:
                last_label_by_call[call_id] = label
            else:
                label = last_label_by_call.get(call_id, call_id)
            events.append(
                AgentCallEvent(
                    run_id=run_id,
                    phase_id=record.get("phase_id") or "",
                    call_id=call_id,
                    label=label,
                    state=record["state"],
                    text=record.get("text"),
                    reason=record.get("reason"),
                    parent_call_id=record.get("parent_call_id"),
                )
            )
        elif kind == "log":
            log_lines.append((
                record.get("level") or "info",
                record.get("message") or "",
            ))
    return events, log_lines


async def resume_run(
    run_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    extra_emit: Callable[[WorkflowEvent], None] | None = None,
    extra_log: Callable[[str, str], None] | None = None,
) -> None:
    """Re-attach to `run_id` and continue it from its last journal record.

    Only valid when `show_run(run_id).status in {"lost", "failed"}` (a
    `"completed"`/`"cancelled"` run has nothing to resume, and a genuinely
    `"running"` run already has an owning process). Replays `call-{n}` /
    `phase-{n}` ids from the journal so already-`"ok"` calls are not
    re-issued -- this is the reason `vibe/workflows/events.py` freezes
    deterministic id-minting from call order alone.

    `extra_emit`/`extra_log` are forwarded unchanged to `execute_run` (see
    its docstring) -- a resumed run gets the same live-fan-out hook a fresh
    run does.
    """
    detail = show_run(run_id, runs_root=runs_root)
    if detail.status not in {"lost", "failed"}:
        raise ValueError(
            f"run {run_id!r} is {detail.status!r}; only a 'lost' or 'failed' "
            "run can be resumed"
        )
    await execute_run(
        run_id, runs_root=runs_root, extra_emit=extra_emit, extra_log=extra_log
    )


async def skip_run(
    run_id: str, call_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT
) -> None:
    """Mark one pending/failed call as permanently `"skipped"` on resume.

    Used when a human decides a stuck or errored call should not be
    retried; appends a `kind: "call"`, `state: "skipped"` record so the
    next `resume_run` treats it as already terminal.

    Honored on resume: `execute_run` folds every `"skipped"`-latest-state
    call position out of the journal (`_skipped_positions_from_journal`)
    and hands it to `WorkflowRuntime(skipped_positions=...)`, which
    short-circuits that call-id position to a `"skipped"` result before it
    ever reaches `agent_call.call_agent` -- no network call, no budget
    spend. (This closes the gap the `run_manager` lane's own report
    flagged: skip used to only update the journal display, never actually
    prevent a re-issue on resume.)
    """
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"no run directory for {run_id!r} under {runs_root}")
    journal = JournalWriter(paths.journal_path, run_id=run_id)
    phase_id = _phase_id_for_call(paths.journal_path, call_id)
    journal.write_call_state(
        phase_id=phase_id,
        call_id=call_id,
        state="skipped",
        reason="skipped by operator",
    )


async def cancel_run(run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> None:
    """Signal the owning process (if alive) to stop, then write a terminal
    `kind: "run_status"`, `run_status: "cancelled"` record.

    Must be safe to call on a `"lost"` run (no live owner to signal) --
    in that case it only appends the terminal record. Never signals the
    *current* process even if it happens to be the recorded pid (a caller
    that owns the run in-process, e.g. the CLI's own `asyncio.Task`, should
    cancel that task directly and let `execute_run`'s own
    `except asyncio.CancelledError` handler write the terminal record --
    calling this afterward is still safe, just redundant).
    """
    paths = RunPaths.for_run(run_id, runs_root=runs_root)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"no run directory for {run_id!r} under {runs_root}")
    meta_doc = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    status, _last_heartbeat_at, pid = _compute_status(
        _read_journal_records(paths.journal_path),
        created_at=meta_doc.get("created_at", ""),
    )
    if (
        status == "running"
        and pid is not None
        and pid != os.getpid()
        and _pid_alive(pid)
    ):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGTERM)
    journal = JournalWriter(paths.journal_path, run_id=run_id)
    journal.write_run_status("cancelled")
