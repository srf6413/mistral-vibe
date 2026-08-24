"""Headless, Textual-free entry point that drives one workflow run to
completion (or until the process itself is killed), with no controlling
terminal, no stdin reads, and no reliance on anything the parent process
(the Textual TUI) owned.

Invoked as:

    python -m vibe.workflows.background_runner --run-id <run_id> \\
        [--runs-root <path>] [--endpoint <url>] [--workspace <name>]

This is the child half of `/background <run_id>` (see
`vibe/cli/textual_ui/app.py`'s `_background_command` /
`_handoff_workflow_to_background`): the CLI cancels the run's local
`asyncio.Task` and fully awaits it, THEN spawns this module via
`subprocess.Popen(..., start_new_session=True)` so the process survives the
parent TUI exiting -- mirroring the detachment shape FounderOS's own
`start_backgrounded_ask_dispatch` uses server-side for the same reason
(OS-level process detachment, not an asyncio-level handoff).

WHY THIS CALLS `run_manager.execute_run`, NOT `run_manager.resume_run`
-----------------------------------------------------------------------

`resume_run(run_id)` is a thin wrapper around `execute_run` that ALSO
asserts `show_run(run_id).status in {"lost", "failed"}` before proceeding
(see its docstring) -- a safety gate that exists for the *interactive*
`/workflows resume <run_id>` command, so a human cannot accidentally
resume a run some other process still legitimately owns.

That gate does not fit the `/background` handoff path this module is the
other half of. Per the handoff sequence in `_handoff_workflow_to_background`,
the in-process `asyncio.Task` driving the run is `task.cancel()`-ed and
FULLY AWAITED before this process is ever spawned. `execute_run`'s own
`except asyncio.CancelledError` handler (in `run_manager.py`) reacts to
that cancellation by writing a terminal `run_status: "cancelled"` record to
the journal before the cancellation finishes propagating -- so by the time
this process starts, `show_run(run_id).status` is `"cancelled"`, not
`"lost"` or `"failed"`. Calling `resume_run` here would therefore always
raise `ValueError` on the exact handoff path this module exists for.

Calling `execute_run` directly is the correct choice instead: it performs
the identical resume-by-journal-position work (see its own and
`_resume_cache_by_position`'s docstrings -- already-`"ok"`/`"cached"` calls
are replayed, not re-issued) with no status precondition at all. The
precondition `resume_run` enforces for a human typing `/workflows resume`
is not a safety net this call site needs -- the caller (the `/background`
handler) already establishes "the local owner just cleanly stopped, and
this process is its sole intended successor" as an invariant before ever
spawning this process, which is a strictly stronger guarantee than
`resume_run`'s own status check would provide.

RACE SAFETY (journal write-ordering)
-------------------------------------

`run_manager.JournalWriter._append` opens, writes, and closes
`journal.jsonl` synchronously on every single record (see `run_manager.py`
-- there is no buffering held open across calls). Awaiting the cancelled
local task to completion (per the handoff sequence's step (a)) therefore
guarantees its `except asyncio.CancelledError` handler has already
completed its own synchronous "cancelled" write -- and the `finally`
block's heartbeat-task teardown -- before that `await` returns control to
the CLI. Only once that has happened does the CLI spawn this process
(step (b)). Because `JournalWriter` never holds the file open between
writes, sequential cancel-then-await-then-spawn is sufficient on its own:
there is never a window where the old (local) writer and the new
(subprocess) writer could both be mid-append to the same file, so no
explicit flush/close step on the local `JournalWriter` instance beyond
what `execute_run`'s own cancellation handling already does is needed.
This process constructs its own fresh `JournalWriter` (inside the
`execute_run` it calls), which reads the file's current tail at
construction to continue the `seq` counter monotonically -- exactly the
"survives process restarts" behavior `JournalWriter`'s own docstring
promises.

UNCAUGHT-EXCEPTION HANDLING
-----------------------------

`execute_run` already writes a `run_status: "failed"` journal record (plus
a `kind: "log"` error line) for any exception raised from inside the
workflow's `main()` or from constructing its runtime -- see its own
`except Exception` handler -- so this module does not need to duplicate
that. The one gap `execute_run` cannot itself cover is a `FileNotFoundError`
raised before any journal even exists (an unknown `--run-id`, or a run
directory that was never created) -- there is nothing to write a journal
record TO in that case. This module's `main()` catches that (and any other
exception that somehow escapes `execute_run`) at the top level and prints
it to stderr, which the parent's `subprocess.Popen(..., stderr=...)`
redirect always points at a log file under the run's own directory (never
`/tmp`) -- so an uncaught exception in this detached, terminal-less process
is always visible somewhere, never silently lost.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import traceback

from vibe.workflows import run_manager


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vibe.workflows.background_runner",
        description=(
            "Drive one workflow run to completion with no Textual UI, no "
            "stdin reads, and no reliance on a parent process -- the "
            "detached-process half of the `/background <run_id>` command."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The run_id to resume (its run directory must already exist "
        "under --runs-root).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Override the runs root directory (defaults to "
        "run_manager.DEFAULT_RUNS_ROOT). Mainly for tests.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override the /ask endpoint for this run (defaults to the "
        "endpoint the run was created with).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override the /ask workspace for this run (defaults to the "
        "workspace the run was created with).",
    )
    return parser


async def run_to_completion(
    run_id: str,
    *,
    runs_root: Path,
    endpoint: str | None = None,
    workspace: str | None = None,
) -> int:
    """Drive `run_id` to completion via `run_manager.execute_run`.

    Returns a process exit code (0 on success, 1 on failure) rather than
    raising, except for `asyncio.CancelledError` on a genuine kill signal,
    which propagates so `asyncio.run` unwinds normally -- `execute_run`'s
    own cancellation handling has already written the terminal journal
    record by the time that happens (see the module docstring).
    """
    try:
        await run_manager.execute_run(
            run_id, runs_root=runs_root, endpoint=endpoint, workspace=workspace
        )
    except FileNotFoundError as exc:
        # No run directory exists for `run_id` -- there is no journal to
        # write this failure into (see the module docstring's "uncaught
        # exception handling" section). Surface on stderr only.
        print(f"background_runner: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # `execute_run` already wrote `run_status="failed"` (plus a log
        # record) to the run's own journal before re-raising -- this is a
        # belt-and-suspenders surface onto stderr (captured by the
        # parent's log-file redirect) so a detached, terminal-less process
        # never fails silently even if the journal write itself somehow
        # didn't happen.
        traceback.print_exc()
        print(
            f"background_runner: workflow run {run_id!r} failed: {exc}", file=sys.stderr
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runs_root = (
        args.runs_root if args.runs_root is not None else run_manager.DEFAULT_RUNS_ROOT
    )

    return asyncio.run(
        run_to_completion(
            args.run_id,
            runs_root=runs_root,
            endpoint=args.endpoint,
            workspace=args.workspace,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
