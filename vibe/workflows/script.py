"""Load and validate a workflow script file.

A workflow script is a plain Python file (never JavaScript -- this is a
from-scratch mirror of the Claude Code Workflow tool's *contract*, not a
port of its implementation) with two required top-level members:

    meta = {
        "name": "...",
        "description": "...",
        "phases": [{"title": "...", "detail": "..."}, ...],
    }

    async def main(wf, args):
        ...

`meta` must be a pure Python literal -- it is parsed with
`ast.literal_eval`, exactly like Claude Code's workflow tool, and loading
fails loudly if it references a name, calls a function, or does anything
else `literal_eval` cannot evaluate. This keeps `meta` inspectable (for the
UI to render phase lists, for `run_manager` to write `meta.json`) without
ever executing the script just to discover its shape.

The whole script (not just `main`) is also linted at parse time to reject
direct wall-clock / randomness / uuid use, because the run journal is
append-only and replay/resume depends on a workflow script being a pure
function of (its args, the `AgentCallResult`s it already received) -- see
`vibe/workflows/events.py` for the corresponding id-minting contract that
the *runtime* (not the script) is responsible for keeping deterministic.
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.workflows.events import PhaseSpec, WorkflowMeta

if TYPE_CHECKING:
    from vibe.workflows.runtime import WorkflowRuntime

_BANNED_MODULES = ("time", "datetime", "random", "uuid")
"""Modules a workflow script may not import at all, in any form."""

_BANNED_ATTR_CHAINS = {
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("time", "monotonic_ns"),
    ("time", "perf_counter"),
    ("time", "perf_counter_ns"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("random",),
    ("uuid",),
}
"""Dotted call targets banned even if reached through a re-exported alias
of an otherwise-allowed module (e.g. `import os.path as p; p.time.time()`
is out of scope -- this catches the direct, common forms)."""


class WorkflowScriptError(ValueError):
    """The script at `path` is not a valid workflow script."""


@dataclass(frozen=True, slots=True)
class LoadedWorkflowScript:
    """A parsed, linted, not-yet-executed workflow script."""

    path: Path
    source: str
    meta: WorkflowMeta
    tree: ast.Module


def load_workflow_script(path: Path) -> LoadedWorkflowScript:
    """Read, parse, lint, and extract `meta` from `path`.

    Raises `WorkflowScriptError` for anything wrong with the script itself
    (syntax error, missing/non-literal `meta`, malformed `meta` shape, or a
    banned import/call caught by the AST lint). Does not execute the
    script -- see `compile_workflow_main` for that step.
    """
    try:
        source = path.read_text()
    except OSError as exc:
        raise WorkflowScriptError(f"cannot read workflow script {path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise WorkflowScriptError(f"{path} is not valid Python: {exc}") from exc

    _lint_ast(tree, path=path)
    meta = _extract_meta(tree, path=path)
    return LoadedWorkflowScript(path=path, source=source, meta=meta, tree=tree)


def _extract_meta(tree: ast.Module, *, path: Path) -> WorkflowMeta:
    meta_node: ast.expr | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "meta"
        ):
            meta_node = node.value
            break
    if meta_node is None:
        raise WorkflowScriptError(
            f"{path} has no top-level `meta = {{...}}` assignment"
        )
    try:
        raw = ast.literal_eval(meta_node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise WorkflowScriptError(
            f"{path}: `meta` must be a pure literal (no names, no calls): {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise WorkflowScriptError(f"{path}: `meta` must be a dict literal")

    missing = {"name", "description"} - raw.keys()
    if missing:
        raise WorkflowScriptError(
            f"{path}: `meta` is missing required key(s): {sorted(missing)}"
        )
    name = raw["name"]
    description = raw["description"]
    if not isinstance(name, str) or not isinstance(description, str):
        raise WorkflowScriptError(f"{path}: `meta.name`/`meta.description` must be str")

    raw_phases = raw.get("phases", [])
    if not isinstance(raw_phases, list):
        raise WorkflowScriptError(f"{path}: `meta.phases` must be a list")
    phases: list[PhaseSpec] = []
    for i, entry in enumerate(raw_phases):
        if not isinstance(entry, dict) or "title" not in entry:
            raise WorkflowScriptError(
                f"{path}: `meta.phases[{i}]` must be a dict with a 'title' key"
            )
        title = entry["title"]
        detail = entry.get("detail", "")
        if not isinstance(title, str) or not isinstance(detail, str):
            raise WorkflowScriptError(
                f"{path}: `meta.phases[{i}]` title/detail must be str"
            )
        phases.append(PhaseSpec(title=title, detail=detail))

    return WorkflowMeta(name=name, description=description, phases=phases)


def _lint_ast(tree: ast.Module, *, path: Path) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BANNED_MODULES:
                    raise WorkflowScriptError(
                        f"{path}:{node.lineno}: `import {alias.name}` is banned "
                        "(nondeterministic module) -- workflow scripts must not "
                        "import time, datetime, random, or uuid"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.split(".", 1)[0] in (
                _BANNED_MODULES
            ):
                raise WorkflowScriptError(
                    f"{path}:{node.lineno}: `from {node.module} import ...` is "
                    "banned (nondeterministic module) -- workflow scripts must "
                    "not import time, datetime, random, or uuid"
                )
        elif isinstance(node, ast.Call):
            chain = _dotted_chain(node.func)
            if chain is not None and _matches_banned_chain(chain):
                dotted = ".".join(chain)
                raise WorkflowScriptError(
                    f"{path}:{node.lineno}: `{dotted}(...)` is banned "
                    "(nondeterministic call) -- breaks journal replay/resume"
                )


def _matches_banned_chain(chain: tuple[str, ...]) -> bool:
    for banned in _BANNED_ATTR_CHAINS:
        if chain[: len(banned)] == banned:
            return True
    return False


def _dotted_chain(node: ast.expr) -> tuple[str, ...] | None:
    """`a.b.c` -> `("a", "b", "c")`; anything else (calls, subscripts) -> None."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return tuple(reversed(parts))


_ALLOWED_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "StopIteration",
    "True",
    "False",
    "None",
)
"""Builtins exposed to an exec'd workflow script. Deliberately excludes
`__import__`, `open`, `eval`, `exec`, `compile`, `input`, and `exit`/`quit`
-- a workflow script gets everything it needs through `wf`, not through
ambient filesystem/process access."""


def build_restricted_globals(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Globals dict a linted script is exec'd into.

    `__builtins__` is trimmed to `_ALLOWED_BUILTINS` so `import` inside the
    exec'd code raises `ImportError: __import__ not found` even if the AST
    lint's import check were somehow bypassed -- the lint and this
    restricted-globals trim are two independent layers of the same rule.
    """
    builtins_ns = {name: getattr(_builtins_module, name) for name in _ALLOWED_BUILTINS}
    globals_dict: dict[str, Any] = {"__builtins__": builtins_ns}
    if extra:
        globals_dict.update(extra)
    return globals_dict


def compile_workflow_main(
    loaded: LoadedWorkflowScript, globals_dict: dict[str, Any]
) -> Any:
    """Exec the linted script's AST into `globals_dict` and return `main`.

    Raises `WorkflowScriptError` if the script does not define an
    `async def main(wf, args)` at module scope. Callers should build
    `globals_dict` with `build_restricted_globals(...)`.
    """
    code = compile(loaded.tree, filename=str(loaded.path), mode="exec")
    exec(code, globals_dict)
    main = globals_dict.get("main")
    if main is None or not _is_async_callable(main):
        raise WorkflowScriptError(
            f"{loaded.path} must define `async def main(wf, args)` at module scope"
        )
    return main


def _is_async_callable(value: Any) -> bool:
    return inspect.iscoroutinefunction(value)


async def run_workflow_script(
    path: Path, wf: WorkflowRuntime, args: dict[str, Any]
) -> None:
    """Load, lint, compile, and run one workflow script's `main(wf, args)`.

    The glue between this module (script loading/linting) and
    `vibe.workflows.runtime.WorkflowRuntime` (the `wf` the script's `main`
    receives) -- everything else in this file is deliberately usable on its
    own (e.g. a UI that only wants `load_workflow_script(...).meta` to list
    phases without running anything), so this helper is additive, not a
    replacement for calling the pieces individually.

    Sets the same nesting-guard `ContextVar` `runtime.py` uses around
    `agent()` for the whole span of `main(...)`, not just around individual
    `agent()` calls -- so an attempt to construct a second `WorkflowRuntime`
    anywhere during a script's execution (not only mid-`agent()`-call) is
    caught, matching the "workflows nest one level only" rule.

    Not wired into `run_manager` (run creation / resume / the journal
    writer) -- those remain that lane's `NotImplementedError` stubs to fill
    in; this only covers "given a loaded script and a constructed `wf`, run
    it."
    """
    # Local import: avoids a module-level import cycle (`runtime.py` does
    # not import `script.py`, but importing `WorkflowRuntime` at module
    # scope here would still be an unnecessary hard dependency for callers
    # of this file who only want the loader/linter, not the runtime).
    from vibe.workflows.runtime import _NESTING_GUARD

    loaded = load_workflow_script(path)
    globals_dict = build_restricted_globals()
    main = compile_workflow_main(loaded, globals_dict)

    token = _NESTING_GUARD.set(_NESTING_GUARD.get() + 1)
    try:
        await main(wf, args)
    finally:
        _NESTING_GUARD.reset(token)
