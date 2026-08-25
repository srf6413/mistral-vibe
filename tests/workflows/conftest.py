"""Shared, network-free fixtures for testing `vibe.workflows`.

`FakeAskTransport` is the `_FakeAskTransport` pattern already proven in
`tests/app_server/test_founderos_ask.py`, extended to hold a *sequence* of
programmed turns (one popped per `.stream(...)` call) instead of one fixed
reply, plus a shorthand for the `awaiting_permission` /
`awaiting_clarification` pause. It is injected into a REAL
`FounderOSAskSession`, never a fake session -- so any test built on this
fixture exercises the real SSE-translation path in `_founderos_ask.py`,
and `agent_call.call_agent` (once implemented) is tested against the same
session type it runs against in production.

`make_fake_session_factory` mirrors the shape `call_agent` and
`WorkflowRuntime` require: a zero-argument `session_factory` that returns a
*fresh, not-yet-used* session on every call, matching "each `call_agent`
call owns exactly one session instance" (see `agent_call.py`).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from vibe.app_server import FounderOSAskSession

Turn = list[dict[str, Any]] | str
"""One programmed turn: a list of raw SSE event dicts, or the literal
string `"awaiting_permission"` / `"awaiting_clarification"`."""


class FakeAskTransport:
    """Programmable `AskTransport`: no network, one queue of canned turns.

    Each element of `turns` is either:

    - a list of raw SSE-decoded event dicts, in the same shape
      `HttpFounderOSAskTransport.stream` yields (see
      `FakeAskTransport.ok(...)` for the common case of a single
      `final_result`); or
    - the literal string `"awaiting_permission"` or
      `"awaiting_clarification"`, which streams one event of that type so
      `FounderOSAskSession.act(...)` raises `FounderOSAskStreamError`
      exactly as it would against the real local /ask boundary.

    `.stream(...)` pops the next turn on every call; calling it more times
    than turns were programmed raises `AssertionError` immediately, rather
    than silently repeating the last turn or hanging.
    """

    def __init__(self, turns: Sequence[Turn] | None = None) -> None:
        self._turns: list[Turn] = list(turns or [])
        self.payloads: list[dict[str, object]] = []
        self.cancel_count = 0
        self.close_count = 0

    @staticmethod
    def ok(text: str) -> list[dict[str, Any]]:
        """Shorthand turn: a single `final_result` carrying `text`."""
        return [{"type": "final_result", "data": {"answer": text}}]

    @staticmethod
    def ok_with_deltas(*deltas: str, final: str | None = None) -> list[dict[str, Any]]:
        """Shorthand turn: streamed `text_delta`s then a `final_result`."""
        events: list[dict[str, Any]] = [
            {"type": "text_delta", "content": delta} for delta in deltas
        ]
        events.append({
            "type": "final_result",
            "data": {"answer": final or "".join(deltas)},
        })
        return events

    def program(self, turn: Turn) -> None:
        self._turns.append(turn)

    def stream(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.payloads.append(dict(payload))
        if not self._turns:
            raise AssertionError(
                "FakeAskTransport.stream() called more times than turns were "
                "programmed -- call .program(...) with one more turn"
            )
        turn = self._turns.pop(0)

        async def generate() -> AsyncGenerator[dict[str, Any], None]:
            if turn in ("awaiting_permission", "awaiting_clarification"):
                yield {"type": turn}
                return
            for event in turn:  # type: ignore[union-attr]
                yield event

        return generate()

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def close(self) -> None:
        self.close_count += 1


def make_fake_session_factory(
    *, cwd: Path, turns: Sequence[Turn]
) -> tuple[Callable[[], FounderOSAskSession], list[FakeAskTransport]]:
    """Build a `session_factory` where the Nth call gets a fresh
    `FounderOSAskSession` wired to a fresh `FakeAskTransport` pre-loaded
    with `turns[N]`.

    Returns `(session_factory, transports)`: `transports` starts empty and
    is appended to (in call order) as `session_factory()` runs, so a test
    can assert per-call `.payloads` / `.cancel_count` / `.close_count`
    after exercising the code under test.
    """
    remaining = list(turns)
    transports: list[FakeAskTransport] = []

    def factory() -> FounderOSAskSession:
        if not remaining:
            raise AssertionError(
                "fake session_factory() called more times than turns were "
                "programmed for this test"
            )
        transport = FakeAskTransport([remaining.pop(0)])
        transports.append(transport)
        session_id = f"fake-session-{len(transports) - 1}"
        return FounderOSAskSession(transport=transport, cwd=cwd, session_id=session_id)

    return factory, transports


@pytest.fixture
def fake_session_factory(
    tmp_path: Path,
) -> Callable[
    [Sequence[Turn]], tuple[Callable[[], FounderOSAskSession], list[FakeAskTransport]]
]:
    """`fake_session_factory([...turns...]) -> (session_factory, transports)`."""

    def build(
        turns: Sequence[Turn],
    ) -> tuple[Callable[[], FounderOSAskSession], list[FakeAskTransport]]:
        return make_fake_session_factory(cwd=tmp_path, turns=turns)

    return build
