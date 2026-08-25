from __future__ import annotations

import pytest

from vibe.cli.commands import CommandRegistry
from vibe.cli.textual_ui.widgets.chat_input.input_kinds import (
    Prompt,
    SlashCommand,
    classify,
)


def _classify(value: str) -> object:
    return classify(
        value, commands=CommandRegistry(), resolve_skill=lambda _value: None
    )


@pytest.mark.parametrize("alias", ["/exit", "exit", "quit", ":q", ":quit"])
def test_classify_treats_bare_exit_synonyms_as_slash_command(alias: str) -> None:
    assert isinstance(_classify(alias), SlashCommand)


@pytest.mark.parametrize("value", ["exit the function early", "quit your job"])
def test_classify_keeps_bare_synonym_with_trailing_text_as_prompt(value: str) -> None:
    result = _classify(value)
    assert isinstance(result, Prompt)
    assert result.text == value


@pytest.mark.parametrize("value", ["/compact", "/compact summarize the last hour"])
def test_classify_sends_compact_through_as_a_plain_prompt(value: str) -> None:
    """`/compact` has no client-side command registration (unlike `/exit`) and
    no matching skill, so it must fall through to Prompt and travel to
    FounderOS /ask as ordinary text — exactly like /remember, /recall,
    /lessons, and /synth already do. FounderOSAskSession.compact() has no
    working client endpoint; routing `/compact` through the normal turn path
    lets the server-side memory-command parser handle it instead.
    """
    result = _classify(value)
    assert isinstance(result, Prompt)
    assert result.text == value
