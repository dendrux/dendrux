"""``Agent(exclusive_subagents=...)`` type-checks on both ``__init__`` overloads."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

mypy_api = pytest.importorskip("mypy.api")

_BODY = """
    from dendrux import Agent

    child = Agent(name="a", description="d", prompt="p", provider="anthropic:x")
    # first overload: provider and prompt given
    Agent(provider="anthropic:x", prompt="p", subagents=[child], exclusive_subagents=["a"])


    class Inherits(Agent):
        prompt = "inherited"


    # second overload: prompt inherited from the class, provider optional
    Inherits(subagents=[child], exclusive_subagents=["a"])
    Inherits(exclusive_subagents=None)
    Agent(exclusive_subagents=["a"], provider=None)
"""


def test_exclusive_subagents_is_accepted_by_both_overloads(tmp_path: Path) -> None:
    sample = tmp_path / "consumer.py"
    sample.write_text(textwrap.dedent(_BODY))
    stdout, _err, _status = mypy_api.run(
        [
            "--strict",
            "--no-incremental",
            "--follow-imports=silent",
            "--config-file=",
            "--show-error-codes",
            "--no-error-summary",
            str(sample),
        ]
    )
    errors = [line for line in stdout.splitlines() if ": error:" in line]
    assert errors == []
