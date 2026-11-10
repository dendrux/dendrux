"""``@tool(...)`` keeps the decorated function's static signature.

The decorator returns the same function object at runtime, so a type checker
should see the original parameters and return type, not ``Any``. These tests run
mypy on small consumer files: one that must type-check cleanly and one whose
wrong calls must each be reported.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

mypy_api = pytest.importorskip("mypy.api")

_SRC = Path(__file__).resolve().parents[2] / "src"

_TOOLS = """
    from dendrux import tool


    @tool()
    async def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b


    @tool(target="client", parallel=False)
    def label(name: str, *, upper: bool = False) -> str:
        '''Format a label.'''
        return name.upper() if upper else name


    class Sheet:
        @tool(timeout_seconds=5.0)
        async def read(self, cell: str) -> list[float]:
            '''Read one cell.'''
            return [float(len(cell))]
"""


def _mypy(tmp_path: Path, body: str) -> tuple[int, list[str]]:
    sample = tmp_path / "consumer.py"
    sample.write_text(textwrap.dedent(_TOOLS) + textwrap.dedent(body))
    stdout, _stderr, status = mypy_api.run(
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
    return status, errors


@pytest.fixture
def mypy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYPYPATH", str(_SRC))


@pytest.mark.usefixtures("mypy_path")
def test_decorated_tools_keep_their_signatures(tmp_path: Path) -> None:
    status, errors = _mypy(
        tmp_path,
        """
        from typing import assert_type

        from dendrux.tool import get_tool_def
        from dendrux.types import ToolDef


        async def use() -> None:
            assert_type(get_tool_def(add), ToolDef)
            assert_type(get_tool_def(Sheet.read).name, str)
            assert_type(await add(1, 2), int)
            assert_type(label("x"), str)
            assert_type(label("x", upper=True), str)
            assert_type(await Sheet().read("A1"), list[float])
        """,
    )
    assert errors == []
    assert status == 0


@pytest.mark.usefixtures("mypy_path")
def test_wrong_calls_to_decorated_tools_are_reported(tmp_path: Path) -> None:
    _status, errors = _mypy(
        tmp_path,
        """
        async def misuse() -> None:
            await add("1", 2)
            label("x", lower=True)
            await Sheet().read(1)
            total: str = await add(1, 2)
        """,
    )
    codes = sorted(line.rsplit("[", 1)[-1].rstrip("]") for line in errors)
    assert codes == ["arg-type", "arg-type", "assignment", "call-arg"], errors
