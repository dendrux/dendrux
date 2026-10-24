"""The snippet runner: shared namespace, skips, static import checks, mock provider."""

from __future__ import annotations

from tests.docs._runner import DocsProvider, run_page
from tests.docs._snippets import Snippet


def _s(code: str, line: int = 1, skip: str | None = None) -> Snippet:
    return Snippet("page.mdx", line, code, skip)


def test_blocks_share_one_namespace_in_order() -> None:
    assert run_page([_s("x = 2\n"), _s("assert x == 2\n", 5)]) == []


def test_a_failing_block_is_reported_with_its_place_and_the_rest_still_run() -> None:
    failures = run_page([_s("1 / 0\n", 3), _s("y = 1\n", 9), _s("assert y == 1\n", 12)])
    assert len(failures) == 1
    assert failures[0].startswith("page.mdx:3:")
    assert "ZeroDivisionError" in failures[0]


def test_skipped_blocks_are_not_executed() -> None:
    assert run_page([_s("raise RuntimeError('ran')\n", skip="fragment")]) == []


def test_skipped_blocks_still_have_dendrux_imports_checked() -> None:
    bad = _s("from dendrux import NoSuchThing\n", 4, skip="fragment")
    failures = run_page([bad])
    assert failures and "NoSuchThing" in failures[0]
    missing_module = _s("from dendrux.nope import x\n", 6, skip="fragment")
    assert "dendrux.nope" in run_page([missing_module])[0]


def test_skipped_blocks_that_are_not_python_are_left_alone() -> None:
    assert run_page([_s("async def f(x) -> int\n", skip="signature listing")]) == []


def test_top_level_await_works() -> None:
    code = "import asyncio\nawait asyncio.sleep(0)\nvalue = 3\n"
    assert run_page([_s(code), _s("assert value == 3\n", 4)]) == []


def test_asyncio_run_inside_a_block_works() -> None:
    assert run_page([_s("import asyncio\n\nasyncio.run(asyncio.sleep(0))\n")]) == []


def test_runaway_blocks_time_out(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("tests.docs._runner.BLOCK_TIMEOUT_SECONDS", 1)
    failures = run_page([_s("import time\ntime.sleep(5)\n")])
    assert failures and "TimeoutError" in failures[0]


async def test_docs_provider_calls_the_first_tool_then_answers() -> None:
    from dendrux import tool
    from dendrux.tool import get_tool_def
    from dendrux.types import Message, Role, ToolCall

    @tool()
    async def lookup(order_id: int, note: str) -> str:
        """Look it up."""
        return "x"

    provider = DocsProvider(model="whatever", api_key="ignored")
    first = await provider.complete([Message(role=Role.USER, content="hi")], [get_tool_def(lookup)])
    assert first.tool_calls and first.tool_calls[0].name == "lookup"
    assert first.tool_calls[0].params == {"order_id": 1, "note": "example"}
    result = Message(role=Role.TOOL, content="x", call_id=first.tool_calls[0].id, name="lookup")
    second = await provider.complete(
        [Message(role=Role.USER, content="hi"), result], [get_tool_def(lookup)]
    )
    assert not second.tool_calls
    assert second.text
    assert isinstance(first.tool_calls[0], ToolCall)
