"""The snippet runner: shared namespace, skips, static import checks, mock provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tests.docs._runner import DocsProvider, run_page
from tests.docs._snippets import Snippet

if TYPE_CHECKING:
    import pytest


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

    provider = DocsProvider(model="whatever")
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


class TestProviderMocksCheckTheRealConstructor:
    """The docs mock must reject what the real provider constructor rejects."""

    def _run(self, code: str) -> list[str]:
        return run_page([_s(code)])

    def test_a_valid_call_runs(self, docs_sandbox: None) -> None:
        code = (
            "from dendrux.llm.anthropic import AnthropicProvider\n"
            "AnthropicProvider(model='claude-haiku-4-5', max_tokens=100)\n"
        )
        assert self._run(code) == []

    def test_an_unknown_keyword_fails(self, docs_sandbox: None) -> None:
        code = (
            "from dendrux.llm.anthropic import AnthropicProvider\n"
            "AnthropicProvider(model='claude-haiku-4-5', removed_option=True)\n"
        )
        failures = self._run(code)
        assert failures and "removed_option" in failures[0]

    def test_a_missing_required_model_fails(self, docs_sandbox: None) -> None:
        for module, cls in (
            ("anthropic", "AnthropicProvider"),
            ("openai", "OpenAIProvider"),
            ("openai_responses", "OpenAIResponsesProvider"),
        ):
            failures = self._run(f"from dendrux.llm.{module} import {cls}\n{cls}()\n")
            assert failures and "model" in failures[0], cls

    def test_discovery_only_construction_is_still_allowed(self, docs_sandbox: None) -> None:
        code = (
            "from dendrux.llm.openrouter import OpenRouterProvider\n"
            "OpenRouterProvider(api_key='k')\n"
        )
        assert self._run(code) == []

    def test_openrouter_forwards_extra_options_to_the_openai_provider(
        self, docs_sandbox: None
    ) -> None:
        head = "from dendrux.llm.openrouter import OpenRouterProvider\n"
        assert self._run(head + "OpenRouterProvider(model='m', max_tokens=10)\n") == []
        bad = head + "OpenRouterProvider(model='m', nope=1)\n"
        failures = self._run(bad)
        assert failures and "nope" in failures[0]

    def test_the_mock_still_answers_like_the_provider(self, docs_sandbox: None) -> None:
        code = (
            "from dendrux.llm.openai import OpenAIProvider\n"
            "p = OpenAIProvider(model='gpt-x')\n"
            "assert p.model == 'gpt-x'\n"
        )
        assert self._run(code) == []


class TestSandboxNeverReadsDotenv:
    """`load_dotenv()` walks up from the caller to a real `.env`; the sandbox must stop that."""

    def test_the_dotenv_functions_are_stubbed(self, docs_sandbox: None) -> None:
        import dotenv
        import dotenv.main

        for module in (dotenv, dotenv.main):
            for name in ("load_dotenv", "find_dotenv", "dotenv_values"):
                assert getattr(getattr(module, name), "docs_sandbox_stub", False), (module, name)

    def test_running_the_dotenv_snippet_opens_and_searches_no_env_file(
        self, docs_sandbox: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins
        import io
        import os

        touched: list[str] = []
        real_open, real_io_open, real_isfile = builtins.open, io.open, os.path.isfile

        def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if ".env" in str(file):
                touched.append(str(file))
            return real_open(file, *args, **kwargs)

        def spy_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if ".env" in str(file):
                touched.append(str(file))
            return real_io_open(file, *args, **kwargs)

        def spy_isfile(path: Any) -> bool:
            if ".env" in str(path):
                touched.append(str(path))
            return real_isfile(path)

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(io, "open", spy_io_open)
        monkeypatch.setattr(os.path, "isfile", spy_isfile)
        code = (
            "from dotenv import dotenv_values, find_dotenv, load_dotenv\n"
            "assert load_dotenv() is False\n"
            "assert find_dotenv() == ''\n"
            "assert dotenv_values() == {}\n"
        )
        assert self._run(code) == []
        assert touched == []

    def _run(self, code: str) -> list[str]:
        return run_page([_s(code)])

    def test_the_quickstart_dotenv_block_runs_through_the_stub(self, docs_sandbox: None) -> None:
        from pathlib import Path

        from tests.docs._snippets import extract_snippets

        repo = Path(__file__).resolve().parents[4]
        blocks = [
            s
            for s in extract_snippets(repo / "docs" / "quickstart.mdx", repo)
            if "load_dotenv" in s.code
        ]
        assert blocks and blocks[0].skip_reason is None
        assert run_page(blocks) == []
