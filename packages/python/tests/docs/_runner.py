"""Run the docs' Python snippets against a mock model, and check skipped ones statically."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib
import signal
import traceback
from typing import TYPE_CHECKING, Any

from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, Role, ToolCall

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.types import Message, ToolDef
    from tests.docs._snippets import Snippet

BLOCK_TIMEOUT_SECONDS = 20
_SAMPLE = {"string": "example", "integer": 1, "number": 1.0, "boolean": True}


def _sample_args(tool: ToolDef) -> dict[str, Any]:
    props = tool.parameters.get("properties", {})
    required = tool.parameters.get("required", list(props))
    return {
        name: _SAMPLE.get(props.get(name, {}).get("type"))
        if props.get(name, {}).get("type") not in ("array", "object")
        else ([] if props[name]["type"] == "array" else {})
        for name in required
    }


class DocsProvider(MockLLM):
    """Stands in for every real provider in the docs: no network, no key, deterministic.

    A turn that offers tools and has no tool result yet calls the first tool with
    sample arguments, so approval and client-tool pauses can happen. Every other turn
    answers with fixed text.
    """

    def __init__(self, *args: Any, model: str = "mock-model", **kwargs: Any) -> None:
        super().__init__([], model=model)

    async def complete(  # type: ignore[override]
        self, messages: list[Message], tools: list[ToolDef] | None = None, **kwargs: Any
    ) -> LLMResponse:
        self._call_count += 1
        model = kwargs.get("model") or self._model
        last_is_tool_result = bool(messages) and messages[-1].role == Role.TOOL
        if tools and not last_is_tool_result:
            call = ToolCall(name=tools[0].name, params=_sample_args(tools[0]))
            return LLMResponse(tool_calls=[call], model=model)
        return LLMResponse(text="ok (docs mock reply)", model=model)


@contextlib.contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _expired(*_: object) -> None:
        raise TimeoutError(f"snippet ran longer than {seconds}s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_snippet(snippet: Snippet, namespace: dict[str, Any]) -> None:
    """Execute one block in the page's shared namespace; top-level ``await`` works."""
    code = compile(snippet.code, snippet.id, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with _deadline(BLOCK_TIMEOUT_SECONDS):
        result = eval(code, namespace)  # noqa: S307
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(result)
            finally:
                loop.close()


def check_static(snippet: Snippet) -> str | None:
    """For a skipped block: if it parses, every ``dendrux`` import in it must resolve."""
    try:
        tree = ast.parse(snippet.code)
    except SyntaxError:
        return None  # a signature listing or pseudo-code; the skip reason says so
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] != "dendrux":
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError as exc:
                return f"{snippet.id}: cannot import {node.module}: {exc}"
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        return f"{snippet.id}: {node.module} has no {alias.name!r}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "dendrux":
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as exc:
                        return f"{snippet.id}: cannot import {alias.name}: {exc}"
    return None


def run_page(snippets: list[Snippet]) -> list[str]:
    """Run a page's blocks in order, sharing one namespace. Returns failure messages."""
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    failures: list[str] = []
    for snippet in snippets:
        if snippet.skip_reason is not None:
            problem = check_static(snippet)
            if problem:
                failures.append(problem)
            continue
        try:
            run_snippet(snippet, namespace)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            tail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append(f"{snippet.id}: {tail}")
    return failures
