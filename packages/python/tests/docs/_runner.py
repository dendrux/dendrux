"""Run the docs' Python snippets against a mock model, and check skipped ones statically."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib
import inspect
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

    def __init__(self, *, model: str = "mock-model") -> None:
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


def checked_provider_mock(real: type, forwards_to: type | None = None) -> type[DocsProvider]:
    """A ``DocsProvider`` that accepts exactly the constructor calls ``real`` accepts.

    The call is bound against the real signature, so an unknown keyword or a missing
    required argument (``model``) fails as it would for a reader. ``forwards_to`` is the
    class that receives a ``**kwargs`` passthrough (OpenRouter forwards to the OpenAI
    provider); those names are checked against it.
    """
    signature = inspect.signature(real)
    var_keyword = next(
        (p.name for p in signature.parameters.values() if p.kind is p.VAR_KEYWORD), None
    )
    forwarded = inspect.signature(forwards_to).parameters if forwards_to else None

    class _Checked(DocsProvider):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            bound = signature.bind(*args, **kwargs)
            if var_keyword and forwarded is not None:
                for name in bound.arguments.get(var_keyword, {}):
                    if name not in forwarded:
                        raise TypeError(
                            f"{real.__name__}() got an unexpected keyword argument {name!r}"
                        )
            super().__init__(model=bound.arguments.get("model") or "mock-model")

    _Checked.__name__ = _Checked.__qualname__ = real.__name__
    return _Checked


def _stub(value: Any) -> Any:
    def stub(*_args: Any, **_kwargs: Any) -> Any:
        return value() if callable(value) else value

    stub.docs_sandbox_stub = True  # type: ignore[attr-defined]
    return stub


def dotenv_stubs() -> list[tuple[Any, str, Any]]:
    """(module, attribute, stub) so no snippet can search for or read a ``.env`` file.

    python-dotenv walks up from the caller's directory to the first ``.env`` it finds, which
    a scratch working directory does not prevent. Returns nothing when it is not installed.
    """
    try:
        import dotenv
        import dotenv.main
    except ImportError:
        return []
    stubs = {"load_dotenv": _stub(False), "find_dotenv": _stub(""), "dotenv_values": _stub(dict)}
    return [(module, name, stubs[name]) for module in (dotenv, dotenv.main) for name in stubs]


def provider_mocks() -> list[tuple[Any, str, type]]:
    """(module, attribute, mock) for every provider class the docs construct."""
    import dendrux.llm.anthropic as anthropic
    import dendrux.llm.openai as openai
    import dendrux.llm.openai_responses as openai_responses
    import dendrux.llm.openrouter as openrouter

    return [
        (anthropic, "AnthropicProvider", checked_provider_mock(anthropic.AnthropicProvider)),
        (openai, "OpenAIProvider", checked_provider_mock(openai.OpenAIProvider)),
        (
            openai_responses,
            "OpenAIResponsesProvider",
            checked_provider_mock(openai_responses.OpenAIResponsesProvider),
        ),
        (
            openrouter,
            "OpenRouterProvider",
            checked_provider_mock(openrouter.OpenRouterProvider, openai.OpenAIProvider),
        ),
    ]


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
