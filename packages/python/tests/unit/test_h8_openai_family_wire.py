"""Captured-request snapshots for the OpenAI-family adapters (H8 pins).

Each cell runs one scenario through the real adapter (Chat, Responses or
OpenRouter) in batch or stream mode with the SDK ``create`` method replaced by
a recorder. The cell snapshot holds the exact keyword arguments handed to the
SDK, the adapter's ``captured_request`` (``provider_request``), the stream
events and the normalised response. Fixtures were captured from the code
before the H8 helper move; a cell that differs fails with the first differing
path.

Regenerate (only when a wire change is intended and reviewed)::

    DENDRUX_REGEN_H8_WIRE=1 pytest tests/unit/test_h8_openai_family_wire.py

Long strings (image data URLs) are pinned by length and sha256, which is still
a byte-exact comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm import openrouter as openrouter_module  # noqa: E402
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402
from dendrux.llm.openrouter import OpenRouterModel, OpenRouterProvider  # noqa: E402
from dendrux.types import (  # noqa: E402
    ImagePart,
    ImageRef,
    LLMResponse,
    Message,
    Role,
    StreamEventType,
    TextPart,
    ToolCall,
    ToolDef,
)
from tests._helpers.v030_oracle import (  # noqa: E402
    Turn,
    chat_completion,
    chat_stream,
    responses_response,
    responses_stream,
    usage_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURES = Path(__file__).parent / "fixtures" / "h8_wire"
REGEN = os.environ.get("DENDRUX_REGEN_H8_WIRE") == "1"
LONG = 160

KIWI = (Path(__file__).parent.parent / "fixtures" / "images" / "kiwi77.png").read_bytes()


def _kiwi() -> ImagePart:
    from dendrux.media import inspect_image

    header = inspect_image(KIWI)
    ref = ImageRef(
        sha256=hashlib.sha256(KIWI).hexdigest(),
        media_type=header.media_type,
        byte_length=len(KIWI),
        width=header.width,
        height=header.height,
    )
    return ImagePart(ref=ref, data=KIWI)


# ---------------------------------------------------------------------------
# Snapshot encoding
# ---------------------------------------------------------------------------


def snap(obj: Any) -> Any:
    """Plain-JSON snapshot; SDK sentinels and unknown objects become type tags."""
    if type(obj).__name__ in {"NotGiven", "Omit"}:
        return "<NOT_GIVEN>"
    if obj is None or isinstance(obj, bool | int | float):
        return obj
    if isinstance(obj, str):
        if len(obj) > LONG:
            return f"<str len={len(obj)} sha256={hashlib.sha256(obj.encode()).hexdigest()}>"
        return obj
    if isinstance(obj, Enum):
        return snap(obj.value)
    if isinstance(obj, dict):
        return {str(k): snap(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [snap(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: snap(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return f"<{type(obj).__name__}>"


def response_snap(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "tool_calls": [
            {
                "name": tc.name,
                "params": snap(tc.params),
                "provider_tool_call_id": tc.provider_tool_call_id,
                "parse_error": tc.parse_error,
            }
            for tc in response.tool_calls or []
        ],
        "usage": snap(response.usage),
        "reasoning": response.reasoning,
        "reasoning_blocks": snap(response.reasoning_blocks),
        "stop_reason": snap(response.stop_reason),
        "raw_stop_reason": response.raw_stop_reason,
        "model": response.model,
        "provider_request": snap(response.provider_request),
        "provider_response": snap(response.provider_response),
    }


def canonical(cell: Any) -> str:
    return json.dumps(cell, sort_keys=True, ensure_ascii=False)


def first_diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: {expected!r} != {actual!r}"
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return f"{path}.{key}: missing on one side"
            found = first_diff(expected[key], actual[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            found = first_diff(e, a, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if expected == actual else f"{path}: {expected!r} != {actual!r}"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

ADD = ToolDef(
    name="add",
    description="Add two numbers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)
SCREENSHOT = ToolDef(
    name="take_screenshot",
    description="Capture the screen.",
    parameters={"type": "object", "properties": {}},
    strict=True,
)
CITY_SCHEMA = {
    "title": "City",
    "type": "object",
    "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["name"],
}


def _text() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are terse."),
        Message(role=Role.USER, content="Say hi."),
    ]


def _server_tool() -> list[Message]:
    call = ToolCall(name="add", params={"a": 2, "b": 3}, provider_tool_call_id="call_A1")
    return [
        Message(role=Role.SYSTEM, content="Use tools."),
        Message(role=Role.USER, content="What is 2+3?"),
        Message(role=Role.ASSISTANT, content="Let me add.", tool_calls=[call]),
        Message(role=Role.TOOL, content="5", name="add", call_id=call.id),
    ]


def _client_tool_image() -> list[Message]:
    call = ToolCall(name="take_screenshot", params={}, provider_tool_call_id="call_S1")
    return [
        Message(role=Role.USER, content="Take a screenshot."),
        Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
        Message.with_parts(
            Role.TOOL, [TextPart('"Captured."'), _kiwi()], name=call.name, call_id=call.id
        ),
        Message.with_parts(Role.USER, [TextPart("And this one?"), _kiwi()]),
    ]


def _structured() -> list[Message]:
    return [Message(role=Role.USER, content="Name a city.")]


TEXT_TURN = Turn(text="Hi there, friend.")
TOOL_TURN = Turn(text="Adding again.", calls=(("add", {"a": 5, "b": 1}, "call_B2"),))
STRUCT_TURN = Turn(text='{"name": "Oslo", "population": 700000}')


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    factory: Callable[[], Any]
    messages: Callable[[], list[Message]]
    turn: Turn
    tools: tuple[ToolDef, ...] = ()
    call: dict[str, Any] = dataclasses.field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    env_base_url: str | None = None


def _chat(**kw: Any) -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o", api_key="test-key", **kw)


def _responses(**kw: Any) -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-5-mini", api_key="test-key", **kw)


def _router(**kw: Any) -> OpenRouterProvider:
    return OpenRouterProvider(
        model=kw.pop("model", "deepseek/deepseek-chat"), api_key="test-key", **kw
    )


CHAT_SCENARIOS = (
    Scenario("text", _chat, _text, TEXT_TURN),
    Scenario("server_tool", _chat, _server_tool, TOOL_TURN, tools=(ADD,)),
    Scenario("client_tool_image", _chat, _client_tool_image, TEXT_TURN, tools=(SCREENSHOT,)),
    Scenario("structured", _chat, _structured, STRUCT_TURN, output_schema=CITY_SCHEMA),
    Scenario(
        "reasoning",
        lambda: _chat(reasoning_effort="low", temperature=0.2),
        _text,
        TEXT_TURN,
        call={"effort": "extra", "thinking": True, "top_p": 0.9, "seed": 7, "unknown_kw": 1},
    ),
    Scenario(
        "reasoning_legacy",
        lambda: _chat(reasoning_effort="medium", max_tokens=99),
        _text,
        TEXT_TURN,
        call={"max_completion_tokens": 77, "stop": ["END"]},
    ),
    Scenario(
        "cache_official",
        lambda: _chat(prompt_cache_retention="24h"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-4o"},
    ),
    Scenario(
        "cache_official_ctor_prefix",
        lambda: _chat(cache_key_prefix="tenant-1"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-4o"},
    ),
    Scenario(
        "cache_custom_base_url",
        lambda: _chat(
            base_url="http://localhost:8000/v1",
            prompt_cache_retention="24h",
            extra_body={"top_k": 3},
        ),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-4o", "extra_body": {"repetition_penalty": 1.1}},
    ),
)

RESPONSES_SCENARIOS = (
    Scenario("text", _responses, _text, TEXT_TURN),
    Scenario(
        "server_tool",
        lambda: _responses(builtin_tools=["web_search_preview"]),
        _server_tool,
        TOOL_TURN,
        tools=(ADD,),
    ),
    Scenario(
        "client_tool_image",
        _responses,
        _client_tool_image,
        TEXT_TURN,
        tools=(SCREENSHOT,),
    ),
    Scenario(
        "client_tool_image_portable",
        lambda: _responses(tool_result_images="portable"),
        _client_tool_image,
        TEXT_TURN,
        tools=(SCREENSHOT,),
    ),
    Scenario("structured", _responses, _structured, STRUCT_TURN, output_schema=CITY_SCHEMA),
    Scenario(
        "reasoning",
        lambda: _responses(thinking=True, temperature=0.3),
        _text,
        TEXT_TURN,
        call={"effort": "extra", "top_p": 0.5, "unknown_kw": 1},
    ),
    Scenario(
        "reasoning_legacy",
        lambda: _responses(reasoning_effort="medium"),
        _text,
        TEXT_TURN,
        call={"max_tokens": 55, "tool_choice": "auto"},
    ),
    Scenario(
        "cache_official",
        lambda: _responses(prompt_cache_retention="24h"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-5-mini"},
    ),
    Scenario(
        "cache_official_ctor_prefix",
        lambda: _responses(cache_key_prefix="tenant-1"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-5-mini"},
    ),
    # OpenAI cache routing fields must stay off compatible endpoints.
    Scenario(
        "cache_custom_base_url_env",
        lambda: _responses(prompt_cache_retention="24h"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:gpt-5-mini"},
        env_base_url="http://localhost:8000/v1",
    ),
)

ROUTER_SCENARIOS = (
    Scenario("text", _router, _text, TEXT_TURN),
    Scenario("server_tool", _router, _server_tool, TOOL_TURN, tools=(ADD,)),
    Scenario("client_tool_image", _router, _client_tool_image, TEXT_TURN, tools=(SCREENSHOT,)),
    Scenario("structured", _router, _structured, STRUCT_TURN, output_schema=CITY_SCHEMA),
    Scenario(
        "reasoning",
        lambda: _router(thinking=True, effort="low"),
        _text,
        TEXT_TURN,
        call={"effort": "extra", "seed": 3},
    ),
    Scenario(
        "reasoning_off",
        _router,
        _text,
        TEXT_TURN,
        call={"thinking": False},
    ),
    Scenario(
        "cache_anthropic",
        lambda: _router(model="anthropic/claude-sonnet-4", app_name="Fixture"),
        _text,
        TEXT_TURN,
        call={"cache_key_prefix": "agent:claude"},
    ),
    Scenario(
        "cache_anthropic_user_control",
        lambda: _router(model="anthropic/claude-sonnet-4"),
        _text,
        TEXT_TURN,
        call={"extra_body": {"cache_control": {"type": "ephemeral", "ttl": "1h"}}},
    ),
)

ADAPTERS = {
    "openai_chat": CHAT_SCENARIOS,
    "openai_responses": RESPONSES_SCENARIOS,
    "openrouter": ROUTER_SCENARIOS,
}

CELLS = [
    (adapter, scenario, mode)
    for adapter, scenarios in ADAPTERS.items()
    for scenario in scenarios
    for mode in ("batch", "stream")
]


def _router_catalog() -> dict[str, OpenRouterModel]:
    def entry(model_id: str) -> OpenRouterModel:
        return OpenRouterModel(
            id=model_id,
            name=model_id,
            context_length=128_000,
            supported_parameters=("tools", "reasoning"),
            prompt_price=0.0,
            completion_price=0.0,
            input_modalities=("text", "image"),
            output_modalities=("text",),
        )

    return {m: entry(m) for m in ("deepseek/deepseek-chat", "anthropic/claude-sonnet-4")}


@pytest.fixture
def router_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = dict(openrouter_module._catalog_cache)
    cache[openrouter_module.OPENROUTER_BASE_URL + "/models"] = _router_catalog()
    monkeypatch.setattr(openrouter_module, "_catalog_cache", cache)


def _install(adapter: str, provider: Any, turn: Turn, model: str, calls: list[Any]) -> None:
    usage = usage_for(1)

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        stream = kwargs.get("stream", False)
        if adapter == "openai_responses":
            return (responses_stream if stream else responses_response)(turn, usage, 1, model)
        builder = chat_stream if stream else chat_completion
        return builder(turn, usage, 1, model, openrouter=adapter == "openrouter")

    mock = AsyncMock(side_effect=create)
    if adapter == "openai_responses":
        provider._client.responses.create = mock
    else:
        provider._client.chat.completions.create = mock


async def run_cell(adapter: str, scenario: Scenario, mode: str) -> dict[str, Any]:
    provider = scenario.factory()
    calls: list[Any] = []
    _install(adapter, provider, scenario.turn, provider.model, calls)
    kwargs = dict(scenario.call)
    messages = scenario.messages()
    tools = list(scenario.tools) or None
    cell: dict[str, Any] = {}
    if mode == "batch":
        response = await provider.complete(
            messages, tools, output_schema=scenario.output_schema, **kwargs
        )
        cell["events"] = []
    else:
        events = [
            e
            async for e in provider.complete_stream(
                messages, tools, output_schema=scenario.output_schema, **kwargs
            )
        ]
        cell["events"] = [
            {
                "type": e.type.value,
                "text": e.text,
                "tool_name": e.tool_name,
                "tool_call_id": e.tool_call_id,
            }
            for e in events
            if e.type != StreamEventType.DONE
        ]
        assert events[-1].type == StreamEventType.DONE
        response = events[-1].raw
    cell["sdk_calls"] = [snap(c) for c in calls]
    cell["response"] = response_snap(response)
    return json.loads(canonical(cell))


def _fixture(adapter: str) -> Path:
    return FIXTURES / f"{adapter}.json"


_regen_store: dict[str, dict[str, Any]] = {}


@pytest.mark.parametrize(
    ("adapter", "scenario", "mode"),
    CELLS,
    ids=[f"{a}-{s.name}-{m}" for a, s, m in CELLS],
)
async def test_captured_request_snapshot(
    adapter: str,
    scenario: Scenario,
    mode: str,
    router_catalog: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if scenario.env_base_url is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", scenario.env_base_url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    actual = await run_cell(adapter, scenario, mode)
    key = f"{scenario.name}__{mode}"
    path = _fixture(adapter)
    if REGEN:
        store = _regen_store.setdefault(adapter, {})
        if path.exists() and not store:
            store.update(json.loads(path.read_text()))
        store[key] = actual
        FIXTURES.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, sort_keys=True, indent=1, ensure_ascii=False) + "\n")
        return
    expected = json.loads(path.read_text())[key]
    assert canonical(actual) == canonical(expected), first_diff(expected, actual)


def test_every_fixture_cell_is_exercised() -> None:
    """No stale or missing cells: the fixture keys are exactly the parametrised cells."""
    for adapter, scenarios in ADAPTERS.items():
        keys = set(json.loads(_fixture(adapter).read_text()))
        want = {f"{s.name}__{m}" for s in scenarios for m in ("batch", "stream")}
        assert keys == want, adapter


def test_snapshots_cover_the_brief() -> None:
    """The pinned shapes the H8 brief names are present in the fixtures."""
    chat = json.loads(_fixture("openai_chat").read_text())
    responses = json.loads(_fixture("openai_responses").read_text())
    assert chat["structured__batch"]["sdk_calls"][0]["response_format"]["type"] == "json_schema"
    fmt = responses["structured__batch"]["sdk_calls"][0]["text"]["format"]
    assert fmt["name"] == "City" and fmt["strict"] is True
    # Both adapters keep OpenAI cache routing fields off compatible endpoints.
    assert "prompt_cache_key" not in chat["cache_custom_base_url__batch"]["sdk_calls"][0]
    custom = responses["cache_custom_base_url_env__batch"]["sdk_calls"][0]
    assert "prompt_cache_key" not in custom
    assert "prompt_cache_retention" not in custom


@pytest.mark.parametrize("mode", ["batch", "stream"])
async def test_responses_custom_endpoint_omits_cache_routing(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = next(s for s in RESPONSES_SCENARIOS if s.name == "cache_custom_base_url_env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    cell = await run_cell("openai_responses", scenario, mode)
    for payload in [*cell["sdk_calls"], cell["response"]["provider_request"]]:
        assert "prompt_cache_key" not in payload
        assert "prompt_cache_retention" not in payload
