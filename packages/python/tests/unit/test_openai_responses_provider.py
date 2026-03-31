"""Tests for OpenAI Responses API provider — conversion logic only, no API calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm.openai_responses import OpenAIResponsesProvider  # noqa: E402
from dendrux.types import Message, Role, ToolCall, ToolDef  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Responses API objects for testing _normalize_response
# ---------------------------------------------------------------------------
@dataclass
class FakeFunctionCall:
    type: str = "function_call"
    call_id: str = "call_1"
    name: str = "add"
    arguments: str = '{"a": 1}'


@dataclass
class FakeOutputText:
    type: str = "output_text"
    text: str = "Hello"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150


@dataclass
class FakeResponse:
    output: list[Any] = field(default_factory=list)
    output_text: str | None = None
    usage: FakeUsage = field(default_factory=FakeUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


# ---------------------------------------------------------------------------
# Provider fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-4o", api_key="test-key")


@pytest.fixture
def provider_with_search() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(
        model="gpt-4o", api_key="test-key", builtin_tools=["web_search_preview"],
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_model_property(self, provider: OpenAIResponsesProvider) -> None:
        assert provider.model == "gpt-4o"

    def test_repr_no_builtins(self, provider: OpenAIResponsesProvider) -> None:
        assert repr(provider) == "OpenAIResponsesProvider(model='gpt-4o')"

    def test_repr_with_builtins(self, provider_with_search: OpenAIResponsesProvider) -> None:
        r = repr(provider_with_search)
        assert "web_search_preview" in r
        assert "OpenAIResponsesProvider" in r

    def test_invalid_builtin_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown built-in tool 'invalid'"):
            OpenAIResponsesProvider(model="gpt-4o", api_key="test", builtin_tools=["invalid"])

    def test_capabilities(self) -> None:
        assert OpenAIResponsesProvider.capabilities.supports_native_tools is True
        assert OpenAIResponsesProvider.capabilities.supports_tool_call_ids is True


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------
class TestConvertMessages:
    def test_system_becomes_instructions(self, provider: OpenAIResponsesProvider) -> None:
        msgs = [Message(role=Role.SYSTEM, content="You are helpful.")]
        instructions, items = provider._convert_messages(msgs)
        assert instructions == "You are helpful."
        assert items == []

    def test_user_message(self, provider: OpenAIResponsesProvider) -> None:
        msgs = [Message(role=Role.USER, content="Hello")]
        instructions, items = provider._convert_messages(msgs)
        assert instructions == ""
        assert items == [{"role": "user", "content": "Hello"}]

    def test_assistant_text_only(self, provider: OpenAIResponsesProvider) -> None:
        msgs = [Message(role=Role.ASSISTANT, content="Hi")]
        _, items = provider._convert_messages(msgs)
        assert items == [{"role": "assistant", "content": "Hi"}]

    def test_assistant_with_tool_calls(self, provider: OpenAIResponsesProvider) -> None:
        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="call_abc")
        msgs = [Message(role=Role.ASSISTANT, content="Let me add", tool_calls=[tc])]
        _, items = provider._convert_messages(msgs)

        # Text content + function_call item
        assert len(items) == 2
        assert items[0] == {"role": "assistant", "content": "Let me add"}
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_abc"
        assert items[1]["name"] == "add"
        assert json.loads(items[1]["arguments"]) == {"a": 1}

    def test_assistant_tool_calls_no_text(self, provider: OpenAIResponsesProvider) -> None:
        tc = ToolCall(name="add", params={}, provider_tool_call_id="call_1")
        msgs = [Message(role=Role.ASSISTANT, content="", tool_calls=[tc])]
        _, items = provider._convert_messages(msgs)
        # No text item, only function_call
        assert len(items) == 1
        assert items[0]["type"] == "function_call"

    def test_tool_result(self, provider: OpenAIResponsesProvider) -> None:
        tc = ToolCall(name="add", params={}, provider_tool_call_id="call_abc")
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content="3", name="add", call_id=tc.id),
        ]
        _, items = provider._convert_messages(msgs)

        assert len(items) == 2
        assert items[1]["type"] == "function_call_output"
        assert items[1]["call_id"] == "call_abc"
        assert items[1]["output"] == "3"

    def test_duplicate_call_id_raises(self, provider: OpenAIResponsesProvider) -> None:
        tc = ToolCall(name="add", params={})
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
        ]
        with pytest.raises(ValueError, match="Duplicate Dendrux call_id"):
            provider._convert_messages(msgs)

    def test_tool_result_missing_call_raises(self, provider: OpenAIResponsesProvider) -> None:
        msgs = [Message(role=Role.TOOL, content="3", name="add", call_id="nonexistent")]
        with pytest.raises(ValueError, match="no matching ToolCall"):
            provider._convert_messages(msgs)

    def test_multi_turn(self, provider: OpenAIResponsesProvider) -> None:
        tc = ToolCall(name="search", params={"q": "test"}, provider_tool_call_id="call_1")
        msgs = [
            Message(role=Role.SYSTEM, content="Be helpful."),
            Message(role=Role.USER, content="Search"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content="results", name="search", call_id=tc.id),
            Message(role=Role.ASSISTANT, content="Found it."),
        ]
        instructions, items = provider._convert_messages(msgs)
        assert instructions == "Be helpful."
        assert len(items) == 4  # user + function_call + function_call_output + assistant


# ---------------------------------------------------------------------------
# Tool building tests
# ---------------------------------------------------------------------------
class TestBuildTools:
    def test_dendrux_tools_only(self, provider: OpenAIResponsesProvider) -> None:
        tools = [
            ToolDef(name="add", description="Add", parameters={"type": "object", "properties": {}}),
        ]
        result = provider._build_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "add"
        # Flatter format than Chat Completions — no nested "function" key
        assert "function" not in result[0]

    def test_builtin_tools_only(self, provider_with_search: OpenAIResponsesProvider) -> None:
        result = provider_with_search._build_tools(None)
        assert len(result) == 1
        assert result[0] == {"type": "web_search_preview"}

    def test_mixed_tools(self, provider_with_search: OpenAIResponsesProvider) -> None:
        tools = [
            ToolDef(name="custom", description="Custom", parameters={"type": "object", "properties": {}}),
        ]
        result = provider_with_search._build_tools(tools)
        assert len(result) == 2
        assert result[0] == {"type": "web_search_preview"}
        assert result[1]["type"] == "function"
        assert result[1]["name"] == "custom"


# ---------------------------------------------------------------------------
# Response normalization tests
# ---------------------------------------------------------------------------
class TestNormalizeResponse:
    def test_text_response(self, provider: OpenAIResponsesProvider) -> None:
        response = FakeResponse(
            output=[FakeOutputText()],
            output_text="Hello world",
        )
        result = provider._normalize_response(response)
        assert result.text == "Hello world"
        assert result.tool_calls is None
        assert result.usage.input_tokens == 100

    def test_function_call_response(self, provider: OpenAIResponsesProvider) -> None:
        response = FakeResponse(
            output=[FakeFunctionCall(call_id="call_1", name="add", arguments='{"a": 1}')],
            output_text=None,
        )
        result = provider._normalize_response(response)
        assert result.text is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "add"
        assert result.tool_calls[0].params == {"a": 1}
        assert result.tool_calls[0].provider_tool_call_id == "call_1"

    def test_malformed_arguments_raises(self, provider: OpenAIResponsesProvider) -> None:
        response = FakeResponse(
            output=[FakeFunctionCall(arguments='{"bad":')],
        )
        with pytest.raises(ValueError, match="invalid JSON"):
            provider._normalize_response(response)

    def test_empty_arguments(self, provider: OpenAIResponsesProvider) -> None:
        response = FakeResponse(
            output=[FakeFunctionCall(arguments="")],
        )
        result = provider._normalize_response(response)
        assert result.tool_calls is not None
        assert result.tool_calls[0].params == {}

    def test_no_usage(self, provider: OpenAIResponsesProvider) -> None:
        response = FakeResponse(output=[], output_text="hi")
        response.usage = None  # type: ignore[assignment]
        result = provider._normalize_response(response)
        assert result.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# complete() tests with mocked client
# ---------------------------------------------------------------------------
class TestComplete:
    async def test_text_response(self, provider: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(output=[], output_text="Hello!")
        provider._client.responses.create = AsyncMock(return_value=fake)

        result = await provider.complete([Message(role=Role.USER, content="Hi")])
        assert result.text == "Hello!"
        assert result.provider_request is not None

    async def test_tool_call_response(self, provider: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(
            output=[FakeFunctionCall(call_id="call_1", name="add", arguments='{"a":1}')],
        )
        provider._client.responses.create = AsyncMock(return_value=fake)

        tools = [ToolDef(name="add", description="Add", parameters={"type": "object", "properties": {}})]
        result = await provider.complete(
            [Message(role=Role.USER, content="add")], tools=tools,
        )
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "add"

    async def test_timeout_raises_clean_error(self, provider: OpenAIResponsesProvider) -> None:
        provider._client.responses.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=None),  # type: ignore[arg-type]
        )
        with pytest.raises(TimeoutError, match="timed out.*timeout=300"):
            await provider.complete([Message(role=Role.USER, content="Hi")])

    async def test_builtin_tools_included(self, provider_with_search: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        provider_with_search._client.responses.create = mock_create

        await provider_with_search.complete([Message(role=Role.USER, content="search")])

        call_kwargs = mock_create.call_args[1]
        tools = call_kwargs["tools"]
        assert any(t["type"] == "web_search_preview" for t in tools)

    async def test_default_max_output_tokens(self, provider: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        provider._client.responses.create = mock_create

        await provider.complete([Message(role=Role.USER, content="Hi")])

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_output_tokens"] == 16_000

    async def test_reasoning_effort_forwarded(self) -> None:
        p = OpenAIResponsesProvider(
            model="gpt-5", api_key="test", reasoning_effort="high",
        )
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        p._client.responses.create = mock_create

        await p.complete([Message(role=Role.USER, content="think hard")])

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high"}

    async def test_instructions_from_system_message(self, provider: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        provider._client.responses.create = mock_create

        msgs = [
            Message(role=Role.SYSTEM, content="Be helpful."),
            Message(role=Role.USER, content="Hi"),
        ]
        await provider.complete(msgs)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["instructions"] == "Be helpful."

    async def test_unsupported_kwargs_dropped(self, provider: OpenAIResponsesProvider) -> None:
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        provider._client.responses.create = mock_create

        await provider.complete(
            [Message(role=Role.USER, content="Hi")], unknown_param="ignored",
        )
        call_kwargs = mock_create.call_args[1]
        assert "unknown_param" not in call_kwargs

    async def test_reasoning_effort_per_call_override(self) -> None:
        """Per-call reasoning_effort overrides constructor default."""
        p = OpenAIResponsesProvider(
            model="gpt-5", api_key="test", reasoning_effort="high",
        )
        fake = FakeResponse(output=[], output_text="ok")
        mock_create = AsyncMock(return_value=fake)
        p._client.responses.create = mock_create

        await p.complete(
            [Message(role=Role.USER, content="quick")], reasoning_effort="low",
        )
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "low"}


class TestEdgeCases:
    def test_null_arguments_produce_empty_dict(self, provider: OpenAIResponsesProvider) -> None:
        """json.loads("null") should not produce params=None."""
        response = FakeResponse(
            output=[FakeFunctionCall(arguments="null")],
        )
        result = provider._normalize_response(response)
        assert result.tool_calls is not None
        assert result.tool_calls[0].params == {}

    def test_parallel_tool_calls(self, provider: OpenAIResponsesProvider) -> None:
        """Multiple function_call items in one assistant turn."""
        tc1 = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="call_1")
        tc2 = ToolCall(name="mul", params={"x": 2}, provider_tool_call_id="call_2")
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, content="2", name="add", call_id=tc1.id),
            Message(role=Role.TOOL, content="6", name="mul", call_id=tc2.id),
        ]
        _, items = provider._convert_messages(msgs)

        # 2 function_call items + 2 function_call_output items
        func_calls = [i for i in items if i.get("type") == "function_call"]
        func_outputs = [i for i in items if i.get("type") == "function_call_output"]
        assert len(func_calls) == 2
        assert len(func_outputs) == 2

    def test_multiple_system_messages_joined(self, provider: OpenAIResponsesProvider) -> None:
        """Multiple SYSTEM messages are joined into one instructions string."""
        msgs = [
            Message(role=Role.SYSTEM, content="Rule 1."),
            Message(role=Role.SYSTEM, content="Rule 2."),
            Message(role=Role.USER, content="Hi"),
        ]
        instructions, items = provider._convert_messages(msgs)
        assert instructions == "Rule 1.\n\nRule 2."
        assert len(items) == 1  # Only the user message

    def test_code_interpreter_rejected(self) -> None:
        """code_interpreter requires config — not supported via builtin_tools."""
        with pytest.raises(ValueError, match="Unknown built-in tool"):
            OpenAIResponsesProvider(
                model="gpt-4o", api_key="test", builtin_tools=["code_interpreter"],
            )
