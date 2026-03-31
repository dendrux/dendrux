"""Tests for OpenAI Chat Completions provider — conversion logic only, no API calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.types import Message, Role, ToolCall, ToolDef  # noqa: E402


# ---------------------------------------------------------------------------
# Fake OpenAI response objects for testing _normalize_response
# ---------------------------------------------------------------------------
@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    type: str
    function: FakeFunction


@dataclass
class FakeMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage = field(default_factory=FakeMessage)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class FakeChatCompletion:
    choices: list[FakeChoice] = field(default_factory=lambda: [FakeChoice()])
    usage: FakeUsage = field(default_factory=FakeUsage)

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


# ---------------------------------------------------------------------------
# Provider instance (never makes real API calls in these tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o", api_key="test-key")


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------
class TestConvertMessages:
    def test_system_message(self, provider: OpenAIProvider) -> None:
        msgs = [Message(role=Role.SYSTEM, content="You are helpful.")]
        result = provider._convert_messages(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message(self, provider: OpenAIProvider) -> None:
        msgs = [Message(role=Role.USER, content="Hello")]
        result = provider._convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_text_only(self, provider: OpenAIProvider) -> None:
        msgs = [Message(role=Role.ASSISTANT, content="Hi there")]
        result = provider._convert_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_with_tool_calls(self, provider: OpenAIProvider) -> None:
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="call_abc")
        msgs = [Message(role=Role.ASSISTANT, content="Let me add", tool_calls=[tc])]
        result = provider._convert_messages(msgs)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me add"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_abc"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "add"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_tool_result_message(self, provider: OpenAIProvider) -> None:
        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="call_abc")
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content="3", name="add", call_id=tc.id),
        ]
        result = provider._convert_messages(msgs)

        # Assistant + tool result
        assert len(result) == 2
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_abc"
        assert result[1]["content"] == "3"

    def test_tool_result_uses_dendrux_id_when_no_provider_id(self, provider: OpenAIProvider) -> None:
        """Falls back to Dendrux ULID when provider_tool_call_id is None."""
        tc = ToolCall(name="add", params={})  # No provider_tool_call_id
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content="result", name="add", call_id=tc.id),
        ]
        result = provider._convert_messages(msgs)
        assert result[1]["tool_call_id"] == tc.id

    def test_tool_result_missing_call_raises(self, provider: OpenAIProvider) -> None:
        """TOOL message referencing unknown call_id raises ValueError."""
        msgs = [
            Message(role=Role.TOOL, content="3", name="add", call_id="nonexistent"),
        ]
        with pytest.raises(ValueError, match="no matching ToolCall"):
            provider._convert_messages(msgs)

    def test_multi_turn_conversation(self, provider: OpenAIProvider) -> None:
        """Full conversation with system, user, assistant, tool, assistant."""
        tc = ToolCall(name="search", params={"q": "test"}, provider_tool_call_id="call_1")
        msgs = [
            Message(role=Role.SYSTEM, content="You are a helper."),
            Message(role=Role.USER, content="Search for test"),
            Message(role=Role.ASSISTANT, content="Searching", tool_calls=[tc]),
            Message(role=Role.TOOL, content="Found 5 results", name="search", call_id=tc.id),
            Message(role=Role.ASSISTANT, content="I found 5 results."),
        ]
        result = provider._convert_messages(msgs)
        assert len(result) == 5
        assert [m["role"] for m in result] == ["system", "user", "assistant", "tool", "assistant"]

    def test_parallel_tool_calls(self, provider: OpenAIProvider) -> None:
        """Assistant with multiple tool calls in one message."""
        tc1 = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="call_1")
        tc2 = ToolCall(name="mul", params={"x": 2}, provider_tool_call_id="call_2")
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, content="2", name="add", call_id=tc1.id),
            Message(role=Role.TOOL, content="6", name="mul", call_id=tc2.id),
        ]
        result = provider._convert_messages(msgs)

        assert len(result[0]["tool_calls"]) == 2
        assert result[1]["tool_call_id"] == "call_1"
        assert result[2]["tool_call_id"] == "call_2"


# ---------------------------------------------------------------------------
# Tool definition conversion tests
# ---------------------------------------------------------------------------
class TestConvertTools:
    def test_single_tool(self, provider: OpenAIProvider) -> None:
        tools = [
            ToolDef(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            )
        ]
        result = provider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "add"
        assert result[0]["function"]["description"] == "Add two numbers"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_multiple_tools(self, provider: OpenAIProvider) -> None:
        tools = [
            ToolDef(name="a", description="A", parameters={"type": "object", "properties": {}}),
            ToolDef(name="b", description="B", parameters={"type": "object", "properties": {}}),
        ]
        result = provider._convert_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


# ---------------------------------------------------------------------------
# Response normalization tests
# ---------------------------------------------------------------------------
class TestNormalizeResponse:
    def test_text_response(self, provider: OpenAIProvider) -> None:
        response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="Hello world"))],
        )
        result = provider._normalize_response(response)
        assert result.text == "Hello world"
        assert result.tool_calls is None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_tool_call_response(self, provider: OpenAIProvider) -> None:
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content=None,
                        tool_calls=[
                            FakeToolCall(
                                id="call_abc123",
                                type="function",
                                function=FakeFunction(
                                    name="add",
                                    arguments='{"a": 1, "b": 2}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        result = provider._normalize_response(response)
        assert result.text is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "add"
        assert result.tool_calls[0].params == {"a": 1, "b": 2}
        assert result.tool_calls[0].provider_tool_call_id == "call_abc123"

    def test_parallel_tool_calls_response(self, provider: OpenAIProvider) -> None:
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call_1",
                                type="function",
                                function=FakeFunction(name="add", arguments='{"a": 1}'),
                            ),
                            FakeToolCall(
                                id="call_2",
                                type="function",
                                function=FakeFunction(name="mul", arguments='{"x": 2}'),
                            ),
                        ],
                    ),
                )
            ],
        )
        result = provider._normalize_response(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "add"
        assert result.tool_calls[1].name == "mul"
        assert result.tool_calls[0].provider_tool_call_id == "call_1"
        assert result.tool_calls[1].provider_tool_call_id == "call_2"

    def test_text_with_tool_calls(self, provider: OpenAIProvider) -> None:
        """Response with both text and tool calls."""
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content="Let me calculate",
                        tool_calls=[
                            FakeToolCall(
                                id="call_1",
                                type="function",
                                function=FakeFunction(name="add", arguments='{"a": 1}'),
                            ),
                        ],
                    ),
                )
            ],
        )
        result = provider._normalize_response(response)
        assert result.text == "Let me calculate"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_no_usage(self, provider: OpenAIProvider) -> None:
        """Some compatible APIs may not return usage."""
        response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="hi"))],
        )
        response.usage = None  # type: ignore[assignment]
        result = provider._normalize_response(response)
        assert result.usage.total_tokens == 0

    def test_empty_arguments(self, provider: OpenAIProvider) -> None:
        """Tool call with empty arguments string."""
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call_1",
                                type="function",
                                function=FakeFunction(name="noop", arguments=""),
                            ),
                        ],
                    ),
                )
            ],
        )
        result = provider._normalize_response(response)
        assert result.tool_calls is not None
        assert result.tool_calls[0].params == {}


# ---------------------------------------------------------------------------
# Provider construction tests
# ---------------------------------------------------------------------------
class TestProviderConstruction:
    def test_model_property(self, provider: OpenAIProvider) -> None:
        assert provider.model == "gpt-4o"

    def test_repr_default_base_url(self, provider: OpenAIProvider) -> None:
        assert repr(provider) == "OpenAIProvider(model='gpt-4o')"

    def test_repr_custom_base_url(self) -> None:
        p = OpenAIProvider(model="llama3", api_key="test", base_url="http://localhost:8000/v1")
        assert "base_url=" in repr(p)
        assert "localhost:8000" in repr(p)

    def test_capabilities(self) -> None:
        assert OpenAIProvider.capabilities.supports_native_tools is True
        assert OpenAIProvider.capabilities.supports_tool_call_ids is True
        assert OpenAIProvider.capabilities.supports_parallel_tool_calls is True


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_malformed_json_arguments_raises_helpful_error(self, provider: OpenAIProvider) -> None:
        """Compatible APIs may return truncated JSON — error should identify the tool."""
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call_1",
                                type="function",
                                function=FakeFunction(name="broken", arguments='{"a": 1, "b":'),
                            ),
                        ],
                    ),
                )
            ],
        )
        with pytest.raises(ValueError, match="broken.*invalid JSON"):
            provider._normalize_response(response)

    def test_duplicate_call_id_raises(self, provider: OpenAIProvider) -> None:
        """Duplicate Dendrux call_id should raise, not silently overwrite."""
        tc = ToolCall(name="add", params={}, provider_tool_call_id="call_1")
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),  # Same call_id
        ]
        with pytest.raises(ValueError, match="Duplicate Dendrux call_id"):
            provider._convert_messages(msgs)

    def test_assistant_tool_calls_no_text(self, provider: OpenAIProvider) -> None:
        """Empty content with tool calls should omit content key."""
        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="call_1")
        msgs = [Message(role=Role.ASSISTANT, content="", tool_calls=[tc])]
        result = provider._convert_messages(msgs)
        # Empty string is falsy — content should not be in the dict
        assert "content" not in result[0]

    def test_no_usage_returns_zero(self, provider: OpenAIProvider) -> None:
        """Compatible APIs may omit usage — should default to zeros."""
        response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="ok"))],
        )
        response.usage = None  # type: ignore[assignment]
        result = provider._normalize_response(response)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0


class TestComplete:
    """Test complete() method with mocked API client."""

    async def test_complete_text_response(self, provider: OpenAIProvider) -> None:
        """complete() normalizes a text response and attaches evidence payloads."""
        from unittest.mock import AsyncMock

        fake_response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="Hello!"))],
        )
        provider._client.chat.completions.create = AsyncMock(return_value=fake_response)

        msgs = [Message(role=Role.USER, content="Hi")]
        result = await provider.complete(msgs)

        assert result.text == "Hello!"
        assert result.tool_calls is None
        assert result.provider_request is not None
        assert result.provider_response is not None

    async def test_complete_tool_call_response(self, provider: OpenAIProvider) -> None:
        """complete() normalizes tool call responses."""
        from unittest.mock import AsyncMock

        fake_response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call_1",
                                type="function",
                                function=FakeFunction(name="add", arguments='{"a": 1}'),
                            ),
                        ],
                    ),
                )
            ],
        )
        provider._client.chat.completions.create = AsyncMock(return_value=fake_response)

        tools = [ToolDef(name="add", description="Add", parameters={"type": "object", "properties": {}})]
        msgs = [Message(role=Role.USER, content="add 1")]
        result = await provider.complete(msgs, tools=tools)

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "add"

    async def test_complete_timeout_raises_clean_error(self, provider: OpenAIProvider) -> None:
        """APITimeoutError is caught and re-raised as TimeoutError with hint."""
        from unittest.mock import AsyncMock

        import openai

        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=None),  # type: ignore[arg-type]
        )

        msgs = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(TimeoutError, match="timed out.*timeout=300"):
            await provider.complete(msgs)

    async def test_complete_forwards_kwargs(self, provider: OpenAIProvider) -> None:
        """Supported kwargs are forwarded to the API call."""
        from unittest.mock import AsyncMock

        fake_response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="ok"))],
        )
        mock_create = AsyncMock(return_value=fake_response)
        provider._client.chat.completions.create = mock_create

        msgs = [Message(role=Role.USER, content="Hi")]
        await provider.complete(msgs, temperature=0.5, seed=42)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["seed"] == 42

    async def test_complete_ignores_unsupported_kwargs(self, provider: OpenAIProvider) -> None:
        """Unknown kwargs are silently dropped."""
        from unittest.mock import AsyncMock

        fake_response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="ok"))],
        )
        mock_create = AsyncMock(return_value=fake_response)
        provider._client.chat.completions.create = mock_create

        msgs = [Message(role=Role.USER, content="Hi")]
        await provider.complete(msgs, unknown_param="ignored")

        call_kwargs = mock_create.call_args[1]
        assert "unknown_param" not in call_kwargs

    async def test_complete_default_max_tokens(self, provider: OpenAIProvider) -> None:
        """Default max_tokens is 16_000 when not specified."""
        from unittest.mock import AsyncMock

        fake_response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeMessage(content="ok"))],
        )
        mock_create = AsyncMock(return_value=fake_response)
        provider._client.chat.completions.create = mock_create

        msgs = [Message(role=Role.USER, content="Hi")]
        await provider.complete(msgs)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 16_000
