"""Tests for AnthropicProvider — conversion logic only, no API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    TextBlock,
    ToolUseBlock,
    Usage,
)

from dendrite.llm.anthropic import AnthropicProvider
from dendrite.types import (
    Message,
    Role,
    ToolCall,
    ToolDef,
)


@pytest.fixture
def provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-6")


def _make_anthropic_response(
    content: list,
    input_tokens: int = 100,
    output_tokens: int = 50,
    stop_reason: str = "end_turn",
) -> AnthropicMessage:
    """Build a fake Anthropic Message response."""
    return AnthropicMessage(
        id="msg_test123",
        content=content,
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ------------------------------------------------------------------
# Capabilities — reflect implemented behavior, not API potential
# ------------------------------------------------------------------


class TestCapabilities:
    def test_declares_native_tools(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_native_tools is True

    def test_declares_tool_call_ids(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_tool_call_ids is True

    def test_streaming_not_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_streaming is False

    def test_streaming_tool_deltas_not_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_streaming_tool_deltas is False

    def test_thinking_not_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_thinking is False

    def test_multimodal_not_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_multimodal is False

    def test_declares_system_prompt(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_system_prompt is True

    def test_declares_parallel_tool_calls(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_parallel_tool_calls is True


# ------------------------------------------------------------------
# Outbound: _convert_messages
# ------------------------------------------------------------------


class TestConvertMessages:
    def test_system_messages_extracted(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hi"),
        ]
        system, api_msgs = provider._convert_messages(messages)
        assert system == "You are helpful"
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_multiple_system_messages_joined(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(role=Role.SYSTEM, content="Be helpful"),
            Message(role=Role.SYSTEM, content="Be concise"),
            Message(role=Role.USER, content="Hi"),
        ]
        system, _ = provider._convert_messages(messages)
        assert system == "Be helpful\n\nBe concise"

    def test_user_message_passthrough(self, provider: AnthropicProvider) -> None:
        messages = [Message(role=Role.USER, content="Hello")]
        _, api_msgs = provider._convert_messages(messages)
        assert api_msgs == [{"role": "user", "content": "Hello"}]

    def test_assistant_text_only(self, provider: AnthropicProvider) -> None:
        messages = [Message(role=Role.ASSISTANT, content="Hi there")]
        _, api_msgs = provider._convert_messages(messages)
        assert api_msgs == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_with_tool_calls(self, provider: AnthropicProvider) -> None:
        tc = ToolCall(
            name="search",
            params={"q": "test"},
            provider_tool_call_id="toolu_abc123",
        )
        messages = [Message(role=Role.ASSISTANT, content="Let me search", tool_calls=[tc])]
        _, api_msgs = provider._convert_messages(messages)

        assert len(api_msgs) == 1
        content = api_msgs[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Let me search"}
        assert content[1] == {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "search",
            "input": {"q": "test"},
        }

    def test_assistant_tool_calls_no_text(self, provider: AnthropicProvider) -> None:
        tc = ToolCall(
            name="add",
            params={"a": 1},
            provider_tool_call_id="toolu_xyz",
        )
        messages = [Message(role=Role.ASSISTANT, content="", tool_calls=[tc])]
        _, api_msgs = provider._convert_messages(messages)

        content = api_msgs[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_tool_message_becomes_user_tool_result(self, provider: AnthropicProvider) -> None:
        tc = ToolCall(
            name="add",
            params={"a": 1},
            provider_tool_call_id="toolu_abc",
        )
        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(
                role=Role.TOOL,
                content='{"result": 3}',
                name="add",
                call_id=tc.id,
            ),
        ]
        _, api_msgs = provider._convert_messages(messages)

        tool_msg = api_msgs[1]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        block = tool_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_abc"
        assert block["content"] == '{"result": 3}'

    def test_consecutive_tool_messages_merge(self, provider: AnthropicProvider) -> None:
        """Parallel tool results should merge into a single user message."""
        tc1 = ToolCall(name="add", params={}, provider_tool_call_id="toolu_1")
        tc2 = ToolCall(name="mul", params={}, provider_tool_call_id="toolu_2")
        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2]),
            Message(role=Role.TOOL, content='{"r": 3}', name="add", call_id=tc1.id),
            Message(role=Role.TOOL, content='{"r": 6}', name="mul", call_id=tc2.id),
        ]
        _, api_msgs = provider._convert_messages(messages)

        assert len(api_msgs) == 2  # assistant + single merged user
        tool_results = api_msgs[1]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "toolu_1"
        assert tool_results[1]["tool_use_id"] == "toolu_2"

    def test_tool_without_provider_id_becomes_plain_user(self, provider: AnthropicProvider) -> None:
        """TOOL message without native correlation falls back to plain user message."""
        tc = ToolCall(name="add", params={})  # No provider_tool_call_id
        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content='{"r": 3}', name="add", call_id=tc.id),
        ]
        _, api_msgs = provider._convert_messages(messages)

        # Falls back to plain user message (no tool_result block)
        assert api_msgs[1]["role"] == "user"
        assert api_msgs[1]["content"] == '{"r": 3}'

    def test_tool_error_message(self, provider: AnthropicProvider) -> None:
        tc = ToolCall(name="div", params={}, provider_tool_call_id="toolu_err")
        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(
                role=Role.TOOL,
                content="Division by zero",
                name="div",
                call_id=tc.id,
                meta={"is_error": True},
            ),
        ]
        _, api_msgs = provider._convert_messages(messages)

        block = api_msgs[1]["content"][0]
        assert block["is_error"] is True

    def test_tool_missing_call_id_raises(self, provider: AnthropicProvider) -> None:
        """TOOL message referencing unknown call_id must fail deterministically."""
        tc = ToolCall(name="add", params={}, provider_tool_call_id="toolu_1")
        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(
                role=Role.TOOL,
                content="result",
                name="add",
                call_id="nonexistent_id",
            ),
        ]
        with pytest.raises(ValueError, match="no matching ToolCall found"):
            provider._convert_messages(messages)

    def test_duplicate_call_id_raises(self, provider: AnthropicProvider) -> None:
        """Duplicate Dendrite call_ids in the conversation must fail."""
        shared_id = "01SHARED"
        # Create two ToolCalls with the same id (bypass frozen by using object.__setattr__)
        tc1 = ToolCall.__new__(ToolCall)
        object.__setattr__(tc1, "name", "add")
        object.__setattr__(tc1, "params", {})
        object.__setattr__(tc1, "id", shared_id)
        object.__setattr__(tc1, "provider_tool_call_id", "toolu_1")

        tc2 = ToolCall.__new__(ToolCall)
        object.__setattr__(tc2, "name", "mul")
        object.__setattr__(tc2, "params", {})
        object.__setattr__(tc2, "id", shared_id)
        object.__setattr__(tc2, "provider_tool_call_id", "toolu_2")

        messages = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc1]),
            Message(role=Role.TOOL, content="3", name="add", call_id=shared_id),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc2]),
        ]
        with pytest.raises(ValueError, match="Duplicate Dendrite call_id"):
            provider._convert_messages(messages)


# ------------------------------------------------------------------
# Outbound: _convert_tools
# ------------------------------------------------------------------


class TestConvertTools:
    def test_converts_tool_defs(self, provider: AnthropicProvider) -> None:
        tools = [
            ToolDef(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        result = provider._convert_tools(tools)

        assert len(result) == 1
        assert result[0] == {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }

    def test_converts_multiple_tools(self, provider: AnthropicProvider) -> None:
        tools = [
            ToolDef(name="a", description="Tool A", parameters={}),
            ToolDef(name="b", description="Tool B", parameters={}),
        ]
        result = provider._convert_tools(tools)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"


# ------------------------------------------------------------------
# Inbound: _normalize_response
# ------------------------------------------------------------------


class TestNormalizeResponse:
    def test_text_response(self, provider: AnthropicProvider) -> None:
        response = _make_anthropic_response(
            content=[TextBlock(type="text", text="Hello world", citations=None)]
        )
        result = provider._normalize_response(response)

        assert result.text == "Hello world"
        assert result.tool_calls is None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.raw is response

    def test_tool_use_response(self, provider: AnthropicProvider) -> None:
        response = _make_anthropic_response(
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_abc123",
                    name="search",
                    input={"q": "test"},
                )
            ],
            stop_reason="tool_use",
        )
        result = provider._normalize_response(response)

        assert result.text is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "search"
        assert tc.params == {"q": "test"}
        assert tc.provider_tool_call_id == "toolu_abc123"
        assert tc.id is not None  # Dendrite ULID auto-generated

    def test_mixed_text_and_tool_use(self, provider: AnthropicProvider) -> None:
        response = _make_anthropic_response(
            content=[
                TextBlock(type="text", text="Let me search", citations=None),
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_xyz",
                    name="search",
                    input={"q": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        result = provider._normalize_response(response)

        assert result.text == "Let me search"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_multiple_tool_calls(self, provider: AnthropicProvider) -> None:
        response = _make_anthropic_response(
            content=[
                ToolUseBlock(type="tool_use", id="toolu_1", name="add", input={"a": 1}),
                ToolUseBlock(type="tool_use", id="toolu_2", name="mul", input={"b": 2}),
            ],
            stop_reason="tool_use",
        )
        result = provider._normalize_response(response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].provider_tool_call_id == "toolu_1"
        assert result.tool_calls[1].provider_tool_call_id == "toolu_2"

    def test_multiple_text_blocks_concatenated_exactly(self, provider: AnthropicProvider) -> None:
        """Text blocks are joined with no separator — no injected content."""
        response = _make_anthropic_response(
            content=[
                TextBlock(type="text", text="First part", citations=None),
                TextBlock(type="text", text="Second part", citations=None),
            ]
        )
        result = provider._normalize_response(response)
        assert result.text == "First partSecond part"

    def test_empty_input_dict_becomes_empty_params(self, provider: AnthropicProvider) -> None:
        response = _make_anthropic_response(
            content=[
                ToolUseBlock(type="tool_use", id="toolu_1", name="noop", input={}),
            ],
            stop_reason="tool_use",
        )
        result = provider._normalize_response(response)
        assert result.tool_calls[0].params == {}


# ------------------------------------------------------------------
# End-to-end: complete() with mocked API
# ------------------------------------------------------------------


class TestComplete:
    async def test_text_completion(self, provider: AnthropicProvider) -> None:
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="The answer is 42", citations=None)]
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await provider.complete([Message(role=Role.USER, content="What is 6*7?")])

        assert result.text == "The answer is 42"
        assert result.tool_calls is None

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 16_000
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0] == {
            "role": "user",
            "content": "What is 6*7?",
        }

    async def test_tool_call_completion(self, provider: AnthropicProvider) -> None:
        mock_response = _make_anthropic_response(
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_calc",
                    name="add",
                    input={"a": 1, "b": 2},
                )
            ],
            stop_reason="tool_use",
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            tools = [ToolDef(name="add", description="Add numbers", parameters={})]
            result = await provider.complete(
                [Message(role=Role.USER, content="add 1+2")],
                tools=tools,
            )

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "add"
        assert result.tool_calls[0].provider_tool_call_id == "toolu_calc"

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tools"] == [
            {"name": "add", "description": "Add numbers", "input_schema": {}}
        ]

    async def test_system_message_extracted(self, provider: AnthropicProvider) -> None:
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="Hi", citations=None)]
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await provider.complete(
                [
                    Message(role=Role.SYSTEM, content="Be helpful"),
                    Message(role=Role.USER, content="Hi"),
                ]
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"
        assert len(call_kwargs["messages"]) == 1

    async def test_kwargs_override_constructor_defaults(self, provider: AnthropicProvider) -> None:
        """complete() kwargs override constructor defaults."""
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="Hi", citations=None)]
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await provider.complete(
                [Message(role=Role.USER, content="Hi")],
                temperature=0.5,
                max_tokens=1000,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000

    async def test_unsupported_kwargs_ignored(self, provider: AnthropicProvider) -> None:
        """Unknown kwargs are silently ignored, not forwarded to API."""
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="Hi", citations=None)]
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await provider.complete(
                [Message(role=Role.USER, content="Hi")],
                some_unknown_param="value",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert "some_unknown_param" not in call_kwargs

    async def test_model_override(self, provider: AnthropicProvider) -> None:
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="Hi", citations=None)]
        )

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            await provider.complete(
                [Message(role=Role.USER, content="Hi")],
                model="claude-opus-4-6",
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"

    async def test_full_tool_round_trip(self, provider: AnthropicProvider) -> None:
        """Complete conversation: user → assistant(tool_call) → tool_result → answer."""
        mock_response = _make_anthropic_response(
            content=[TextBlock(type="text", text="3", citations=None)]
        )

        tc = ToolCall(
            name="add",
            params={"a": 1, "b": 2},
            provider_tool_call_id="toolu_round",
        )
        messages = [
            Message(role=Role.SYSTEM, content="You are a calculator"),
            Message(role=Role.USER, content="add 1+2"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(
                role=Role.TOOL,
                content='{"result": 3}',
                name="add",
                call_id=tc.id,
            ),
        ]

        with patch.object(
            provider._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response
            result = await provider.complete(messages)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are a calculator"
        api_msgs = call_kwargs["messages"]
        assert len(api_msgs) == 3  # user, assistant, user(tool_result)

        # Assistant message has tool_use block
        assert api_msgs[1]["content"][0]["type"] == "tool_use"
        assert api_msgs[1]["content"][0]["id"] == "toolu_round"

        # Tool result in user message
        assert api_msgs[2]["content"][0]["type"] == "tool_result"
        assert api_msgs[2]["content"][0]["tool_use_id"] == "toolu_round"

        assert result.text == "3"
