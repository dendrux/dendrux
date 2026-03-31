"""Tests for AnthropicProvider — conversion logic only, no API calls."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    TextBlock,
    ToolUseBlock,
    Usage,
)

from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import (
    Message,
    Role,
    StreamEventType,
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

    def test_streaming_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_streaming is True

    def test_streaming_tool_deltas_implemented(self, provider: AnthropicProvider) -> None:
        assert provider.capabilities.supports_streaming_tool_deltas is True

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
        """Duplicate Dendrux call_ids in the conversation must fail."""
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
        with pytest.raises(ValueError, match="Duplicate Dendrux call_id"):
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
        assert tc.id is not None  # Dendrux ULID auto-generated

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


# ------------------------------------------------------------------
# Provider hardening (F-07, F-08, F-09)
# ------------------------------------------------------------------


class TestProviderHardening:
    def test_client_has_timeout(self) -> None:
        """F-07: Client should be constructed with a timeout."""
        provider = AnthropicProvider(api_key="sk-test", model="test")
        assert provider._client.timeout.read == 120.0
        assert provider._client.timeout.connect == 10.0

    def test_custom_timeout(self) -> None:
        """F-07: Custom timeout is forwarded to the client."""
        provider = AnthropicProvider(api_key="sk-test", model="test", timeout=60.0)
        assert provider._client.timeout.read == 60.0

    def test_client_has_retries(self) -> None:
        """F-08: Client should be constructed with retry support."""
        provider = AnthropicProvider(api_key="sk-test", model="test")
        assert provider._client.max_retries == 3

    def test_custom_retries(self) -> None:
        """F-08: Custom max_retries is forwarded to the client."""
        provider = AnthropicProvider(api_key="sk-test", model="test", max_retries=5)
        assert provider._client.max_retries == 5

    def test_assert_replaced_with_valueerror(self, provider: AnthropicProvider) -> None:
        """F-09: TOOL message with None call_id raises ValueError, not AssertionError."""
        # Manually craft a message that bypasses __post_init__ to simulate
        # a broken invariant (e.g. from deserialization or future code change)
        msg = Message.__new__(Message)
        object.__setattr__(msg, "role", Role.TOOL)
        object.__setattr__(msg, "content", "result")
        object.__setattr__(msg, "name", "add")
        object.__setattr__(msg, "call_id", None)
        object.__setattr__(msg, "tool_calls", None)
        object.__setattr__(msg, "meta", {})

        with pytest.raises(ValueError, match="TOOL message missing call_id"):
            provider._convert_messages([msg])


# ------------------------------------------------------------------
# Streaming — complete_stream()
# ------------------------------------------------------------------


# Lightweight fakes for Anthropic streaming events.
# These mirror the shape that anthropic SDK yields from messages.stream().


@dataclass
class _FakeContentBlock:
    type: str
    name: str | None = None
    id: str | None = None
    text: str | None = None


@dataclass
class _FakeDelta:
    type: str
    text: str | None = None
    partial_json: str | None = None


@dataclass
class _FakeStreamEvent:
    type: str
    content_block: _FakeContentBlock | None = None
    delta: _FakeDelta | None = None
    index: int = 0


class _FakeStream:
    """Async iterator that replays a list of events, mimicking AsyncMessageStream."""

    def __init__(self, events: list[_FakeStreamEvent], final_message: AnthropicMessage):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event

    async def get_final_message(self) -> AnthropicMessage:
        return self._final_message


class TestCompleteStream:
    """Tests for AnthropicProvider.complete_stream()."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        return AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-6")

    @staticmethod
    def _text_events(chunks: list[str]) -> list[_FakeStreamEvent]:
        """Build stream events for a text-only response."""
        events = [
            _FakeStreamEvent(
                type="content_block_start",
                content_block=_FakeContentBlock(type="text"),
            ),
        ]
        for chunk in chunks:
            events.append(
                _FakeStreamEvent(
                    type="content_block_delta",
                    delta=_FakeDelta(type="text_delta", text=chunk),
                )
            )
        events.append(_FakeStreamEvent(type="content_block_stop"))
        return events

    @staticmethod
    def _tool_events(
        name: str, tool_id: str, json_parts: list[str]
    ) -> list[_FakeStreamEvent]:
        """Build stream events for a tool call."""
        events = [
            _FakeStreamEvent(
                type="content_block_start",
                content_block=_FakeContentBlock(type="tool_use", name=name, id=tool_id),
            ),
        ]
        for part in json_parts:
            events.append(
                _FakeStreamEvent(
                    type="content_block_delta",
                    delta=_FakeDelta(type="input_json_delta", partial_json=part),
                )
            )
        events.append(_FakeStreamEvent(type="content_block_stop"))
        return events

    async def test_text_only_stream(self, provider: AnthropicProvider) -> None:
        """Text-only response yields TEXT_DELTA events + DONE."""
        final_msg = _make_anthropic_response(
            [TextBlock(type="text", text="Hello world")],
        )
        events = self._text_events(["Hello", " world"])
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")]
        ):
            collected.append(event)

        # Should have: 2 text deltas + 1 DONE
        assert len(collected) == 3
        assert collected[0].type == StreamEventType.TEXT_DELTA
        assert collected[0].text == "Hello"
        assert collected[1].type == StreamEventType.TEXT_DELTA
        assert collected[1].text == " world"
        assert collected[2].type == StreamEventType.DONE
        # DONE carries the full LLMResponse
        llm_response = collected[2].raw
        assert llm_response.text == "Hello world"
        assert llm_response.tool_calls is None
        assert llm_response.usage.input_tokens == 100

    async def test_tool_call_stream(self, provider: AnthropicProvider) -> None:
        """Tool call response yields TOOL_USE_START + TOOL_USE_END + DONE."""
        final_msg = _make_anthropic_response(
            [ToolUseBlock(type="tool_use", id="toolu_123", name="add", input={"a": 1, "b": 2})],
            stop_reason="tool_use",
        )
        events = self._tool_events("add", "toolu_123", ['{"a":', " 1, ", '"b": 2}'])
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Add 1 and 2")]
        ):
            collected.append(event)

        # TOOL_USE_START + TOOL_USE_END + DONE
        assert len(collected) == 3
        assert collected[0].type == StreamEventType.TOOL_USE_START
        assert collected[0].tool_name == "add"
        assert collected[1].type == StreamEventType.TOOL_USE_END
        assert collected[1].tool_call.name == "add"
        assert collected[1].tool_call.params == {"a": 1, "b": 2}
        assert collected[1].tool_call.provider_tool_call_id == "toolu_123"
        assert collected[2].type == StreamEventType.DONE

    async def test_text_then_tool_stream(self, provider: AnthropicProvider) -> None:
        """Mixed response: text first, then tool call."""
        final_msg = _make_anthropic_response(
            [
                TextBlock(type="text", text="Let me calculate."),
                ToolUseBlock(type="tool_use", id="toolu_456", name="add", input={"a": 15, "b": 27}),
            ],
            stop_reason="tool_use",
        )
        events = (
            self._text_events(["Let me", " calculate."])
            + self._tool_events("add", "toolu_456", ['{"a": 15, "b":', " 27}"])
        )
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Add 15 and 27")]
        ):
            collected.append(event)

        types = [e.type for e in collected]
        assert types == [
            StreamEventType.TEXT_DELTA,      # "Let me"
            StreamEventType.TEXT_DELTA,      # " calculate."
            StreamEventType.TOOL_USE_START,  # add starts
            StreamEventType.TOOL_USE_END,    # add complete
            StreamEventType.DONE,
        ]
        # Text accumulated correctly
        assert collected[-1].raw.text == "Let me calculate."
        # Tool call assembled correctly
        tc = collected[3].tool_call
        assert tc.name == "add"
        assert tc.params == {"a": 15, "b": 27}

    async def test_multiple_tool_calls_stream(self, provider: AnthropicProvider) -> None:
        """Parallel tool calls in a single response."""
        final_msg = _make_anthropic_response(
            [
                ToolUseBlock(type="tool_use", id="t1", name="add", input={"a": 1, "b": 2}),
                ToolUseBlock(type="tool_use", id="t2", name="multiply", input={"x": 3, "y": 4}),
            ],
            stop_reason="tool_use",
        )
        events = (
            self._tool_events("add", "t1", ['{"a": 1, "b": 2}'])
            + self._tool_events("multiply", "t2", ['{"x": 3, "y": 4}'])
        )
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Do both")]
        ):
            collected.append(event)

        types = [e.type for e in collected]
        assert types == [
            StreamEventType.TOOL_USE_START,
            StreamEventType.TOOL_USE_END,
            StreamEventType.TOOL_USE_START,
            StreamEventType.TOOL_USE_END,
            StreamEventType.DONE,
        ]
        # Both tool calls in the final LLMResponse
        llm_response = collected[-1].raw
        assert len(llm_response.tool_calls) == 2

    async def test_done_carries_usage_stats(self, provider: AnthropicProvider) -> None:
        """DONE event carries correct usage stats from final message."""
        final_msg = _make_anthropic_response(
            [TextBlock(type="text", text="Hi")],
            input_tokens=200,
            output_tokens=75,
        )
        events = self._text_events(["Hi"])
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")]
        ):
            collected.append(event)

        done = collected[-1]
        assert done.type == StreamEventType.DONE
        assert done.raw.usage.input_tokens == 200
        assert done.raw.usage.output_tokens == 75
        assert done.raw.usage.total_tokens == 275

    async def test_stream_timeout_raises_clean_error(self, provider: AnthropicProvider) -> None:
        """API timeout during streaming raises clean TimeoutError."""
        import anthropic as anthropic_mod

        provider._client.messages.stream = MagicMock(
            side_effect=anthropic_mod.APITimeoutError(request=MagicMock())
        )

        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in provider.complete_stream(
                [Message(role=Role.USER, content="Hi")]
            ):
                pass

    async def test_malformed_tool_json_falls_back_to_empty_params(
        self, provider: AnthropicProvider
    ) -> None:
        """Malformed JSON in tool call fragments falls back to empty params."""
        final_msg = _make_anthropic_response(
            [ToolUseBlock(type="tool_use", id="t1", name="broken", input={})],
            stop_reason="tool_use",
        )
        # Send broken JSON fragments that won't parse
        events = self._tool_events("broken", "t1", ['{"a": ', "INVALID"])
        fake_stream = _FakeStream(events, final_msg)

        provider._client.messages.stream = MagicMock(return_value=fake_stream)

        collected = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")]
        ):
            collected.append(event)

        tool_end = [e for e in collected if e.type == StreamEventType.TOOL_USE_END][0]
        assert tool_end.tool_call.params == {}
        assert tool_end.tool_call.name == "broken"
