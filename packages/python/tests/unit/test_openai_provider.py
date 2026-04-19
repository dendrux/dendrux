"""Tests for OpenAI Chat Completions provider — conversion logic and streaming."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.types import (  # noqa: E402
    Message,
    Role,
    StreamEventType,
    ToolCall,
    ToolDef,
)


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

    def test_tool_result_uses_dendrux_id_when_no_provider_id(
        self, provider: OpenAIProvider
    ) -> None:
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

        tools = [
            ToolDef(name="add", description="Add", parameters={"type": "object", "properties": {}})
        ]
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


# ---------------------------------------------------------------------------
# Streaming fakes — simulate OpenAI SDK streaming chunks
# ---------------------------------------------------------------------------
@dataclass
class FakeDeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeDeltaToolCall:
    index: int
    id: str | None = None
    function: FakeDeltaFunction | None = None


@dataclass
class FakeDelta:
    content: str | None = None
    tool_calls: list[FakeDeltaToolCall] | None = None


@dataclass
class FakeStreamChoice:
    delta: FakeDelta = field(default_factory=FakeDelta)
    index: int = 0
    finish_reason: str | None = None


@dataclass
class FakeChunkUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FakeChunk:
    choices: list[FakeStreamChoice] = field(default_factory=list)
    usage: FakeChunkUsage | None = None


class MockAsyncStream:
    """Async iterable that yields pre-built chunks."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._idx = 0

    def __aiter__(self) -> MockAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _text_chunks(text: str) -> list[FakeChunk]:
    """Build streaming chunks for a text-only response."""
    chunks: list[FakeChunk] = []
    for char in text:
        chunks.append(FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(content=char))]))
    # Terminal chunk with finish_reason
    chunks.append(FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="stop")]))
    # Usage chunk (no choices)
    chunks.append(
        FakeChunk(
            choices=[],
            usage=FakeChunkUsage(
                prompt_tokens=10, completion_tokens=len(text), total_tokens=10 + len(text)
            ),
        )
    )
    return chunks


def _tool_call_chunks(
    tools: list[tuple[str, str, str]],
) -> list[FakeChunk]:
    """Build streaming chunks for tool call responses.

    Each tool is (name, provider_id, arguments_json).
    Arguments are split into 2-char fragments to simulate incremental delivery.
    """
    chunks: list[FakeChunk] = []
    for idx, (name, tool_id, args_json) in enumerate(tools):
        # First chunk: name + id
        chunks.append(
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=idx,
                                    id=tool_id,
                                    function=FakeDeltaFunction(name=name, arguments=""),
                                )
                            ]
                        )
                    )
                ]
            )
        )
        # Argument fragments
        for i in range(0, len(args_json), 2):
            fragment = args_json[i : i + 2]
            chunks.append(
                FakeChunk(
                    choices=[
                        FakeStreamChoice(
                            delta=FakeDelta(
                                tool_calls=[
                                    FakeDeltaToolCall(
                                        index=idx,
                                        function=FakeDeltaFunction(arguments=fragment),
                                    )
                                ]
                            )
                        )
                    ]
                )
            )
    # Terminal chunk
    chunks.append(
        FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")])
    )
    # Usage chunk
    chunks.append(
        FakeChunk(
            choices=[],
            usage=FakeChunkUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        )
    )
    return chunks


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------
class TestCompleteStream:
    """Tests for complete_stream() — real streaming via mocked SDK chunks."""

    async def test_text_only_stream(self, provider: OpenAIProvider) -> None:
        """Text deltas are yielded as TEXT_DELTA events, followed by DONE."""
        chunks = _text_chunks("Hello")
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream([Message(role=Role.USER, content="Hi")]):
            events.append(event)

        text_events = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert len(text_events) == 5
        assert "".join(e.text for e in text_events) == "Hello"

        done_events = [e for e in events if e.type == StreamEventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].raw.text == "Hello"

    async def test_single_tool_call_stream(self, provider: OpenAIProvider) -> None:
        """Single tool call: TOOL_USE_START → TOOL_USE_END → DONE."""
        chunks = _tool_call_chunks([("add", "call_1", '{"a": 1, "b": 2}')])
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="add 1+2")],
            tools=[ToolDef(name="add", description="Add", parameters={"type": "object"})],
        ):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TOOL_USE_START in types
        assert StreamEventType.TOOL_USE_END in types
        assert StreamEventType.DONE in types

        start = next(e for e in events if e.type == StreamEventType.TOOL_USE_START)
        assert start.tool_name == "add"
        assert start.tool_call_id == "call_1"

        end = next(e for e in events if e.type == StreamEventType.TOOL_USE_END)
        assert end.tool_call is not None
        assert end.tool_call.name == "add"
        assert end.tool_call.params == {"a": 1, "b": 2}
        assert end.tool_call.provider_tool_call_id == "call_1"

    async def test_parallel_tool_calls_stream(self, provider: OpenAIProvider) -> None:
        """Multiple tool calls flushed in index order on finish_reason."""
        chunks = _tool_call_chunks(
            [
                ("add", "call_1", '{"a": 1}'),
                ("mul", "call_2", '{"x": 2}'),
            ]
        )
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="calc")],
        ):
            events.append(event)

        starts = [e for e in events if e.type == StreamEventType.TOOL_USE_START]
        ends = [e for e in events if e.type == StreamEventType.TOOL_USE_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_name == "add"
        assert starts[1].tool_name == "mul"
        assert ends[0].tool_call.name == "add"
        assert ends[1].tool_call.name == "mul"

    async def test_interleaved_tool_chunks(self, provider: OpenAIProvider) -> None:
        """Tool call chunks can arrive interleaved by index — defensive buffering."""
        chunks = [
            # Tool 0 starts
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_a",
                                    function=FakeDeltaFunction(name="foo", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Tool 0 args fragment
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0, function=FakeDeltaFunction(arguments='{"x"')
                                )
                            ]
                        )
                    )
                ]
            ),
            # Tool 1 starts (before tool 0 finishes args)
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=1,
                                    id="call_b",
                                    function=FakeDeltaFunction(name="bar", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Tool 0 args continue
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0, function=FakeDeltaFunction(arguments=": 1}")
                                )
                            ]
                        )
                    )
                ]
            ),
            # Tool 1 args
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=1, function=FakeDeltaFunction(arguments='{"y": 2}')
                                )
                            ]
                        )
                    )
                ]
            ),
            # Terminal
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            # Usage
            FakeChunk(
                choices=[],
                usage=FakeChunkUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="go")],
        ):
            events.append(event)

        ends = [e for e in events if e.type == StreamEventType.TOOL_USE_END]
        assert len(ends) == 2
        # Flushed in index order regardless of interleaving
        assert ends[0].tool_call.name == "foo"
        assert ends[0].tool_call.params == {"x": 1}
        assert ends[1].tool_call.name == "bar"
        assert ends[1].tool_call.params == {"y": 2}

    async def test_usage_extracted_from_final_chunk(self, provider: OpenAIProvider) -> None:
        """Usage stats come from the final chunk with stream_options=include_usage."""
        chunks = _text_chunks("Hi")
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        done = next(e for e in events if e.type == StreamEventType.DONE)
        assert done.raw.usage.input_tokens == 10
        assert done.raw.usage.output_tokens == 2
        assert done.raw.usage.total_tokens == 12

    async def test_done_carries_complete_llm_response(self, provider: OpenAIProvider) -> None:
        """DONE event carries text + tool_calls + usage in LLMResponse."""
        # Text + tool call in same response
        chunks = [
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(content="Let me "))]),
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(content="check"))]),
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=FakeDeltaFunction(
                                        name="search", arguments='{"q": "test"}'
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            FakeChunk(
                choices=[],
                usage=FakeChunkUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="search")],
        ):
            events.append(event)

        done = next(e for e in events if e.type == StreamEventType.DONE)
        assert done.raw.text == "Let me check"
        assert done.raw.tool_calls is not None
        assert done.raw.tool_calls[0].name == "search"
        assert done.raw.usage.total_tokens == 35

    async def test_stream_passes_stream_options(self, provider: OpenAIProvider) -> None:
        """complete_stream() passes stream=True and stream_options."""
        chunks = _text_chunks("ok")
        mock_create = AsyncMock(return_value=MockAsyncStream(chunks))
        provider._client.chat.completions.create = mock_create

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_stream_bad_request_raises_not_implemented(
        self, provider: OpenAIProvider
    ) -> None:
        """BadRequestError from incompatible backend → NotImplementedError with base_url."""
        import httpx

        mock_response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="streaming not supported",
                response=mock_response,
                body=None,
            ),
        )

        with pytest.raises(NotImplementedError, match="Streaming request rejected"):
            async for _ in provider.complete_stream(
                [Message(role=Role.USER, content="Hi")],
            ):
                pass  # pragma: no cover

    async def test_stream_timeout_raises(self, provider: OpenAIProvider) -> None:
        """APITimeoutError → TimeoutError with hint."""
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=None),  # type: ignore[arg-type]
        )

        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in provider.complete_stream(
                [Message(role=Role.USER, content="Hi")],
            ):
                pass  # pragma: no cover

    async def test_stream_forwards_kwargs(self, provider: OpenAIProvider) -> None:
        """Supported kwargs are forwarded alongside stream options."""
        chunks = _text_chunks("ok")
        mock_create = AsyncMock(return_value=MockAsyncStream(chunks))
        provider._client.chat.completions.create = mock_create

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")],
            temperature=0.3,
            seed=42,
        ):
            events.append(event)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["seed"] == 42

    async def test_stream_malformed_tool_args_default_empty(self, provider: OpenAIProvider) -> None:
        """Malformed JSON in tool arguments defaults to empty dict, not crash."""
        chunks = [
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=FakeDeltaFunction(name="broken", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0, function=FakeDeltaFunction(arguments="{bad json")
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            FakeChunk(choices=[], usage=FakeChunkUsage()),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="go")],
        ):
            events.append(event)

        end = next(e for e in events if e.type == StreamEventType.TOOL_USE_END)
        assert end.tool_call.params == {}

    async def test_stream_args_before_name(self, provider: OpenAIProvider) -> None:
        """Arguments arriving before name for the same index are not dropped."""
        chunks = [
            # Index 0: id + args arrive first, no name yet
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=FakeDeltaFunction(arguments='{"a"'),
                                )
                            ]
                        )
                    )
                ]
            ),
            # More args, still no name
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    function=FakeDeltaFunction(arguments=": 1}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Name arrives late
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    function=FakeDeltaFunction(name="add"),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Terminal
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            # Usage
            FakeChunk(
                choices=[],
                usage=FakeChunkUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="go")],
        ):
            events.append(event)

        # START should have been emitted when name arrived
        start = next(e for e in events if e.type == StreamEventType.TOOL_USE_START)
        assert start.tool_name == "add"
        assert start.tool_call_id == "call_1"

        # END should have all args (not dropped)
        end = next(e for e in events if e.type == StreamEventType.TOOL_USE_END)
        assert end.tool_call.name == "add"
        assert end.tool_call.params == {"a": 1}
        assert end.tool_call.provider_tool_call_id == "call_1"

    async def test_stream_provider_response_populated(self, provider: OpenAIProvider) -> None:
        """Streaming DONE event carries provider_response for evidence layer."""
        chunks = _text_chunks("Hi")
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        events = []
        async for event in provider.complete_stream(
            [Message(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        done = next(e for e in events if e.type == StreamEventType.DONE)
        assert done.raw.provider_request is not None
        assert done.raw.provider_response is not None
        assert done.raw.provider_response["object"] == "chat.completion.chunked"
        assert done.raw.provider_response["finish_reason"] == "stop"
        assert done.raw.provider_response["usage"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# Phase 1 hardening tests
# ---------------------------------------------------------------------------
class TestBoundaryHardening:
    """Crash-proofing and degradation logging tests."""

    def test_normalize_empty_choices_raises(self, provider: OpenAIProvider) -> None:
        """_normalize_response raises RuntimeError if choices list is empty."""
        response = FakeChatCompletion(choices=[])
        with pytest.raises(RuntimeError, match="no choices"):
            provider._normalize_response(response)

    async def test_stream_malformed_json_logs_warning(
        self, provider: OpenAIProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed tool call JSON in stream logs a warning with context."""
        chunks = [
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=FakeDeltaFunction(name="broken", arguments="{bad"),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            FakeChunk(choices=[], usage=FakeChunkUsage()),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        import logging

        with caplog.at_level(logging.WARNING, logger="dendrux.llm.openai"):
            events = []
            async for event in provider.complete_stream(
                [Message(role=Role.USER, content="go")],
            ):
                events.append(event)

        assert any("Malformed tool call JSON" in r.message for r in caplog.records)
        assert any("provider=openai" in r.message for r in caplog.records)
        # Tool call should still be emitted with empty params
        end = next(e for e in events if e.type == StreamEventType.TOOL_USE_END)
        assert end.tool_call.params == {}

    async def test_stream_unknown_tool_name_logs_warning(
        self, provider: OpenAIProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tool call with no name in stream logs a warning."""
        chunks = [
            # Arg fragment without name
            FakeChunk(
                choices=[
                    FakeStreamChoice(
                        delta=FakeDelta(
                            tool_calls=[
                                FakeDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    function=FakeDeltaFunction(name=None, arguments='{"a":1}'),
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(), finish_reason="tool_calls")]),
            FakeChunk(choices=[], usage=FakeChunkUsage()),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=MockAsyncStream(chunks))

        import logging

        with caplog.at_level(logging.WARNING, logger="dendrux.llm.openai"):
            events = []
            async for event in provider.complete_stream(
                [Message(role=Role.USER, content="go")],
            ):
                events.append(event)

        assert any("Tool call completed with no name" in r.message for r in caplog.records)
        end = next(e for e in events if e.type == StreamEventType.TOOL_USE_END)
        assert end.tool_call.name == "unknown"

    async def test_complete_connection_error_mapped(self, provider: OpenAIProvider) -> None:
        """APIConnectionError → ConnectionError with model context."""
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=None),  # type: ignore[arg-type]
        )
        with pytest.raises(ConnectionError, match="Connection to OpenAI API failed"):
            await provider.complete([Message(role=Role.USER, content="Hi")])

    async def test_stream_connection_error_mapped(self, provider: OpenAIProvider) -> None:
        """Streaming APIConnectionError → ConnectionError with model context."""
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=None),  # type: ignore[arg-type]
        )
        with pytest.raises(ConnectionError, match="Connection to OpenAI API failed"):
            async for _ in provider.complete_stream(
                [Message(role=Role.USER, content="Hi")],
            ):
                pass  # pragma: no cover


class TestApiKeyValidation:
    """Fail-fast at __init__ when no api_key + no env var."""

    def test_raises_when_no_api_key_and_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider(model="gpt-4o")

    def test_accepts_explicit_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        OpenAIProvider(model="gpt-4o", api_key="sk-test")

    def test_accepts_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        OpenAIProvider(model="gpt-4o")

    def test_local_server_works_with_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local servers (vLLM, Ollama) work via base_url + placeholder api_key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        OpenAIProvider(
            model="llama-3",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )
