"""Tests for LLM provider abstraction and MockLLM."""

import pytest

from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolDef,
    UsageStats,
)


class TestMockLLM:
    """MockLLM returns predetermined responses in order."""

    async def test_returns_responses_in_order(self):
        llm = MockLLM(
            [
                LLMResponse(text="first"),
                LLMResponse(text="second"),
            ]
        )
        messages = [Message(role=Role.USER, content="hi")]

        r1 = await llm.complete(messages)
        r2 = await llm.complete(messages)

        assert r1.text == "first"
        assert r2.text == "second"

    async def test_raises_when_exhausted(self):
        llm = MockLLM([LLMResponse(text="only one")])
        messages = [Message(role=Role.USER, content="hi")]

        await llm.complete(messages)
        with pytest.raises(IndexError, match="1 calls made but only 1 responses"):
            await llm.complete(messages)

    async def test_tracks_call_count(self):
        llm = MockLLM(
            [
                LLMResponse(text="a"),
                LLMResponse(text="b"),
            ]
        )
        messages = [Message(role=Role.USER, content="hi")]

        assert llm.calls_made == 0
        assert not llm.exhausted

        await llm.complete(messages)
        assert llm.calls_made == 1
        assert not llm.exhausted

        await llm.complete(messages)
        assert llm.calls_made == 2
        assert llm.exhausted

    async def test_records_call_history(self):
        tools = [ToolDef(name="add", description="Add numbers", parameters={})]
        llm = MockLLM([LLMResponse(text="ok")])
        messages = [Message(role=Role.USER, content="hi")]

        await llm.complete(messages, tools=tools, temperature=0.5)

        assert len(llm.call_history) == 1
        assert llm.call_history[0]["messages"] == messages
        assert llm.call_history[0]["tools"] == tools
        assert llm.call_history[0]["kwargs"] == {"temperature": 0.5}

    async def test_returns_tool_calls(self):
        llm = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="add", params={"a": 1, "b": 2})],
                ),
            ]
        )
        messages = [Message(role=Role.USER, content="add 1 + 2")]

        response = await llm.complete(messages)
        assert response.text is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "add"
        assert response.tool_calls[0].params == {"a": 1, "b": 2}

    async def test_returns_usage_stats(self):
        llm = MockLLM(
            [
                LLMResponse(
                    text="hi",
                    usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
            ]
        )
        messages = [Message(role=Role.USER, content="hi")]

        response = await llm.complete(messages)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15


class TestLLMProviderABC:
    """LLMProvider ABC enforces the contract."""

    async def test_cannot_instantiate_without_complete(self):
        class BadProvider(LLMProvider):
            pass

        with pytest.raises(TypeError, match="complete"):
            BadProvider()

    async def test_subclass_with_complete_is_valid(self):
        llm = MockLLM([LLMResponse(text="hi")])
        assert isinstance(llm, LLMProvider)


class TestDefaultCompleteStream:
    """Default complete_stream() wraps complete() into stream events."""

    async def test_text_response_streams_as_delta_then_done(self):
        llm = MockLLM([LLMResponse(text="hello world")])
        messages = [Message(role=Role.USER, content="hi")]

        events: list[StreamEvent] = []
        async for event in llm.complete_stream(messages):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == StreamEventType.TEXT_DELTA
        assert events[0].text == "hello world"
        assert events[1].type == StreamEventType.DONE

    async def test_tool_call_response_streams_as_tool_use_end(self):
        tool_call = ToolCall(name="search", params={"q": "test"})
        llm = MockLLM([LLMResponse(tool_calls=[tool_call])])
        messages = [Message(role=Role.USER, content="search")]

        events: list[StreamEvent] = []
        async for event in llm.complete_stream(messages):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == StreamEventType.TOOL_USE_END
        assert events[0].tool_call == tool_call
        assert events[0].tool_name == "search"
        assert events[1].type == StreamEventType.DONE

    async def test_text_and_tool_calls_streams_both(self):
        tool_call = ToolCall(name="add", params={"a": 1, "b": 2})
        llm = MockLLM([LLMResponse(text="Let me add those", tool_calls=[tool_call])])
        messages = [Message(role=Role.USER, content="add")]

        events: list[StreamEvent] = []
        async for event in llm.complete_stream(messages):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == StreamEventType.TEXT_DELTA
        assert events[0].text == "Let me add those"
        assert events[1].type == StreamEventType.TOOL_USE_END
        assert events[1].tool_call == tool_call
        assert events[2].type == StreamEventType.DONE

    async def test_empty_response_streams_only_done(self):
        llm = MockLLM([LLMResponse()])
        messages = [Message(role=Role.USER, content="hi")]

        events: list[StreamEvent] = []
        async for event in llm.complete_stream(messages):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == StreamEventType.DONE
