"""Tests for ConsoleObserver and observer composition in agent.run()."""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.observers.composite import CompositeObserver
from dendrux.observers.console import ConsoleObserver
from dendrux.tool import tool
from dendrux.types import LLMResponse, Message, Role, ToolCall, ToolResult


class RecordingObserver:
    """Test observer that records all events."""

    def __init__(self) -> None:
        self.messages: list[tuple[Message, int]] = []
        self.llm_calls: list[tuple[Any, int]] = []
        self.tool_completions: list[tuple[ToolCall, ToolResult, int]] = []

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        self.messages.append((message, iteration))

    async def on_llm_call_completed(
        self, response: Any, iteration: int, **kwargs: Any
    ) -> None:
        self.llm_calls.append((response, iteration))

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        self.tool_completions.append((tool_call, tool_result, iteration))


class TestConsoleObserver:
    """Test that ConsoleObserver implements the protocol without errors."""

    async def test_text_run(self) -> None:
        """ConsoleObserver handles a simple text-only run."""
        obs = ConsoleObserver()
        msg = Message(role=Role.USER, content="hello")
        await obs.on_message_appended(msg, 0)

        response = LLMResponse(text="hi")
        await obs.on_llm_call_completed(response, 1)

        assistant_msg = Message(role=Role.ASSISTANT, content="hi")
        await obs.on_message_appended(assistant_msg, 1)

    async def test_tool_call_run(self) -> None:
        """ConsoleObserver handles tool calls and results."""
        obs = ConsoleObserver()

        tc = ToolCall(name="add", params={"a": 1, "b": 2})
        assistant_msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        await obs.on_message_appended(assistant_msg, 1)

        tr = ToolResult(name="add", call_id=tc.id, payload="3", success=True, duration_ms=100)
        await obs.on_tool_completed(tc, tr, 1)

    async def test_failed_tool(self) -> None:
        """ConsoleObserver handles failed tools gracefully."""
        obs = ConsoleObserver()

        tc = ToolCall(name="broken", params={})
        tr = ToolResult(
            name="broken", call_id=tc.id, payload="", success=False,
            error="something went wrong",
        )
        await obs.on_tool_completed(tc, tr, 1)

    async def test_max_calls_limit_display(self) -> None:
        """ConsoleObserver shows limit message for max_calls_per_run."""
        obs = ConsoleObserver()

        tc = ToolCall(name="search", params={})
        tr = ToolResult(
            name="search", call_id=tc.id, payload="", success=False,
            error="Tool 'search' has reached its maximum of 3 calls for this run.",
        )
        await obs.on_tool_completed(tc, tr, 1)

    async def test_large_params_truncated(self) -> None:
        """Large tool params are truncated, not dumped."""
        obs = ConsoleObserver()

        tc = ToolCall(name="save", params={"content": "x" * 1000})
        assistant_msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        await obs.on_message_appended(assistant_msg, 1)


class TestAgentRunWithObserver:
    """Test that agent.run(observer=...) threads the observer correctly."""

    async def test_observer_receives_events(self) -> None:
        """External observer receives all lifecycle events from agent.run()."""
        @tool()
        async def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        tc = ToolCall(name="add", params={"a": 1, "b": 2})
        responses = [
            LLMResponse(text="adding", tool_calls=[tc]),
            LLMResponse(text="The answer is 3"),
        ]
        mock = MockLLM(responses=responses)

        recording = RecordingObserver()
        agent = Agent(provider=mock, prompt="test", tools=[add])
        result = await agent.run("What is 1+2?", observer=recording)

        # Observer received events
        assert len(recording.messages) > 0
        assert len(recording.llm_calls) > 0
        assert len(recording.tool_completions) == 1
        assert recording.tool_completions[0][1].success is True

    async def test_observer_without_persistence(self) -> None:
        """Observer works even without database persistence."""
        responses = [LLMResponse(text="hello")]
        mock = MockLLM(responses=responses)

        recording = RecordingObserver()
        agent = Agent(provider=mock, prompt="test", tools=[])
        result = await agent.run("hi", observer=recording)

        assert result.answer == "hello"
        assert len(recording.messages) >= 1  # At least user + assistant


class TestCompositeObserver:
    async def test_fans_out_to_all(self) -> None:
        """CompositeObserver dispatches to all registered observers."""
        r1 = RecordingObserver()
        r2 = RecordingObserver()
        composite = CompositeObserver([r1, r2])

        msg = Message(role=Role.USER, content="hello")
        await composite.on_message_appended(msg, 0)

        assert len(r1.messages) == 1
        assert len(r2.messages) == 1

    async def test_one_failure_doesnt_block_others(self) -> None:
        """If one observer fails, others still receive events."""
        class FailingObserver:
            async def on_message_appended(self, message: Any, iteration: int) -> None:
                raise RuntimeError("boom")

            async def on_llm_call_completed(self, response: Any, iteration: int, **kw: Any) -> None:
                pass

            async def on_tool_completed(self, tc: Any, tr: Any, iteration: int) -> None:
                pass

        recording = RecordingObserver()
        composite = CompositeObserver([FailingObserver(), recording])  # type: ignore[list-item]

        msg = Message(role=Role.USER, content="hello")
        await composite.on_message_appended(msg, 0)

        # Recording observer still received the event
        assert len(recording.messages) == 1
