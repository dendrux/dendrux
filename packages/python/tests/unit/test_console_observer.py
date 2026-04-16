"""Tests for ConsoleNotifier and notifier composition in agent.run()."""

from __future__ import annotations

from typing import Any

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.tool import tool
from dendrux.types import LLMResponse, Message, Role, ToolCall, ToolResult


class RecordingNotifier:
    """Test notifier that records all events."""

    def __init__(self) -> None:
        self.messages: list[tuple[Message, int]] = []
        self.llm_calls: list[tuple[Any, int]] = []
        self.tool_completions: list[tuple[ToolCall, ToolResult, int]] = []

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        self.messages.append((message, iteration))

    async def on_llm_call_completed(self, response: Any, iteration: int, **kwargs: Any) -> None:
        self.llm_calls.append((response, iteration))

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        self.tool_completions.append((tool_call, tool_result, iteration))


class TestConsoleNotifier:
    """Test that ConsoleNotifier implements the protocol without errors."""

    async def test_text_run(self) -> None:
        """ConsoleNotifier handles a simple text-only run."""
        obs = ConsoleNotifier()
        msg = Message(role=Role.USER, content="hello")
        await obs.on_message_appended(msg, 0)

        response = LLMResponse(text="hi")
        await obs.on_llm_call_completed(response, 1)

        assistant_msg = Message(role=Role.ASSISTANT, content="hi")
        await obs.on_message_appended(assistant_msg, 1)

    async def test_tool_call_run(self) -> None:
        """ConsoleNotifier handles tool calls and results."""
        obs = ConsoleNotifier()

        tc = ToolCall(name="add", params={"a": 1, "b": 2})
        assistant_msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        await obs.on_message_appended(assistant_msg, 1)

        tr = ToolResult(name="add", call_id=tc.id, payload="3", success=True, duration_ms=100)
        await obs.on_tool_completed(tc, tr, 1)

    async def test_failed_tool(self) -> None:
        """ConsoleNotifier handles failed tools gracefully."""
        obs = ConsoleNotifier()

        tc = ToolCall(name="broken", params={})
        tr = ToolResult(
            name="broken",
            call_id=tc.id,
            payload="",
            success=False,
            error="something went wrong",
        )
        await obs.on_tool_completed(tc, tr, 1)

    async def test_max_calls_limit_display(self) -> None:
        """ConsoleNotifier shows limit message for max_calls_per_run."""
        obs = ConsoleNotifier()

        tc = ToolCall(name="search", params={})
        tr = ToolResult(
            name="search",
            call_id=tc.id,
            payload="",
            success=False,
            error="Tool 'search' has reached its maximum of 3 calls for this run.",
        )
        await obs.on_tool_completed(tc, tr, 1)

    async def test_large_params_truncated(self) -> None:
        """Large tool params are truncated, not dumped."""
        obs = ConsoleNotifier()

        tc = ToolCall(name="save", params={"content": "x" * 1000})
        assistant_msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        await obs.on_message_appended(assistant_msg, 1)

    async def test_llm_done_accumulates_cache_totals(self) -> None:
        """on_llm_call_completed adds cache_read / cache_creation to the
        running totals so print_summary can report a hit ratio."""
        from dendrux.types import UsageStats

        obs = ConsoleNotifier()
        response = LLMResponse(
            text="ok",
            usage=UsageStats(
                input_tokens=500,
                output_tokens=50,
                total_tokens=550,
                cache_read_input_tokens=900,
                cache_creation_input_tokens=200,
            ),
        )
        await obs.on_llm_call_completed(response, 1)
        await obs.on_llm_call_completed(response, 2)
        assert obs._total_cache_read == 1800
        assert obs._total_cache_creation == 400

    async def test_llm_done_handles_none_cache_fields(self) -> None:
        """A provider that didn't report cache (older runs, compatible
        backends) leaves totals unchanged — None is treated as 0."""
        from dendrux.types import UsageStats

        obs = ConsoleNotifier()
        response = LLMResponse(
            text="ok",
            usage=UsageStats(input_tokens=200, output_tokens=30, total_tokens=230),
        )
        await obs.on_llm_call_completed(response, 1)
        assert obs._total_cache_read == 0
        assert obs._total_cache_creation == 0


class TestAgentRunWithNotifier:
    """Test that agent.run(notifier=...) threads the notifier correctly."""

    async def test_notifier_receives_events(self) -> None:
        """External notifier receives all lifecycle events from agent.run()."""

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

        recording = RecordingNotifier()
        agent = Agent(provider=mock, prompt="test", tools=[add])
        await agent.run("What is 1+2?", notifier=recording)

        # Notifier received events
        assert len(recording.messages) > 0
        assert len(recording.llm_calls) > 0
        assert len(recording.tool_completions) == 1
        assert recording.tool_completions[0][1].success is True

    async def test_notifier_without_persistence(self) -> None:
        """Notifier works even without database persistence."""
        responses = [LLMResponse(text="hello")]
        mock = MockLLM(responses=responses)

        recording = RecordingNotifier()
        agent = Agent(provider=mock, prompt="test", tools=[])
        result = await agent.run("hi", notifier=recording)

        assert result.answer == "hello"
        assert len(recording.messages) >= 1  # At least user + assistant


class TestCompositeNotifier:
    async def test_fans_out_to_all(self) -> None:
        """CompositeNotifier dispatches to all registered notifiers."""
        r1 = RecordingNotifier()
        r2 = RecordingNotifier()
        composite = CompositeNotifier([r1, r2])

        msg = Message(role=Role.USER, content="hello")
        await composite.on_message_appended(msg, 0)

        assert len(r1.messages) == 1
        assert len(r2.messages) == 1

    async def test_one_failure_doesnt_block_others(self) -> None:
        """If one notifier fails, others still receive events."""

        class FailingNotifier:
            async def on_message_appended(self, message: Any, iteration: int) -> None:
                raise RuntimeError("boom")

            async def on_llm_call_completed(self, response: Any, iteration: int, **kw: Any) -> None:
                pass

            async def on_tool_completed(self, tc: Any, tr: Any, iteration: int) -> None:
                pass

        recording = RecordingNotifier()
        composite = CompositeNotifier([FailingNotifier(), recording])  # type: ignore[list-item]

        msg = Message(role=Role.USER, content="hello")
        await composite.on_message_appended(msg, 0)

        # Recording notifier still received the event
        assert len(recording.messages) == 1
