"""Tests for SingleCall loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import Loop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    ToolCall,
    UsageStats,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_agent(**overrides) -> Agent:
    defaults: dict[str, Any] = {
        "prompt": "Classify as positive, negative, or neutral.",
        "tools": [],
        "loop": SingleCall(),
    }
    defaults.update(overrides)
    return Agent(**defaults)


def _response(text: str) -> LLMResponse:
    return LLMResponse(text=text)


# ------------------------------------------------------------------
# SingleCall is a Loop
# ------------------------------------------------------------------


class TestSingleCallABC:
    def test_is_a_loop(self) -> None:
        assert isinstance(SingleCall(), Loop)


# ------------------------------------------------------------------
# Agent validation
# ------------------------------------------------------------------


class TestSingleCallValidation:
    def test_rejects_agent_with_tools(self) -> None:
        @tool()
        async def dummy() -> str:
            """A dummy tool."""
            return "hi"

        with pytest.raises(ValueError, match="SingleCall loop but has 1 tool"):
            Agent(
                prompt="test",
                tools=[dummy],
                loop=SingleCall(),
            )

    def test_accepts_agent_without_tools(self) -> None:
        agent = _make_agent()
        assert agent.loop is not None
        assert isinstance(agent.loop, SingleCall)
        assert agent.tools == []


# ------------------------------------------------------------------
# SingleCall.run() — batch
# ------------------------------------------------------------------


class TestSingleCallRun:
    async def test_returns_text_as_answer(self) -> None:
        llm = MockLLM([_response("positive")])
        agent = _make_agent()

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="I love this product!",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "positive"
        assert result.iteration_count == 1
        assert result.steps == []

    async def test_unique_run_ids(self) -> None:
        agent = _make_agent()

        r1 = await SingleCall().run(
            agent=agent,
            provider=MockLLM([_response("a")]),
            strategy=NativeToolCalling(),
            user_input="Hi",
        )
        r2 = await SingleCall().run(
            agent=agent,
            provider=MockLLM([_response("b")]),
            strategy=NativeToolCalling(),
            user_input="Hi",
        )

        assert r1.run_id != r2.run_id

    async def test_usage_stats_populated(self) -> None:
        resp = LLMResponse(
            text="neutral",
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        llm = MockLLM([resp])
        agent = _make_agent()

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="test",
        )

        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    async def test_raises_on_unexpected_tool_calls(self) -> None:
        """Provider should not return tool_calls when no tools are sent."""
        resp = LLMResponse(
            text="calling a tool",
            tool_calls=[ToolCall(name="phantom", params={})],
        )
        llm = MockLLM([resp])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="unexpected tool_calls"):
            await SingleCall().run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
            )

    async def test_accepts_initial_history_as_chat_context(self) -> None:
        """SingleCall now accepts initial_history (chat history seeding).

        It is no longer treated as a resume signal. The full message list
        comes from initial_history when provided, with the runner being
        responsible for appending the new user_input to it.
        """
        llm = MockLLM([_response("ok")])
        agent = _make_agent()

        history = [
            Message(role=Role.USER, content="prior question"),
            Message(role=Role.ASSISTANT, content="prior answer"),
            Message(role=Role.USER, content="next question"),
        ]

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="next question",
            initial_history=history,
        )
        assert result.status == RunStatus.SUCCESS

        # Provider should have seen the entire seeded history as messages.
        sent = llm.call_history[0]["messages"]
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert [m.content for m in user_assistant] == [
            "prior question",
            "prior answer",
            "next question",
        ]

    async def test_rejects_resume_with_initial_steps(self) -> None:
        """SingleCall must reject initial_steps (genuine resume signal)."""
        from dendrux.types import AgentStep, Finish

        llm = MockLLM([_response("ok")])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="does not support resume"):
            await SingleCall().run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
                initial_steps=[
                    AgentStep(reasoning=None, action=Finish(answer="x")),
                ],
            )

    async def test_rejects_resume_with_iteration_offset(self) -> None:
        """SingleCall must reject non-zero iteration_offset."""
        llm = MockLLM([_response("ok")])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="does not support resume"):
            await SingleCall().run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
                iteration_offset=3,
            )

    async def test_rejects_resume_with_initial_usage(self) -> None:
        """SingleCall must reject initial_usage."""
        llm = MockLLM([_response("ok")])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="does not support resume"):
            await SingleCall().run(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
                initial_usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
            )

    async def test_stream_rejects_resume(self) -> None:
        """SingleCall streaming must reject genuine resume parameters."""
        llm = MockLLM([_response("ok")])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="does not support resume"):
            async for _ in SingleCall().run_stream(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
                initial_usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
            ):
                pass

    async def test_stream_accepts_initial_history(self) -> None:
        """SingleCall streaming now accepts initial_history (chat seeding)."""
        llm = MockLLM([_response("streamed answer")])
        agent = _make_agent()

        history = [
            Message(role=Role.USER, content="prior"),
            Message(role=Role.ASSISTANT, content="prior reply"),
            Message(role=Role.USER, content="next"),
        ]

        events = []
        async for ev in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="next",
            initial_history=history,
        ):
            events.append(ev)

        # Verify provider saw the seeded history.
        sent = llm.call_history[0]["messages"]
        user_assistant = [m for m in sent if m.role in (Role.USER, Role.ASSISTANT)]
        assert [m.content for m in user_assistant] == [
            "prior",
            "prior reply",
            "next",
        ]

    async def test_handles_none_text(self) -> None:
        """If provider returns None text, answer should be None."""
        llm = MockLLM([LLMResponse(text=None)])
        agent = _make_agent()

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="test",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer is None


# ------------------------------------------------------------------
# SingleCall.run() — notifier hooks
# ------------------------------------------------------------------


class TestSingleCallNotifier:
    async def test_notifier_hooks_fire(self) -> None:
        notifier = AsyncMock()
        notifier.on_message_appended = AsyncMock()
        notifier.on_llm_call_completed = AsyncMock()
        notifier.on_tool_completed = AsyncMock()

        llm = MockLLM([_response("positive")])
        agent = _make_agent()

        await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="test",
            notifier=notifier,
        )

        # user message + assistant message = 2 calls
        assert notifier.on_message_appended.call_count == 2
        # one LLM call
        assert notifier.on_llm_call_completed.call_count == 1
        # no tool calls
        assert notifier.on_tool_completed.call_count == 0

    async def test_notifier_messages_correct(self) -> None:
        notifier = AsyncMock()
        notifier.on_message_appended = AsyncMock()
        notifier.on_llm_call_completed = AsyncMock()
        notifier.on_tool_completed = AsyncMock()

        llm = MockLLM([_response("negative")])
        agent = _make_agent()

        await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="I hate bugs",
            notifier=notifier,
        )

        # First call: user message
        first_msg: Message = notifier.on_message_appended.call_args_list[0][0][0]
        assert first_msg.role == Role.USER
        assert first_msg.content == "I hate bugs"

        # Second call: assistant message
        second_msg: Message = notifier.on_message_appended.call_args_list[1][0][0]
        assert second_msg.role == Role.ASSISTANT
        assert second_msg.content == "negative"


# ------------------------------------------------------------------
# SingleCall.run_stream() — streaming
# ------------------------------------------------------------------


class TestSingleCallStream:
    async def test_streams_text_deltas(self) -> None:
        """Streaming should yield TEXT_DELTA events followed by RUN_COMPLETED."""
        llm = MockLLM([_response("positive")])
        agent = _make_agent()

        events = []
        async for event in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="I love this!",
        ):
            events.append(event)

        # At minimum we get RUN_COMPLETED (MockLLM fallback stream)
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert events[-1].run_result is not None
        assert events[-1].run_result.status == RunStatus.SUCCESS
        assert events[-1].run_result.answer == "positive"

    async def test_stream_answer_from_done_response(self) -> None:
        """Answer should come from the final LLMResponse, not concatenated deltas."""
        llm = MockLLM([_response("the full answer")])
        agent = _make_agent()

        events = []
        async for event in SingleCall().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="test",
        ):
            events.append(event)

        completed = events[-1]
        assert completed.type == RunEventType.RUN_COMPLETED
        assert completed.run_result.answer == "the full answer"

    async def test_stream_raises_on_tool_use_events(self) -> None:
        """If provider streams TOOL_USE_* events, SingleCall should raise."""

        class ToolUseStreamLLM(MockLLM):
            """MockLLM that streams a tool_use event."""

            async def complete_stream(self, messages, tools=None, **kwargs):
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_name="phantom",
                    tool_call_id="tc_1",
                )
                yield StreamEvent(
                    type=StreamEventType.DONE,
                    raw=LLMResponse(text="done"),
                )

        llm = ToolUseStreamLLM([_response("fallback")])
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="unexpected.*tool_use_start"):
            async for _ in SingleCall().run_stream(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="test",
            ):
                pass


# ------------------------------------------------------------------
# Import smoke test
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Full runner integration (agent.run() → runner → SingleCall)
# ------------------------------------------------------------------


class TestSingleCallRunnerIntegration:
    async def test_agent_run_uses_single_call(self) -> None:
        """agent.run() should use SingleCall when set on the agent."""
        llm = MockLLM([_response("positive")])
        agent = Agent(
            provider=llm,
            loop=SingleCall(),
            prompt="Classify as positive, negative, or neutral.",
        )

        result = await agent.run("I love this!")

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "positive"
        assert result.iteration_count == 1

    async def test_agent_run_persists_loop_name(self) -> None:
        """Runner should persist dendrux.loop in run metadata."""
        from dataclasses import dataclass, field

        @dataclass
        class RecordingStore:
            created_runs: list[dict[str, Any]] = field(default_factory=list)
            finalized_runs: list[dict[str, Any]] = field(default_factory=list)

            async def create_run(self, run_id, agent_name, **kwargs):
                self.created_runs.append({"run_id": run_id, "agent_name": agent_name, **kwargs})
                from dendrux.types import CreateRunResult, RunStatus

                return CreateRunResult(run_id=run_id, outcome="created", status=RunStatus.RUNNING)

            async def finalize_run(self, run_id, **kwargs):
                self.finalized_runs.append({"run_id": run_id, **kwargs})
                return True

            async def save_run_event(self, *args, **kwargs):
                pass

            async def save_trace(self, *args, **kwargs):
                pass

            async def save_tool_call(self, *args, **kwargs):
                pass

            async def save_usage(self, *args, **kwargs):
                pass

            async def save_llm_interaction(self, *args, **kwargs):
                pass

            async def touch_progress(self, *args, **kwargs):
                pass

        store = RecordingStore()
        llm = MockLLM([_response("neutral")])
        agent = Agent(
            provider=llm,
            loop=SingleCall(),
            prompt="Classify.",
            state_store=store,
        )

        await agent.run("test")

        meta = store.created_runs[0]["meta"]
        assert meta["dendrux.loop"] == "SingleCall"


# ------------------------------------------------------------------
# Import smoke test
# ------------------------------------------------------------------


class TestSingleCallExports:
    def test_importable_from_loops(self) -> None:
        from dendrux.loops import SingleCall as SingleCallFromLoops

        assert SingleCallFromLoops is SingleCall

    def test_importable_from_dendrux(self) -> None:
        from dendrux import SingleCall as SingleCallFromRoot

        assert SingleCallFromRoot is SingleCall
