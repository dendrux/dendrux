"""Tests for Loop protocol and ReActLoop."""

from __future__ import annotations

import pytest

from dendrite.agent import Agent
from dendrite.llm.mock import MockLLM
from dendrite.loops.base import Loop
from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.tool import tool
from dendrite.types import (
    AgentStep,
    Clarification,
    Finish,
    LLMResponse,
    Message,
    Role,
    RunStatus,
    ToolCall,
)

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool()
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool()
def sync_add(a: int, b: int) -> int:
    """Add two numbers (sync)."""
    return a + b


@tool()
async def greet(name: str) -> str:
    """Return a greeting string."""
    return f"hello {name}"


@tool()
async def failing_tool() -> str:
    """A tool that always fails."""
    raise RuntimeError("Something broke")


@tool()
async def return_datetime() -> dict:
    """Return a datetime object (non-JSON-serializable)."""
    import datetime

    return {"timestamp": datetime.datetime(2025, 1, 15, 12, 0, 0)}


@tool()
async def return_set() -> dict:
    """Return a set (non-JSON-serializable)."""
    return {"items": {1, 2, 3}}


# ------------------------------------------------------------------
# Test agent
# ------------------------------------------------------------------


def _make_agent(**overrides) -> Agent:
    defaults = {
        "model": "mock",
        "prompt": "You are a calculator.",
        "tools": [add, multiply],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Loop ABC
# ------------------------------------------------------------------


class TestLoopABC:
    def test_cannot_instantiate_without_run(self) -> None:
        with pytest.raises(TypeError):
            Loop()  # type: ignore[abstract]

    def test_react_loop_is_a_loop(self) -> None:
        assert isinstance(ReActLoop(), Loop)


# ------------------------------------------------------------------
# ReActLoop — simple finish (no tools)
# ------------------------------------------------------------------


class TestReActLoopFinish:
    async def test_immediate_finish(self) -> None:
        """LLM answers immediately without calling any tools."""
        llm = MockLLM([LLMResponse(text="42")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="What is the meaning of life?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "42"
        assert result.iteration_count == 1
        assert len(result.steps) == 1
        assert isinstance(result.steps[0].action, Finish)

    async def test_run_has_unique_id(self) -> None:
        llm = MockLLM([LLMResponse(text="done")])
        agent = _make_agent()

        r1 = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
        )

        llm2 = MockLLM([LLMResponse(text="done")])
        r2 = await ReActLoop().run(
            agent=agent,
            provider=llm2,
            strategy=NativeToolCalling(),
            user_input="Hi",
        )

        assert r1.run_id != r2.run_id


# ------------------------------------------------------------------
# ReActLoop — tool calling
# ------------------------------------------------------------------


class TestReActLoopToolCalling:
    async def test_single_tool_call_then_finish(self) -> None:
        """LLM calls a tool, gets the result, then finishes."""
        tc = ToolCall(
            name="add",
            params={"a": 15, "b": 27},
            provider_tool_call_id="toolu_1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Let me add those", tool_calls=[tc]),
                LLMResponse(text="15 + 27 = 42"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="What is 15 + 27?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "15 + 27 = 42"
        assert result.iteration_count == 2
        assert len(result.steps) == 2

        # First step was a tool call
        assert isinstance(result.steps[0].action, ToolCall)
        assert result.steps[0].action.name == "add"

        # Second step was finish
        assert isinstance(result.steps[1].action, Finish)

    async def test_tool_receives_correct_params(self) -> None:
        """Verify the tool function is called with the right arguments."""
        tc = ToolCall(
            name="multiply",
            params={"a": 6, "b": 7},
            provider_tool_call_id="toolu_m",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="42"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="6 * 7?",
        )

        assert result.status == RunStatus.SUCCESS
        # The tool was called and result fed back — if params were wrong,
        # the tool would fail and result would be an error

    async def test_multiple_tool_calls_sequential(self) -> None:
        """LLM calls tools across multiple iterations."""
        tc1 = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        tc2 = ToolCall(name="multiply", params={"a": 3, "b": 4}, provider_tool_call_id="t2")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc1]),
                LLMResponse(tool_calls=[tc2]),
                LLMResponse(text="1+2=3, 3*4=12"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Compute both",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 3
        assert isinstance(result.steps[0].action, ToolCall)
        assert isinstance(result.steps[1].action, ToolCall)
        assert isinstance(result.steps[2].action, Finish)

    async def test_string_tool_result_is_json_encoded(self) -> None:
        """Tools returning strings must have JSON-encoded payloads."""
        tc = ToolCall(name="greet", params={"name": "world"}, provider_tool_call_id="t_str")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="done"),
            ]
        )
        agent = _make_agent(tools=[greet])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Say hi",
        )

        assert result.status == RunStatus.SUCCESS
        # The tool result fed back to the LLM should be valid JSON.
        # "hello world" must be serialized as '"hello world"', not bare hello world.
        history_call = llm.call_history[1]  # second LLM call has tool result in messages
        tool_msg = [m for m in history_call["messages"] if m.role == Role.TOOL][0]
        assert tool_msg.content == '"hello world"'

    async def test_multiple_tool_calls_in_one_turn(self) -> None:
        """All tool calls from a single assistant turn are executed and results appended."""
        tc1 = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        tc2 = ToolCall(name="multiply", params={"a": 3, "b": 4}, provider_tool_call_id="t2")
        llm = MockLLM(
            [
                LLMResponse(text="I'll compute both", tool_calls=[tc1, tc2]),
                LLMResponse(text="1+2=3, 3*4=12"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Add 1+2 and multiply 3*4",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "1+2=3, 3*4=12"
        assert result.iteration_count == 2

        # Second LLM call should have both tool results in order
        second_call_msgs = llm.call_history[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].name == "add"
        assert tool_msgs[1].name == "multiply"

    async def test_multiple_tool_calls_preserve_order(self) -> None:
        """Tool results are appended in the same order as assistant's tool_calls."""
        tc1 = ToolCall(name="multiply", params={"a": 5, "b": 6}, provider_tool_call_id="t_m")
        tc2 = ToolCall(name="add", params={"a": 10, "b": 20}, provider_tool_call_id="t_a")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc1, tc2]),
                LLMResponse(text="done"),
            ]
        )
        agent = _make_agent()

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Go",
        )

        second_call_msgs = llm.call_history[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m.role == Role.TOOL]
        assert tool_msgs[0].name == "multiply"
        assert tool_msgs[0].call_id == tc1.id
        assert tool_msgs[1].name == "add"
        assert tool_msgs[1].call_id == tc2.id

    async def test_sync_tool_works(self) -> None:
        """Sync tools are executed via asyncio.to_thread."""
        tc = ToolCall(name="sync_add", params={"a": 10, "b": 20}, provider_tool_call_id="t_sync")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="30"),
            ]
        )
        agent = _make_agent(tools=[sync_add])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="10 + 20?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "30"


# ------------------------------------------------------------------
# ReActLoop — error handling
# ------------------------------------------------------------------


class TestReActLoopErrors:
    async def test_unknown_tool_returns_error_result(self) -> None:
        """If LLM calls a tool that doesn't exist, loop returns error in result."""
        tc = ToolCall(
            name="nonexistent",
            params={},
            provider_tool_call_id="toolu_bad",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="I couldn't find that tool"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do something",
        )

        # Loop should continue — the error result is fed back to LLM
        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 2

    async def test_tool_exception_returns_error_result(self) -> None:
        """If a tool raises an exception, it becomes an error ToolResult."""
        tc = ToolCall(
            name="failing_tool",
            params={},
            provider_tool_call_id="toolu_fail",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="The tool failed, sorry"),
            ]
        )
        agent = _make_agent(tools=[failing_tool])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do the thing",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 2


# ------------------------------------------------------------------
# ReActLoop — max iterations
# ------------------------------------------------------------------


class TestReActLoopMaxIterations:
    async def test_stops_at_max_iterations(self) -> None:
        """Loop terminates with MAX_ITERATIONS when limit is hit."""
        tc = ToolCall(name="add", params={"a": 1, "b": 1}, provider_tool_call_id="t_loop")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(tool_calls=[tc]),
                LLMResponse(tool_calls=[tc]),
            ]
        )
        agent = _make_agent(max_iterations=3)

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Keep going",
        )

        assert result.status == RunStatus.MAX_ITERATIONS
        assert result.iteration_count == 3
        assert result.answer is None

    async def test_max_iterations_one_with_finish(self) -> None:
        """max_iterations=1 works as single-shot when LLM finishes immediately."""
        llm = MockLLM([LLMResponse(text="Quick answer")])
        agent = _make_agent(max_iterations=1)

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Be quick",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Quick answer"


# ------------------------------------------------------------------
# ReActLoop — usage tracking
# ------------------------------------------------------------------


class TestReActLoopUsage:
    async def test_accumulates_usage_across_iterations(self) -> None:
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t_u")
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[tc],
                    usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
                ),
                LLMResponse(
                    text="3",
                    usage=UsageStats(input_tokens=200, output_tokens=30, total_tokens=230),
                ),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
        )

        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 80
        assert result.usage.total_tokens == 380


# ------------------------------------------------------------------
# ReActLoop — Clarification action (M4)
# ------------------------------------------------------------------


class TestReActLoopClarification:
    async def test_clarification_returns_waiting_human_input(self) -> None:
        """Clarification action returns WAITING_HUMAN_INPUT status with the question as answer."""
        from dendrite.strategies.base import Strategy

        class ClarifyStrategy(Strategy):
            """Strategy that always returns a Clarification on the first call."""

            def build_messages(self, *, system_prompt, history, tool_defs):
                return [Message(role=Role.SYSTEM, content=system_prompt), *history], None

            def parse_response(self, response):
                return AgentStep(
                    reasoning="I need more info",
                    action=Clarification(question="Which format do you want?"),
                )

            def format_tool_result(self, result):
                return Message(
                    role=Role.TOOL,
                    content=result.payload,
                    name=result.name,
                    call_id=result.call_id,
                )

        llm = MockLLM([LLMResponse(text="doesn't matter")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=ClarifyStrategy(),
            user_input="Do something",
        )

        assert result.status == RunStatus.WAITING_HUMAN_INPUT
        assert result.answer == "Which format do you want?"
        assert isinstance(result.steps[0].action, Clarification)


# ------------------------------------------------------------------
# ReActLoop — non-serializable tool results (C2)
# ------------------------------------------------------------------


class TestReActLoopNonSerializableResults:
    async def test_tool_returning_datetime_doesnt_crash(self) -> None:
        """C2: Tool returning datetime uses default=str fallback."""
        tc = ToolCall(
            name="return_datetime",
            params={},
            provider_tool_call_id="t_dt",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="Got the timestamp"),
            ]
        )
        agent = _make_agent(tools=[return_datetime])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Get time",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Got the timestamp"

    async def test_tool_returning_set_doesnt_crash(self) -> None:
        """C2: Tool returning set uses default=str fallback."""
        tc = ToolCall(
            name="return_set",
            params={},
            provider_tool_call_id="t_set",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="Got the items"),
            ]
        )
        agent = _make_agent(tools=[return_set])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Get items",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Got the items"


# ------------------------------------------------------------------
# ReActLoop — non-server tool guard (H2)
# ------------------------------------------------------------------


class TestReActLoopToolTargetGuard:
    async def test_client_tool_pauses_loop(self) -> None:
        """Sprint 3: Non-server tools pause the loop instead of erroring."""

        @tool(target="client")
        async def read_range(sheet: str) -> str:
            """Read from Excel."""
            return "should never run"

        tc = ToolCall(
            name="read_range",
            params={"sheet": "Sheet1"},
            provider_tool_call_id="t_client",
        )
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent(tools=[read_range])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Read sheet",
        )

        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        pause_state = result.meta["pause_state"]
        assert len(pause_state.pending_tool_calls) == 1
        assert pause_state.pending_tool_calls[0].name == "read_range"
        # Tool should NOT have been executed
        assert llm.calls_made == 1


# ------------------------------------------------------------------
# ReActLoop — cost_usd accumulation (M1-billing)
# ------------------------------------------------------------------


class TestReActLoopCostAccumulation:
    async def test_cost_usd_accumulated_across_iterations(self) -> None:
        """cost_usd should be summed when providers report it."""
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t_c")
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[tc],
                    usage=UsageStats(
                        input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003
                    ),
                ),
                LLMResponse(
                    text="3",
                    usage=UsageStats(
                        input_tokens=200, output_tokens=30, total_tokens=230, cost_usd=0.005
                    ),
                ),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
        )

        assert result.usage.cost_usd == pytest.approx(0.008)

    async def test_cost_usd_stays_none_when_not_reported(self) -> None:
        """If no provider reports cost, total stays None."""
        usage = UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
        llm = MockLLM([LLMResponse(text="hi", usage=usage)])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
        )

        assert result.usage.cost_usd is None

    async def test_cost_usd_partial_reporting(self) -> None:
        """If only some calls report cost, sum only those."""
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t_p")
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[tc],
                    usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
                ),
                LLMResponse(
                    text="3",
                    usage=UsageStats(
                        input_tokens=200, output_tokens=30, total_tokens=230, cost_usd=0.005
                    ),
                ),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
        )

        assert result.usage.cost_usd == pytest.approx(0.005)


# ------------------------------------------------------------------
# ReActLoop — final message persistence (F-01, F-03)
# ------------------------------------------------------------------


class TestReActLoopFinalMessagePersistence:
    async def test_finish_notifies_observer_with_assistant_message(self) -> None:
        """F-01: The final assistant message must be notified to the observer on Finish."""
        from dendrite.loops.base import LoopObserver

        recorded_messages: list[Message] = []

        class RecordingObserver(LoopObserver):
            async def on_message_appended(self, message: Message, iteration: int) -> None:
                recorded_messages.append(message)

            async def on_llm_call_completed(self, response, iteration: int, **kwargs) -> None:
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration: int) -> None:
                pass

        llm = MockLLM([LLMResponse(text="The answer is 42")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Question?",
            observer=RecordingObserver(),
        )

        assert result.status == RunStatus.SUCCESS
        # Should have: user message + final assistant message
        assert len(recorded_messages) == 2
        assert recorded_messages[0].role == Role.USER
        assert recorded_messages[1].role == Role.ASSISTANT
        assert recorded_messages[1].content == "The answer is 42"

    async def test_clarification_notifies_observer_with_assistant_message(self) -> None:
        """F-03: The assistant message must be notified on Clarification too."""
        from dendrite.loops.base import LoopObserver
        from dendrite.strategies.base import Strategy

        recorded_messages: list[Message] = []

        class RecordingObserver(LoopObserver):
            async def on_message_appended(self, message: Message, iteration: int) -> None:
                recorded_messages.append(message)

            async def on_llm_call_completed(self, response, iteration: int, **kwargs) -> None:
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration: int) -> None:
                pass

        class ClarifyStrategy(Strategy):
            def build_messages(self, *, system_prompt, history, tool_defs):
                return [Message(role=Role.SYSTEM, content=system_prompt), *history], None

            def parse_response(self, response):
                return AgentStep(
                    reasoning="Need info",
                    action=Clarification(question="Which one?"),
                )

            def format_tool_result(self, result):
                return Message(
                    role=Role.TOOL, content=result.payload, name=result.name, call_id=result.call_id
                )

        llm = MockLLM([LLMResponse(text="clarifying")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=ClarifyStrategy(),
            user_input="Do it",
            observer=RecordingObserver(),
        )

        assert result.status == RunStatus.WAITING_HUMAN_INPUT
        # Should have: user message + final assistant message
        assert len(recorded_messages) == 2
        assert recorded_messages[1].role == Role.ASSISTANT
        assert recorded_messages[1].content == "clarifying"

    async def test_tool_call_then_finish_persists_all_messages(self) -> None:
        """Complete flow: tool call + finish should persist all messages including final."""
        from dendrite.loops.base import LoopObserver

        recorded_messages: list[Message] = []

        class RecordingObserver(LoopObserver):
            async def on_message_appended(self, message: Message, iteration: int) -> None:
                recorded_messages.append(message)

            async def on_llm_call_completed(self, response, iteration: int, **kwargs) -> None:
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration: int) -> None:
                pass

        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(text="Computing", tool_calls=[tc]),
                LLMResponse(text="The sum is 3"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            observer=RecordingObserver(),
        )

        assert result.status == RunStatus.SUCCESS
        roles = [m.role for m in recorded_messages]
        # user, assistant (tool call), tool result, assistant (final answer)
        assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
        assert recorded_messages[-1].content == "The sum is 3"


# ------------------------------------------------------------------
# ReActLoop — observer warning surfacing (F-04)
# ------------------------------------------------------------------


class TestReActLoopObserverWarnings:
    async def test_observer_failure_surfaces_in_meta(self) -> None:
        """F-04: Observer exceptions should be captured in RunResult.meta, not silently lost."""
        from dendrite.loops.base import LoopObserver

        class FailingObserver(LoopObserver):
            async def on_message_appended(self, message: Message, iteration: int) -> None:
                raise RuntimeError("DB connection lost")

            async def on_llm_call_completed(self, response, iteration: int, **kwargs) -> None:
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration: int) -> None:
                pass

        llm = MockLLM([LLMResponse(text="done")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
            observer=FailingObserver(),
        )

        # Run should still succeed
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "done"
        # But warnings should be surfaced
        assert "observer_warnings" in result.meta
        assert len(result.meta["observer_warnings"]) > 0

    async def test_no_warnings_when_observer_healthy(self) -> None:
        """No observer_warnings key when everything works fine."""
        from dendrite.loops.base import LoopObserver

        class HealthyObserver(LoopObserver):
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kwargs):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

        llm = MockLLM([LLMResponse(text="ok")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
            observer=HealthyObserver(),
        )

        assert result.status == RunStatus.SUCCESS
        assert "observer_warnings" not in result.meta


# ------------------------------------------------------------------
# ReActLoop — tool execution timeout (F-10)
# ------------------------------------------------------------------


class TestReActLoopToolTimeout:
    async def test_slow_tool_times_out(self) -> None:
        """F-10: Tool exceeding timeout_seconds returns an error result."""
        import asyncio

        @tool(timeout_seconds=0.1)
        async def slow_tool() -> str:
            """A tool that takes too long."""
            await asyncio.sleep(10)
            return "never reached"

        tc = ToolCall(name="slow_tool", params={}, provider_tool_call_id="t_slow")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="tool timed out"),
            ]
        )
        agent = _make_agent(tools=[slow_tool])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do slow thing",
        )

        assert result.status == RunStatus.SUCCESS
        # The timeout error should have been fed back to the LLM
        second_call_msgs = llm.call_history[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "timed out" in tool_msgs[0].content


class TestReActLoopDuplicateToolNames:
    async def test_duplicate_tool_names_raise(self) -> None:
        """A-08: Two tools with the same name should raise at loop start."""
        from dendrite.loops.react import _build_tool_lookup
        from dendrite.types import ToolDef

        @tool()
        async def samename(x: int) -> int:
            """One."""
            return x

        @tool()
        async def samename2(x: int) -> int:
            """Two."""
            return x

        # Override the name on the second tool's ToolDef to collide
        samename2.__tool_def__ = ToolDef(  # noqa: B010
            name="samename",
            description="Two.",
            parameters=samename2.__tool_def__.parameters,
        )

        with pytest.raises(ValueError, match="Duplicate tool name"):
            _build_tool_lookup([samename, samename2])


# Need to import UsageStats for the usage test
from dendrite.types import UsageStats  # noqa: E402
