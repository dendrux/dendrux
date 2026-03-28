"""Tests for the agent runner — the top-level API."""

from __future__ import annotations

from dendrite import Agent, run, tool
from dendrite.llm.mock import MockLLM
from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.types import (
    LLMResponse,
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


# ------------------------------------------------------------------
# run() API
# ------------------------------------------------------------------


class TestRun:
    async def test_simple_text_answer(self) -> None:
        """Agent answers without tools."""
        llm = MockLLM([LLMResponse(text="Hello!")])
        agent = Agent(prompt="Be friendly.", tools=[add])

        result = await run(agent, provider=llm, user_input="Hi")

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Hello!"

    async def test_tool_call_and_finish(self) -> None:
        """Agent calls a tool then finishes."""
        tc = ToolCall(name="add", params={"a": 3, "b": 4}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="7"),
            ]
        )
        agent = Agent(prompt="Calculate.", tools=[add])

        result = await run(agent, provider=llm, user_input="3+4?")

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "7"
        assert result.iteration_count == 2

    async def test_uses_default_strategy_and_loop(self) -> None:
        """Defaults to NativeToolCalling and ReActLoop."""
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(agent, provider=llm, user_input="Hi")

        assert result.status == RunStatus.SUCCESS

    async def test_custom_strategy(self) -> None:
        """Accepts explicit strategy."""
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(
            agent,
            provider=llm,
            user_input="Hi",
            strategy=NativeToolCalling(),
        )

        assert result.status == RunStatus.SUCCESS

    async def test_custom_loop(self) -> None:
        """Accepts explicit loop."""
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(
            agent,
            provider=llm,
            user_input="Hi",
            loop=ReActLoop(),
        )

        assert result.status == RunStatus.SUCCESS

    async def test_max_iterations_respected(self) -> None:
        """Agent's max_iterations is respected."""
        tc = ToolCall(name="add", params={"a": 1, "b": 1}, provider_tool_call_id="t_loop")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(tool_calls=[tc]),
            ]
        )
        agent = Agent(prompt="Loop.", tools=[add], max_iterations=2)

        result = await run(agent, provider=llm, user_input="Go")

        assert result.status == RunStatus.MAX_ITERATIONS

    async def test_imports_from_top_level(self) -> None:
        """Agent, run, and tool are importable from dendrite."""
        import dendrite

        assert hasattr(dendrite, "Agent")
        assert hasattr(dendrite, "run")
        assert hasattr(dendrite, "tool")
