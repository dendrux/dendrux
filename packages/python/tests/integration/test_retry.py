"""Integration tests for agent.retry() across the backend matrix.

The ``engine`` / ``store`` fixtures live in
``tests/integration/conftest.py`` and parametrize across SQLite and Postgres.
"""

from __future__ import annotations

import pytest

from dendrux.errors import RunNotFoundError
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestRetryBasic:
    """Tests for runner.retry() core behavior."""

    async def test_retry_creates_new_run_with_lineage(self, store):
        """retry() creates a new run linked via retry_of_run_id."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry, run

        mock = MockLLM(
            [
                LLMResponse(
                    text=None, tool_calls=[ToolCall(name="greet", params={"name": "World"})]
                ),
                LLMResponse(text="Hello World!"),
            ]
        )

        from dendrux import tool

        @tool()
        async def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        agent = Agent(provider=mock, prompt="test", tools=[greet])

        # First run succeeds
        result1 = await run(
            agent,
            provider=mock,
            user_input="say hello",
            state_store=store,
        )
        assert result1.status == RunStatus.SUCCESS

        # Retry from the first run
        mock2 = MockLLM([LLMResponse(text="Retried hello!")])
        result2 = await retry(
            result1.run_id,
            agent=agent,
            provider=mock2,
            state_store=store,
        )

        assert result2.status == RunStatus.SUCCESS
        assert result2.run_id != result1.run_id

        # Check lineage
        retry_run = await store.get_run(result2.run_id)
        assert retry_run.retry_of_run_id == result1.run_id

    async def test_retry_seeds_history_from_traces(self, store):
        """retry() passes initial_history to the loop from traces."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry, run

        mock = MockLLM([LLMResponse(text="original answer")])
        agent = Agent(provider=mock, prompt="test")

        result1 = await run(
            agent,
            provider=mock,
            user_input="what is 2+2?",
            state_store=store,
        )

        # Verify traces exist
        traces = await store.get_traces(result1.run_id)
        assert len(traces) >= 2  # at least user + assistant

        # Retry — the new LLM gets the history
        mock2 = MockLLM([LLMResponse(text="retried answer")])
        result2 = await retry(
            result1.run_id,
            agent=agent,
            provider=mock2,
            state_store=store,
        )

        assert result2.status == RunStatus.SUCCESS
        assert result2.answer == "retried answer"

    async def test_retry_rejects_running_run(self, store):
        """retry() raises ValueError for non-terminal runs."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry

        mock = MockLLM([LLMResponse(text="done")])
        agent = Agent(provider=mock, prompt="test")

        # Create a running run (don't finalize)
        await store.create_run("run1", "TestAgent")

        with pytest.raises(ValueError, match="only terminal runs"):
            await retry("run1", agent=agent, provider=mock, state_store=store)

    async def test_retry_rejects_single_call_source(self, store):
        """retry() raises ValueError when source run used SingleCall."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry

        mock = MockLLM([LLMResponse(text="done")])
        agent = Agent(provider=mock, prompt="test")

        # Create a terminal run that was originally SingleCall
        await store.create_run(
            "run1",
            "TestAgent",
            meta={"dendrux.loop": "SingleCall"},
        )
        await store.finalize_run(
            "run1",
            status="error",
            error="boom",
            expected_current_status="running",
        )

        with pytest.raises(ValueError, match="SingleCall"):
            await retry("run1", agent=agent, provider=mock, state_store=store)

    async def test_retry_rejects_nonexistent_run(self, store):
        """retry() raises RunNotFoundError (a LookupError) for an unknown run_id (B13)."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry

        mock = MockLLM([LLMResponse(text="done")])
        agent = Agent(provider=mock, prompt="test")

        with pytest.raises(RunNotFoundError, match="not found"):
            await retry("nonexistent", agent=agent, provider=mock, state_store=store)

    async def test_retry_rejects_no_traces(self, store):
        """retry() raises ValueError if the source run has no traces."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry

        mock = MockLLM([LLMResponse(text="done")])
        agent = Agent(provider=mock, prompt="test")

        # Create a terminal run with no traces (directly via store)
        await store.create_run("run1", "TestAgent")
        await store.finalize_run(
            "run1",
            status="error",
            error="boom",
            expected_current_status="running",
        )

        with pytest.raises(ValueError, match="no persisted traces"):
            await retry("run1", agent=agent, provider=mock, state_store=store)

    async def test_retry_emits_started_event_with_retry_of(self, store):
        """The retry run's run.started event includes retry_of."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry, run

        mock = MockLLM([LLMResponse(text="original")])
        agent = Agent(provider=mock, prompt="test")

        result1 = await run(
            agent,
            provider=mock,
            user_input="hello",
            state_store=store,
        )

        mock2 = MockLLM([LLMResponse(text="retried")])
        result2 = await retry(
            result1.run_id,
            agent=agent,
            provider=mock2,
            state_store=store,
        )

        events = await store.get_run_events(result2.run_id)
        started = [e for e in events if e.event_type == "run.started"]
        assert len(started) == 1
        assert started[0].data["retry_of"] == result1.run_id

    async def test_retry_metadata_stored(self, store):
        """retry_of_run_id and dendrux.retry_of are persisted in meta."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import retry, run

        mock = MockLLM([LLMResponse(text="original")])
        agent = Agent(provider=mock, prompt="test")

        result1 = await run(
            agent,
            provider=mock,
            user_input="hello",
            state_store=store,
        )

        mock2 = MockLLM([LLMResponse(text="retried")])
        result2 = await retry(
            result1.run_id,
            agent=agent,
            provider=mock2,
            state_store=store,
        )

        retry_run = await store.get_run(result2.run_id)
        assert retry_run.retry_of_run_id == result1.run_id
        assert retry_run.meta["dendrux.retry_of"] == result1.run_id


class TestAgentRetryMethod:
    """Tests for Agent.retry() public API."""

    async def test_agent_retry_without_persistence_raises(self):
        """agent.retry() without persistence raises ValueError."""
        from unittest.mock import AsyncMock

        from dendrux.agent import Agent

        agent = Agent(provider=AsyncMock(), prompt="test")

        with pytest.raises(ValueError, match="retry.*requires persistence"):
            await agent.retry("run1")

    async def test_agent_retry_delegates_to_runner(self, store):
        """agent.retry() delegates to runner.retry()."""
        from dendrux.agent import Agent
        from dendrux.runtime.runner import run

        mock = MockLLM([LLMResponse(text="original")])
        agent = Agent(provider=mock, prompt="test", state_store=store)

        result1 = await run(
            agent,
            provider=mock,
            user_input="hello",
            state_store=store,
        )

        mock2 = MockLLM([LLMResponse(text="retried")])
        agent2 = Agent(provider=mock2, prompt="test", state_store=store)
        result2 = await agent2.retry(result1.run_id)

        assert result2.status == RunStatus.SUCCESS
        assert result2.run_id != result1.run_id

        retry_run = await store.get_run(result2.run_id)
        assert retry_run.retry_of_run_id == result1.run_id


_BLOCKS = [{"type": "thinking", "thinking": "plan", "signature": "sig-1"}]


class TestRetryReasoningBlocks:
    """Reasoning replay blocks round-trip through react_traces into retry()."""

    async def _source_run(self, store, model: str = "model-a"):
        from dendrux import tool
        from dendrux.agent import Agent
        from dendrux.runtime.runner import run

        @tool()
        async def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        mock = MockLLM(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(name="greet", params={"name": "W"})],
                    reasoning_blocks=_BLOCKS,
                ),
                LLMResponse(text="done"),
            ],
            model=model,
        )
        agent = Agent(provider=mock, prompt="test", tools=[greet])
        result = await run(agent, provider=mock, user_input="hi", state_store=store)
        assert result.status == RunStatus.SUCCESS
        return agent, result

    async def test_trace_meta_carries_blocks_and_model(self, store):
        from dendrux.types import REASONING_MODEL_META_KEY

        _, result = await self._source_run(store)
        traces = await store.get_traces(result.run_id)
        tool_turn = next(t for t in traces if t.role == "assistant" and "tool_calls" in t.meta)
        assert tool_turn.meta["reasoning_blocks"] == _BLOCKS
        assert tool_turn.meta[REASONING_MODEL_META_KEY] == "model-a"
        final = [t for t in traces if t.role == "assistant"][-1]
        assert "reasoning_blocks" not in final.meta

    async def test_retry_restores_blocks_for_same_model(self, store):
        from dendrux.runtime.runner import retry
        from dendrux.types import REASONING_MODEL_META_KEY, Role

        agent, result = await self._source_run(store)
        mock2 = MockLLM([LLMResponse(text="again")], model="model-a")
        await retry(result.run_id, agent=agent, provider=mock2, state_store=store)

        seen = mock2.call_history[0]["messages"]
        tool_turn = next(m for m in seen if m.role == Role.ASSISTANT and m.tool_calls)
        assert tool_turn.reasoning_blocks == _BLOCKS
        assert tool_turn.meta[REASONING_MODEL_META_KEY] == "model-a"

    async def test_retry_drops_blocks_for_other_model(self, store):
        from dendrux.runtime.runner import retry
        from dendrux.types import Role

        agent, result = await self._source_run(store)
        mock2 = MockLLM([LLMResponse(text="again")], model="model-b")
        await retry(result.run_id, agent=agent, provider=mock2, state_store=store)

        seen = mock2.call_history[0]["messages"]
        tool_turn = next(m for m in seen if m.role == Role.ASSISTANT and m.tool_calls)
        assert tool_turn.reasoning_blocks is None

    async def test_retry_honors_per_call_model_override(self, store):
        from dendrux.runtime.runner import retry
        from dendrux.types import Role

        agent, result = await self._source_run(store)
        mock2 = MockLLM([LLMResponse(text="again")], model="model-b")
        await retry(result.run_id, agent=agent, provider=mock2, state_store=store, model="model-a")

        seen = mock2.call_history[0]["messages"]
        tool_turn = next(m for m in seen if m.role == Role.ASSISTANT and m.tool_calls)
        assert tool_turn.reasoning_blocks == _BLOCKS
