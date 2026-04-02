"""Tests for runner persistence wiring — state_store, observer, finalize_run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dendrux import Agent, run, tool
from dendrux.llm.mock import MockLLM
from dendrux.types import (
    LLMResponse,
    RunStatus,
    ToolCall,
    UsageStats,
)

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ------------------------------------------------------------------
# Mock StateStore — records all calls
# ------------------------------------------------------------------


@dataclass
class RecordingStateStore:
    """Fake StateStore that records create_run/finalize_run calls."""

    created_runs: list[dict[str, Any]] = field(default_factory=list)
    finalized_runs: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_saved: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any) -> None:
        self.created_runs.append({"run_id": run_id, "agent_name": agent_name, **kwargs})

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        kwargs.pop("expected_current_status", None)
        self.finalized_runs.append({"run_id": run_id, **kwargs})
        return True

    async def save_trace(self, run_id: str, role: str, content: str, **kwargs: Any) -> None:
        self.traces.append({"run_id": run_id, "role": role, "content": content, **kwargs})

    async def save_tool_call(self, run_id: str, **kwargs: Any) -> None:
        self.tool_calls_saved.append({"run_id": run_id, **kwargs})

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        self.usages.append({"run_id": run_id, **kwargs})

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self._events.setdefault(run_id, []).append(kwargs)

    async def get_run_events(self, run_id: str) -> list[Any]:
        @dataclass
        class _Event:
            sequence_index: int
            event_type: str = ""
            iteration_index: int = 0
            correlation_id: str | None = None
            data: dict[str, Any] | None = None

        return [
            _Event(
                sequence_index=e.get("sequence_index", 0),
                event_type=e.get("event_type", ""),
                iteration_index=e.get("iteration_index", 0),
                correlation_id=e.get("correlation_id"),
                data=e.get("data"),
            )
            for e in self._events.get(run_id, [])
        ]


# ------------------------------------------------------------------
# Runner creates and finalizes runs
# ------------------------------------------------------------------


class TestRunnerPersistence:
    async def test_creates_run_before_loop(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="hi")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="hello", state_store=store)

        assert len(store.created_runs) == 1
        created = store.created_runs[0]
        assert created["agent_name"] == "Agent"
        assert created["input_data"] == {"input": "hello"}
        assert created["model"] == "mock"

    async def test_finalizes_on_success(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        assert len(store.finalized_runs) == 1
        finalized = store.finalized_runs[0]
        assert finalized["status"] == "success"
        assert finalized["answer"] == "done"
        assert finalized["iteration_count"] == 1

    async def test_finalizes_on_max_iterations(self) -> None:
        store = RecordingStateStore()
        tc = ToolCall(name="add", params={"a": 1, "b": 1}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = Agent(prompt="Test.", tools=[add], max_iterations=1)

        await run(agent, provider=llm, user_input="go", state_store=store)

        finalized = store.finalized_runs[0]
        assert finalized["status"] == "max_iterations"

    async def test_finalizes_on_error(self) -> None:
        store = RecordingStateStore()

        class ExplodingLLM(MockLLM):
            async def complete(self, messages, tools=None, **kwargs):
                raise RuntimeError("LLM exploded")

        llm = ExplodingLLM([])
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(RuntimeError, match="LLM exploded"):
            await run(agent, provider=llm, user_input="go", state_store=store)

        assert len(store.finalized_runs) == 1
        finalized = store.finalized_runs[0]
        assert finalized["status"] == "error"
        assert "LLM exploded" in finalized["error"]
        # F-05: Error path should pass total_usage=None (not zeroed UsageStats)
        # so that finalize_run doesn't overwrite summary columns with zeros.
        # Per-call token_usage rows are the source of truth for errored runs.
        assert finalized["total_usage"] is None

    async def test_run_id_consistent_across_create_and_finalize(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        assert store.created_runs[0]["run_id"] == store.finalized_runs[0]["run_id"]

    async def test_passes_tenant_id(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store, tenant_id="t_123")

        assert store.created_runs[0]["tenant_id"] == "t_123"

    async def test_passes_metadata(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        meta = {"thread_id": "th_1", "user_id": "u_42"}
        await run(agent, provider=llm, user_input="go", state_store=store, metadata=meta)

        persisted_meta = store.created_runs[0]["meta"]
        assert persisted_meta["thread_id"] == "th_1"
        assert persisted_meta["user_id"] == "u_42"
        assert persisted_meta["dendrux.loop"] == "ReActLoop"

    async def test_passes_strategy_name(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="ok")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        assert store.created_runs[0]["strategy"] == "NativeToolCalling"

    async def test_no_store_means_no_persistence(self) -> None:
        """Without state_store, run() works the same as Sprint 1."""
        llm = MockLLM([LLMResponse(text="hi")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(agent, provider=llm, user_input="hello")

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "hi"


# ------------------------------------------------------------------
# Observer records traces and tool calls
# ------------------------------------------------------------------


class TestRunnerObserverWiring:
    async def test_traces_recorded(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM([LLMResponse(text="42")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="what?", state_store=store)

        # Should have at least user message trace
        roles = [t["role"] for t in store.traces]
        assert "user" in roles

    async def test_tool_call_traces_recorded(self) -> None:
        store = RecordingStateStore()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="3")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="1+2?", state_store=store)

        # Should have: user, assistant (with tool_call), tool (result)
        roles = [t["role"] for t in store.traces]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    async def test_tool_calls_saved(self) -> None:
        store = RecordingStateStore()
        tc = ToolCall(name="add", params={"a": 5, "b": 3}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="8")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="5+3?", state_store=store)

        assert len(store.tool_calls_saved) == 1
        assert store.tool_calls_saved[0]["tool_name"] == "add"

    async def test_usage_saved(self) -> None:
        store = RecordingStateStore()
        llm = MockLLM(
            [
                LLMResponse(
                    text="hi",
                    usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
                )
            ]
        )
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="go", state_store=store)

        assert len(store.usages) == 1
        assert store.usages[0]["usage"].input_tokens == 100

    async def test_finalize_includes_total_usage(self) -> None:
        store = RecordingStateStore()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
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
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="1+2?", state_store=store)

        finalized = store.finalized_runs[0]
        assert finalized["total_usage"].input_tokens == 300
        assert finalized["total_usage"].output_tokens == 80
