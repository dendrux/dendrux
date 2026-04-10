"""Tests for Wave 3 — budget (governance v1, advisory)."""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import Budget, LLMResponse, RunEventType, RunStatus, ToolCall, UsageStats

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


def _make_agent(**overrides) -> Agent:
    defaults = {
        "prompt": "You are a helpful assistant.",
        "tools": [search],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


def _response(text: str, total_tokens: int, tool_calls=None) -> LLMResponse:
    """Helper to create LLMResponse with specific token usage."""
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=UsageStats(
            input_tokens=total_tokens // 2,
            output_tokens=total_tokens - total_tokens // 2,
            total_tokens=total_tokens,
        ),
    )


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestBudgetConstruction:
    def test_budget_none_is_default(self):
        """Agent without budget= has budget=None."""
        agent = _make_agent()
        assert agent.budget is None

    def test_budget_accepted_on_react(self):
        """Budget is accepted on ReAct agents."""
        agent = _make_agent(budget=Budget(max_tokens=10_000))
        assert agent.budget is not None
        assert agent.budget.max_tokens == 10_000

    def test_budget_accepted_on_single_call(self):
        """Budget is accepted on SingleCall (observational)."""
        agent = Agent(
            prompt="Test",
            loop=SingleCall(),
            budget=Budget(max_tokens=5_000),
        )
        assert agent.budget is not None

    def test_budget_default_warn_at(self):
        """Default warn_at is (0.5, 0.75, 0.9)."""
        b = Budget(max_tokens=10_000)
        assert b.warn_at == (0.5, 0.75, 0.9)

    def test_budget_custom_warn_at(self):
        """Custom warn_at tuple is accepted."""
        b = Budget(max_tokens=10_000, warn_at=(0.8,))
        assert b.warn_at == (0.8,)

    def test_budget_zero_max_tokens_raises(self):
        """max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(max_tokens=0)

    def test_budget_negative_max_tokens_raises(self):
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(max_tokens=-100)

    def test_budget_warn_at_out_of_range_raises(self):
        """warn_at fraction outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Budget(max_tokens=10_000, warn_at=(1.5,))

    def test_budget_warn_at_zero_raises(self):
        """warn_at=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Budget(max_tokens=10_000, warn_at=(0.0,))

    def test_budget_warn_at_one_raises(self):
        """warn_at=1.0 raises ValueError (exclusive)."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Budget(max_tokens=10_000, warn_at=(1.0,))

    def test_budget_warn_at_negative_raises(self):
        """Negative warn_at raises ValueError (blocks sentinel collision)."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Budget(max_tokens=10_000, warn_at=(-1.0,))


# ------------------------------------------------------------------
# Threshold events
# ------------------------------------------------------------------


class TestBudgetThresholds:
    async def test_threshold_fires_once(self):
        """Each threshold fires exactly once, even across multiple iterations."""
        # Two iterations: 6000 tokens each → crosses 50% after first, 100% after second
        tc = ToolCall(name="search", params={"query": "test"}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                _response("Searching...", 6000, tool_calls=[tc]),
                _response("Done.", 6000),
            ]
        )
        agent = _make_agent(budget=Budget(max_tokens=10_000))

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append({"event_type": event_type, "data": data})

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search for test",
            recorder=SpyRecorder(),
        )

        threshold_events = [e for e in governance_events if e["event_type"] == "budget.threshold"]
        exceeded_events = [e for e in governance_events if e["event_type"] == "budget.exceeded"]

        # 6000/10000 = 60% → crosses 0.5 threshold
        # 12000/10000 = 120% → crosses 0.75 and 0.9 thresholds
        assert len(threshold_events) == 3  # 0.5, 0.75, 0.9
        fractions = [e["data"]["fraction"] for e in threshold_events]
        assert sorted(fractions) == [0.5, 0.75, 0.9]

        # Exceeded fires once
        assert len(exceeded_events) == 1
        assert exceeded_events[0]["data"]["used"] == 12000
        assert exceeded_events[0]["data"]["max"] == 10_000

    async def test_no_threshold_when_under_budget(self):
        """No events fire when usage stays under all thresholds."""
        llm = MockLLM([_response("Done.", 100)])
        agent = _make_agent(
            tools=[],
            budget=Budget(max_tokens=10_000),
        )

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append({"event_type": event_type})

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
            recorder=SpyRecorder(),
        )

        assert len(governance_events) == 0

    async def test_custom_warn_at(self):
        """Custom warn_at fractions are respected."""
        llm = MockLLM([_response("Done.", 8500)])
        agent = _make_agent(
            tools=[],
            budget=Budget(max_tokens=10_000, warn_at=(0.8,)),
        )

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append({"event_type": event_type, "data": data})

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
            recorder=SpyRecorder(),
        )

        threshold_events = [e for e in governance_events if e["event_type"] == "budget.threshold"]
        assert len(threshold_events) == 1
        assert threshold_events[0]["data"]["fraction"] == 0.8

    async def test_single_call_fires_budget_events(self):
        """SingleCall loop fires budget events after its one LLM call."""
        llm = MockLLM([_response("Classified.", 8000)])
        agent = Agent(
            prompt="Classify the input.",
            loop=SingleCall(),
            budget=Budget(max_tokens=10_000, warn_at=(0.5,)),
        )

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append({"event_type": event_type, "data": data})

        result = await SingleCall().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="I love this!",
            recorder=SpyRecorder(),
        )

        assert result.status == RunStatus.SUCCESS
        threshold_events = [e for e in governance_events if e["event_type"] == "budget.threshold"]
        assert len(threshold_events) == 1
        assert threshold_events[0]["data"]["fraction"] == 0.5


# ------------------------------------------------------------------
# Advisory behavior (no pause)
# ------------------------------------------------------------------


class TestBudgetAdvisory:
    async def test_exceeded_does_not_pause(self):
        """Budget exceeded does NOT pause — run completes normally."""
        llm = MockLLM([_response("Done.", 20_000)])
        agent = _make_agent(
            tools=[],
            budget=Budget(max_tokens=10_000),
        )

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
        )

        # Run completes SUCCESS, not WAITING_BUDGET
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Done."

    async def test_exceeded_with_tools_still_executes(self):
        """Tools still execute even when budget is exceeded."""
        tc = ToolCall(name="search", params={"query": "q"}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                _response("Searching...", 15_000, tool_calls=[tc]),
                _response("Found it.", 5000),
            ]
        )
        agent = _make_agent(budget=Budget(max_tokens=10_000))

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Search",
        )

        assert result.status == RunStatus.SUCCESS
        # Tool executed — result in LLM history
        second_call = llm.call_history[1]
        tool_msgs = [m for m in second_call["messages"] if m.role.value == "tool"]
        assert len(tool_msgs) == 1

    async def test_no_budget_is_noop(self):
        """Agent without budget= behaves identically to before."""
        llm = MockLLM([_response("Done.", 5000)])
        agent = _make_agent(tools=[])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
        )

        assert result.status == RunStatus.SUCCESS


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------


class TestBudgetStreaming:
    async def test_threshold_fires_in_stream(self):
        """Budget threshold events are visible during streaming."""
        llm = MockLLM([_response("Done.", 8000)])
        agent = _make_agent(
            tools=[],
            budget=Budget(max_tokens=10_000, warn_at=(0.5,)),
        )

        governance_events: list[dict] = []

        class SpyRecorder:
            async def on_message_appended(self, message, iteration):
                pass

            async def on_llm_call_completed(self, response, iteration, **kw):
                pass

            async def on_tool_completed(self, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(self, event_type, iteration, data, correlation_id=None):
                governance_events.append({"event_type": event_type})

        events = []
        async for event in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
            recorder=SpyRecorder(),
        ):
            events.append(event)

        # Run completes
        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1

        # Threshold fired via recorder
        assert any(e["event_type"] == "budget.threshold" for e in governance_events)

    async def test_exceeded_in_stream_still_completes(self):
        """Budget exceeded in stream does not prevent completion."""
        llm = MockLLM([_response("Done.", 20_000)])
        agent = _make_agent(
            tools=[],
            budget=Budget(max_tokens=10_000),
        )

        events = []
        async for event in ReActLoop().run_stream(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hello",
        ):
            events.append(event)

        completed = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(completed) == 1
        assert completed[0].run_result.status == RunStatus.SUCCESS
