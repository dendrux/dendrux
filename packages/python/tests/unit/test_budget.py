"""Budget: advisory events (Wave 3) and 0.3.0 dollar limit, hard actions, fired keys."""

from __future__ import annotations

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops._budget import BudgetCarry, BudgetTracker
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.runtime.runner import _accepts_kwarg, _budget_carry_kwargs
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import (
    Budget,
    LLMResponse,
    PauseState,
    RunEventType,
    RunStatus,
    ToolCall,
    UsageStats,
)

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

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
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

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
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

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
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

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
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

        class SpyRecorder(BaseRecorder):
            async def on_message_appended(self, run_id, message, iteration):
                pass

            async def on_llm_call_completed(self, run_id, response, iteration, **kw):
                pass

            async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
                pass

            async def on_governance_event(
                self, run_id, event_type, iteration, data, correlation_id=None
            ):
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


# ------------------------------------------------------------------
# 0.3.0: dollar limit, hard actions, persisted fired keys
# ------------------------------------------------------------------


class _GovSpy(BaseRecorder):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def on_message_appended(self, run_id, message, iteration):
        pass

    async def on_llm_call_completed(self, run_id, response, iteration, **kw):
        pass

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
        pass

    async def on_governance_event(self, run_id, event_type, iteration, data, correlation_id=None):
        self.events.append((event_type, data))

    def of(self, event_type: str) -> list[dict]:
        return [d for t, d in self.events if t == event_type]


def _priced(text: str | None, tokens: int, cost: float | None, tool_calls=None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=UsageStats(
            input_tokens=tokens // 2,
            output_tokens=tokens - tokens // 2,
            total_tokens=tokens,
            cost_usd=cost,
        ),
    )


def _search_call(pid: str = "t1") -> list[ToolCall]:
    return [ToolCall(name="search", params={"query": "q"}, provider_tool_call_id=pid)]


class TestBudgetV2Construction:
    def test_positional_max_tokens_still_works(self):
        b = Budget(10_000)
        assert (b.max_tokens, b.max_cost_usd, b.on_exceed, b.include_children) == (
            10_000,
            None,
            "warn",
            True,
        )

    def test_cost_only(self):
        assert Budget(max_cost_usd=1.5).max_tokens is None

    def test_needs_a_limit(self):
        with pytest.raises(ValueError, match="max_tokens and/or max_cost_usd"):
            Budget()

    @pytest.mark.parametrize("cost", [0, -1.0, float("inf"), float("nan"), True])
    def test_bad_cost(self, cost):
        with pytest.raises(ValueError, match="max_cost_usd must be > 0"):
            Budget(max_cost_usd=cost)

    @pytest.mark.parametrize("tokens", [True, 1.5])
    def test_bad_token_type(self, tokens):
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            Budget(max_tokens=tokens)

    def test_bad_action(self):
        with pytest.raises(ValueError, match="on_exceed"):
            Budget(max_tokens=10, on_exceed="stop")  # type: ignore[arg-type]

    def test_bad_include_children(self):
        with pytest.raises(ValueError, match="include_children"):
            Budget(max_tokens=10, include_children=1)  # type: ignore[arg-type]


class TestPauseStateBudgetFields:
    def _pause(self, **kw) -> PauseState:
        return PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
            **kw,
        )

    def test_round_trip(self):
        pause = self._pause(
            pause_reason="budget",
            budget_fired=[("tokens", 0.5), ("cost_usd", 1.0)],
            budget_unpriced_reported=True,
            budget_exceeded=[{"metric": "cost_usd", "used": 2.0, "max": 1.0}],
        )
        back = PauseState.from_dict(pause.to_dict())
        assert back.budget_fired == [("tokens", 0.5), ("cost_usd", 1.0)]
        assert back.budget_unpriced_reported is True
        assert back.budget_exceeded == [{"metric": "cost_usd", "used": 2.0, "max": 1.0}]
        assert BudgetCarry.from_pause(back).fired == (("tokens", 0.5), ("cost_usd", 1.0))

    def test_defaults_add_no_keys(self):
        d = self._pause().to_dict()
        assert not {"budget_fired", "budget_unpriced_reported", "budget_exceeded"} & set(d)
        back = PauseState.from_dict(d)
        assert back.budget_fired is None and back.budget_unpriced_reported is False


class TestBudgetActionsInLoop:
    async def test_pause_waits_for_next_checkpoint(self):
        llm = MockLLM([_priced("s", 600, 0.0, _search_call()), _priced("never", 1, 0.0)])
        agent = _make_agent(budget=Budget(max_tokens=500, on_exceed="pause"))
        result = await ReActLoop().run(
            agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
        )
        assert result.status == RunStatus.PAUSED
        assert len(llm.call_history) == 1
        pause = result.meta["pause_state"]
        assert pause.pause_reason == "budget"
        assert pause.iteration == 1
        assert pause.history[-1].role.value == "tool"  # tool result appended first
        assert pause.budget_exceeded == [{"metric": "tokens", "used": 600, "max": 500}]

    async def test_cancel(self):
        llm = MockLLM([_priced("s", 600, 0.0, _search_call()), _priced("never", 1, 0.0)])
        spy = _GovSpy()
        agent = _make_agent(budget=Budget(max_tokens=500, on_exceed="cancel"))
        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=spy,
        )
        assert result.status == RunStatus.CANCELLED
        assert result.meta["budget_exceeded"][0]["metric"] == "tokens"
        assert spy.of("budget.exceeded")[0]["action"] == "cancel"

    async def test_stream_pause_and_cancel(self):
        for action, terminal in (
            ("pause", RunEventType.RUN_PAUSED),
            ("cancel", RunEventType.RUN_CANCELLED),
        ):
            llm = MockLLM([_priced("s", 600, 0.0, _search_call()), _priced("never", 1, 0.0)])
            agent = _make_agent(budget=Budget(max_tokens=500, on_exceed=action))
            events = [
                e
                async for e in ReActLoop().run_stream(
                    agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
                )
            ]
            assert events[-1].type == terminal
            assert len(llm.call_history) == 1

    async def test_cost_limit_in_memory_with_unpriced_call(self):
        llm = MockLLM(
            [
                _priced("a", 10, None, _search_call("t1")),
                _priced("b", 10, 0.06, _search_call("t2")),
                _priced("never", 1, 0.0),
            ]
        )
        spy = _GovSpy()
        agent = _make_agent(budget=Budget(max_cost_usd=0.05, on_exceed="pause"))
        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=spy,
        )
        assert result.status == RunStatus.PAUSED
        assert result.meta["budget_exceeded"] == [{"metric": "cost_usd", "used": 0.06, "max": 0.05}]
        assert spy.of("budget.unpriced") == [
            {
                "metric": "cost_usd",
                "used": 0.0,
                "max": 0.05,
                "unpriced_calls": 1,
                "known_cost_usd": 0.0,
            }
        ]
        assert result.meta["pause_state"].budget_unpriced_reported is True

    async def test_events_carry_metric(self):
        llm = MockLLM([_priced("Done.", 900, 0.9)])
        spy = _GovSpy()
        agent = _make_agent(tools=[], budget=Budget(max_tokens=1000, max_cost_usd=1.0))
        await ReActLoop().run(
            agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go", recorder=spy
        )
        keys = sorted((d["metric"], d["fraction"]) for d in spy.of("budget.threshold"))
        assert keys == [(m, f) for m in ("cost_usd", "tokens") for f in (0.5, 0.75, 0.9)]
        assert all({"metric", "used", "max"} <= set(d) for d in spy.of("budget.threshold"))

    async def test_carry_prevents_refire(self):
        llm = MockLLM([_priced("Done.", 600, 0.0)])
        spy = _GovSpy()
        agent = _make_agent(tools=[], budget=Budget(max_tokens=1000))
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=spy,
            budget_carry=BudgetCarry(fired=(("tokens", 0.5),)),
        )
        assert spy.of("budget.threshold") == []

    async def test_single_call_stream_degrades(self):
        llm = MockLLM([_priced("Classified.", 900, 0.0)])
        spy = _GovSpy()
        agent = Agent(
            prompt="Classify.",
            loop=SingleCall(),
            budget=Budget(max_tokens=500, on_exceed="cancel"),
        )
        events = [
            e
            async for e in SingleCall().run_stream(
                agent=agent,
                provider=llm,
                strategy=NativeToolCalling(),
                user_input="x",
                recorder=spy,
            )
        ]
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert spy.of("budget.exceeded")[0]["action"] == "warn"


class TestBudgetTrackerSources:
    async def test_store_without_rollups_uses_own_usage(self, caplog):
        class OldStore:
            pass

        usage = UsageStats(total_tokens=700, cost_usd=0.2)
        tracker = BudgetTracker(
            Budget(max_tokens=500, max_cost_usd=1.0, on_exceed="pause"),
            run_id="r",
            usage=usage,
            recorder=BaseRecorder(),
            notifier=None,
            warnings=[],
            state_store=OldStore(),  # type: ignore[arg-type]
        )
        assert "include_children ignored" in caplog.text
        action, breaches = await tracker.checkpoint(2)
        assert action == "pause"
        assert [b.metric for b in breaches] == ["tokens"]

    async def test_failed_live_read_falls_back(self):
        class BrokenStore:
            async def subtree_usage(self, run_id):
                raise RuntimeError("db down")

        tracker = BudgetTracker(
            Budget(max_tokens=500, on_exceed="pause"),
            run_id="r",
            usage=UsageStats(total_tokens=700),
            recorder=BaseRecorder(),
            notifier=None,
            warnings=[],
            state_store=BrokenStore(),  # type: ignore[arg-type]
        )
        _, breaches = await tracker.checkpoint(2)
        assert [b.metric for b in breaches] == ["tokens"]

    async def test_restored_unknown_cost_counts_as_unpriced(self):
        tracker = BudgetTracker(
            Budget(max_cost_usd=1.0),
            run_id="r",
            usage=UsageStats(total_tokens=10, cost_usd=None, cost_unknown=True),
            recorder=None,
            notifier=None,
            warnings=[],
        )
        spy_events = await tracker.evaluate(1)
        assert spy_events == []
        assert tracker.unpriced_reported is True

    async def test_no_budget_is_inert(self):
        tracker = BudgetTracker(
            None, run_id="r", usage=UsageStats(), recorder=None, notifier=None, warnings=[]
        )
        assert await tracker.checkpoint(1) == ("warn", [])
        assert await tracker.evaluate(1) == []
        assert tracker.action == "warn"


class TestRunnerBudgetHelpers:
    def test_carry_only_for_loops_that_accept_it(self):
        pause = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
            budget_fired=[("tokens", 0.5)],
        )
        assert _budget_carry_kwargs(ReActLoop().run, pause) == {
            "budget_carry": BudgetCarry(fired=(("tokens", 0.5),))
        }

        async def legacy_run(*, agent, provider, strategy, user_input, **kwargs):
            return None

        async def old_run(*, agent, provider, strategy, user_input, state_store=None):
            return None

        assert _budget_carry_kwargs(legacy_run, pause) == {}
        assert _budget_carry_kwargs(old_run, pause) == {}

    def test_accepts_kwarg(self):
        def explicit(*, total_usage=None):
            return None

        def var_kw(**kwargs):
            return None

        def neither(*, other=None):
            return None

        assert _accepts_kwarg(explicit, "total_usage")
        assert _accepts_kwarg(var_kw, "total_usage")
        assert not _accepts_kwarg(neither, "total_usage")
        assert not _accepts_kwarg(object(), "total_usage")


class TestBudgetCarryFixRound:
    def _pause(self, **kw) -> PauseState:
        return PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
            **kw,
        )

    def test_accounting_round_trip_and_gate_only_for_budget_pauses(self):
        exceeded = [{"metric": "cost_usd", "used": 0.2, "max": 0.1}]
        pause = PauseState.from_dict(
            self._pause(
                pause_reason="budget",
                budget_exceeded=exceeded,
                budget_known_cost_usd=0.04,
                budget_unpriced_calls=2,
            ).to_dict()
        )
        carry = BudgetCarry.from_pause(pause)
        assert (carry.known_cost_usd, carry.unpriced_calls, carry.gate) == (0.04, 2, ("cost_usd",))
        requested = self._pause(pause_reason="requested", budget_exceeded=exceeded)
        assert BudgetCarry.from_pause(requested).gate == ()

    async def test_carried_subtotal_seeds_fallback_accounting(self):
        tracker = BudgetTracker(
            Budget(max_cost_usd=0.05, on_exceed="pause"),
            run_id="r",
            usage=UsageStats(total_tokens=20, cost_usd=None, cost_unknown=True),
            recorder=None,
            notifier=None,
            warnings=[],
            carry=BudgetCarry(known_cost_usd=0.04, unpriced_calls=1),
        )
        tracker.add_response(UsageStats(total_tokens=10, cost_usd=0.02))
        breaches = await tracker.evaluate(3)
        assert [(b.metric, b.used) for b in breaches] == [("cost_usd", 0.06)]
        assert tracker.pause_fields()["budget_known_cost_usd"] == 0.06
        assert tracker.pause_fields()["budget_unpriced_calls"] == 1

    @pytest.mark.parametrize(
        "budget", [None, Budget(max_tokens=10, on_exceed="pause")], ids=["none", "no-cost-limit"]
    )
    async def test_gate_passes_without_a_limit_on_the_metric(self, budget):
        tracker = BudgetTracker(
            budget,
            run_id="r",
            usage=UsageStats(total_tokens=5, cost_usd=1.0),
            recorder=None,
            notifier=None,
            warnings=[],
            carry=BudgetCarry(gate=("cost_usd",)),
        )
        assert (await tracker.checkpoint(2))[1] == []
