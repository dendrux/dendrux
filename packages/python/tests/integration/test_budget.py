"""Dollar/token budget with child roll-up and hard actions (0.3.0 Track F).

SQLite + Postgres via the ``store`` matrix fixture. Covers: pause/cancel at
the next top-of-iteration checkpoint, both limits independently, the
root-inclusive child rollup (incl. unknown cost), the budget resume rule,
persisted fired keys (no re-fire after resume), running totals on paused
rows, precedence vs cancel / requested pause / approval pause, streaming,
SingleCall degrade and the pre-0.3 store fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.loops.single import SingleCall
from dendrux.runtime.context import get_delegation_context
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
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_run_control_races import Pre03Store

pytestmark = pytest.mark.asyncio

NAME = "Budgeted"


class _CostReportingLLM(ScriptedLLM):
    """Scripted responses carry their own ``cost_usd``, like a provider that reports cost."""

    @property
    def reports_cost(self) -> bool:
        return True


def _usage(tokens: int, cost: float | None) -> UsageStats:
    return UsageStats(
        input_tokens=tokens // 2,
        output_tokens=tokens - tokens // 2,
        total_tokens=tokens,
        cost_usd=cost,
    )


def _call(name: str, pid: str, *, tokens: int = 0, cost: float | None = 0.0) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(name=name, params={}, provider_tool_call_id=pid)],
        usage=_usage(tokens, cost),
    )


def _text(text: str, *, tokens: int = 0, cost: float | None = 0.0) -> LLMResponse:
    return LLMResponse(text=text, usage=_usage(tokens, cost))


def _own_run_id() -> str:
    ctx = get_delegation_context()
    assert ctx is not None
    return ctx.run_id


async def _events(store: Any, run_id: str, event_type: str) -> list[Any]:
    return [e for e in await store.get_run_events(run_id) if e.event_type == event_type]


async def _pause_state(store: Any, run_id: str) -> PauseState:
    data = await store.get_pause_state(run_id)
    assert data is not None
    return PauseState.from_dict(data)


class _Counter:
    def __init__(self) -> None:
        self.n = 0


def _work_tool(counter: _Counter) -> Any:
    @tool()
    async def work() -> str:
        """Do work."""
        counter.n += 1
        return "worked"

    return work


def _agent(
    store: Any,
    responses: list[LLMResponse],
    budget: Budget | None,
    *,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> tuple[Agent, ScriptedLLM]:
    llm = _CostReportingLLM(responses)
    agent = Agent(
        name=NAME,
        provider=llm,
        prompt="P.",
        tools=tools or [],
        state_store=store,
        budget=budget,
        **kwargs,
    )
    return agent, llm


# ---------------------------------------------------------------------------
# Pause / cancel at the next checkpoint
# ---------------------------------------------------------------------------


class TestBudgetPause:
    async def test_pauses_at_next_checkpoint_after_tools_ran(self, store) -> None:
        counter = _Counter()
        agent, llm = _agent(
            store,
            [_call("work", "t1", tokens=600, cost=0.02), _text("done", tokens=100)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(counter)],
        )
        result = await agent.run("go")

        assert result.status == RunStatus.PAUSED
        assert counter.n == 1  # the crossing response's tools ran
        assert llm.calls == 1  # no LLM call after the limit
        assert result.meta["budget_exceeded"] == [{"metric": "tokens", "used": 600, "max": 500}]

        record = await store.get_run(result.run_id)
        assert record.status == "paused"
        pause = await _pause_state(store, result.run_id)
        assert pause.pause_reason == "budget"
        assert pause.budget_exceeded == [{"metric": "tokens", "used": 600, "max": 500}]
        assert pause.pending_tool_calls == []
        assert ("tokens", 1.0) in (pause.budget_fired or [])

        (paused_event,) = await _events(store, result.run_id, "run.paused")
        assert paused_event.data["reason"] == "budget"
        assert paused_event.data["exceeded"] == [{"metric": "tokens", "used": 600, "max": 500}]
        (exceeded,) = await _events(store, result.run_id, "budget.exceeded")
        assert exceeded.data["metric"] == "tokens"
        assert exceeded.data["action"] == "pause"

    async def test_paused_row_carries_running_totals(self, store) -> None:
        """Regression: pause_run wrote no totals, so paused runs read 0 / None."""
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=600, cost=0.25)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")
        record = await store.get_run(result.run_id)
        assert record.status == "paused"
        assert record.total_input_tokens == 300
        assert record.total_output_tokens == 300
        assert record.total_cost_usd == pytest.approx(0.25)

    async def test_waiting_approval_row_carries_running_totals(self, store) -> None:
        @tool()
        async def refund() -> str:
            """Refund."""
            return "refunded"

        agent, _ = _agent(
            store,
            [_call("refund", "t1", tokens=40, cost=0.5)],
            None,
            tools=[refund],
            require_approval=["refund"],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.WAITING_APPROVAL
        record = await store.get_run(result.run_id)
        assert record.total_input_tokens == 20
        assert record.total_output_tokens == 20
        assert record.total_cost_usd == pytest.approx(0.5)

    async def test_finishing_response_completes(self, store) -> None:
        agent, _ = _agent(
            store, [_text("done", tokens=900)], Budget(max_tokens=500, on_exceed="pause")
        )
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        assert len(await _events(store, result.run_id, "budget.exceeded")) == 1

    async def test_cancel_sets_failure_reason_and_distinct_event(self, store) -> None:
        counter = _Counter()
        agent, llm = _agent(
            store,
            [_call("work", "t1", tokens=600), _text("done")],
            Budget(max_tokens=500, on_exceed="cancel"),
            tools=[_work_tool(counter)],
        )
        result = await agent.run("go")

        assert result.status == RunStatus.CANCELLED
        assert counter.n == 1 and llm.calls == 1
        assert result.meta["budget_exceeded"] == [{"metric": "tokens", "used": 600, "max": 500}]
        record = await store.get_run(result.run_id)
        assert record.status == "cancelled"
        assert record.failure_reason == "budget_exceeded"
        assert record.total_input_tokens == 300
        (cancelled,) = await _events(store, result.run_id, "run.cancelled")
        assert cancelled.data["reason"] == "budget_exceeded"
        assert cancelled.data["exceeded"][0]["metric"] == "tokens"

    async def test_warn_default_keeps_running(self, store) -> None:
        counter = _Counter()
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=600), _text("done", tokens=100)],
            Budget(max_tokens=500),
            tools=[_work_tool(counter)],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        (exceeded,) = await _events(store, result.run_id, "budget.exceeded")
        assert exceeded.data["action"] == "warn"


# ---------------------------------------------------------------------------
# Both limits, cost, unpriced calls
# ---------------------------------------------------------------------------


class TestBudgetLimits:
    async def test_cost_limit_independent_of_tokens(self, store) -> None:
        agent, llm = _agent(
            store,
            [
                _call("work", "t1", tokens=10, cost=0.03),
                _call("work", "t2", tokens=10, cost=0.03),
                _text("done"),
            ],
            Budget(max_tokens=100_000, max_cost_usd=0.05, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")

        assert result.status == RunStatus.PAUSED
        assert llm.calls == 2
        (breach,) = result.meta["budget_exceeded"]
        assert breach["metric"] == "cost_usd"
        assert breach["used"] == pytest.approx(0.06)
        assert breach["max"] == 0.05
        thresholds = await _events(store, result.run_id, "budget.threshold")
        assert {e.data["metric"] for e in thresholds} == {"cost_usd"}
        assert sorted(e.data["fraction"] for e in thresholds) == [0.5, 0.75, 0.9]

    async def test_both_limits_fire_once_each_and_resume_needs_both(self, store) -> None:
        budget = Budget(max_tokens=500, max_cost_usd=0.05, on_exceed="pause")
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=600, cost=0.10)],
            budget,
            tools=[_work_tool(_Counter())],
        )
        paused = await agent.run("go")
        assert {b["metric"] for b in paused.meta["budget_exceeded"]} == {"tokens", "cost_usd"}
        exceeded = await _events(store, paused.run_id, "budget.exceeded")
        assert sorted(e.data["metric"] for e in exceeded) == ["cost_usd", "tokens"]

        # Raising only the token limit: cost is still exceeded -> re-pause.
        tokens_only, llm_t = _agent(
            store,
            [_text("never")],
            Budget(max_tokens=10_000, max_cost_usd=0.05, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        again = await tokens_only.resume(paused.run_id)
        assert again.status == RunStatus.PAUSED
        assert llm_t.calls == 0
        assert again.meta["budget_exceeded"] == [
            {"metric": "cost_usd", "used": pytest.approx(0.10), "max": 0.05}
        ]

        both, llm_b = _agent(
            store,
            [_text("done", tokens=10, cost=0.01)],
            Budget(max_tokens=10_000, max_cost_usd=1.0, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        done = await both.resume(paused.run_id)
        assert done.status == RunStatus.SUCCESS
        assert llm_b.calls == 1
        # No re-fire across the two resumes.
        exceeded = await _events(store, paused.run_id, "budget.exceeded")
        assert len(exceeded) == 2

    async def test_unpriced_calls_use_known_cost_and_report_once(self, store) -> None:
        agent, llm = _agent(
            store,
            [
                _call("work", "t1", tokens=10, cost=None),
                _call("work", "t2", tokens=10, cost=0.04),
                _call("work", "t3", tokens=10, cost=None),
                _call("work", "t4", tokens=10, cost=0.02),
                _text("done"),
            ],
            Budget(max_cost_usd=0.05, on_exceed="warn"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert llm.calls == 5
        (unpriced,) = await _events(store, result.run_id, "budget.unpriced")
        assert unpriced.data == {
            "metric": "cost_usd",
            "used": 0.0,
            "max": 0.05,
            "unpriced_calls": 1,
            "known_cost_usd": 0.0,
        }
        (exceeded,) = await _events(store, result.run_id, "budget.exceeded")
        assert exceeded.data["used"] == pytest.approx(0.06)  # priced part only
        assert exceeded.data["reason"] == "budget_exceeded"
        # Run total stays unknown (one call unpriced); persisted as None.
        assert (await store.get_run(result.run_id)).total_cost_usd is None

    @pytest.mark.parametrize("on_exceed", ["pause", "cancel"])
    async def test_unpriced_call_counts_as_exceeded_when_enforcing(self, store, on_exceed) -> None:
        agent, llm = _agent(
            store,
            [_call("work", "t1", tokens=10, cost=None), _text("never")],
            Budget(max_cost_usd=0.05, on_exceed=on_exceed),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")

        assert result.status == (RunStatus.PAUSED if on_exceed == "pause" else RunStatus.CANCELLED)
        assert llm.calls == 1
        assert result.meta["budget_exceeded"] == [{"metric": "cost_usd", "used": 0.0, "max": 0.05}]
        (exceeded,) = await _events(store, result.run_id, "budget.exceeded")
        assert exceeded.data["reason"] == "unpriced_call"
        assert exceeded.data["action"] == on_exceed
        assert len(await _events(store, result.run_id, "budget.unpriced")) == 1


# ---------------------------------------------------------------------------
# Delegated children
# ---------------------------------------------------------------------------


def _delegating_parent(
    store: Any, child_response: LLMResponse, budget: Budget
) -> tuple[Agent, ScriptedLLM, ScriptedLLM]:
    child_llm = _CostReportingLLM([child_response])
    child = Agent(name="Child", provider=child_llm, prompt="C.", state_store=store)

    @tool()
    async def delegate() -> str:
        """Delegate to the child."""
        result = await child.run("sub-task")
        return result.answer or ""

    parent, parent_llm = _agent(
        store,
        [_call("delegate", "t1", tokens=100, cost=0.01), _text("done", tokens=10, cost=0.01)],
        budget,
        tools=[delegate],
    )
    return parent, parent_llm, child_llm


class TestBudgetChildren:
    async def test_child_spend_counts_toward_parent(self, store) -> None:
        parent, parent_llm, child_llm = _delegating_parent(
            store,
            _text("child done", tokens=1000, cost=0.10),
            Budget(max_cost_usd=0.05, on_exceed="pause"),
        )
        result = await parent.run("go")

        assert child_llm.calls == 1
        assert result.status == RunStatus.PAUSED
        assert parent_llm.calls == 1
        (breach,) = result.meta["budget_exceeded"]
        # Root-inclusive rollup, no double count: 0.01 (parent) + 0.10 (child).
        assert breach == {"metric": "cost_usd", "used": pytest.approx(0.11), "max": 0.05}
        runs = await store.list_runs(parent_run_id=result.run_id)
        assert len(runs) == 1

    async def test_include_children_false_counts_own_usage_only(self, store) -> None:
        parent, parent_llm, _ = _delegating_parent(
            store,
            _text("child done", tokens=1000, cost=0.10),
            Budget(max_cost_usd=0.05, on_exceed="pause", include_children=False),
        )
        result = await parent.run("go")
        assert result.status == RunStatus.SUCCESS
        assert parent_llm.calls == 2
        assert await _events(store, result.run_id, "budget.exceeded") == []

    async def test_unknown_child_cost_with_both_limits(self, store) -> None:
        parent, parent_llm, _ = _delegating_parent(
            store,
            _text("child done", tokens=1000, cost=None),
            Budget(max_tokens=500, max_cost_usd=1.0, on_exceed="pause"),
        )
        result = await parent.run("go")

        assert result.status == RunStatus.PAUSED
        assert parent_llm.calls == 1
        assert result.meta["budget_exceeded"] == [
            {"metric": "tokens", "used": 1100, "max": 500},
            {"metric": "cost_usd", "used": pytest.approx(0.01), "max": 1.0},
        ]
        (unpriced,) = await _events(store, result.run_id, "budget.unpriced")
        assert unpriced.data["unpriced_calls"] == 1
        assert unpriced.data["known_cost_usd"] == pytest.approx(0.01)
        # The unknown child spend stands in for the cost limit while enforcing.
        reasons = {
            e.data["metric"]: e.data["reason"]
            for e in await _events(store, result.run_id, "budget.exceeded")
        }
        assert reasons == {"tokens": "budget_exceeded", "cost_usd": "unpriced_call"}

    async def test_warn_budget_reports_child_spend_after_next_response(self, store) -> None:
        parent, _, _ = _delegating_parent(
            store,
            _text("child done", tokens=1000, cost=0.10),
            Budget(max_cost_usd=0.05),
        )
        result = await parent.run("go")
        assert result.status == RunStatus.SUCCESS
        (exceeded,) = await _events(store, result.run_id, "budget.exceeded")
        # Warn checkpoints are inert: seen by the evaluation after call 2.
        assert exceeded.iteration_index == 2
        assert exceeded.data["used"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Resume rule
# ---------------------------------------------------------------------------


class TestBudgetResume:
    async def _paused(self, store: Any) -> str:
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.PAUSED
        return result.run_id

    @pytest.mark.parametrize("limit", [500, 600])
    async def test_resume_without_strictly_higher_limit_repauses(self, store, limit) -> None:
        run_id = await self._paused(store)
        same, llm = _agent(
            store,
            [_text("never")],
            Budget(max_tokens=limit, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await same.submit_resume(run_id)

        assert result.status == RunStatus.PAUSED
        assert llm.calls == 0
        pause = await _pause_state(store, run_id)
        assert pause.pause_reason == "budget"
        assert pause.budget_exceeded == [{"metric": "tokens", "used": 600, "max": limit}]
        assert len(await _events(store, run_id, "run.paused")) == 2
        assert len(await _events(store, run_id, "budget.exceeded")) == 1

    async def test_resume_with_higher_limit_continues(self, store) -> None:
        run_id = await self._paused(store)
        higher, llm = _agent(
            store,
            [_text("done", tokens=100)],
            Budget(max_tokens=601, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await higher.resume(run_id)
        assert result.status == RunStatus.SUCCESS
        assert llm.calls == 1
        record = await store.get_run(run_id)
        assert record.total_input_tokens + record.total_output_tokens == 700

    async def test_unpriced_pause_resumes_without_changing_the_policy(self, store) -> None:
        budget = Budget(max_cost_usd=5.0, on_exceed="pause", count_cache_tokens=False)
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=10, cost=None)],
            budget,
            tools=[_work_tool(_Counter())],
        )
        paused = await agent.run("go")
        assert paused.status == RunStatus.PAUSED
        state = await _pause_state(store, paused.run_id)
        assert state.budget_unpriced_calls == 1

        same, llm = _agent(
            store,
            [_text("done", tokens=10, cost=0.01)],
            budget,
            tools=[_work_tool(_Counter())],
        )
        result = await same.resume(paused.run_id)
        assert result.status == RunStatus.SUCCESS
        assert llm.calls == 1

    async def test_a_new_unpriced_call_after_resume_pauses_again(self, store) -> None:
        budget = Budget(max_cost_usd=5.0, on_exceed="pause", count_cache_tokens=False)
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=10, cost=None)],
            budget,
            tools=[_work_tool(_Counter())],
        )
        paused = await agent.run("go")
        again, llm = _agent(
            store,
            [_call("work", "t2", tokens=10, cost=None), _text("never", cost=0.0)],
            budget,
            tools=[_work_tool(_Counter())],
        )
        result = await again.resume(paused.run_id)
        assert result.status == RunStatus.PAUSED
        assert llm.calls == 1
        state = await _pause_state(store, paused.run_id)
        assert state.budget_unpriced_calls == 2

    async def test_resume_stream_repauses(self, store) -> None:
        run_id = await self._paused(store)
        same, llm = _agent(
            store,
            [_text("never")],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        events = [e async for e in same.resume_stream(run_id)]
        assert events[-1].type == RunEventType.RUN_PAUSED
        assert events[-1].run_result.status == RunStatus.PAUSED
        assert llm.calls == 0

    async def test_fired_thresholds_do_not_refire_after_approval_resume(self, store) -> None:
        """Regression: fired thresholds lived in loop memory and re-fired on resume."""

        @tool()
        async def refund() -> str:
            """Refund."""
            return "refunded"

        agent, _ = _agent(
            store,
            [_call("refund", "t1", tokens=600), _text("done", tokens=100)],
            Budget(max_tokens=1000),
            tools=[refund],
            require_approval=["refund"],
        )
        waiting = await agent.run("go")
        assert waiting.status == RunStatus.WAITING_APPROVAL
        assert (await _pause_state(store, waiting.run_id)).budget_fired == [("tokens", 0.5)]

        done = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert done.status == RunStatus.SUCCESS
        fractions = [
            e.data["fraction"] for e in await _events(store, waiting.run_id, "budget.threshold")
        ]
        assert fractions == [0.5]


# ---------------------------------------------------------------------------
# Precedence: cancel > requested pause > budget pause; approval first
# ---------------------------------------------------------------------------


class TestBudgetPrecedence:
    async def test_cancel_wins_over_budget(self, store) -> None:
        @tool()
        async def stop() -> str:
            """Request cancellation of this run."""
            await store.request_cancel(_own_run_id())
            return "ok"

        agent, _ = _agent(
            store,
            [_call("stop", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[stop],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        record = await store.get_run(result.run_id)
        assert record.failure_reason is None
        (cancelled,) = await _events(store, result.run_id, "run.cancelled")
        assert cancelled.data["reason"] == "cancel_requested"

    async def test_requested_pause_wins_then_budget_on_resume(self, store) -> None:
        holder: dict[str, Agent] = {}

        @tool()
        async def hold() -> str:
            """Ask this run to pause."""
            await holder["agent"].request_pause(_own_run_id())
            return "ok"

        agent, llm = _agent(
            store,
            [_call("hold", "t1", tokens=600), _text("never")],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[hold],
        )
        holder["agent"] = agent
        first = await agent.run("go")
        assert first.status == RunStatus.PAUSED
        assert (await _pause_state(store, first.run_id)).pause_reason == "requested"

        second = await agent.submit_resume(first.run_id)
        assert second.status == RunStatus.PAUSED
        assert (await _pause_state(store, first.run_id)).pause_reason == "budget"
        assert llm.calls == 1
        reasons = [e.data["reason"] for e in await _events(store, first.run_id, "run.paused")]
        assert reasons == ["requested", "budget"]

    async def test_approval_pause_first_then_budget_after_approved_tools(self, store) -> None:
        ran = _Counter()

        @tool()
        async def refund() -> str:
            """Refund."""
            ran.n += 1
            return "refunded"

        agent, llm = _agent(
            store,
            [_call("refund", "t1", tokens=600), _text("never")],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[refund],
            require_approval=["refund"],
        )
        waiting = await agent.run("go")
        assert waiting.status == RunStatus.WAITING_APPROVAL

        result = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert ran.n == 1
        assert result.status == RunStatus.PAUSED
        assert llm.calls == 1
        assert (await _pause_state(store, waiting.run_id)).pause_reason == "budget"


# ---------------------------------------------------------------------------
# Streaming, SingleCall, pre-0.3 stores
# ---------------------------------------------------------------------------


class TestBudgetStreamingAndDegrade:
    async def test_stream_pause(self, store) -> None:
        agent, llm = _agent(
            store,
            [_call("work", "t1", tokens=600), _text("never")],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        async with agent.stream("go") as stream:
            events = [e async for e in stream]
        assert events[-1].type == RunEventType.RUN_PAUSED
        assert events[-1].run_result.status == RunStatus.PAUSED
        assert llm.calls == 1
        assert (await _pause_state(store, stream.run_id)).pause_reason == "budget"
        record = await store.get_run(stream.run_id)
        assert record.total_input_tokens == 300

    async def test_stream_cancel(self, store) -> None:
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=600), _text("never")],
            Budget(max_tokens=500, on_exceed="cancel"),
            tools=[_work_tool(_Counter())],
        )
        async with agent.stream("go") as stream:
            events = [e async for e in stream]
        assert events[-1].type == RunEventType.RUN_CANCELLED
        record = await store.get_run(stream.run_id)
        assert record.status == "cancelled"
        assert record.failure_reason == "budget_exceeded"

    @pytest.mark.parametrize("on_exceed", ["pause", "cancel"])
    async def test_single_call_degrades_to_warn(self, store, on_exceed) -> None:
        llm = _CostReportingLLM([_text("classified", tokens=900, cost=0.2)])
        agent = Agent(
            provider=llm,
            prompt="Classify.",
            loop=SingleCall(),
            state_store=store,
            budget=Budget(max_tokens=500, max_cost_usd=0.1, on_exceed=on_exceed),
        )
        result = await agent.run("x")
        assert result.status == RunStatus.SUCCESS
        exceeded = await _events(store, result.run_id, "budget.exceeded")
        assert sorted(e.data["metric"] for e in exceeded) == ["cost_usd", "tokens"]
        assert {e.data["action"] for e in exceeded} == {"warn"}


class TestBudgetPre03Store:
    async def test_budget_pause_and_resume_on_legacy_store(self, store) -> None:
        legacy = Pre03Store(store)
        agent, _ = _agent(
            legacy,
            [_call("work", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        paused = await agent.run("go")
        assert paused.status == RunStatus.PAUSED
        assert (await _pause_state(store, paused.run_id)).pause_reason == "budget"

        same, llm_same = _agent(
            legacy,
            [_text("never")],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        assert (await same.resume(paused.run_id)).status == RunStatus.PAUSED
        assert llm_same.calls == 0

        higher, _ = _agent(
            legacy,
            [_text("done", tokens=10)],
            Budget(max_tokens=10_000, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        assert (await higher.resume(paused.run_id)).status == RunStatus.SUCCESS

    async def test_budget_cancel_on_legacy_store(self, store) -> None:
        class OldSignatureStore(Pre03Store):
            """finalize_run_if_status_in with the 0.2 signature (no failure_reason)."""

            async def finalize_run_if_status_in(
                self,
                run_id: str,
                *,
                status: str,
                allowed_current_statuses: list[str],
                answer: str | None = None,
                error: str | None = None,
                iteration_count: int | None = None,
                total_usage: UsageStats | None = None,
                pii_mapping: dict[str, str] | None = None,
            ) -> bool:
                return await self._inner.finalize_run_if_status_in(
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                )

        legacy = OldSignatureStore(store)
        agent, _ = _agent(
            legacy,
            [_call("work", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="cancel"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        record = await store.get_run(result.run_id)
        assert record.status == "cancelled"
        # A pre-0.3 CAS cannot store failure_reason; the event still says why.
        assert record.failure_reason is None
        (cancelled,) = await _events(store, result.run_id, "run.cancelled")
        assert cancelled.data["reason"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# Fix round: resume gate, Track-E-era stores, legacy accounting, read counts
# ---------------------------------------------------------------------------


async def _token_paused(store: Any) -> str:
    """A run budget-paused on tokens (600 used / 500 max) with $0.10 spent."""
    agent, _ = _agent(
        store,
        [_call("work", "t1", tokens=600, cost=0.10)],
        Budget(max_tokens=500, on_exceed="pause"),
        tools=[_work_tool(_Counter())],
    )
    result = await agent.run("go")
    assert result.status == RunStatus.PAUSED
    return result.run_id


async def _resume(agent: Agent, run_id: str, *, stream: bool) -> Any:
    if not stream:
        return await agent.resume(run_id)
    events = [e async for e in agent.resume_stream(run_id)]
    return events[-1].run_result


class TestBudgetResumeGate:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("on_exceed", ["warn", "cancel"])
    async def test_gate_repauses_regardless_of_new_action(self, store, on_exceed, stream) -> None:
        run_id = await _token_paused(store)
        resumer, llm = _agent(
            store,
            [_text("never")],
            Budget(max_tokens=500, on_exceed=on_exceed),
            tools=[_work_tool(_Counter())],
        )
        result = await _resume(resumer, run_id, stream=stream)
        assert result.status == RunStatus.PAUSED
        assert llm.calls == 0
        pause = await _pause_state(store, run_id)
        assert pause.pause_reason == "budget"
        assert [b["metric"] for b in pause.budget_exceeded or []] == ["tokens"]
        assert (await store.get_run(run_id)).failure_reason is None

    @pytest.mark.parametrize("stream", [False, True])
    async def test_gate_checks_only_previously_exceeded_metrics(self, store, stream) -> None:
        run_id = await _token_paused(store)
        # Tokens raised above live use; a newly added cost limit is already
        # below known spend but was not a previously exceeded metric.
        resumer, llm = _agent(
            store,
            [_text("done", tokens=10, cost=0.0)],
            Budget(max_tokens=10_000, max_cost_usd=0.05, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        result = await _resume(resumer, run_id, stream=stream)
        assert llm.calls == 1
        assert result.status == RunStatus.SUCCESS


def _track_e_store(inner: Any) -> Any:
    """A store with Track E's atomic signatures (no total_usage / failure_reason)."""

    class TrackEStore:
        def __getattr__(self, name: str) -> Any:
            return getattr(inner, name)

        async def pause_with_event(
            self,
            run_id: str,
            *,
            status: str,
            pause_data: dict[str, Any],
            event_data: dict[str, Any] | None = None,
            iteration_count: int | None = None,
            pii_mapping: dict[str, str] | None = None,
        ) -> int | None:
            return await inner.pause_with_event(
                run_id,
                status=status,
                pause_data=pause_data,
                event_data=event_data,
                iteration_count=iteration_count,
                pii_mapping=pii_mapping,
            )

        async def finalize_with_event(
            self,
            run_id: str,
            *,
            status: str,
            allowed_current_statuses: list[str],
            event_type: str,
            event_data: dict[str, Any] | None = None,
            answer: str | None = None,
            error: str | None = None,
            iteration_count: int | None = None,
            total_usage: UsageStats | None = None,
            pii_mapping: dict[str, str] | None = None,
        ) -> int | None:
            return await inner.finalize_with_event(
                run_id,
                status=status,
                allowed_current_statuses=allowed_current_statuses,
                event_type=event_type,
                event_data=event_data,
                answer=answer,
                error=error,
                iteration_count=iteration_count,
                total_usage=total_usage,
                pii_mapping=pii_mapping,
            )

    return TrackEStore()


class TestTrackEStoreCompat:
    async def test_budget_pause_and_approval_pause(self, store) -> None:
        compat = _track_e_store(store)
        agent, _ = _agent(
            compat,
            [_call("work", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        assert (await agent.run("go")).status == RunStatus.PAUSED

        @tool()
        async def refund() -> str:
            """Refund."""
            return "refunded"

        approver, _ = _agent(
            compat, [_call("refund", "t1")], None, tools=[refund], require_approval=["refund"]
        )
        assert (await approver.run("go")).status == RunStatus.WAITING_APPROVAL

    async def test_budget_cancel_keeps_distinct_reason(self, store) -> None:
        agent, _ = _agent(
            _track_e_store(store),
            [_call("work", "t1", tokens=600)],
            Budget(max_tokens=500, on_exceed="cancel"),
            tools=[_work_tool(_Counter())],
        )
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        (cancelled,) = await _events(store, result.run_id, "run.cancelled")
        assert cancelled.data["reason"] == "budget_exceeded"


class TestLegacyAccounting:
    async def test_mixed_priced_spend_survives_pause_without_rollups(self, store) -> None:
        legacy = Pre03Store(store)

        @tool()
        async def refund() -> str:
            """Refund."""
            return "refunded"

        budget = Budget(max_tokens=10_000, max_cost_usd=0.05, on_exceed="pause")
        agent, _ = _agent(
            legacy,
            [
                _call("work", "t1", tokens=10, cost=0.04),
                _call("refund", "t2", tokens=10, cost=None),
                _call("work", "t3", tokens=10, cost=0.02),
                _text("never"),
            ],
            budget,
            tools=[_work_tool(_Counter()), refund],
            require_approval=["refund"],
        )
        waiting = await agent.run("go")
        assert waiting.status == RunStatus.WAITING_APPROVAL
        result = await agent.submit_approval(
            waiting.run_id, approved=True, expected_pause_id=waiting.pause_id
        )
        assert result.status == RunStatus.PAUSED
        # The unpriced refund call stops the run; the priced part is carried.
        assert result.meta["budget_exceeded"] == [
            {"metric": "cost_usd", "used": pytest.approx(0.04), "max": 0.05}
        ]
        state = await _pause_state(legacy, result.run_id)
        assert state.budget_known_cost_usd == pytest.approx(0.04)
        assert state.budget_unpriced_calls == 1


def _count_reads(store: Any) -> dict[str, int]:
    counts = {"subtree_usage": 0, "run_usage": 0}
    for name in counts:
        original = getattr(store, name)

        async def wrapped(run_id: str, _o: Any = original, _n: str = name) -> Any:
            counts[_n] += 1
            return await _o(run_id)

        setattr(store, name, wrapped)
    return counts


class TestBudgetReads:
    async def test_warn_checkpoints_are_inert(self, store) -> None:
        counts = _count_reads(store)
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=10), _call("work", "t2", tokens=10), _text("done")],
            Budget(max_tokens=10_000),
            tools=[_work_tool(_Counter())],
        )
        assert (await agent.run("go")).status == RunStatus.SUCCESS
        assert counts["subtree_usage"] == 3  # one per LLM response only

    async def test_enforcing_own_usage_reuses_post_response_read(self, store) -> None:
        counts = _count_reads(store)
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=10), _call("work", "t2", tokens=10), _text("done")],
            Budget(max_tokens=10_000, on_exceed="pause", include_children=False),
            tools=[_work_tool(_Counter())],
        )
        assert (await agent.run("go")).status == RunStatus.SUCCESS
        assert counts["run_usage"] == 3

    async def test_enforcing_children_skips_fresh_first_checkpoint(self, store) -> None:
        counts = _count_reads(store)
        agent, _ = _agent(
            store,
            [_call("work", "t1", tokens=10), _text("done")],
            Budget(max_tokens=10_000, on_exceed="pause"),
            tools=[_work_tool(_Counter())],
        )
        assert (await agent.run("go")).status == RunStatus.SUCCESS
        # 2 post-response reads + 1 checkpoint read before call 2 (children may have spent).
        assert counts["subtree_usage"] == 3
