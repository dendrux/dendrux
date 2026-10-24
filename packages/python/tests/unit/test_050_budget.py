"""0.5.0 T3: budget configuration (B4), cache-token budgets (B17) and unpriced calls."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import BudgetConfigurationError
from dendrux.llm.mock import MockLLM
from dendrux.loops._budget import BudgetCarry, BudgetTracker
from dendrux.loops._helpers import price_response
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.pricing import ModelPricing, PriceTable, check_budget_config, resolve_priced_model
from dendrux.runtime.state import UsageRollup
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import Budget, LLMResponse, RunStatus, ToolCall, UsageStats

_RATE = ModelPricing(input=1.0, output=1.0)
_OTHER_RATE = ModelPricing(input=10.0, output=10.0)


class _CostReportingLLM(MockLLM):
    @property
    def reports_cost(self) -> bool:
        return True


def _agent(
    *,
    model: str = "mock",
    pricing: PriceTable | None = None,
    budget: Budget | None = None,
    provider: MockLLM | None = None,
) -> Agent:
    return Agent(
        provider=provider or MockLLM([], model=model),
        prompt="p",
        pricing=pricing,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# B4: BudgetConfigurationError
# ---------------------------------------------------------------------------


class TestBudgetConfigurationAtConstruction:
    def test_dollar_budget_without_pricing_source_raises(self) -> None:
        with pytest.raises(BudgetConfigurationError) as info:
            _agent(budget=Budget(max_cost_usd=1.0))
        assert info.value.model == "mock"
        assert info.value.reason == "unpriced_model"
        assert isinstance(info.value, ValueError)

    def test_table_without_the_model_raises(self) -> None:
        with pytest.raises(BudgetConfigurationError):
            _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"other": _RATE}))

    def test_exact_entry_passes(self) -> None:
        _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"mock": _RATE}))

    def test_glob_entry_passes(self) -> None:
        _agent(
            model="gpt-5-mini",
            budget=Budget(max_cost_usd=1.0),
            pricing=PriceTable({"gpt-5*": _RATE}),
        )

    def test_dated_snapshot_id_needs_a_glob(self) -> None:
        dated = "gpt-5-2025-08-07"
        _agent(
            model=dated,
            budget=Budget(max_cost_usd=1.0),
            pricing=PriceTable({"gpt-5-*": _RATE}),
        )
        with pytest.raises(BudgetConfigurationError):
            _agent(
                model=dated,
                budget=Budget(max_cost_usd=1.0),
                pricing=PriceTable({"gpt-5": _RATE}),
            )

    def test_provider_that_reports_cost_needs_no_table(self) -> None:
        _agent(budget=Budget(max_cost_usd=1.0), provider=_CostReportingLLM([]))

    def test_token_only_budget_needs_no_pricing(self) -> None:
        _agent(budget=Budget(max_tokens=1000, count_cache_tokens=True))

    def test_no_budget_needs_no_pricing(self) -> None:
        _agent()

    def test_agent_without_provider_is_not_checked(self) -> None:
        Agent(prompt="p", budget=Budget(max_cost_usd=1.0))

    def test_error_names_the_model_not_the_budget_value(self) -> None:
        with pytest.raises(BudgetConfigurationError, match="PriceTable") as info:
            _agent(model="m-x", budget=Budget(max_cost_usd=123.0))
        assert "m-x" in str(info.value)


class TestBudgetConfigurationAtRunStart:
    async def test_per_call_model_override_is_checked(self) -> None:
        agent = _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"mock": _RATE}))
        with pytest.raises(BudgetConfigurationError) as info:
            await agent.run("hi", model="unlisted")
        assert info.value.model == "unlisted"

    def test_stream_checks_the_override_too(self) -> None:
        agent = _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"mock": _RATE}))
        with pytest.raises(BudgetConfigurationError):
            agent.stream("hi", model="unlisted")

    async def test_priced_override_is_accepted(self) -> None:
        table = PriceTable({"mock": _RATE, "other": _OTHER_RATE})
        agent = _agent(budget=Budget(max_cost_usd=1.0), pricing=table)
        check_budget_config(agent, model="other")

    async def test_override_falls_back_to_provider_model_when_absent(self) -> None:
        agent = _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"mock": _RATE}))
        check_budget_config(agent)
        check_budget_config(agent, model=None)

    def test_provider_reporting_cost_covers_any_override(self) -> None:
        agent = _agent(budget=Budget(max_cost_usd=1.0), provider=_CostReportingLLM([]))
        check_budget_config(agent, model="anything")


# ---------------------------------------------------------------------------
# B4: pricing resolves response.model, then the requested model
# ---------------------------------------------------------------------------


def _response(model: str | None) -> LLMResponse:
    return LLMResponse(
        text="x",
        usage=UsageStats(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000),
        model=model,
    )


class TestPricingFallsBackToTheRequestedModel:
    def test_dated_snapshot_prices_through_the_requested_alias(self) -> None:
        table = PriceTable({"gpt-5": _RATE})
        priced = price_response(_response("gpt-5-2025-08-07"), pricing=table, default_model="gpt-5")
        assert priced.usage.cost_usd == pytest.approx(1.0)
        assert priced.usage.cost_source == "table"

    def test_response_model_wins_when_both_are_priced(self) -> None:
        table = PriceTable({"gpt-5": _RATE, "gpt-5-2025-08-07": _OTHER_RATE})
        priced = price_response(_response("gpt-5-2025-08-07"), pricing=table, default_model="gpt-5")
        assert priced.usage.cost_usd == pytest.approx(10.0)

    def test_per_call_model_is_the_fallback_not_the_provider_default(self) -> None:
        table = PriceTable({"mock": _RATE, "other": _OTHER_RATE})
        priced = price_response(
            _response("other-2025"), pricing=table, default_model="mock", model="other"
        )
        assert priced.usage.cost_usd == pytest.approx(10.0)

    def test_unpriced_per_call_model_does_not_borrow_the_default(self) -> None:
        table = PriceTable({"mock": _RATE})
        priced = price_response(_response(None), pricing=table, default_model="mock", model="x")
        assert priced.usage.cost_usd is None

    def test_default_model_prices_when_nothing_else_is_known(self) -> None:
        table = PriceTable({"mock": _RATE})
        priced = price_response(_response(None), pricing=table, default_model="mock")
        assert priced.usage.cost_usd == pytest.approx(1.0)

    def test_provider_reported_cost_is_kept(self) -> None:
        response = _response("gpt-5-2025-08-07")
        response.usage = UsageStats(
            input_tokens=10, output_tokens=0, total_tokens=10, cost_usd=0.5, cost_source="provider"
        )
        priced = price_response(
            response, pricing=PriceTable({"gpt-5": _RATE}), default_model="gpt-5"
        )
        assert priced.usage.cost_usd == 0.5

    def test_exact_requested_entry_beats_a_catch_all_glob_on_the_vendor_id(self) -> None:
        table = PriceTable({"claude-*": _RATE, "claude-opus-4": _OTHER_RATE})
        priced = price_response(
            _response("claude-opus-4-20250514"), pricing=table, default_model="claude-opus-4"
        )
        assert priced.usage.cost_usd == pytest.approx(10.0)

    def test_lookup_order_exact_then_glob_response_before_requested(self) -> None:
        wide = ModelPricing(input=2.0, output=2.0)
        narrow = ModelPricing(input=3.0, output=3.0)
        table = PriceTable({"a-*": wide, "b-*": narrow})
        # Neither is exact: the glob on response.model wins over the glob on the request.
        assert resolve_priced_model(table, "a-1", "b-1") == "a-1"
        assert resolve_priced_model(table, "zzz", "b-1") == "b-1"
        exact = PriceTable({"a-*": wide, "b-1": narrow})
        # An exact entry for the request beats a glob on the response.
        assert resolve_priced_model(exact, "a-1", "b-1") == "b-1"
        both = PriceTable({"a-1": wide, "b-1": narrow})
        assert resolve_priced_model(both, "a-1", "b-1") == "a-1"

    def test_cost_unknown_call_stays_unpriced(self) -> None:
        response = _response("mock")
        response.usage = UsageStats(
            input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000, cost_unknown=True
        )
        priced = price_response(response, pricing=PriceTable({"mock": _RATE}), default_model="mock")
        assert priced.usage.cost_usd is None
        assert priced.usage.cost_source is None

    def test_resolve_priced_model(self) -> None:
        table = PriceTable({"a": _RATE})
        assert resolve_priced_model(table, "b", "a") == "a"
        assert resolve_priced_model(table, "b", None, "c") == "b"
        assert resolve_priced_model(table, None, None) is None
        assert resolve_priced_model(None, "b", "a") == "b"


# ---------------------------------------------------------------------------
# B4 at run time: an unpriced call under max_cost_usd
# ---------------------------------------------------------------------------


class _Spy(BaseRecorder):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_message_appended(self, run_id, message, iteration):
        pass

    async def on_llm_call_completed(self, run_id, response, iteration, **kw):
        pass

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration):
        pass

    async def on_governance_event(self, run_id, event_type, iteration, data, correlation_id=None):
        self.events.append((event_type, data))

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [d for t, d in self.events if t == event_type]


def _tracker(budget: Budget, rollup: UsageRollup, **kw: Any) -> BudgetTracker:
    class Store:
        async def subtree_usage(self, run_id: str) -> UsageRollup:
            return rollup

        run_usage = subtree_usage

    return BudgetTracker(
        budget,
        run_id="r",
        usage=UsageStats(),
        recorder=_Spy(),
        notifier=None,
        warnings=[],
        state_store=Store(),  # type: ignore[arg-type]
        **kw,
    )


class TestUnpricedCallUnderCostLimit:
    @pytest.mark.parametrize("action", ["pause", "cancel"])
    async def test_counts_as_exceeded_when_enforcing(self, action: str) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed=action),  # type: ignore[arg-type]
            UsageRollup(known_cost_usd=0.1, unpriced_calls=1, llm_calls=2),
        )
        breaches = await tracker.evaluate(1)
        assert [(b.metric, b.used, b.max) for b in breaches] == [("cost_usd", 0.1, 1.0)]

    async def test_warn_only_reports(self) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed="warn"),
            UsageRollup(known_cost_usd=0.1, unpriced_calls=1, llm_calls=2),
        )
        assert await tracker.evaluate(1) == []
        assert tracker.unpriced_reported is True

    async def test_fully_priced_usage_is_unaffected(self) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed="cancel"),
            UsageRollup(known_cost_usd=0.1, unpriced_calls=0, llm_calls=2),
        )
        assert await tracker.evaluate(1) == []

    async def test_token_only_budget_ignores_unpriced_calls(self) -> None:
        tracker = _tracker(
            Budget(max_tokens=10_000, on_exceed="cancel", count_cache_tokens=False),
            UsageRollup(unpriced_calls=3, input_tokens=5),
        )
        assert await tracker.evaluate(1) == []

    @pytest.mark.parametrize("action", ["pause", "cancel"])
    async def test_resume_gate_repauses_while_a_call_is_unpriced(self, action: str) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed=action),  # type: ignore[arg-type]
            UsageRollup(known_cost_usd=0.1, unpriced_calls=1),
            carry=BudgetCarry(gate=("cost_usd",)),
        )
        _, breaches = await tracker.checkpoint(2)
        assert [b.metric for b in breaches] == ["cost_usd"]

    @pytest.mark.parametrize("action", ["pause", "cancel"])
    async def test_resume_accepts_the_unpriced_calls_that_caused_the_pause(
        self, action: str
    ) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed=action),  # type: ignore[arg-type]
            UsageRollup(known_cost_usd=0.1, unpriced_calls=1),
            carry=BudgetCarry(gate=("cost_usd",), unpriced_calls=1),
        )
        _, breaches = await tracker.checkpoint(2)
        assert breaches == []
        assert await tracker.evaluate(2) == []

    async def test_a_new_unpriced_call_after_resume_stops_the_run_again(self) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed="pause"),
            UsageRollup(known_cost_usd=0.1, unpriced_calls=2),
            carry=BudgetCarry(gate=("cost_usd",), unpriced_calls=1),
        )
        _, breaches = await tracker.checkpoint(2)
        assert [b.metric for b in breaches] == ["cost_usd"]

    async def test_pause_fields_record_the_measured_unpriced_count(self) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed="pause"),
            UsageRollup(known_cost_usd=0.1, unpriced_calls=3),
        )
        await tracker.evaluate(1)
        assert tracker.pause_fields()["budget_unpriced_calls"] == 3

    async def test_resume_gate_passes_when_priced_and_below_the_limit(self) -> None:
        tracker = _tracker(
            Budget(max_cost_usd=1.0, on_exceed="pause"),
            UsageRollup(known_cost_usd=0.1),
            carry=BudgetCarry(gate=("cost_usd",)),
        )
        _, breaches = await tracker.checkpoint(2)
        assert breaches == []


# ---------------------------------------------------------------------------
# B17: count_cache_tokens
# ---------------------------------------------------------------------------

_ROLLUP = UsageRollup(
    input_tokens=100, output_tokens=50, cache_read_tokens=400, cache_write_tokens=100
)
_CACHE_USAGE = UsageStats(
    input_tokens=100,
    output_tokens=50,
    total_tokens=150,
    cache_read_input_tokens=400,
    cache_creation_input_tokens=100,
)


def _memory_tracker(budget: Budget) -> BudgetTracker:
    return BudgetTracker(
        budget,
        run_id="r",
        usage=_CACHE_USAGE,
        recorder=None,
        notifier=None,
        warnings=[],
    )


class TestCountCacheTokens:
    @pytest.mark.parametrize(("choice", "expected"), [(True, 650), (False, 150), (None, 150)])
    async def test_store_rollup_formula(self, choice: bool | None, expected: int) -> None:
        tracker = _tracker(Budget(max_tokens=10_000, count_cache_tokens=choice), _ROLLUP)
        assert (await tracker._measure()).tokens == expected

    @pytest.mark.parametrize(("choice", "expected"), [(True, 650), (False, 150), (None, 150)])
    async def test_in_memory_formula(self, choice: bool | None, expected: int) -> None:
        tracker = _memory_tracker(Budget(max_tokens=10_000, count_cache_tokens=choice))
        assert (await tracker._measure()).tokens == expected

    async def test_in_memory_formula_does_not_need_total_tokens(self) -> None:
        tracker = BudgetTracker(
            Budget(max_tokens=500, on_exceed="cancel", count_cache_tokens=True),
            run_id="r",
            usage=UsageStats(input_tokens=900, output_tokens=100, cache_read_input_tokens=100),
            recorder=None,
            notifier=None,
            warnings=[],
        )
        assert (await tracker._measure()).tokens == 1100
        assert [b.metric for b in await tracker.evaluate(1)] == ["tokens"]

    async def test_store_and_memory_paths_agree(self) -> None:
        usage = UsageStats(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=100,
        )
        memory = BudgetTracker(
            Budget(max_tokens=10_000, count_cache_tokens=True),
            run_id="r",
            usage=usage,
            recorder=None,
            notifier=None,
            warnings=[],
        )
        stored = _tracker(Budget(max_tokens=10_000, count_cache_tokens=True), _ROLLUP)
        assert (await memory._measure()).tokens == (await stored._measure()).tokens == 650

    async def test_missing_cache_fields_count_as_zero(self) -> None:
        tracker = BudgetTracker(
            Budget(max_tokens=10_000, count_cache_tokens=True),
            run_id="r",
            usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
            recorder=None,
            notifier=None,
            warnings=[],
        )
        assert (await tracker._measure()).tokens == 10

    async def test_opt_in_trips_the_limit_that_the_default_misses(self) -> None:
        opted = _memory_tracker(Budget(max_tokens=500, on_exceed="cancel", count_cache_tokens=True))
        default = _memory_tracker(
            Budget(max_tokens=500, on_exceed="cancel", count_cache_tokens=False)
        )
        assert [b.metric for b in await opted.evaluate(1)] == ["tokens"]
        assert await default.evaluate(1) == []

    async def test_cancels_a_cache_heavy_run_end_to_end(self) -> None:
        @tool()
        async def look() -> str:
            """Look."""
            return "ok"

        def turn(text: str | None, calls: list[ToolCall] | None) -> LLMResponse:
            return LLMResponse(
                text=text,
                tool_calls=calls,
                usage=UsageStats(
                    input_tokens=10,
                    output_tokens=10,
                    total_tokens=20,
                    cache_read_input_tokens=900,
                ),
            )

        for choice, expected in ((True, RunStatus.CANCELLED), (False, RunStatus.SUCCESS)):
            llm = MockLLM(
                [turn(None, [ToolCall(name="look", params={}, id="c1")]), turn("done", None)]
            )
            agent = Agent(
                prompt="p",
                tools=[look],
                budget=Budget(max_tokens=500, on_exceed="cancel", count_cache_tokens=choice),
            )
            result = await ReActLoop().run(
                agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="go"
            )
            assert result.status == expected, choice


class TestCountCacheTokensNotChosen:
    """B17 warns with ``FutureWarning`` (visible by default) at ``Agent(...)``."""

    @staticmethod
    def _construct(budget: Budget) -> None:
        _agent(budget=budget)

    def test_warns_at_the_agent_line_when_a_token_budget_did_not_choose(self) -> None:
        with pytest.warns(FutureWarning, match="count_cache_tokens") as seen:
            self._construct(Budget(max_tokens=100))
        assert seen[0].filename == __file__

    def test_dedup_is_left_to_the_warnings_module_per_call_site(self) -> None:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("default")
            for _ in range(3):
                self._construct(Budget(max_tokens=100))
        assert len([w for w in seen if issubclass(w.category, FutureWarning)]) == 1

    def test_a_filtered_warning_does_not_use_up_the_notice(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._construct(Budget(max_tokens=100))
        with pytest.warns(FutureWarning):
            self._construct(Budget(max_tokens=100))

    @pytest.mark.parametrize("choice", [True, False])
    def test_silent_when_chosen(self, choice: bool) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._construct(Budget(max_tokens=100, count_cache_tokens=choice))

    def test_silent_for_cost_only_budget(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _agent(budget=Budget(max_cost_usd=1.0), pricing=PriceTable({"mock": _RATE}))

    def test_runs_of_a_built_agent_do_not_warn_again(self) -> None:
        agent = _agent(budget=Budget(max_tokens=100, count_cache_tokens=False))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_budget_config(agent)


class TestCostUnknownCallEndsEnforcingRun:
    async def test_cost_unknown_call_cancels_under_a_cost_budget(self) -> None:
        @tool()
        async def look() -> str:
            """Look."""
            return "ok"

        unknown = LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="look", params={}, id="c1")],
            usage=UsageStats(input_tokens=10, output_tokens=10, total_tokens=20, cost_unknown=True),
        )
        llm = MockLLM([unknown, LLMResponse(text="never", usage=UsageStats())])
        agent = Agent(
            prompt="p",
            tools=[look],
            provider=llm,
            pricing=PriceTable({"mock": _RATE}),
            budget=Budget(max_cost_usd=100.0, on_exceed="cancel"),
        )
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        assert llm.calls_made == 1
