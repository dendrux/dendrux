"""Tests for developer-owned pricing (``dendrux.pricing``).

Covers:
  - ModelPricing arithmetic and validation
  - PriceTable lookup precedence (exact > longest glob > none)
  - price_usage resolution order (provider cost wins; table; unpriced)
  - Loop integration: ReAct run/stream + SingleCall fill cost_usd per call,
    per-call ``model`` overrides are honored, unpriced steps make the run
    total None, mixed sources are labelled
  - Accumulator: a None total that already carries tokens stays None (resume)
  - Persistence: cost_source rides the llm.completed event + token_usage meta
  - UsageStats dict round-trip carries cost_source
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux import ModelPricing, PriceTable
from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop, _accumulate_usage
from dendrux.loops.single import SingleCall
from dendrux.pricing import price_usage
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import (
    LLMResponse,
    RunEventType,
    ToolCall,
    UsageStats,
    _usage_from_dict,
    _usage_to_dict,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SONNET = ModelPricing(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75)
GPT = ModelPricing(input=1.25, output=10.0, cache_read=0.125)


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent(**overrides: Any) -> Agent:
    defaults: dict[str, Any] = {
        "prompt": "You are a calculator.",
        "tools": [add],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


def _usage(
    inp: int = 1_000_000,
    out: int = 1_000_000,
    *,
    cache_read: int | None = None,
    cache_write: int | None = None,
    cost: float | None = None,
    source: str | None = None,
) -> UsageStats:
    return UsageStats(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        cost_usd=cost,
        cost_source=source,
    )


# ------------------------------------------------------------------
# ModelPricing
# ------------------------------------------------------------------


class TestModelPricing:
    def test_cost_uses_per_million_rates(self) -> None:
        assert SONNET.cost(_usage(1_000_000, 1_000_000)) == pytest.approx(18.0)

    def test_cache_fields_use_their_own_rates(self) -> None:
        usage = _usage(0, 0, cache_read=1_000_000, cache_write=1_000_000)
        assert SONNET.cost(usage) == pytest.approx(0.30 + 3.75)

    def test_cache_rates_default_to_input_rate(self) -> None:
        plain = ModelPricing(input=2.0, output=4.0)
        usage = _usage(0, 0, cache_read=1_000_000, cache_write=500_000)
        assert plain.cost(usage) == pytest.approx(2.0 + 1.0)

    def test_none_cache_counts_contribute_nothing(self) -> None:
        assert GPT.cost(_usage(100, 10)) == pytest.approx((100 * 1.25 + 10 * 10.0) / 1e6)

    def test_zero_usage_costs_zero(self) -> None:
        assert SONNET.cost(UsageStats()) == 0.0

    @pytest.mark.parametrize("field_name", ["input", "output", "cache_read", "cache_write"])
    def test_negative_rate_rejected(self, field_name: str) -> None:
        kwargs: dict[str, float] = {"input": 1.0, "output": 1.0, field_name: -0.01}
        with pytest.raises(ValueError, match=field_name):
            ModelPricing(**kwargs)


# ------------------------------------------------------------------
# PriceTable
# ------------------------------------------------------------------


class TestPriceTable:
    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PriceTable({})

    def test_blank_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            PriceTable({"  ": SONNET})

    def test_non_pricing_value_rejected(self) -> None:
        with pytest.raises(TypeError, match="ModelPricing"):
            PriceTable({"m": {"input": 1.0}})  # type: ignore[dict-item]

    def test_exact_match(self) -> None:
        table = PriceTable({"claude-sonnet-4-6": SONNET})
        assert table.lookup("claude-sonnet-4-6") is SONNET

    def test_unknown_model_is_none(self) -> None:
        table = PriceTable({"claude-sonnet-4-6": SONNET})
        assert table.lookup("gpt-5") is None
        assert table.lookup(None) is None
        assert table.lookup("") is None

    def test_glob_match(self) -> None:
        table = PriceTable({"gpt-5*": GPT})
        assert table.lookup("gpt-5-mini-2026-01-01") is GPT

    def test_exact_beats_glob(self) -> None:
        exact = ModelPricing(input=9.0, output=9.0)
        table = PriceTable({"gpt-5*": GPT, "gpt-5-pro": exact})
        assert table.lookup("gpt-5-pro") is exact
        assert table.lookup("gpt-5-mini") is GPT

    def test_longest_glob_wins(self) -> None:
        narrow = ModelPricing(input=7.0, output=7.0)
        table = PriceTable({"gpt-*": GPT, "gpt-5-mini*": narrow})
        assert table.lookup("gpt-5-mini-2026") is narrow
        assert table.lookup("gpt-4o") is GPT

    def test_match_is_case_sensitive(self) -> None:
        table = PriceTable({"gpt-5*": GPT})
        assert table.lookup("GPT-5") is None

    def test_cost_for_and_contains(self) -> None:
        table = PriceTable({"m": SONNET})
        assert "m" in table
        assert "other" not in table
        assert 42 not in table
        assert table.cost_for("m", _usage(1_000_000, 0)) == pytest.approx(3.0)
        assert table.cost_for("other", _usage()) is None
        assert len(table) == 1
        assert repr(table) == "PriceTable(1 entries)"


# ------------------------------------------------------------------
# price_usage resolution
# ------------------------------------------------------------------


class TestPriceUsage:
    def test_provider_cost_kept_and_tagged(self) -> None:
        table = PriceTable({"m": SONNET})
        priced = price_usage(_usage(cost=0.5), model="m", pricing=table)
        assert priced.cost_usd == 0.5
        assert priced.cost_source == "provider"

    def test_existing_source_preserved(self) -> None:
        usage = _usage(cost=0.5, source="provider")
        assert price_usage(usage, model="m", pricing=None) is usage

    def test_table_prices_unreported_cost(self) -> None:
        priced = price_usage(_usage(1_000_000, 0), model="m", pricing=PriceTable({"m": SONNET}))
        assert priced.cost_usd == pytest.approx(3.0)
        assert priced.cost_source == "table"

    def test_no_table_is_identity(self) -> None:
        usage = _usage()
        assert price_usage(usage, model="m", pricing=None) is usage

    def test_unknown_model_stays_unpriced(self) -> None:
        usage = _usage()
        priced = price_usage(usage, model="other", pricing=PriceTable({"m": SONNET}))
        assert priced is usage
        assert priced.cost_usd is None
        assert priced.cost_source is None

    def test_never_mutates_input(self) -> None:
        usage = _usage(1_000_000, 0)
        price_usage(usage, model="m", pricing=PriceTable({"m": SONNET}))
        assert usage.cost_usd is None
        assert usage.cost_source is None


# ------------------------------------------------------------------
# Accumulator semantics
# ------------------------------------------------------------------


class TestAccumulator:
    def test_priced_steps_sum_with_uniform_source(self) -> None:
        total = UsageStats()
        _accumulate_usage(total, _usage(10, 5, cost=0.1, source="table"))
        _accumulate_usage(total, _usage(10, 5, cost=0.2, source="table"))
        assert total.cost_usd == pytest.approx(0.3)
        assert total.cost_source == "table"

    def test_unpriced_step_makes_total_unknown_and_sticky(self) -> None:
        total = UsageStats()
        _accumulate_usage(total, _usage(10, 5, cost=0.1, source="table"))
        _accumulate_usage(total, _usage(10, 5))
        _accumulate_usage(total, _usage(10, 5, cost=0.2, source="table"))
        assert total.cost_usd is None
        assert total.cost_source is None
        assert total.input_tokens == 30

    def test_resumed_unknown_total_stays_unknown(self) -> None:
        """A None total that already carries tokens (reloaded from the DB
        after a pause) must not be re-seeded from zero by the next step."""
        total = _usage(100, 50)
        _accumulate_usage(total, _usage(10, 5, cost=0.2, source="table"))
        assert total.cost_usd is None

    def test_fresh_total_accepts_first_priced_step(self) -> None:
        total = UsageStats()
        _accumulate_usage(total, _usage(10, 5, cost=0.2, source="provider"))
        assert total.cost_usd == pytest.approx(0.2)
        assert total.cost_source == "provider"

    def test_mixed_sources_labelled(self) -> None:
        total = UsageStats()
        _accumulate_usage(total, _usage(10, 5, cost=0.1, source="provider"))
        _accumulate_usage(total, _usage(10, 5, cost=0.2, source="table"))
        assert total.cost_usd == pytest.approx(0.3)
        assert total.cost_source == "mixed"


# ------------------------------------------------------------------
# Loop integration
# ------------------------------------------------------------------


def _tool_turn(**usage: Any) -> LLMResponse:
    tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
    return LLMResponse(tool_calls=[tc], usage=_usage(**usage))


def _final(text: str = "3", **usage: Any) -> LLMResponse:
    return LLMResponse(text=text, usage=_usage(**usage))


class TestReActPricing:
    async def test_run_prices_every_call_from_table(self) -> None:
        llm = MockLLM([_tool_turn(inp=100, out=10), _final(inp=200, out=20)])
        agent = _agent(pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}))

        result = await ReActLoop().run(
            agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="1+2?"
        )

        expected = (100 * 1.0 + 10 * 2.0 + 200 * 1.0 + 20 * 2.0) / 1e6
        assert result.usage.cost_usd == pytest.approx(expected)
        assert result.usage.cost_source == "table"

    async def test_no_pricing_leaves_cost_none(self) -> None:
        llm = MockLLM([_final(inp=200, out=20)])
        result = await ReActLoop().run(
            agent=_agent(), provider=llm, strategy=NativeToolCalling(), user_input="hi"
        )
        assert result.usage.cost_usd is None
        assert result.usage.cost_source is None

    async def test_per_call_model_override_is_priced_by_its_own_entry(self) -> None:
        table = PriceTable(
            {
                "mock": ModelPricing(input=1.0, output=1.0),
                "fancy": ModelPricing(input=100.0, output=100.0),
            }
        )
        llm = MockLLM([_final(inp=10, out=10)])
        result = await ReActLoop().run(
            agent=_agent(pricing=table),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="hi",
            provider_kwargs={"model": "fancy"},
        )
        assert result.usage.cost_usd == pytest.approx(20 * 100.0 / 1e6)

    async def test_provider_model_used_when_response_has_none(self) -> None:
        """A provider that leaves ``LLMResponse.model`` unset is priced by its
        configured model."""
        table = PriceTable({"configured": ModelPricing(input=1.0, output=1.0)})
        llm = MockLLM([_final(inp=10, out=10)], model="configured")
        result = await ReActLoop().run(
            agent=_agent(pricing=table),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="hi",
        )
        assert result.usage.cost_usd == pytest.approx(20 / 1e6)

    async def test_model_missing_from_table_makes_run_total_unknown(self) -> None:
        table = PriceTable({"mock": ModelPricing(input=1.0, output=1.0)})
        llm = MockLLM([_tool_turn(inp=100, out=10), _final(inp=10, out=10)])
        result = await ReActLoop().run(
            agent=_agent(pricing=table),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            provider_kwargs={"model": "unlisted"},
        )
        assert result.usage.cost_usd is None
        assert result.usage.cost_source is None

    async def test_provider_cost_wins_over_table(self) -> None:
        table = PriceTable({"mock": ModelPricing(input=1000.0, output=1000.0)})
        llm = MockLLM([_final(inp=10, out=10, cost=0.001)])
        result = await ReActLoop().run(
            agent=_agent(pricing=table),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="hi",
        )
        assert result.usage.cost_usd == pytest.approx(0.001)
        assert result.usage.cost_source == "provider"

    async def test_mixed_sources_across_steps(self) -> None:
        table = PriceTable({"mock": ModelPricing(input=1.0, output=1.0)})
        llm = MockLLM([_tool_turn(inp=10, out=10, cost=0.5), _final(inp=10, out=10)])
        result = await ReActLoop().run(
            agent=_agent(pricing=table),
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
        )
        assert result.usage.cost_usd == pytest.approx(0.5 + 20 / 1e6)
        assert result.usage.cost_source == "mixed"

    async def test_stream_path_prices_calls(self) -> None:
        llm = MockLLM([_tool_turn(inp=100, out=10), _final(inp=200, out=20)])
        agent = _agent(pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}))

        events = [
            e
            async for e in ReActLoop().run_stream(
                agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="1+2?"
            )
        ]
        done = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert len(done) == 1
        assert done[0].run_result is not None
        expected = (100 * 1.0 + 10 * 2.0 + 200 * 1.0 + 20 * 2.0) / 1e6
        assert done[0].run_result.usage.cost_usd == pytest.approx(expected)
        assert done[0].run_result.usage.cost_source == "table"


class TestSingleCallPricing:
    async def test_run_priced(self) -> None:
        llm = MockLLM([_final(inp=100, out=10)])
        agent = Agent(
            prompt="Answer.",
            loop=SingleCall(),
            pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}),
        )
        result = await SingleCall().run(
            agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="hi"
        )
        assert result.usage.cost_usd == pytest.approx((100 + 20) / 1e6)
        assert result.usage.cost_source == "table"

    async def test_stream_priced(self) -> None:
        llm = MockLLM([_final(inp=100, out=10)])
        agent = Agent(
            prompt="Answer.",
            loop=SingleCall(),
            pricing=PriceTable({"mock": ModelPricing(input=1.0, output=2.0)}),
        )
        events = [
            e
            async for e in SingleCall().run_stream(
                agent=agent, provider=llm, strategy=NativeToolCalling(), user_input="hi"
            )
        ]
        done = [e for e in events if e.type == RunEventType.RUN_COMPLETED]
        assert done[0].run_result is not None
        assert done[0].run_result.usage.cost_usd == pytest.approx((100 + 20) / 1e6)


class TestAgentConstruction:
    def test_pricing_defaults_to_none(self) -> None:
        assert _agent().pricing is None

    def test_pricing_exposed(self) -> None:
        table = PriceTable({"mock": SONNET})
        assert _agent(pricing=table).pricing is table


# ------------------------------------------------------------------
# Persistence + serialization
# ------------------------------------------------------------------


@dataclass
class _SpyStore:
    usages: list[dict[str, Any]] = field(default_factory=list)
    llm_interactions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    async def save_usage(self, run_id: str, **kwargs: Any) -> None:
        self.usages.append(kwargs)

    async def save_llm_interaction(self, run_id: str, **kwargs: Any) -> None:
        self.llm_interactions.append(kwargs)

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self.events.append(kwargs)

    async def touch_progress(self, run_id: str) -> None:
        pass


class TestPersistence:
    async def test_cost_source_rides_event_interaction_and_usage_meta(self) -> None:
        store = _SpyStore()
        recorder = PersistenceRecorder(store, "run_1")  # type: ignore[arg-type]
        response = LLMResponse(text="hi", usage=_usage(10, 5, cost=0.01, source="table"))

        await recorder.on_llm_call_completed("run_1", response, iteration=1)

        event = next(e for e in store.events if e["event_type"] == "llm.completed")
        assert event["data"]["cost_usd"] == 0.01
        assert event["data"]["cost_source"] == "table"
        assert store.usages[0]["meta"] == {"cost_source": "table"}
        assert store.llm_interactions[0]["semantic_response"]["usage"]["cost_source"] == "table"

    async def test_unpriced_call_has_no_source(self) -> None:
        store = _SpyStore()
        recorder = PersistenceRecorder(store, "run_1")  # type: ignore[arg-type]
        await recorder.on_llm_call_completed(
            "run_1", LLMResponse(text="hi", usage=_usage(10, 5)), iteration=1
        )
        event = next(e for e in store.events if e["event_type"] == "llm.completed")
        assert event["data"]["cost_usd"] is None
        assert event["data"]["cost_source"] is None
        assert store.usages[0]["meta"] is None


class TestSerialization:
    def test_usage_dict_round_trip_carries_cost_source(self) -> None:
        usage = _usage(10, 5, cost=0.25, source="mixed")
        data = _usage_to_dict(usage)
        assert data["cost_source"] == "mixed"
        restored = _usage_from_dict(data)
        assert restored.cost_usd == 0.25
        assert restored.cost_source == "mixed"

    def test_legacy_dict_without_cost_source(self) -> None:
        restored = _usage_from_dict({"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.1})
        assert restored.cost_source is None


class TestUnknownUsage:
    @pytest.mark.parametrize("responses_api", [False, True])
    def test_provider_missing_usage_is_not_free(self, responses_api):
        from dendrux.llm.openai import OpenAIProvider
        from dendrux.llm.openai_responses import OpenAIResponsesProvider
        from dendrux.loops._helpers import price_response

        provider_cls = OpenAIResponsesProvider if responses_api else OpenAIProvider
        provider = provider_cls(model="test", api_key="unused")
        raw = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=None))],
            output=[],
            usage=None,
        )
        response = provider._normalize_response(raw)
        priced = price_response(response, default_model="test", pricing=PriceTable({"test": GPT}))
        assert priced.usage.cost_usd is None
        assert not priced.usage.usage_reported

    @pytest.mark.parametrize("loop", [ReActLoop(), SingleCall()])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_custom_provider_missing_usage_is_not_free(self, loop, stream):
        agent = Agent(
            provider=MockLLM([LLMResponse(text="answer")]),
            prompt="test",
            loop=loop,
            pricing=PriceTable({"mock": GPT}),
        )
        if stream:
            async with agent.stream("hi") as running:
                async for _ in running:
                    pass
            result = running.result
        else:
            result = await agent.run("hi")
        assert result.usage.cost_usd is None
        assert result.usage.cost_source is None
        assert not result.usage.usage_reported

    def test_explicit_zero_usage_is_priced(self):
        result = price_usage(UsageStats(), model="test", pricing=PriceTable({"test": GPT}))
        assert result.cost_usd == 0
        assert result.cost_source == "table"

    def test_provider_cost_wins_even_without_usage(self):
        result = price_usage(
            UsageStats(cost_usd=0, usage_reported=False),
            model="test",
            pricing=PriceTable({"test": GPT}),
        )
        assert result.cost_usd == 0
        assert result.cost_source == "provider"

    @pytest.mark.parametrize("cache_tokens", [None, 100])
    def test_unknown_cost_survives_serialization(self, cache_tokens):
        total = UsageStats()
        _accumulate_usage(
            total, UsageStats(cache_read_input_tokens=cache_tokens, usage_reported=False)
        )
        total = _usage_from_dict(_usage_to_dict(total))
        _accumulate_usage(total, _usage(10, 5, cost=0.1, source="provider"))
        assert total.cost_usd is None
        assert total.cost_unknown
        assert not total.usage_reported
