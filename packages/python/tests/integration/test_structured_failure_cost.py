"""The failed structured call's usage and cost reach the store and budget (#26)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from dendrux import Agent, Budget, ModelPricing, PriceTable, SingleCall
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    RunStatus,
    StructuredOutputValidationError,
    UsageStats,
)

_PRICING = PriceTable({"mock": ModelPricing(input=1.0, output=2.0)})


def _usage() -> UsageStats:
    return UsageStats(input_tokens=100, output_tokens=10, total_tokens=110)


class Out(BaseModel):
    label: str


def _structured_llm(text: str) -> MockLLM:
    llm = MockLLM([LLMResponse(text=text, usage=_usage(), stop_reason="max_tokens")])
    llm.capabilities = ProviderCapabilities(supports_structured_output=True)
    return llm


async def test_failed_structured_call_is_recorded_and_priced(store) -> None:
    agent = Agent(
        strict_tenancy=False,
        provider=_structured_llm('{"lab'),
        prompt="test",
        loop=SingleCall(),
        output_type=Out,
        state_store=store,
        pricing=_PRICING,
    )
    with pytest.raises(StructuredOutputValidationError) as info:
        await agent.run("hi")

    err = info.value
    assert err.model == "mock"
    assert err.usage is not None
    assert err.usage.input_tokens == 100
    assert err.usage.output_tokens == 10
    assert err.usage.cost_usd == pytest.approx((100 * 1.0 + 10 * 2.0) / 1_000_000)

    runs = await store.list_runs()
    run_id = runs[0].id
    record = await store.get_run(run_id)
    assert record.status == RunStatus.ERROR.value

    [call] = await store.get_llm_interactions(run_id)
    assert call.input_tokens == 100
    assert call.output_tokens == 10
    assert call.cost_usd == pytest.approx(err.usage.cost_usd)
    assert call.semantic_response["text"] == '{"lab'
    assert call.semantic_response["stop_reason"] == "max_tokens"

    types = [e.event_type for e in await store.get_run_events(run_id)]
    assert "llm.completed" in types
    assert "llm.failed" not in types
    assert types[-1] == "run.error"

    usage = (await RunStore(store).get_subtree(run_id)).usage
    assert usage.llm_calls == 1
    assert usage.input_tokens == 100
    assert usage.cost_usd == pytest.approx(err.usage.cost_usd)


async def test_failed_structured_call_counts_toward_budget(store) -> None:
    agent = Agent(
        strict_tenancy=False,
        provider=_structured_llm("not json"),
        prompt="test",
        loop=SingleCall(),
        output_type=Out,
        state_store=store,
        pricing=_PRICING,
        budget=Budget(max_cost_usd=1e-6),
    )
    with pytest.raises(StructuredOutputValidationError):
        await agent.run("hi")
    run_id = (await store.list_runs())[0].id
    exceeded = await store.get_run_events(run_id, event_types=["budget.exceeded"])
    assert len(exceeded) == 1
    assert exceeded[0].data["metric"] == "cost_usd"
