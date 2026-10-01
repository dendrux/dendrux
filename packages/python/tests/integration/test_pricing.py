from __future__ import annotations

import pytest

from dendrux import Agent, ModelPricing, PriceTable, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import _build_cached_result
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult, UsageStats


@tool(target="client")
async def lookup() -> str:
    return ""


@pytest.mark.parametrize("source", ["table", "provider", "mixed", None])
@pytest.mark.parametrize("cas", [False, True])
async def test_cached_usage_preserves_final_summary(store, source, cas):
    await store.create_run("priced", "test")
    usage = UsageStats(
        input_tokens=100,
        output_tokens=10,
        total_tokens=210,
        cache_read_input_tokens=100,
        reasoning_tokens=5,
        cost_usd=0.1 if source else None,
        cost_source=source,
        usage_reported=source is not None,
        cost_unknown=source is None,
    )
    if cas:
        await store.finalize_run_if_status_in(
            "priced",
            status="success",
            allowed_current_statuses=["running"],
            total_usage=usage,
        )
    else:
        await store.finalize_run("priced", status="success", total_usage=usage)
    cached = await _build_cached_result(store, "priced")
    assert cached.usage == usage


async def test_idempotent_retry_preserves_table_source(store):
    llm = MockLLM([LLMResponse(text="done", usage=UsageStats(input_tokens=100, output_tokens=10))])
    agent = Agent(
        provider=llm,
        prompt="test",
        state_store=store,
        pricing=PriceTable({"mock": ModelPricing(input=1, output=2)}),
    )
    first = await agent.run("hello", idempotency_key="pricing")
    cached = await agent.run("hello", idempotency_key="pricing")
    assert first.usage.cost_source == "table"
    assert cached.usage == first.usage
    assert llm.calls_made == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cache_tokens", [None, 100])
@pytest.mark.parametrize("legacy", [False, True])
async def test_unpriced_call_stays_unknown_after_pause_resume(store, stream, cache_tokens, legacy):
    call = ToolCall(name="lookup", params={}, provider_tool_call_id="call1")
    llm = MockLLM(
        [
            LLMResponse(tool_calls=[call], usage=UsageStats(cache_read_input_tokens=cache_tokens)),
            LLMResponse(
                text="done", usage=UsageStats(input_tokens=10, output_tokens=5, cost_usd=0.1)
            ),
        ]
    )
    agent = Agent(provider=llm, prompt="test", tools=[lookup], state_store=store)
    if stream:
        async with agent.stream("hello") as running:
            async for _ in running:
                pass
        paused = running.result
    else:
        paused = await agent.run("hello")
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    if legacy:
        pause_data = await store.get_pause_state(paused.run_id)
        for name in ("cost_source", "usage_reported", "cost_unknown"):
            pause_data["usage"].pop(name)
        await store.pause_run(paused.run_id, status=paused.status, pause_data=pause_data)
    result = await agent.submit_tool_results(
        paused.run_id, [ToolResult(name="lookup", call_id=call.id, payload="ok")]
    )
    assert result.status == RunStatus.SUCCESS
    assert result.usage.cost_usd is None
    assert result.usage.cost_source is None
    assert result.usage.cost_unknown
    assert (await _build_cached_result(store, result.run_id)).usage == result.usage


@pytest.mark.parametrize("cas", [False, True])
async def test_usage_summary_preserves_existing_answer(store, cas):
    await store.create_run("priced", "test")
    await store.finalize_run("priced", status="running", answer="saved answer")
    usage = UsageStats(cost_usd=0, cost_source="provider")
    if cas:
        await store.finalize_run_if_status_in(
            "priced", status="success", allowed_current_statuses=["running"], total_usage=usage
        )
    else:
        await store.finalize_run("priced", status="success", total_usage=usage)
    cached = await _build_cached_result(store, "priced")
    assert cached.answer == "saved answer"
    assert cached.usage == usage


@pytest.mark.parametrize("cas", [False, True])
async def test_losing_finalize_keeps_winning_usage_summary(store, cas):
    await store.create_run("priced", "test")
    winner = UsageStats(cost_usd=0.1, cost_source="provider")
    await store.finalize_run("priced", status="success", answer="winner", total_usage=winner)
    loser = UsageStats(cost_usd=9, cost_source="table")
    if cas:
        won = await store.finalize_run_if_status_in(
            "priced", status="cancelled", allowed_current_statuses=["running"], total_usage=loser
        )
    else:
        won = await store.finalize_run(
            "priced", status="cancelled", expected_current_status="running", total_usage=loser
        )
    assert not won
    cached = await _build_cached_result(store, "priced")
    assert cached.answer == "winner"
    assert cached.usage == winner
