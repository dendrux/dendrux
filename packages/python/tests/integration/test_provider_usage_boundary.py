"""Untrusted provider accounting never reaches numeric columns unchecked."""

from dataclasses import replace

import pytest

from dendrux import Agent, tool
from dendrux.types import Budget, LLMResponse, RunStatus, ToolCall, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM

BAD_USAGE = [
    ("input_tokens", 2**40),
    ("input_tokens", -1000000),
    ("output_tokens", True),
    ("total_tokens", -1),
    ("cache_read_input_tokens", "9"),
    ("cache_creation_input_tokens", 2**31),
    ("reasoning_tokens", 1.5),
    ("cost_usd", float("nan")),
    ("cost_usd", float("inf")),
    ("cost_usd", -5.0),
    ("cost_usd", 1e12),
]


@pytest.mark.parametrize("field,value", BAD_USAGE)
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_provider_usage_is_unreported_and_safe_to_persist(
    store, caplog, field, value, stream
):
    usage = replace(
        UsageStats(input_tokens=1, output_tokens=1, total_tokens=2, cost_usd=0.01), **{field: value}
    )
    provider = ScriptedLLM([LLMResponse(text="done", usage=usage)])
    async with Agent(
        prompt="p", provider=provider, state_store=store, strict_tenancy=False
    ) as agent:
        if stream:
            async with agent.stream("go", tenant_id="A") as source:
                async for _ in source:
                    pass
            result = source.result
        else:
            result = await agent.run("go", tenant_id="A")
        assert result.status == RunStatus.SUCCESS
        assert result.usage.usage_reported is False
        assert result.usage.cost_unknown is True
        assert result.usage.total_tokens == 0 and result.usage.cost_usd is None
        row = await store.get_run(result.run_id)
        assert row.total_input_tokens == 0 and row.total_output_tokens == 0
        assert row.total_cost_usd is None
        assert any(getattr(record, "code", None) == "usage_invalid" for record in caplog.records)


@pytest.mark.parametrize("metric", ["tokens", "cost"])
async def test_invalid_provider_usage_cannot_bypass_enforcing_budget(store, metric):
    class CostProvider(ScriptedLLM):
        @property
        def reports_cost(self):
            return True

    @tool()
    async def read_value() -> str:
        return "value"

    provider = CostProvider(
        [
            LLMResponse(
                tool_calls=[ToolCall("read_value", {})],
                usage=UsageStats(input_tokens=-100, cost_usd=-5),
            ),
            LLMResponse(text="done"),
        ]
    )
    budget = (
        Budget(max_tokens=1000, on_exceed="pause", count_cache_tokens=False)
        if metric == "tokens"
        else Budget(max_cost_usd=1, on_exceed="pause")
    )
    async with Agent(
        prompt="p",
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        tools=[read_value],
        budget=budget,
    ) as agent:
        result = await agent.run("go", tenant_id="A")
        assert result.status == RunStatus.PAUSED
        events = await store.get_run_events(result.run_id, event_types=["budget.exceeded"])
        assert events and events[0].data["reason"] in ("usage_invalid", "unpriced_call")
