"""Early reclaim stops retain committed checkpoint accounting."""

import pytest

from dendrux import Agent, tool
from dendrux.errors import RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall, UsageStats, _usage_from_dict
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire
from tests.integration.test_061_secrets_reclaim import Crash, _mark


async def _interrupted_run(store, engine, fmt):
    await pin_storage_format(engine, fmt)

    @tool()
    async def interrupted_write() -> str:
        raise Crash()

    @tool(idempotent=True)
    async def read_value() -> str:
        return "ready"

    usage = UsageStats(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        cost_usd=0.01,
        cost_source="provider",
        cache_read_input_tokens=3,
        cache_creation_input_tokens=2,
        reasoning_tokens=4,
    )
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read_value")], usage=usage),
            LLMResponse(tool_calls=[ToolCall("interrupted_write")], usage=usage),
        ]
    )
    agent = Agent(
        name="accounting",
        prompt="p",
        state_store=store,
        provider=provider,
        tools=[read_value, interrupted_write],
        strict_tenancy=True,
    )
    try:
        with pytest.raises(Crash):
            await agent.run("go", tenant_id="tenant")
    finally:
        await agent.close()
    (record,) = await store.list_runs()
    return record.id, agent.tools


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("marked", [False, True])
async def test_reclaim_preserves_committed_accounting(store, engine, fmt, marked):
    run_id, tools = await _interrupted_run(store, engine, fmt)
    if marked:
        await _mark(store, run_id)
    before = await store.get_run(run_id)
    provider = MockLLM([LLMResponse(text="done")])
    agent = Agent(
        name="accounting",
        prompt="p",
        state_store=store,
        provider=provider,
        tools=tools,
        strict_tenancy=True,
    )
    try:
        if fmt == 1:
            # Native legacy runs cannot be reclaimed; admission must leave them intact.
            with pytest.raises(RunNotReclaimableError) as caught:
                await agent.reclaim(run_id, tenant_id="tenant")
            assert caught.value.reason == "legacy_run"
            assert await store.get_run(run_id) == before
            assert provider._call_count == 0
            return
        state = await store.get_loop_state(run_id)
        expected_usage = _usage_from_dict(state["usage"])
        assert state["iteration"] == 2
        assert (expected_usage.input_tokens, expected_usage.output_tokens) == (22, 14)
        item = await expire(store, engine, run_id)
        result = await agent.reclaim(run_id, tenant_id="tenant", item=item)
        row = await store.get_run(run_id)
        assert result.iteration_count == row.iteration_count == 2
        assert result.usage == expected_usage
        assert (row.total_input_tokens, row.total_output_tokens) == (22, 14)
        assert row.total_cost_usd == pytest.approx(expected_usage.cost_usd)
        assert row.total_cache_read_tokens == expected_usage.cache_read_input_tokens
        assert row.total_cache_creation_tokens == expected_usage.cache_creation_input_tokens
        assert row.total_reasoning_tokens == expected_usage.reasoning_tokens
        assert provider._call_count == 0
        if marked:
            assert result.status == RunStatus.ERROR
            assert row.failure_reason == result.meta["failure_reason"] == "guardrail_blocked"
            assert row.meta["dendrux.blocked"] is True
        else:
            assert result.status == RunStatus.WAITING_APPROVAL
            assert row.failure_reason is None
    finally:
        await agent.close()
