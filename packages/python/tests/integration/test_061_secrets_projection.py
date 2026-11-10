"""Trusted projections retain the existing stop outcome in both native formats."""

import pytest

from dendrux import Agent, tool
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format

SECRET = "AKIAIOSFODNN7EXAMPLE"


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_projected_initial_block_keeps_zero_provider_calls(store, engine, fmt, single):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = Agent(
        prompt="Answer briefly.",
        state_store=store,
        provider=provider,
        guardrails=[SecretDetection()],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run(SECRET, tenant_id="tenant")
        assert result.status == RunStatus.ERROR
        assert provider._call_count == 0
        assert (await store.get_run(result.run_id)).failure_reason == "guardrail_blocked"
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_projected_tool_observation_keeps_one_provider_call(store, engine, fmt):
    await pin_storage_format(engine, fmt)

    @tool()
    async def lookup() -> str:
        """Return a synthetic fixture."""
        return SECRET

    provider = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("lookup")]), LLMResponse(text="unreachable")]
    )
    agent = Agent(
        prompt="Use lookup.",
        state_store=store,
        provider=provider,
        tools=[lookup],
        guardrails=[SecretDetection()],
    )
    try:
        result = await agent.run("go", tenant_id="tenant")
        assert result.status == RunStatus.ERROR
        assert provider._call_count == 1
        assert (await store.get_run(result.run_id)).failure_reason == "guardrail_blocked"
        assert SECRET not in repr(provider.wire_messages)
    finally:
        await agent.close()
