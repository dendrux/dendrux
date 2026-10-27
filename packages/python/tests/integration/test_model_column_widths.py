"""Provider identities fit the persistence columns on every run path."""

import json

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM


@tool(target="client")
async def read_cells() -> str:
    return ""


@tool()
async def server_tool() -> str:
    return "server"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["name", "provider_id", "call_id"])
async def test_oversized_model_identity_is_recoverable_without_echo(store, field, stream):
    secret = "PRIVATE_MODEL_IDENTITY" * 20
    call = ToolCall(
        secret if field == "name" else "read_cells",
        {},
        provider_tool_call_id=secret if field == "provider_id" else "read",
        **({"id": secret} if field == "call_id" else {}),
    )
    provider = ScriptedLLM([LLMResponse(tool_calls=[call]), LLMResponse(text="done")])
    async with Agent(
        prompt="p",
        tools=[read_cells, server_tool],
        provider=provider,
        require_approval=["server_tool"] if field == "name" else [],
        state_store=store,
        strict_tenancy=False,
    ) as agent:
        if stream:
            async with agent.stream("go", tenant_id="A") as source:
                events = [event async for event in source]
            result = source.result
        else:
            result = await agent.run("go", tenant_id="A")
            events = []
        if field != "name":
            assert result.status == RunStatus.WAITING_CLIENT_TOOL
            pending = result.pending.tool_calls[0]
            result = await agent.submit_tool_results(
                result.run_id, [ToolResult(pending.name, pending.id, '"ok"')], tenant_id="A"
            )
        assert result.status == RunStatus.SUCCESS
        calls = await store.get_tool_calls(result.run_id)
        assert len(calls) == 1
        assert len(calls[0].tool_name) <= 255
        assert len(calls[0].tool_call_id) <= 26
        assert len(calls[0].provider_tool_call_id or "") <= 255
        if field == "name":
            assert calls[0].success is False
            assert calls[0].result["error"]
            assert secret not in json.dumps(calls[0].result)
        for event in events:
            assert secret not in (event.tool_name or "")
            assert secret not in (event.tool_call_id or "")


@pytest.mark.parametrize("stream", [False, True])
async def test_oversized_response_model_still_persists_a_success(store, stream):
    private_model = "PRIVATE_MODEL" * 30

    class ModelProvider(ScriptedLLM):
        async def complete(self, *args, **kwargs):
            response = await super().complete(*args, **kwargs)
            response.model = private_model
            return response

    provider = ModelProvider([LLMResponse(text="done")])
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
        interactions = await store.get_llm_interactions(result.run_id)
        assert len(interactions[0].model) <= 255
