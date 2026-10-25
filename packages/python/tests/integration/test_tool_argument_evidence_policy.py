"""Rejected model arguments use persistable evidence before any callback."""

import json

import pytest

from dendrux import Agent, tool
from dendrux.loops.base import BaseNotifier
from dendrux.types import LLMResponse, RunStatus, ToolCall, _validate_persistable
from tests._helpers.scripted_provider import ScriptedLLM


@pytest.mark.parametrize("bad", ["PRIVATE\x00PARAM", "PRIVATE\ud800PARAM"])
@pytest.mark.parametrize("target", ["server", "client"])
@pytest.mark.parametrize("stream", [False, True])
async def test_bad_model_arguments_are_rejected_before_evidence_sinks(store, bad, target, stream):
    executions = []

    @tool(target=target)
    def record(note: str = "") -> str:
        """Record a note."""
        executions.append(note)
        return "ok"

    class Evidence(BaseNotifier):
        def __init__(self):
            self.sinks = set()

        async def on_llm_call_completed(self, run_id, response, iteration, **kwargs):
            for call in response.tool_calls or []:
                _validate_persistable(call.params)
                self.sinks.add("interaction")
            _validate_persistable(response.provider_response)

        async def on_message_appended(self, run_id, message, iteration):
            for call in message.tool_calls or []:
                _validate_persistable(call.params)
                self.sinks.add("trace")

        async def on_tool_completed(self, run_id, tool_call, result, iteration):
            _validate_persistable(tool_call.params)
            assert json.loads(result.payload)["code"] == "invalid_params"
            self.sinks.add("tool")

    notifier = Evidence()
    provider = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall("record", {"note": bad})], provider_response={"args": bad}
            ),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p", tools=[record], provider=provider, state_store=store, strict_tenancy=False
    )
    try:
        if stream:
            async with agent.stream("go", tenant_id="A", notifier=notifier) as source:
                async for _ in source:
                    pass
            result = source.result
        else:
            result = await agent.run("go", tenant_id="A", notifier=notifier)
        assert result.status == RunStatus.SUCCESS
        assert not executions
        assert notifier.sinks == {"interaction", "trace", "tool"}
        calls = await store.get_tool_calls(result.run_id)
        assert len(calls) == 1 and calls[0].result["code"] == "invalid_params"
        _validate_persistable(calls[0].params)
        for trace in await store.get_traces(result.run_id):
            _validate_persistable(trace.meta)
        for interaction in await store.get_llm_interactions(result.run_id):
            _validate_persistable(interaction.semantic_response)
            _validate_persistable(interaction.provider_response)
    finally:
        await agent.close()
