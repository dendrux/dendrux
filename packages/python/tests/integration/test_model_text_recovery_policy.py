"""Unpersistable model text is replaced uniformly before evidence and display."""

import pytest

from dendrux import Agent
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    Clarification,
    LLMResponse,
    RunStatus,
    ToolCall,
    _validate_persistable,
)
from tests._helpers.scripted_provider import ScriptedLLM


@pytest.mark.parametrize("bad", ["\x00", "\ud800"])
@pytest.mark.parametrize("kind", ["tool", "question", "react_text", "single_text"])
@pytest.mark.parametrize("stream", [False, True])
async def test_model_text_is_recoverable_on_every_run_path(store, monkeypatch, bad, kind, stream):
    value = "model" + bad + "value"
    fixed = "model\ufffdvalue"
    response = LLMResponse(text=value, provider_response={"value": value})
    loop = SingleCall() if kind == "single_text" else ReActLoop()
    if kind == "tool":
        response = LLMResponse(tool_calls=[ToolCall(value, {})], provider_response={"name": value})
    if kind == "question":

        def clarify(self, response):
            return AgentStep(
                reasoning=None,
                action=Clarification(question=response.text, options=[response.text]),
            )

        monkeypatch.setattr(NativeToolCalling, "parse_response", clarify)
    provider = ScriptedLLM([response, LLMResponse(text="done")])
    agent = Agent(prompt="p", provider=provider, loop=loop, state_store=store, strict_tenancy=False)
    try:
        events = []
        if stream:
            async with agent.stream("go", tenant_id="A") as source:
                events = [event async for event in source]
            result = source.result
        else:
            result = await agent.run("go", tenant_id="A")
        expected = RunStatus.WAITING_HUMAN_INPUT if kind == "question" else RunStatus.SUCCESS
        assert result.status == expected
        if kind != "tool":
            assert result.answer == fixed
        else:
            calls = await store.get_tool_calls(result.run_id)
            assert calls[0].tool_name == fixed
        for event in events:
            _validate_persistable(event.text)
            _validate_persistable(event.tool_name)
            if event.tool_call is not None:
                _validate_persistable(event.tool_call.name)
        for trace in await store.get_traces(result.run_id):
            _validate_persistable(trace.content)
            _validate_persistable(trace.meta)
        interactions = await store.get_llm_interactions(result.run_id)
        assert interactions, "sanitizing model text must retain interaction evidence"
        for interaction in interactions:
            _validate_persistable(interaction.semantic_response)
            _validate_persistable(interaction.provider_response)
    finally:
        await agent.close()
