"""Empty successful provider envelopes fail consistently at every invocation boundary."""

import pytest
from pydantic import BaseModel

from dendrux import Agent
from dendrux.errors import ProviderError
from dendrux.llm._calls import complete_call
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests.integration.test_tenant_run_control import _pause
from tests.unit.test_050_providers import (
    c_completion,
    chat_provider,
    json_handler,
    r_response,
    responses_provider,
    wire,
)


@pytest.mark.parametrize("adapter", ["chat", "responses"])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("resume", [False, True])
async def test_empty_completion_is_persisted_error(store, adapter, single, stream, resume):
    if resume and single:
        pytest.skip("SingleCall explicitly rejects resume")
    provider = chat_provider() if adapter == "chat" else responses_provider()
    wire(
        provider,
        json_handler(200, c_completion({"content": ""}) if adapter == "chat" else r_response()),
    )
    # Normalize the real successful wire envelope, then exercise the loop paths
    # through a deterministic provider with batch and streaming parity.
    response = await provider.complete([])
    await provider.close()
    mock = MockLLM([response])
    agent = Agent(
        strict_tenancy=False,
        prompt="p",
        provider=mock,
        state_store=store,
        loop=SingleCall() if single else None,
    )
    paused_agent = None
    try:
        if resume:
            paused = await _pause("input", store, "A")
            paused_agent = paused.agent
            # A continuation may choose either loop on the same named agent.
            agent.name = paused_agent.name
            if stream:
                events = [
                    e
                    async for e in agent.resume_stream(
                        paused.run_id, user_input="yes", tenant_id="A"
                    )
                ]
                result = events[-1].run_result
            else:
                with pytest.raises(ProviderError) as caught:
                    await agent.resume(paused.run_id, user_input="yes", tenant_id="A")
                result = await agent.get_result(caught.value.run_id, tenant_id="A")
        elif stream:
            events = [e async for e in agent.stream("go", tenant_id="A")]
            result = events[-1].run_result
        else:
            with pytest.raises(ProviderError) as caught:
                await agent.run("go", tenant_id="A")
            result = await agent.get_result(caught.value.run_id, tenant_id="A")
        assert result.status == RunStatus.ERROR
        assert result.error_info.category == "rejected"
        assert result.error_info.provider_code == "empty_completion"
        record = await store.get_run(result.run_id)
        assert record.status == "error"
    finally:
        await agent.close()
        if paused_agent:
            await paused_agent.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("single", [False, True])
async def test_empty_token_limited_completion_remains_truncated(store, single, stream):
    agent = Agent(
        strict_tenancy=False,
        prompt="p",
        provider=MockLLM([LLMResponse(text="", stop_reason="max_tokens")]),
        state_store=store,
        loop=SingleCall() if single else None,
    )
    try:
        if stream:
            result = [e async for e in agent.stream("go")][-1].run_result
        else:
            result = await agent.run("go")
        assert result.truncated is True
        assert result.error_info is None
    finally:
        await agent.close()


async def test_tool_only_completion_is_actionable():
    result = await complete_call(
        MockLLM([LLMResponse(tool_calls=[ToolCall(name="f", params={})])]), []
    )
    assert result.tool_calls


async def test_empty_structured_object_is_valid(store):
    class Empty(BaseModel):
        pass

    from dataclasses import replace

    mock = MockLLM([LLMResponse(text="{}")])
    mock.capabilities = replace(mock.capabilities, supports_structured_output=True)
    agent = Agent(
        strict_tenancy=False,
        prompt="p",
        loop=SingleCall(),
        output_type=Empty,
        state_store=store,
        provider=mock,
    )
    try:
        result = await agent.run("go")
        assert result.status == RunStatus.SUCCESS
        assert isinstance(result.output, Empty)
    finally:
        await agent.close()
