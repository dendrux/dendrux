"""Transform failures remain closed through persistence and model requests."""

import json
import logging

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests.unit.test_061_safe_diagnostics import (
    SAFE_TRANSFORM,
    SENTINEL,
    arbitrary_error,
    assert_safe_records,
    diagnostic_value,
)


@pytest.mark.parametrize("kind", ["value", "custom", "chain"])
@pytest.mark.parametrize("entry", ["run", "stream", "approval"])
async def test_061_arbitrary_exception_text_never_leaks(engine, kind, entry, caplog):
    caplog.set_level(logging.WARNING, logger="dendrux")
    error = arbitrary_error(kind)

    def transform(ctx):
        raise error

    call = ToolCall(name="diagnostic_value", params={}, provider_tool_call_id="provider-call")
    llm = MockLLM([LLMResponse(tool_calls=[call]), LLMResponse(text="done")])
    db_store = store_for_engine(engine)
    agent = Agent(
        provider=llm,
        prompt="diagnostics",
        tools=[diagnostic_value],
        state_store=db_store,
        strict_tenancy=False,
        tool_result_transform=transform,
        require_approval=["diagnostic_value"] if entry == "approval" else None,
    )
    emitted = []
    try:
        if entry == "stream":
            async with agent.stream("go") as stream:
                emitted = [event async for event in stream]
            result = stream.result
        else:
            result = await agent.run("go")
            if entry == "approval":
                result = await agent.submit_approval(
                    result.run_id,
                    approved=True,
                    expected_pause_id=result.pause_id,
                )
        assert result.status == RunStatus.SUCCESS
        store = RunStore.from_engine(engine)
        invocations = await store.get_tool_invocations(result.run_id, limit=1000)
        traces = await store.get_traces(result.run_id, limit=1000)
        events = await db_store.get_run_events(result.run_id)
        assert SENTINEL not in repr((emitted, invocations, traces, events, llm.call_history))
        (invocation,) = invocations
        assert invocation.success is False
        assert invocation.error == SAFE_TRANSFORM
        assert invocation.result == {"error": SAFE_TRANSFORM}
        tool_messages = [m for m in llm.call_history[1]["messages"] if m.role.value == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0].content) == {"error": SAFE_TRANSFORM}
        assert_safe_records([r for r in caplog.records if r.name.startswith("dendrux")])
    finally:
        await agent.close()
