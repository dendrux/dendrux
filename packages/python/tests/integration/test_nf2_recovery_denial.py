"""A committed denial survives a crash before recovery consumes it."""

import json
from unittest.mock import patch

import pytest

from dendrux import tool
from dendrux.types import ApprovalDecision, LLMResponse, RunStatus, ToolCall
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import make_agent


@pytest.mark.parametrize("streaming", [False, True])
async def test_reclaimed_denial_matches_approved_result(store, engine, streaming):
    @tool()
    async def work():
        pytest.fail("a denied tool must not execute")

    call = ToolCall("work")
    agent = make_agent(
        store,
        [LLMResponse(tool_calls=[call]), LLMResponse(text="done")],
        tools=[work],
        require_approval=["work"],
    )
    if streaming:
        async with agent.stream("input", tenant_id="tenant") as stream:
            async for _ in stream:
                pass
        paused = stream.result
    else:
        paused = await agent.run("input", tenant_id="tenant")

    class Crash(BaseException):
        pass

    submit = store.submit_and_claim

    async def crash_after_claim(*args, **kwargs):
        await submit(*args, **kwargs)
        raise Crash()

    reason = 'human says "no"'
    with patch.object(store, "submit_and_claim", crash_after_claim), pytest.raises(Crash):
        await agent.submit_approval(
            paused.run_id,
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            decisions={call.id: ApprovalDecision(approved=False, reason=reason)},
        )
    item = await expire(store, engine, paused.run_id)
    result = await agent.reclaim(paused.run_id, tenant_id="tenant", item=item)
    assert result.status == RunStatus.SUCCESS
    observations = [row for row in await store.get_traces(paused.run_id) if row.role == "tool"]
    assert len(observations) == 1
    assert observations[0].content == json.dumps({"denied": reason})
    assert observations[0].meta["is_error"] is True
    assert (await store.get_loop_state(paused.run_id))["submission"] is None
