"""Recovery policy uses current settings and durable execution counts."""

import json

import pytest

from dendrux import tool
from dendrux.runtime import _reclaim
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.recovery_checkpoint import checkpoint
from tests._helpers.runtime_paths import make_agent
from tests.integration.test_cross_path_invariants import paused_recovery


@pytest.mark.parametrize("policy", ["deny", "lowered_cap"])
async def test_recovery_dispatch_rechecks_current_policy(store, engine, policy):
    effects = []

    def build_tool(cap):
        @tool(max_calls_per_run=cap, idempotent=True)
        async def work():
            effects.append(True)
            return "ok"

        return work

    original = make_agent(store, [], tools=[build_tool(5)])
    call = ToolCall("work", id="pending")
    item = await checkpoint(original, store, engine, calls=[call], phases=["dispatchable"])
    state = await store.get_loop_state(item.run_id)
    state["tool_counts"] = {"work": 2}
    await store.checkpoint_step(
        item.run_id, claim_epoch=item.claim_epoch, loop_state=state, pii_mapping=None, trace=None
    )
    current = make_agent(
        store,
        [LLMResponse(text="done")],
        tools=[build_tool(2)],
        deny=["work"] if policy == "deny" else [],
        on_governance_change="warn",
    )
    result = await current.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status == RunStatus.SUCCESS and effects == []
    messages = [t for t in await store.get_traces(item.run_id) if t.role == "tool"]
    assert len(messages) == 1
    if policy == "deny":
        assert messages[0].content == json.dumps({"denied": "Tool 'work' is denied by policy."})
    else:
        assert json.loads(messages[0].content) == {
            "limit": "Tool 'work' has reached its maximum of 2 calls for this run."
        }


@pytest.mark.parametrize("streaming", [False, True])
async def test_stream_recovery_omits_work_callback(store, engine, monkeypatch, streaming):
    agent, paused, effects = await paused_recovery(store, engine)
    resolve = _reclaim.resolve
    callbacks = []

    async def capture(*args, **kwargs):
        callbacks.append(kwargs.get("on_work"))
        return await resolve(*args, **kwargs)

    monkeypatch.setattr(_reclaim, "resolve", capture)
    if streaming:
        async with agent.resume_stream(
            paused.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        ) as stream:
            async for _ in stream:
                pass
        result = stream.result
    else:
        result = await agent.resume(
            paused.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        )
    assert result.status == RunStatus.SUCCESS and effects == [True]
    assert len(callbacks) == 1
    assert (callbacks[0] is None) is streaming
