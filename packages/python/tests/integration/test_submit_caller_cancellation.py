"""Caller cancellation after a committed claim cannot discard the run cleanup."""

import asyncio

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool(target="client")
async def read_range() -> str:
    return ""


async def test_caller_cancel_after_claim_has_no_strands_in_200_trials(store, monkeypatch):
    import dendrux.runtime.submit as submit

    original = submit.claim_or_raise

    async def cancel_after_claim(*args, **kwargs):
        epoch = await original(*args, **kwargs)
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        return epoch

    monkeypatch.setattr(submit, "claim_or_raise", cancel_after_claim)
    for trial in range(200):
        provider = ScriptedLLM([tool_call("read_range", pid="read"), LLMResponse(text="done")])
        async with Agent(
            prompt="p",
            tools=[read_range],
            provider=provider,
            state_store=store,
            strict_tenancy=False,
        ) as agent:
            paused = await agent.run("read", tenant_id="A")
            caller = asyncio.create_task(
                agent.submit_tool_results(
                    paused.run_id,
                    [ToolResult("read_range", paused.pending.tool_calls[0].id, '"ok"')],
                    tenant_id="A",
                )
            )
            with pytest.raises(asyncio.CancelledError):
                await caller
            row = await store.get_run(paused.run_id)
            assert row.status == "cancelled", (trial, row.status)
            events = await store.get_run_events(paused.run_id)
            assert sum(event.event_type == "run.cancelled" for event in events) == 1
            assert not agent._task_manager.is_running(paused.run_id)


@pytest.mark.parametrize("entry", ["input", "approve", "reject", "decisions", "resume"])
async def test_every_other_claimed_submit_cleans_up_caller_cancellation(store, monkeypatch, entry):
    import dendrux.runtime.submit as submit
    from dendrux.types import ApprovalDecision
    from tests.integration.test_tenant_run_control import _pause

    kind = "input" if entry == "input" else "paused" if entry == "resume" else "approval"
    paused = await _pause(kind, store, "A")
    original = submit.claim_or_raise

    async def cancel_after_claim(*args, **kwargs):
        epoch = await original(*args, **kwargs)
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        return epoch

    monkeypatch.setattr(submit, "claim_or_raise", cancel_after_claim)
    agent = paused.agent
    raw = await store.get_pause_state(paused.run_id)
    kwargs = dict(tenant_id="A", expected_pause_id=raw["pause_id"])
    try:
        if entry == "input":
            invocation = agent.submit_input(paused.run_id, "answer", **kwargs)
        elif entry == "resume":
            invocation = agent.submit_resume(paused.run_id, **kwargs)
        elif entry == "decisions":
            invocation = agent.submit_approval(
                paused.run_id,
                decisions={
                    call["id"]: ApprovalDecision.approve() for call in raw["pending_tool_calls"]
                },
                **kwargs,
            )
        else:
            invocation = agent.submit_approval(paused.run_id, approved=entry == "approve", **kwargs)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(invocation)
        assert (await store.get_run(paused.run_id)).status == "cancelled"
        events = await store.get_run_events(paused.run_id)
        assert sum(event.event_type == "run.cancelled" for event in events) == 1
        assert not agent._task_manager.is_running(paused.run_id)
    finally:
        await agent.close()
