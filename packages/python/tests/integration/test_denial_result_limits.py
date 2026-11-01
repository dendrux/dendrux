"""Approval denial results obey the client submission bounds before claiming."""

import asyncio
import json

import pytest

from dendrux import Agent, tool
from dendrux.errors import InvalidToolResultError, RunAlreadyClaimedError
from dendrux.types import ApprovalDecision, Limits, LLMResponse, ToolCall, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM


@tool()
async def gated() -> str:
    return "executed"


@pytest.mark.parametrize("shape", ["batch", "decisions"])
@pytest.mark.parametrize(
    "reason,count,per_result,batch",
    [
        ("x" * 80, 1, 64, 2048),
        ("é" * 20, 1, 64, 2048),
        ("x" * 64, 3, 512, 193),
    ],
)
async def test_denial_limit_leaves_pause_unclaimed(
    store, caplog, shape, reason, count, per_result, batch
):
    calls = [ToolCall(name="gated", params={}) for _ in range(count)]
    agent = Agent(
        strict_tenancy=False,
        prompt="p",
        tools=[gated],
        require_approval=["gated"],
        state_store=store,
        limits=Limits(max_tool_result_bytes=per_result, max_input_bytes=batch),
        provider=ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")]),
    )
    try:
        paused = await agent.run("go")
        before = await store.get_run_control(paused.run_id)
        kwargs = (
            {"approved": False, "rejection_reason": reason}
            if shape == "batch"
            else {"decisions": {call.id: ApprovalDecision.deny(reason) for call in calls}}
        )
        with pytest.raises(InvalidToolResultError):
            await agent.submit_approval(paused.run_id, expected_pause_id=paused.pause_id, **kwargs)
        after = await store.get_run_control(paused.run_id)
        assert after.status == "waiting_approval"
        assert after.claim_epoch == before.claim_epoch
        assert await store.get_tool_calls(paused.run_id) == []
        assert any(
            getattr(r, "run_id", None) == paused.run_id
            and getattr(r, "code", None) == "invalid_tool_result"
            for r in caplog.records
        )
    finally:
        if hasattr(agent.provider, "gate"):
            agent.provider.gate.set()
        await agent.close()


async def test_denial_batch_bound_matches_client_submission(store):
    from dendrux.runtime.submit import prepare_client_results
    from dendrux.types import PauseState

    calls = [ToolCall(name="gated", params={}) for _ in range(3)]
    agent = Agent(prompt="p", limits=Limits(max_tool_result_bytes=512, max_input_bytes=193))
    results = [
        ToolResult(name=c.name, call_id=c.id, payload=json.dumps("x" * 64), success=False)
        for c in calls
    ]
    with pytest.raises(InvalidToolResultError):
        await prepare_client_results(
            agent,
            "00000000000000000000000102",
            PauseState(
                agent_name="a",
                history=[],
                steps=[],
                iteration=0,
                trace_order_offset=0,
                usage=None,
                pending_tool_calls=calls,
            ),
            results,
        )


async def test_parallel_denials_claim_once(store):
    from dendrux.errors import RunAlreadyTerminalError, RunNotPausedError

    agent = Agent(
        strict_tenancy=False,
        prompt="p",
        tools=[gated],
        require_approval=["gated"],
        state_store=store,
        provider=ScriptedLLM(
            [LLMResponse(tool_calls=[ToolCall(name="gated", params={})]), LLMResponse(text="done")],
            gate_at=1,
        ),
    )
    try:
        paused = await agent.run("go")
        tasks = [
            asyncio.create_task(
                agent.submit_approval(
                    paused.run_id, approved=False, expected_pause_id=paused.pause_id
                )
            )
            for _ in range(8)
        ]
        await asyncio.wait_for(agent.provider.entered.wait(), 5)
        pending = set(tasks)
        while len(pending) > 1:
            done, pending = await asyncio.wait(
                pending, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "competing claims did not finish"
        agent.provider.gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(not isinstance(r, BaseException) for r in results) == 1
        assert all(
            not isinstance(r, BaseException)
            or isinstance(r, (RunAlreadyClaimedError, RunAlreadyTerminalError, RunNotPausedError))
            for r in results
        )
        assert len(await store.get_tool_calls(paused.run_id)) == 1
    finally:
        if hasattr(agent.provider, "gate"):
            agent.provider.gate.set()
        await agent.close()
