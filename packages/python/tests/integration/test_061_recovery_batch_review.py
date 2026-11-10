"""Evaluated approval batches and durable recovery observations retain their meaning."""

import json
from unittest.mock import patch

import pytest

from dendrux import Agent, tool
from dendrux.runtime.context import get_delegation_context
from dendrux.types import ApprovalDecision, LLMResponse, ToolCall, ToolDecision
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_061_recovery_batch import Crash
from tests.integration.test_061_recovery_pause import _continue


@pytest.mark.parametrize("entry", ["submit_resume", "resume", "resume_stream"])
async def test_061_approval_barrier_keeps_evaluated_sibling_and_denial(store, entry):
    effects, hooks = [], []

    @tool()
    async def a() -> str:
        """Execute the approved call and request an operator pause."""
        effects.append("a")
        assert await store.request_pause(get_delegation_context().run_id)
        return "a-done"

    @tool()
    async def c() -> str:
        """Execute the already evaluated ungated sibling."""
        effects.append("c")
        return "c-done"

    @tool()
    async def b() -> str:
        """Never execute the denied call."""
        effects.append("b")
        return "b-done"

    def before(context):
        hooks.append(context.tool_name)

    calls = [ToolCall(n) for n in ["a", "c", "b"]]
    llm = ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="Test",
        tools=[a, c, b],
        state_store=store,
        before_tool=before,
        require_approval=["a", "b"],
    )
    waiting = await agent.run("go", tenant_id="tenant")
    ids = {tc.name: tc.id for tc in waiting.pending.tool_calls}
    paused = await agent.submit_approval(
        waiting.run_id,
        tenant_id="tenant",
        expected_pause_id=waiting.pause_id,
        decisions={
            ids["a"]: ApprovalDecision.approve(),
            ids["b"]: ApprovalDecision.deny("not today"),
        },
    )
    assert paused.status.value == "paused" and effects == ["a"]
    raw = await store.get_pause_state(waiting.run_id)
    assert raw["pending_decisions"] == {ids["b"]: {"approved": False, "reason": "not today"}}
    initial_hooks = list(hooks)
    result = await _continue(agent, paused.run_id, paused.pause_id, entry)
    observations = {m.name: m.content for m in llm.messages[-1] if m.role.value == "tool"}
    assert (hooks, observations["b"]) == (initial_hooks, json.dumps({"denied": "not today"}))
    assert result.status.value == "success" and effects == ["a", "c"]
    await agent.close()


@pytest.mark.parametrize("boundary", ["second_crash", "requested"])
async def test_061_recovery_retains_recorded_hook_gate(store, engine, boundary):
    from dendrux.runtime import _reclaim
    from tests.integration.test_060_reclaim_agent import make_agent

    effects = []
    seen = 0

    @tool()
    async def write_region() -> str:
        """Wait for approval."""
        effects.append("write_region")
        return "ok"

    @tool()
    async def post_review_notice() -> str:
        """Wait behind the gate."""
        effects.append("post_review_notice")
        return "ok"

    def before(context):
        nonlocal seen
        if context.tool_name == "write_region":
            seen += 1
            reason = (
                "Review region" if boundary == "second_crash" or seen == 1 else "Changed reason"
            )
            return ToolDecision.require_approval(reason)
        return None

    agent = make_agent(store, tools=[write_region, post_review_notice], before_tool=before)
    calls = [ToolCall("write_region"), ToolCall("post_review_notice")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable"] * 2)
    prepare = _reclaim._dispatch_recovery_call

    async def pause_after_hook(*args, **kwargs):
        outcome = await prepare(*args, **kwargs)
        if outcome.pending_approval:
            assert await store.request_pause(item.run_id)
        return outcome

    async def crash(*args, **kwargs):
        raise Crash()

    if boundary == "second_crash":
        with patch.object(_reclaim, "_recovery_pause_result", crash), pytest.raises(Crash):
            await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
        item = await expire(store, engine, item.run_id)
        result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    else:
        with patch.object(_reclaim, "_dispatch_recovery_call", pause_after_hook):
            paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
        assert paused.status.value == "paused"
        result = await agent.submit_resume(
            item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        )
    assert result.status.value == "waiting_approval" and effects == []
    raw = await store.get_pause_state(item.run_id)
    assert raw["approval_gates"] == {calls[0].id: {"source": "hook", "reason": "Review region"}}
    events = await store.get_run_events(item.run_id)
    approvals = [e for e in events if e.event_type == "approval.requested"]
    assert len(approvals) == 2
    assert all(
        e.data["source"] == "hook" and e.data["reason"] == "Review region" for e in approvals
    )
    assert await store.get_tool_calls(item.run_id) == []
    await agent.close()


async def test_061_whole_decline_observations_match_after_crash(store, engine):
    observations = []
    effects = []
    for crash_after_submit in [False, True]:
        effects.clear()

        @tool()
        async def write_region() -> str:
            """Do not execute declined work."""
            effects.append("write_region")
            return "ok"

        @tool()
        async def post_review_notice() -> str:
            """Do not execute a declined sibling."""
            effects.append("post_review_notice")
            return "ok"

        calls = [ToolCall("write_region"), ToolCall("post_review_notice")]
        llm = ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
        agent = Agent(
            provider=llm,
            prompt="Test",
            state_store=store,
            tools=[write_region, post_review_notice],
            require_approval=["write_region"],
        )
        waiting = await agent.run("go", tenant_id="tenant")
        submit = store.submit_and_claim

        async def crash(*args, _submit=submit, **kwargs):
            await _submit(*args, **kwargs)
            raise Crash()

        if crash_after_submit:
            with patch.object(store, "submit_and_claim", crash), pytest.raises(Crash):
                await agent.submit_approval(
                    waiting.run_id,
                    tenant_id="tenant",
                    expected_pause_id=waiting.pause_id,
                    approved=False,
                    rejection_reason="not today",
                )
            item = await expire(store, engine, waiting.run_id)
            result = await agent.reclaim(waiting.run_id, tenant_id="tenant", item=item)
        else:
            result = await agent.submit_approval(
                waiting.run_id,
                tenant_id="tenant",
                expected_pause_id=waiting.pause_id,
                approved=False,
                rejection_reason="not today",
            )
        assert result.status.value == "success" and effects == []
        observations.append({m.name: m.content for m in llm.messages[-1] if m.role.value == "tool"})
        await agent.close()
    assert observations[0] == {c.name: json.dumps("not today") for c in calls}
    assert observations[1] == observations[0]


async def test_061_unresolved_gate_survives_hook_substitution(store, engine):
    from tests.integration.test_060_reclaim_agent import make_agent

    effects = []

    @tool()
    async def work() -> str:
        """Wait for the outstanding decision."""
        effects.append(True)
        return "ok"

    def before(context):
        return ToolDecision.substitute("replacement")

    agent = make_agent(store, tools=[work], before_tool=before)
    call = ToolCall("work")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["gated"])
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_approval"
    assert effects == [] and await store.get_tool_calls(item.run_id) == []
    await agent.close()
