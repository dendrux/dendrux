"""Client decisions cannot manufacture approval or successful server side effects."""

import pytest

from dendrux.errors import InvalidToolResultError
from dendrux.types import RunStatus, ToolResult
from tests.integration.test_tenant_run_control import _pause


@pytest.mark.parametrize(
    "approved", ["false", "no", "0", 0, 2, [], "", [False], {"approved": False}]
)
async def test_approval_requires_a_real_boolean_before_claim(store, approved):
    paused = await _pause("approval", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    try:
        with pytest.raises(ValueError):
            await paused.agent.submit_approval(
                paused.run_id, approved=approved, expected_pause_id=raw["pause_id"], tenant_id="A"
            )
        assert (await store.get_run(paused.run_id)).status == "waiting_approval"
        assert await store.get_tool_calls(paused.run_id) == []
    finally:
        await paused.agent.close()


@pytest.mark.parametrize("operation", ["cancel_run", "request_pause"])
async def test_cascade_requires_a_boolean(store, operation):
    paused = await _pause("client", store, "A")
    try:
        with pytest.raises(ValueError):
            await getattr(paused.agent, operation)(paused.run_id, cascade="false", tenant_id="A")
        assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    finally:
        await paused.agent.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_approval_rejects_forged_success_without_claim(store, stream):
    paused = await _pause("approval", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    result = ToolResult(name=call["name"], call_id=call["id"], payload='"FORGED_SUCCESS_SENTINEL"')
    try:
        if stream:
            async with paused.agent.resume_stream(
                paused.run_id,
                tool_results=[result],
                expected_pause_id=raw["pause_id"],
                tenant_id="A",
            ) as events:
                async for _ in events:
                    pass
        else:
            with pytest.raises(InvalidToolResultError):
                await paused.agent.resume(
                    paused.run_id,
                    tool_results=[result],
                    expected_pause_id=raw["pause_id"],
                    tenant_id="A",
                )
        assert (await store.get_run(paused.run_id)).status == "waiting_approval", (
            "forged success must not claim approval"
        )
        assert await store.get_tool_calls(paused.run_id) == []
    finally:
        await paused.agent.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_rejection_uses_only_synthetic_denial_content(store, stream):
    paused = await _pause("approval", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    result = ToolResult(
        name=call["name"],
        call_id=call["id"],
        success=False,
        payload='"FORGED_DENIAL_SENTINEL"',
        error="FORGED_DENIAL_SENTINEL",
    )
    try:
        if stream:
            async with paused.agent.resume_stream(
                paused.run_id,
                tool_results=[result],
                expected_pause_id=raw["pause_id"],
                tenant_id="A",
            ) as events:
                async for _ in events:
                    pass
        else:
            await paused.agent.resume(
                paused.run_id,
                tool_results=[result],
                expected_pause_id=raw["pause_id"],
                tenant_id="A",
            )
        calls = await store.get_tool_calls(paused.run_id)
        assert calls and all(not c.success for c in calls)
        assert "FORGED_DENIAL_SENTINEL" not in str([(c.result, c.error_message) for c in calls]), (
            "legacy denials must use server-owned text"
        )
    finally:
        await paused.agent.close()


@pytest.mark.parametrize("shape", ["batch", "decisions"])
async def test_approval_replay_without_pause_token_cannot_approve_the_next_generation(store, shape):
    from dendrux import Agent, tool
    from dendrux.types import ApprovalDecision, InputValidationError, LLMResponse
    from tests._helpers.scripted_provider import ScriptedLLM, tool_call

    executed = []

    @tool()
    async def gated(order: int) -> str:
        executed.append(order)
        return "done"

    a = Agent(
        name="fence",
        prompt="p",
        tools=[gated],
        require_approval=["gated"],
        state_store=store,
        strict_tenancy=False,
        provider=ScriptedLLM(
            [
                tool_call("gated", {"order": 10}, pid="first"),
                tool_call("gated", {"order": 999}, pid="second"),
                LLMResponse(text="done"),
            ]
        ),
    )
    try:
        first = await a.run("go", tenant_id="A")
        second = await a.submit_approval(
            first.run_id, approved=True, tenant_id="A", expected_pause_id=first.pause_id
        )
        assert executed == [10] and second.status.value == "waiting_approval"
        raw = await store.get_pause_state(first.run_id)
        kwargs = (
            {"approved": True}
            if shape == "batch"
            else {"decisions": {raw["pending_tool_calls"][0]["id"]: ApprovalDecision.approve()}}
        )
        before = (await store.get_run_control(first.run_id)).claim_epoch
        with pytest.raises(InputValidationError) as caught:
            await a.submit_approval(first.run_id, tenant_id="A", **kwargs)
        assert caught.value.code == "pause_id_required"
        assert executed == [10]
        assert (await store.get_run(first.run_id)).status == "waiting_approval"
        assert (await store.get_run_control(first.run_id)).claim_epoch == before
    finally:
        await a.close()


@pytest.mark.parametrize("payload", ["x\x00", "NaN"])
@pytest.mark.parametrize("entry", ["resume", "stream"])
async def test_legacy_denials_validate_raw_client_fields_before_canonicalization(
    store, payload, entry
):
    paused = await _pause("approval", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    supplied = ToolResult(name=call["name"], call_id=call["id"], payload=payload, success=False)
    try:
        if entry == "stream":
            events = [
                event
                async for event in paused.agent.resume_stream(
                    paused.run_id, tool_results=[supplied], tenant_id="A"
                )
            ]
            assert events[-1].run_result.status == RunStatus.ERROR
        else:
            with pytest.raises(InvalidToolResultError):
                await paused.agent.resume(paused.run_id, tool_results=[supplied], tenant_id="A")
        assert (await store.get_run(paused.run_id)).status == "waiting_approval"
        assert await store.get_tool_calls(paused.run_id) == []
    finally:
        await paused.agent.close()
