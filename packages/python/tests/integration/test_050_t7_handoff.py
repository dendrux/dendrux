"""A revoked claim cannot begin execution during an asynchronous submit handoff."""

import datetime as dt

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, PauseState, RunStatus, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import client_read, result_for


@tool()
async def approved_effect() -> str:
    raise AssertionError("revoked claim executed a tool")


@pytest.mark.parametrize("entry", ["client", "approval", "paused"])
async def test_sweep_between_claim_and_task_handoff_returns_persisted_result(store, entry):
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    provider = ScriptedLLM(
        [
            tool_call("client_read" if entry == "client" else "approved_effect", pid="a"),
            LLMResponse(text="done"),
        ]
    )
    a = Agent(
        name="handoff",
        prompt="p",
        tools=[client_read, approved_effect],
        require_approval=["approved_effect"],
        state_store=store,
        provider=provider,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    if entry == "paused":
        await store.create_run("00000000000000000000000200", "handoff")
        pause = PauseState(
            agent_name="handoff",
            pending_tool_calls=[],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
            pause_id="generation",
        )
        await store.pause_run(
            "00000000000000000000000200", status="paused", pause_data=pause.to_dict()
        )
        run_id, pause_id = "00000000000000000000000200", "generation"
    else:
        paused = await a.run("go")
        run_id, pause_id = paused.run_id, paused.pause_id
    calls = provider.calls
    method = "submit_and_claim" if entry == "client" else "claim_paused_run"
    original = getattr(store, method)

    async def claim_then_sweep(run_id, **kwargs):
        epoch = await original(run_id, **kwargs)
        if epoch:
            assert await store.sweep_stale_runs(dt.timedelta(0))
        return epoch

    setattr(store, method, claim_then_sweep)
    try:
        if entry == "client":
            result = await a.submit_tool_results(
                run_id, [result_for(paused)], expected_pause_id=pause_id
            )
        elif entry == "approval":
            result = await a.submit_approval(run_id, approved=True, expected_pause_id=pause_id)
        else:
            result = await a.submit_resume(run_id, expected_pause_id=pause_id)
    finally:
        setattr(store, method, original)
    assert result.status == RunStatus.ERROR
    assert (await store.get_run(run_id)).status == "error"
    assert provider.calls == calls


@pytest.mark.parametrize("entry", ["run", "stream", "resume", "resume_stream"])
async def test_sweep_during_llm_stops_the_returned_tool_before_execution(store, entry):
    import asyncio

    from tests.integration.test_050_t7_integrity import consume

    effects = []

    @tool()
    async def effect() -> str:
        effects.append("ran")
        return "ok"

    responses = [tool_call("effect", pid="effect"), LLMResponse(text="done")]
    resumed = entry.startswith("resume")
    if resumed:
        responses.insert(0, tool_call("client_read", pid="client"))
    provider = ScriptedLLM(responses, gate_at=1 if resumed else 0)
    a = Agent(
        name="fenced",
        prompt="p",
        tools=[client_read, effect],
        state_store=store,
        provider=provider,
        strict_tenancy=False,
    )
    if resumed:
        paused = await a.run("go")
        action = (
            a.resume(paused.run_id, tool_results=[result_for(paused)])
            if entry == "resume"
            else consume(a.resume_stream(paused.run_id, tool_results=[result_for(paused)]))
        )
    else:
        action = a.run("go") if entry == "run" else consume(a.stream("go"))
    task = asyncio.create_task(action)
    try:
        await asyncio.wait_for(provider.entered.wait(), 5)
        assert await store.sweep_stale_runs(dt.timedelta(0))
    finally:
        provider.gate.set()
        outcome = await asyncio.wait_for(task, 5)
    assert effects == []
    result = outcome[0] if isinstance(outcome, tuple) else outcome
    assert result.status == RunStatus.ERROR
    assert provider.calls == (2 if resumed else 1)
