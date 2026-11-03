"""Durable execution counters survive changes to model-visible history."""

from functools import wraps

import pytest
from sqlalchemy import update

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.loops import react
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import checkpoint
from tests._helpers.runtime_paths import (
    SUBMIT_KINDS,
    client_read,
    client_result,
    make_agent,
    submission_case,
)
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


async def save_counts(store, engine, run_id, counts):
    state = await store.get_loop_state(run_id)
    state["tool_counts"] = counts
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(loop_state=state))


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("history_count", [0, 1])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
async def test_saved_counts_override_history_with_legacy_fallback(
    store, engine, monkeypatch, storage_format, history_count, entry
):
    await pin_storage_format(engine, storage_format)
    effects = []

    @tool(max_calls_per_run=1)
    async def work() -> str:
        """Record one execution."""
        effects.append(True)
        return "ok"

    first = [tool_call("work", pid="prior")] if history_count else []
    agent = make_agent(
        store,
        [
            *first,
            tool_call("client_read", pid="pause"),
            tool_call("work", pid="next"),
            LLMResponse(text="done"),
        ],
        tools=[work, client_read],
    )
    paused = await agent.run("go")
    assert len(effects) == history_count
    if storage_format == 2:
        # Empty saved counts are authoritative too, even with a tool observation.
        await save_counts(store, engine, paused.run_id, {} if history_count else {"work": 1})
    initial_counts = []
    init = react._init_loop_state

    async def capture_initial_counts(**kwargs):
        state = await init(**kwargs)
        initial_counts.append(dict(state.call_counts))
        return state

    monkeypatch.setattr(react, "_init_loop_state", capture_initial_counts)
    submitted = [client_result(paused)]
    if entry == "resume":
        result = await agent.resume(paused.run_id, tool_results=submitted)
    elif entry == "submit_tool_results":
        result = await agent.submit_tool_results(paused.run_id, submitted)
    else:
        async with agent.resume_stream(paused.run_id, tool_results=submitted) as stream:
            async for _ in stream:
                pass
        result = stream.result
    assert result.status == RunStatus.SUCCESS
    assert len(effects) == (2 * history_count if storage_format == 2 else 1)
    expected = 1 - history_count if storage_format == 2 else history_count
    assert initial_counts[-1].get("work", 0) == expected


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
async def test_submission_continuations_keep_saved_counts(store, engine, kind):
    effects = []

    @tool(max_calls_per_run=1)
    async def work() -> str:
        """Must be refused after a saved execution."""
        effects.append(True)
        return "ok"

    original, run_id, invoke = await submission_case(store, kind)
    agent = Agent(
        name=original.name,
        prompt=original.prompt,
        tools=[*original.tools, work],
        require_approval=original.require_approval,
        provider=ScriptedLLM([tool_call("work", pid="next"), LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    await save_counts(store, engine, run_id, {"work": 1})
    result = await invoke(agent=agent)
    assert result.status == RunStatus.SUCCESS
    assert effects == []


@pytest.mark.parametrize("entry", ["reclaim", "resume", "resume_stream", "submit_resume"])
async def test_recovery_continuation_keeps_saved_counts(store, engine, monkeypatch, entry):
    effects = []

    @tool(max_calls_per_run=1)
    async def work() -> str:
        """Must be refused after a saved execution."""
        effects.append("work")
        return "ok"

    @tool(idempotent=True)
    async def recover() -> str:
        """Exercise pending recovery before the live continuation."""
        effects.append("recover")
        return "ok"

    agent = make_agent(
        store,
        [tool_call("work", pid="next"), LLMResponse(text="done")],
        tools=[work, recover],
    )
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[ToolCall("recover")],
        phases=["dispatchable"],
    )
    await save_counts(store, engine, item.run_id, {"work": 1})
    if entry == "reclaim":
        result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    else:
        reclaim = store.reclaim_run

        @wraps(reclaim)
        async def claim_then_pause(*args, **kwargs):
            claim = await reclaim(*args, **kwargs)
            await store.request_pause(item.run_id)
            return claim

        with monkeypatch.context() as patcher:
            patcher.setattr(store, "reclaim_run", claim_then_pause)
            paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
        assert paused.status == RunStatus.PAUSED and effects == []
        kwargs = dict(tenant_id="tenant", expected_pause_id=paused.pause_id)
        if entry == "resume_stream":
            async with agent.resume_stream(item.run_id, **kwargs) as stream:
                async for _ in stream:
                    pass
            result = stream.result
        elif entry == "resume":
            result = await agent.resume(item.run_id, **kwargs)
        else:
            result = await agent.submit_resume(item.run_id, **kwargs)
    assert result.status == RunStatus.SUCCESS
    assert effects == ["recover"]
