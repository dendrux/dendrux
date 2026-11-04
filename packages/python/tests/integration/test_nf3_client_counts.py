"""Client observations spend the same durable call budget on both formats."""

import pytest

from dendrux import tool
from dendrux.types import LLMResponse, Role, RunStatus, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests._helpers.runtime_paths import make_agent
from tests._helpers.scripted_provider import tool_call


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("origin", ["run", "stream"])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("success", [False, True])
async def test_client_call_limit_survives_each_pause(
    store, engine, storage_format, origin, entry, limit, success
):
    await pin_storage_format(engine, storage_format)

    @tool(target="client", max_calls_per_run=limit)
    async def read() -> str:
        """Read on the client."""
        raise AssertionError("client tool ran on the server")

    initial = make_agent(store, [tool_call("read", pid="first")], tools=[read])
    if origin == "run":
        paused = await initial.run("go")
    else:
        async with initial.stream("go") as stream:
            async for _ in stream:
                pass
        paused = stream.result

    for used in range(1, limit + 1):
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        resumed = make_agent(
            store, [tool_call("read", pid=f"next-{used}"), LLMResponse(text="done")], tools=[read]
        )
        results = [
            ToolResult(
                name="read",
                payload="ok",
                call_id=paused.pending.tool_calls[0].id,
                success=success,
                error=None if success else "client failed",
            )
        ]
        if entry == "resume":
            result = await resumed.resume(paused.run_id, tool_results=results)
        elif entry == "submit_tool_results":
            result = await resumed.submit_tool_results(paused.run_id, results)
        else:
            async with resumed.resume_stream(paused.run_id, tool_results=results) as stream:
                async for _ in stream:
                    pass
            result = stream.result
        assert result.status == (
            RunStatus.SUCCESS if used == limit else RunStatus.WAITING_CLIENT_TOOL
        )
        if storage_format == 2:
            assert (await store.get_loop_state(paused.run_id))["tool_counts"]["read"] == used
        paused = result
    observations = [m for m in resumed.provider.messages[-1] if m.role == Role.TOOL]
    assert observations[-1].meta["dendrux.not_executed"] is True


@pytest.mark.parametrize("cut", ["acknowledgement", "reclaim"])
async def test_client_count_survives_consumption_retry(store, engine, monkeypatch, cut):
    from functools import wraps

    from dendrux.errors import PersistenceError
    from dendrux.runtime.persistence import PersistenceRecorder
    from tests._helpers.h0_races import Gate, running
    from tests._helpers.recovery_checkpoint import expire

    await pin_storage_format(engine, 2)

    @tool(target="client", max_calls_per_run=1)
    async def read() -> str:
        """Read on the client."""
        raise AssertionError("client executed on server")

    paused = await make_agent(store, [tool_call("read", pid="first")], tools=[read]).run("go")
    agent = make_agent(
        store, [tool_call("read", pid="limited"), LLMResponse(text="done")], tools=[read]
    )
    results = [ToolResult(name="read", payload="ok", call_id=paused.pending.tool_calls[0].id)]
    calls = []
    original = store.consume_submission

    @wraps(original)
    async def consume(*args, **kwargs):
        committed = await original(*args, **kwargs)
        calls.append(committed)
        if cut == "acknowledgement" and len(calls) == 1:
            raise PersistenceError(retryable=True)
        return committed

    monkeypatch.setattr(store, "consume_submission", consume)
    if cut == "acknowledgement":
        result = await agent.submit_tool_results(paused.run_id, results)
        assert calls == [True, False]
    else:
        gate = Gate()
        gate.install(monkeypatch, PersistenceRecorder, "on_llm_call_started")
        async with running(agent.submit_tool_results(paused.run_id, results), gate) as task:
            await gate.entered.wait()
            item = await expire(store, engine, paused.run_id)
            gate.release.set()
            await task
        result = await agent.reclaim(paused.run_id, item=item, tenant_id=None)
        assert calls == [True]
    assert result.status == RunStatus.SUCCESS
    assert (await store.get_loop_state(paused.run_id))["tool_counts"] == {"read": 1}
    traces = await store.get_traces(paused.run_id)
    assert sum((t.meta or {}).get("call_id") == results[0].call_id for t in traces) == 1


@pytest.mark.parametrize("engine", ["postgres"], indirect=True)
async def test_concurrent_client_submissions_count_once(store, engine, monkeypatch):
    import asyncio
    from contextlib import AsyncExitStack

    from sqlalchemy import select

    from dendrux.db.models import AgentRun
    from dendrux.errors import RunAlreadyClaimedError, RunAlreadyTerminalError
    from tests._helpers.h0_races import Gate, independent_callers, running, wait_for_lock
    from tests._helpers.scripted_provider import ScriptedLLM

    await pin_storage_format(engine, 2)

    @tool(target="client", max_calls_per_run=1)
    async def read() -> str:
        """Read on the client."""
        raise AssertionError("client executed on server")

    seed = make_agent(store, [tool_call("read", pid="first")], tools=[read])
    paused = await seed.run("go")
    result = ToolResult(name="read", payload="ok", call_id=paused.pending.tool_calls[0].id)
    providers = [
        ScriptedLLM([tool_call("read", pid="limited"), LLMResponse(text="done")]) for _ in range(4)
    ]
    async with independent_callers(engine, 4) as callers, AsyncExitStack() as stack:
        gates, tasks = [], []
        for (caller, _), provider in zip(callers, providers, strict=True):
            agent = make_agent(caller, provider, tools=[read])
            stack.push_async_callback(agent.close)
            gate = Gate()
            gate.install(monkeypatch, caller, "submit_and_claim")
            gates.append(gate)
            tasks.append(
                await stack.enter_async_context(
                    running(
                        agent.submit_tool_results(
                            paused.run_id, [result], expected_pause_id=paused.pause_id
                        ),
                        gate,
                    )
                )
            )
        await asyncio.gather(*(gate.entered.wait() for gate in gates))
        async with engine.connect() as conn:
            tx = await conn.begin()
            await conn.execute(
                select(AgentRun.id).where(AgentRun.id == paused.run_id).with_for_update()
            )
            for gate in gates:
                gate.release.set()
            await wait_for_lock(conn, tasks[0], count=4)
            await tx.commit()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        winners = [i for i, outcome in enumerate(outcomes) if not isinstance(outcome, Exception)]
        assert len(winners) == 1
        for i, outcome in enumerate(outcomes):
            if i == winners[0]:
                assert outcome.status == RunStatus.SUCCESS
                assert providers[i].calls == 2
            else:
                assert isinstance(outcome, (RunAlreadyClaimedError, RunAlreadyTerminalError))
                assert providers[i].calls == 0
    assert (await store.get_loop_state(paused.run_id))["tool_counts"] == {"read": 1}
    traces = await store.get_traces(paused.run_id)
    assert sum((t.meta or {}).get("call_id") == result.call_id for t in traces) == 1
    await seed.close()
