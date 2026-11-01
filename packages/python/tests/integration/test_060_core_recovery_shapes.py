"""Recovery result agreement and loop identity through the actual Agent paths."""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from functools import wraps

import pytest
from sqlalchemy import update

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, ToolCall
from tests.integration.test_060_core_recovery import mark, reclaim, stale


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_d5_direct_cached_and_fresh_process_recovery_agree(store, engine, monkeypatch):
    stamp = datetime(2026, 10, 29, tzinfo=UTC)
    complete = store.complete_tool_call

    @wraps(complete)
    async def replayed(run_id, *, claim_epoch, observation, event, loop_state_patch, **kwargs):
        kwargs["replayed"] = True
        await complete(
            run_id,
            claim_epoch=claim_epoch,
            observation=observation,
            event=event,
            loop_state_patch=loop_state_patch,
            **kwargs,
        )
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(reclaim_attempts=1, last_reclaimed_at=stamp)
            )

    monkeypatch.setattr(store, "complete_tool_call", replayed)

    @tool(idempotent=True)
    async def work() -> str:
        return "ok"

    a = Agent(
        name="shape",
        prompt="p",
        tools=[work],
        state_store=store,
        strict_tenancy=False,
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")]),
    )
    direct = await a.run("go", idempotency_key="same")
    cached = await a.run("go", idempotency_key="same")
    read = await a.get_result(direct.run_id, tenant_id=None)
    assert direct.recovery.reclaim_attempts == 1
    assert len(direct.recovery.replayed_tool_calls) == 1
    shape = direct.to_dict()["recovery"]
    assert shape["last_reclaimed_at"] == "2026-10-29T00:00:00Z"
    assert cached.to_dict()["recovery"] == read.to_dict()["recovery"] == shape
    code = """
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
async def main():
    engine = create_async_engine(sys.argv[1])
    agent = Agent(name="shape", prompt="p", provider=MockLLM([]),
                  state_store=store_for_engine(engine), strict_tenancy=False)
    result = await agent.get_result(sys.argv[2], tenant_id=None)
    print(json.dumps(result.to_dict()["recovery"]))
    await engine.dispose()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        engine.url.render_as_string(hide_password=False),
        direct.run_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
    )
    output, error = await asyncio.wait_for(process.communicate(), 15)
    assert process.returncode == 0, error.decode()
    assert json.loads(output) == shape
    terminal_read = await a.cancel_run(direct.run_id, tenant_id=None)
    assert terminal_read.to_dict()["recovery"] == shape


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_runtime_loop_marker_preserves_strategy(store, engine, single, streaming):
    from dendrux.loops import ReActLoop, SingleCall
    from dendrux.types import RunEventType

    observed = []

    class Model(MockLLM):
        async def complete(self, *args, **kwargs):
            run = (await store.list_runs())[0]
            state = await store.get_loop_state(run.id)
            observed.append(state.get("loop"))
            return await super().complete(*args, **kwargs)

        async def complete_stream(self, *args, **kwargs):
            from dendrux.types import StreamEvent, StreamEventType

            response = await self.complete(*args, **kwargs)
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)
            yield StreamEvent(type=StreamEventType.DONE, raw=response)

    a = Agent(
        prompt="p",
        provider=Model([LLMResponse(text="done")]),
        loop=SingleCall() if single else ReActLoop(),
        state_store=store,
        strict_tenancy=False,
    )
    if streaming:
        async with a.stream("go") as events:
            async for event in events:
                if event.run_result is not None:
                    result = event.run_result
                    assert event.type == RunEventType.RUN_COMPLETED, result.error
    else:
        result = await a.run("go")
    assert observed == ["single_call" if single else "react"]
    row = await store.get_run(result.run_id)
    assert row.strategy == "NativeToolCalling"
    assert (await store.get_loop_state(result.run_id))["loop"] == observed[0]
    assert result.recovery is None


async def test_unknown_loop_never_marks_or_finalizes(store, engine):
    from dendrux.errors import RunNotReclaimableError

    ids = await stale(store, engine)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == ids[0]).values(loop_state={}))
    before = await store.get_run(ids[0])
    result = await mark(store)
    assert result.inspected == 1
    assert not result.reclaimable and not result.finalised and not result.legacy
    assert await store.get_run(ids[0]) == before
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ids[0])
            .values(lease_expired_at=datetime.now(UTC), reclaim_limit=3)
        )
    with pytest.raises(RunNotReclaimableError) as caught:
        await store.reclaim_run(
            ids[0],
            tenant_id="tenant",
            agent_name="recovery",
            expected_claim_epoch=0,
            governance={},
            governance_policy="raise",
            skill_pins={},
            loop_type="react",
        )
    assert caught.value.reason == "loop_unknown"


async def test_cancelled_reclaim_is_distinct_and_audited_once(store, engine):
    from dendrux.errors import RunNotReclaimableError

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    await store.request_cancel(item.run_id)
    with pytest.raises(RunNotReclaimableError) as caught:
        await reclaim(store, item)
    assert caught.value.reason == "cancelled"
    assert (await store.get_run(item.run_id)).status == "cancelled"
    assert [e.event_type for e in await store.get_run_events(item.run_id)].count(
        "run.cancelled"
    ) == 1


async def test_reclaim_lost_commit_ack_refuses_unproven_ownership(store, engine, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.durability import StoreOutcomeUnknown
    from tests._helpers.db_errors import LostConnectionError

    await stale(store, engine)
    item = (await mark(store)).reclaimable[0]
    commit = AsyncSession.commit
    calls = 0

    async def lost_ack(session):
        nonlocal calls
        await commit(session)
        calls += 1
        if calls == 1:
            raise OperationalError("connection reset", {}, LostConnectionError())

    monkeypatch.setattr(AsyncSession, "commit", lost_ack)
    with pytest.raises(StoreOutcomeUnknown):
        await reclaim(store, item)
    assert calls == 1
    row = await store.get_run(item.run_id)
    assert row.claim_epoch == 2
    assert row.reclaim_attempts == 1
    assert [e.event_type for e in await store.get_run_events(item.run_id)].count(
        "run.reclaimed"
    ) == 1


@pytest.mark.parametrize("single", [False, True])
async def test_mark_uses_checkpoint_loop_and_ignores_strategy(store, engine, single):
    ids = await stale(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == ids[0])
            .values(
                strategy="NativeToolCalling" if single else "single_call",
                loop_state={"loop": "single_call" if single else "react"},
            )
        )
    result = await mark(store)
    if single:
        assert result.finalised[0].failure_reason == "stale_running"
    else:
        assert len(result.reclaimable) == 1 and not result.finalised
        assert await reclaim(store, result.reclaimable[0])


@pytest.mark.parametrize("size", [2, 101])
async def test_recovery_wire_caps_replayed_ids(store, engine, size):
    from dendrux.db.models import ToolCallRecord

    await store.create_run("many-replayed", "shape")
    stamp = datetime(2026, 10, 29, tzinfo=UTC)
    async with store._session_factory() as session:
        session.add_all(
            [
                ToolCallRecord(
                    id=f"{i:026d}",
                    agent_run_id="many-replayed",
                    tool_call_id=f"call-{i:03d}",
                    tool_name="work",
                    target="server",
                    result="ok",
                    success=True,
                    duration_ms=1,
                    iteration_index=1,
                    replayed_at=stamp,
                )
                for i in range(size)
            ]
        )
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "many-replayed")
            .values(reclaim_attempts=1, last_reclaimed_at=stamp)
        )
        await session.commit()
    a = Agent(
        name="shape", prompt="p", provider=MockLLM([]), state_store=store, strict_tenancy=False
    )
    result = await a.get_result("many-replayed", tenant_id=None)
    assert len(result.recovery.replayed_tool_calls) == min(size, 100)
    assert result.recovery.replayed_truncated == (size > 100)
