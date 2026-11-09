"""Recovery counts measure crashes since real progress across public result paths."""

from dataclasses import replace

import pytest
from sqlalchemy import update

from dendrux import tool
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotReclaimableError
from dendrux.runtime.persistence import PersistenceRecorder, message_trace
from dendrux.runtime.state import ClaimResult, claimed_store
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import Message, Role, ToolCall
from tests._helpers.recovery_checkpoint import checkpoint, expire
from tests.integration.test_060_reclaim_agent import make_agent
from tests.integration.test_061_progress_store import STAMP, _set_streak


async def _claim(agent, store, item):
    return await store.reclaim_run(
        item.run_id,
        tenant_id="tenant",
        agent_name=agent.name,
        expected_claim_epoch=item.claim_epoch,
        governance=governance_snapshot(agent),
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )


async def test_061_zero_streak_keeps_recovery_summary(store, engine):
    @tool(idempotent=True)
    async def work() -> str:
        return "ok"

    agent = make_agent(store, tools=[work])
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[ToolCall("work")],
        phases=["dispatchable"],
        started=True,
        idempotent=True,
    )
    direct = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert direct.status.value == "success"
    assert direct.recovery is not None
    assert direct.recovery.reclaim_attempts == 0
    assert direct.recovery.last_reclaimed_at is not None
    assert len(direct.recovery.replayed_tool_calls) == 1
    loaded = await make_agent(store).get_result(item.run_id, tenant_id="tenant")
    assert loaded.recovery == direct.recovery
    await agent.close()


@pytest.mark.parametrize("progress", [False, True])
async def test_061_recovery_limit_counts_only_consecutive_crashes(store, engine, progress):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    for index in range(6 if progress else 3):
        claim = await _claim(agent, store, item)
        assert (await store.get_run(item.run_id)).reclaim_attempts == (1 if progress else index + 1)
        if progress:
            state = await store.get_loop_state(item.run_id)
            state["iteration"] += 1
            await store.checkpoint_step(
                item.run_id,
                claim_epoch=claim.claim_epoch,
                loop_state=state,
                pii_mapping=None,
                trace=message_trace(Message(Role.ASSISTANT, "advancing"), 10 + index),
            )
            assert (await store.get_run(item.run_id)).reclaim_attempts == 0
        if progress or index < 2:
            item = await expire(store, engine, item.run_id)
    if not progress:
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).where(AgentRun.id == item.run_id).values(lease_expired_at=STAMP)
            )
        with pytest.raises(RunNotReclaimableError) as caught:
            await _claim(agent, store, replace(item, claim_epoch=claim.claim_epoch))
        assert caught.value.reason == "reclaim_limit"
    await agent.close()


@pytest.mark.parametrize("message", [False, True])
async def test_061_reused_observation_does_not_reset_streak(store, engine, message):
    agent = make_agent(store)
    call = ToolCall("work")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["dispatchable"])
    state = await store.get_loop_state(item.run_id)
    await _set_streak(engine, item.run_id)
    recorder = PersistenceRecorder(
        claimed_store(store, item.run_id, ClaimResult(item.claim_epoch, 2, state, None)),
        item.run_id,
    )
    await recorder.resolve_observation(
        call.id, Message(Role.TOOL, "reused", call_id=call.id, name="work") if message else None
    )
    row = await store.get_run(item.run_id)
    assert row.reclaim_attempts == 2 and row.last_reclaimed_at == STAMP


@pytest.mark.parametrize("operation", ["checkpoint", "heartbeat", "approval", "input", "pause"])
async def test_061_nonprogress_preserves_streak(store, engine, operation):
    from dendrux.runtime.state import LoopStatePatch

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    state = await store.get_loop_state(item.run_id)
    await _set_streak(engine, item.run_id)
    if operation == "checkpoint":
        await store.checkpoint_step(
            item.run_id,
            claim_epoch=item.claim_epoch,
            loop_state=state,
            pii_mapping=None,
            trace=None,
        )
    elif operation == "heartbeat":
        await store.touch_progress(item.run_id, claim_epoch=item.claim_epoch)
    elif operation == "pause":
        await store.request_pause(item.run_id)
        await store.pause_with_event(
            item.run_id, status="paused", pause_data={}, claim_epoch=item.claim_epoch
        )
    else:
        state["submission"] = {"submission_id": "s", "kind": operation}
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).where(AgentRun.id == item.run_id).values(loop_state=state)
            )
        await store.consume_submission(
            item.run_id,
            claim_epoch=item.claim_epoch,
            submission_id="s",
            traces=[message_trace(Message(Role.USER, "reply"), 10)] if operation == "input" else [],
            patch=LoopStatePatch(),
        )
    assert (await store.get_run(item.run_id)).reclaim_attempts == 2


async def test_061_two_process_reclaim_progress_has_one_owner(store, engine):
    import asyncio
    import json
    import os
    import sys

    if engine.dialect.name != "postgresql":
        pytest.skip("Two-process ownership race requires PostgreSQL")
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    await _set_streak(engine, item.run_id)
    code = """
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from dendrux.errors import RunNotReclaimableError
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.persistence import message_trace
from dendrux.types import Message, Role
async def main():
    config = json.loads(sys.stdin.readline())
    engine = create_async_engine(config.pop("url"))
    store = store_for_engine(engine)
    async with engine.connect():
        pass
    print("ready", flush=True)
    sys.stdin.readline()
    try:
        claim = await store.reclaim_run(**config)
        if claim is False:
            print("lost", flush=True)
            return
        state = dict(claim.loop_state)
        state["iteration"] += 1
        await store.checkpoint_step(config["run_id"], claim_epoch=claim.claim_epoch,
            loop_state=state, pii_mapping=None,
            trace=message_trace(Message(Role.ASSISTANT, "progress"), 10))
        print("won", flush=True)
    except RunNotReclaimableError:
        print("lost", flush=True)
    finally:
        await engine.dispose()
asyncio.run(main())
"""
    config = dict(
        url=engine.url.render_as_string(hide_password=False),
        run_id=item.run_id,
        tenant_id="tenant",
        agent_name=agent.name,
        expected_claim_epoch=item.claim_epoch,
        governance=governance_snapshot(agent),
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )
    children = []
    try:
        for _ in range(2):
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
            )
            children.append(child)
            child.stdin.write((json.dumps(config) + "\n").encode())
            await child.stdin.drain()
        for child in children:
            assert await asyncio.wait_for(child.stdout.readline(), 15) == b"ready\n"
        for child in children:
            child.stdin.write(b"go\n")
            await child.stdin.drain()
        output = await asyncio.wait_for(asyncio.gather(*(p.communicate() for p in children)), 20)
        assert all(p.returncode == 0 for p in children), [err.decode() for _, err in output]
        assert sorted(out.decode().strip() for out, _ in output) == ["lost", "won"]
        row = await store.get_run(item.run_id)
        assert row.reclaim_attempts == 0 and row.claim_epoch == item.claim_epoch + 1
        assert row.last_reclaimed_at is not None
        assert sum(t.content == "progress" for t in await store.get_traces(item.run_id)) == 1
        assert (
            sum(e.event_type == "run.reclaimed" for e in await store.get_run_events(item.run_id))
            == 1
        )
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
                await child.wait()
        await agent.close()


async def test_061_pause_resume_approval_cycle_is_not_progress(store, engine, monkeypatch):
    from tests.integration.test_061_recovery_pause import _agent, _continue, _requested_pause

    effects = []
    agent = _agent(store, effects)
    paused, _ = await _requested_pause(agent, store, engine, monkeypatch)
    await _set_streak(engine, paused.run_id)
    result = await _continue(agent, paused.run_id, paused.pause_id, "submit_resume")
    assert result.status.value == "waiting_approval"
    assert (await store.get_run(paused.run_id)).reclaim_attempts == 2
    assert effects == []
    await agent.close()
