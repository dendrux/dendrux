"""Progress from captured tool contexts must not renew inactive run leases."""

import asyncio
import contextvars
import datetime as dt

import pytest
from sqlalchemy import select, update

from dendrux.db.models import AgentRun
from dendrux.runtime import context


@pytest.mark.parametrize(
    "status",
    [
        "running",
        "paused",
        "waiting_client_tool",
        "waiting_human_input",
        "waiting_approval",
        "success",
        "error",
        "cancelled",
        "max_iterations",
    ],
)
async def test_progress_only_touches_running_rows(store, engine, status):
    await store.create_run("progress", "agent")
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(status=status, last_progress_at=old, updated_at=old)
        )
        before = (await conn.execute(select(AgentRun.last_progress_at, AgentRun.updated_at))).one()
    await store.touch_progress("progress")
    async with engine.connect() as conn:
        after = (await conn.execute(select(AgentRun.last_progress_at, AgentRun.updated_at))).one()
    if status == "running":
        assert all(new > previous for new, previous in zip(after, before, strict=True))
    else:
        assert after == before, "progress must not refresh a paused or terminal run"


async def test_zombie_thread_heartbeat_does_not_keep_pause_alive(store, engine, monkeypatch):
    await store.create_run("zombie", "agent")
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(
                status="paused", status_changed_at=old, updated_at=old, last_progress_at=old
            )
        )
    done = asyncio.Event()
    touch = store.touch_progress

    async def observed(run_id):
        try:
            await touch(run_id)
        finally:
            done.set()

    monkeypatch.setattr(store, "touch_progress", observed)
    token = context.set_delegation_context(
        context.DelegationContext(
            run_id="zombie", persisted=True, state_store=store, loop=asyncio.get_running_loop()
        )
    )
    try:
        captured = contextvars.copy_context()
    finally:
        context.reset_delegation_context(token)
    await asyncio.to_thread(captured.run, context.heartbeat)
    await asyncio.wait_for(done.wait(), 20)
    swept = await store.sweep_abandoned_runs(dt.timedelta(days=1))
    assert [r.run_id for r in swept] == ["zombie"], "a zombie tool heartbeat must not renew a pause"
