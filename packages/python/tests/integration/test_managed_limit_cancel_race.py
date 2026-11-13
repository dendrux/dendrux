"""Root-first admission against mark-before-scan cancellation on independent pools."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import RunCancelledError
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import RunStatus
from tests.integration.test_managed_bind import bind_request, managed_parent
from tests.integration.test_managed_cancel import controller
from tests.integration.test_managed_limits import member


async def wait_for_cancel_lock(engine, application_name):
    async def observed():
        async with engine.connect() as connection:
            while not await connection.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                    "WHERE datname = current_database() AND application_name = :name "
                    "AND wait_event_type = 'Lock')"
                ),
                {"name": application_name},
            ):
                await connection.rollback()  # Refresh pg_stat_activity's transaction snapshot.
                await asyncio.sleep(0.01)

    await asyncio.wait_for(observed(), timeout=10)


async def close_tasks(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("round_number", range(3))
@pytest.mark.parametrize("schedule", ["together", "bind_first", "cancel_first"])
async def test_pg_cancel_versus_eight_grandchild_binds(
    store, engine, monkeypatch, schedule, round_number
):
    if engine.dialect.name != "postgresql":
        pytest.skip("independent row-lock races require Postgres; SQLite shares a connection")
    root, _ = await managed_parent(store)
    parents = [await member(store, root, name=f"p{i}") for i in range(8)]
    requests = [
        bind_request(parent, call, root=root, path=(f"p{i}", "reviewer"))
        for i, (parent, call) in enumerate(parents)
    ]
    name = f"t5c-cancel-{root}"
    deadlocks, at_mark = [], []
    root_locked, release_bind, root_marked, release_scan = (asyncio.Event() for _ in range(4))

    def database_error(context):
        if getattr(context.original_exception, "sqlstate", None) == "40P01":
            deadlocks.append("40P01")

    async with AsyncExitStack() as stack:
        stores = []
        for i in range(9):
            pool = create_async_engine(
                engine.url,
                pool_size=1,
                max_overflow=0,
                connect_args={
                    "server_settings": {"application_name": name if i == 8 else "t5c-bind"}
                },
            )
            stack.push_async_callback(pool.dispose)
            event.listen(pool.sync_engine, "handle_error", database_error)
            stores.append(store_for_engine(pool))
        original_lock = stores[0]._lock_managed_root
        original_mark = stores[8].mark_cancel_requested

        async def gated_lock(session, request):
            await original_lock(session, request)
            if schedule == "bind_first":
                root_locked.set()
                await release_bind.wait()

        async def marked(ids, *, tenant_id):
            result = await original_mark(ids, tenant_id=tenant_id)
            if ids == [root]:
                async with AsyncSession(engine) as session:
                    at_mark.extend(
                        (
                            await session.scalars(
                                select(AgentRun.id).where(
                                    AgentRun.id.in_([r.child_run_id for r in requests])
                                )
                            )
                        ).all()
                    )
                root_marked.set()
                if schedule == "cancel_first":
                    await release_scan.wait()
            return result

        async def bind(index):
            if schedule == "cancel_first" or (schedule == "bind_first" and index):
                await root_marked.wait()
            return await stores[index].bind_managed_child(requests[index])

        monkeypatch.setattr(stores[0], "_lock_managed_root", gated_lock)
        monkeypatch.setattr(stores[8], "mark_cancel_requested", marked)
        tasks = []
        stack.push_async_callback(close_tasks, tasks)
        async with asyncio.timeout(30):
            binders = [asyncio.create_task(bind(i)) for i in range(8)]
            tasks.extend(binders)
            if schedule == "bind_first":
                await root_locked.wait()
            cancel = asyncio.create_task(controller(stores[8]).cancel_run(root))
            tasks.append(cancel)
            if schedule == "bind_first":
                await wait_for_cancel_lock(engine, name)
                release_bind.set()
            results = await asyncio.gather(*binders, return_exceptions=True)
            release_scan.set()
            cancelled = await cancel
    # A running owner observes the durable cancel at its next checkpoint.
    assert cancelled.status == RunStatus.RUNNING
    assert not deadlocks
    assert all(
        not isinstance(result, BaseException) or isinstance(result, RunCancelledError)
        for result in results
    ), results
    created = {
        result.child_run_id for result in results if getattr(result, "outcome", None) == "created"
    }
    assert created == set(at_mark)
    if schedule != "together":
        assert len(created) == (1 if schedule == "bind_first" else 0)
    async with AsyncSession(engine) as session:
        rows = (await session.scalars(select(AgentRun))).all()
        intents = (await session.scalars(select(ToolCallRecord))).all()
    assert {row.id for row in rows} == {root, *(parent for parent, _ in parents), *created}
    assert len(intents) == len(created)
    assert {intent.meta["managed"]["child_run_id"] for intent in intents} == created
    assert all(row.cancel_requested or row.status == "cancelled" for row in rows)
    epochs = {row.id: row.claim_epoch for row in rows}
    for (parent, _), request in zip(parents, requests, strict=True):
        assert epochs[parent] == (1 if request.child_run_id in created else 0)
    print(
        f"T5C_CANCEL_RACE schedule={schedule} round={round_number} "
        f"created={len(created)} deadlocks=0 orphans=0"
    )
