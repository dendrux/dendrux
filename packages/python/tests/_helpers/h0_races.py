"""Deterministic rendezvous and independent Postgres callers for H0."""

import asyncio
from contextlib import asynccontextmanager
from functools import wraps

import pytest_asyncio
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import Base
from dendrux.runtime.stores._factory import store_for_engine


class Gate:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.args = None

    async def wait(self, *args):
        self.args = args
        self.entered.set()
        await self.release.wait()

    def install(self, monkeypatch, owner, method, *, after=False, when=None):
        original = getattr(owner, method)

        @wraps(original)
        async def gated(*args, **kwargs):
            selected = not self.entered.is_set() and (when is None or when(*args, **kwargs))
            if selected and not after:
                await self.wait(*args)
            result = await original(*args, **kwargs)
            if selected and after:
                await self.wait(*args)
            return result

        monkeypatch.setattr(owner, method, gated)


@asynccontextmanager
async def running(coro, *gates):
    task = asyncio.create_task(coro)
    try:
        async with asyncio.timeout(15):
            yield task
    finally:
        for gate in gates:
            gate.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest_asyncio.fixture
async def peer(engine):
    """Separate pools; assert that every SQL statement uses distinct backend PIDs."""
    assert engine.dialect.name == "postgresql"
    other = create_async_engine(engine.url, pool_size=2, max_overflow=0)
    pids = [set(), set()]

    def recorder(index):
        def record(conn, cursor, statement, parameters, context, executemany):
            pids[index].add(conn.connection.driver_connection.get_server_pid())

        return record

    listeners = [recorder(0), recorder(1)]
    for source, listener in zip((engine, other), listeners, strict=True):
        event.listen(source.sync_engine, "before_cursor_execute", listener)
    try:
        yield store_for_engine(other), other
        assert all(pids), "both callers must execute real SQL"
        assert pids[0].isdisjoint(pids[1]), "callers must use independent connections"
    finally:
        for source, listener in zip((engine, other), listeners, strict=True):
            event.remove(source.sync_engine, "before_cursor_execute", listener)
        await other.dispose()


async def snapshot(engine):
    async with engine.connect() as conn:
        return await connection_snapshot(conn)


async def connection_snapshot(conn):
    return {
        table.name: (await conn.execute(select(table).order_by(*table.primary_key))).all()
        for table in Base.metadata.sorted_tables
    }


async def wait_for_lock(conn, task):
    """Prove the other caller is waiting on our row lock, not merely scheduled."""
    while True:
        blocked = await conn.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "AND pg_backend_pid() = ANY(pg_blocking_pids(pid))"
            )
        )
        if blocked:
            return
        if task.done():
            raise AssertionError(
                "claim finished before taking the locked row"
            ) from task.exception()
        await asyncio.sleep(0.01)


async def event_counts(store, run_id):
    from collections import Counter

    return Counter(e.event_type for e in await store.get_run_events(run_id))
