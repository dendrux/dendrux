"""N4: force backend reassignment, rather than relying on pool timing.

CI supplies transaction and session pool URLs pointing at the private test DB.
Requires a pool with at least two backend connections and transaction mode's
LISTEN command permitted (startup self-test is deliberately only best effort).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack

import pytest
from sqlalchemy.engine import make_url

from dendrux.store import EventFeed, RunStore
from dendrux.store._follow import _Follower, _RunFeed
from dendrux.store._notifier_pg import PostgresEventNotifier
from dendrux.types import generate_ulid


@pytest.mark.parametrize("session_pool", [False, True], ids=["transaction", "session"])
async def test_n4_controlled_delivery_with_backend_reassignment(
    pg_test_url, engine, store, session_pool, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres listener")
    key = "DENDRUX_TEST_PGBOUNCER_SESSION_URL" if session_pool else "DENDRUX_TEST_PGBOUNCER_URL"
    endpoint = os.environ.get(key)
    if not endpoint:
        pytest.skip(f"{key} not set; PgBouncer is not installed locally")
    parsed = make_url(endpoint)
    assert parsed.database == make_url(pg_test_url).database
    import asyncpg

    connect = asyncpg.connect

    async def uncached_connect(*args, **kwargs):
        # Isolate LISTEN loss from transaction pooling's prepared-statement policy.
        kwargs["statement_cache_size"] = 0
        kwargs.setdefault("timeout", 5)
        return await connect(*args, **kwargs)

    monkeypatch.setattr(asyncpg, "connect", uncached_connect)
    notifier = PostgresEventNotifier(endpoint)
    await notifier.start()
    reader = RunStore.from_engine(engine, notifier=notifier)
    owner = EventFeed(reader, poll_healthy_s=0.2)
    run = generate_ulid()
    await store.create_run(run, "pgbouncer")
    feed = _RunFeed(owner, run, None)
    owner._feeds[run] = feed
    follower = _Follower(until_terminal=False, limit=10)
    feed.add(follower)
    try:
        async with AsyncExitStack() as stack:
            sub = await stack.enter_async_context(notifier.subscribe(run))
            listener = notifier._connection
            # Pin the backend while installing LISTEN and recording its real pid.
            async with listener.transaction():
                await listener.execute('LISTEN "dendrux_events"')
                listened_pid = await listener.fetchval("SELECT pg_backend_pid()", timeout=2)
            blockers = []
            if not session_pool:
                # Hold the listened backend in another frontend's transaction,
                # guaranteeing that the listener frontend must be reassigned.
                for _ in range(16):
                    blocker = await asyncpg.connect(
                        parsed.set(drivername="postgresql").render_as_string(hide_password=False),
                        statement_cache_size=0,
                    )
                    stack.push_async_callback(blocker.close)
                    transaction = blocker.transaction()
                    await transaction.start()
                    stack.push_async_callback(transaction.rollback)
                    blockers.append(blocker)
                    if await blocker.fetchval("SELECT pg_backend_pid()", timeout=2) == listened_pid:
                        break
                else:
                    pytest.fail("Could not reserve the LISTEN backend within 16 connections")
                assert await listener.fetchval("SELECT pg_backend_pid()", timeout=2) != listened_pid
            # A healthy feed must reconcile even when every hint is lost.
            await asyncio.sleep(0.02)
            sub.event.clear()
            await store.save_run_event(run, event_type="llm.completed")
            if session_pool:
                await asyncio.wait_for(sub.event.wait(), 0.1)
            else:
                await asyncio.sleep(0.05)
                assert not sub.event.is_set()
            async with asyncio.timeout(1):
                while not follower.buffer:
                    await asyncio.sleep(0.001)
            assert follower.buffer[0].event_type == "llm.completed"
    finally:
        await owner.aclose()
        await reader.close()
        await notifier.stop()
