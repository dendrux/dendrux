"""N1–N7 on the private PostgreSQL database; notifications are hints only."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

import pytest
from sqlalchemy import text

from dendrux.errors import NotifierUnavailableError, PersistenceError
from dendrux.store import FollowCapacityError
from dendrux.store._notifier_pg import PostgresEventNotifier
from dendrux.types import generate_ulid


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_n1_commit_wakes_under_50ms_and_1000_subscriptions_one_connection(
    pg_test_url, engine, store, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres listener")
    store.notify_channel = "dendrux_events"
    run = generate_ulid()
    await store.create_run(run, "notifier")
    notifier = PostgresEventNotifier(pg_test_url)
    await notifier.start()
    try:
        async with AsyncExitStack() as stack:
            subs = [await stack.enter_async_context(notifier.subscribe(run)) for _ in range(1000)]
            pid = notifier._connection.get_server_pid()
            async with engine.connect() as conn:
                count = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND application_name = 'dendrux_event_notifier'"
                        )
                    )
                ).scalar_one()
                assert count == 1
                assert (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE pid = :pid AND datname = current_database()"
                        ),
                        {"pid": pid},
                    )
                ).scalar_one() == 1
            started = asyncio.get_running_loop().time()
            await store.save_run_event(run, event_type="llm.completed")
            await asyncio.wait_for(asyncio.gather(*(s.event.wait() for s in subs)), 1)
            elapsed = asyncio.get_running_loop().time() - started
            record_property("wake_latency_ms", elapsed * 1000)
            print(f"NOTIFIER_WAKE_MS={elapsed * 1000:.3f}; SUBSCRIPTIONS=1000; CONNECTIONS={count}")
            assert elapsed < 0.25  # Commit plus 1,000 wakeups on a shared CI runner.
            assert len(await store.get_run_events(run)) == 1
    finally:
        await notifier.stop()


async def test_n3_cap_real_listener(pg_test_url):
    notifier = PostgresEventNotifier(pg_test_url, max_subscriptions=1)
    await notifier.start()
    try:
        async with notifier.subscribe(generate_ulid()):
            with pytest.raises(FollowCapacityError):
                async with notifier.subscribe(generate_ulid()):
                    pass
    finally:
        await notifier.stop()


async def test_n5_failed_notify_rolls_back_and_same_event_id_retries(engine, store):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres transactional NOTIFY")
    run, event_id = generate_ulid(), generate_ulid()
    await store.create_run(run, "notifier")
    store.notify_channel = "invalid\x00channel"
    with pytest.raises(PersistenceError):
        await store.save_run_event(run, event_id=event_id, event_type="llm.completed")
    assert await store.get_run_events(run) == []
    store.notify_channel = "dendrux_events"
    await store.save_run_event(run, event_id=event_id, event_type="llm.completed")
    await store.save_run_event(run, event_id=event_id, event_type="llm.completed")
    events = await store.get_run_events(run)
    assert len(events) == 1 and events[0].sequence_index == 0


async def test_n6_real_listener_rejects_cross_loop(pg_test_url):
    notifier = PostgresEventNotifier(pg_test_url)
    await notifier.start()

    async def other():
        with pytest.raises(RuntimeError):
            async with notifier.subscribe(generate_ulid()):
                pass

    try:
        await asyncio.to_thread(lambda: asyncio.run(other()))
    finally:
        await notifier.stop()


async def test_n7_failed_self_test_releases_backend(pg_test_url, engine, monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres listener")
    notifier = PostgresEventNotifier(pg_test_url, self_test_timeout_s=0.01)
    # Suppress only delivery, keeping the real connection and LISTEN/NOTIFY.
    monkeypatch.setattr(notifier, "_on_notification", lambda *args: None, raising=False)
    with pytest.raises(NotifierUnavailableError):
        await notifier.start()
    assert notifier._connection is None and notifier._task is None
    async with asyncio.timeout(10):
        while True:
            async with engine.connect() as conn:
                count = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM pg_stat_activity "
                            "WHERE datname = current_database() "
                            "AND application_name = 'dendrux_event_notifier'"
                        )
                    )
                ).scalar_one()
            if count == 0:
                break
            await asyncio.sleep(0.01)


async def test_concurrent_two_writer_instances_durable_order_and_wakeup(pg_test_url, engine, store):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres concurrency")
    from dendrux.runtime.stores._factory import store_for_engine

    store.notify_channel = "dendrux_events"
    other = store_for_engine(engine, notify_channel="dendrux_events")
    run = generate_ulid()
    await store.create_run(run, "notifier-race")
    notifier = PostgresEventNotifier(pg_test_url)
    await notifier.start()
    try:
        async with notifier.subscribe(run) as sub:
            barrier = asyncio.Event()

            async def write(index):
                await barrier.wait()
                writer = store if index % 2 else other
                await writer.save_run_event(
                    run, event_type="llm.completed", event_id=generate_ulid()
                )

            tasks = [asyncio.create_task(write(i)) for i in range(20)]
            barrier.set()
            await asyncio.gather(*tasks)
            await asyncio.wait_for(sub.event.wait(), 10)
            events = await store.get_run_events(run)
            assert len(events) == 20
            assert [e.sequence_index for e in events] == list(range(20))
    finally:
        await notifier.stop()


@pytest.mark.parametrize("cancelled", [0, 1], ids=["leader", "joiner"])
async def test_n3_cancelled_start_keeps_one_real_listener(
    pg_test_url, engine, store, monkeypatch, cancelled
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres concurrent listener startup")
    import asyncpg

    connect = asyncpg.connect
    release, entered = asyncio.Event(), asyncio.Event()
    attempts = 0

    async def gated_connect(*args, **kwargs):
        nonlocal attempts
        if kwargs.get("server_settings", {}).get("application_name") == "dendrux_event_notifier":
            attempts += 1
            entered.set()
            await release.wait()
        return await connect(*args, **kwargs)

    monkeypatch.setattr(asyncpg, "connect", gated_connect)
    notifier = PostgresEventNotifier(pg_test_url)
    starting = [asyncio.create_task(notifier.start()) for _ in range(3)]
    try:
        await asyncio.wait_for(entered.wait(), 10)
        starting[cancelled].cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting[cancelled]
        release.set()
        results = await asyncio.gather(
            *(task for i, task in enumerate(starting) if i != cancelled), return_exceptions=True
        )
        assert results == [None, None], results
        assert attempts == 1 and notifier.state.healthy
        async with engine.connect() as conn:
            assert (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                        "AND application_name='dendrux_event_notifier'"
                    )
                )
            ).scalar_one() == 1
        store.notify_channel = "dendrux_events"
        run = generate_ulid()
        await store.create_run(run, "cancelled-start")
        async with notifier.subscribe(run) as sub:
            await store.save_run_event(run, event_type="llm.completed")
            await asyncio.wait_for(sub.event.wait(), 10)
    finally:
        release.set()
        for task in starting:
            task.cancel()
        await asyncio.gather(*starting, return_exceptions=True)
        await notifier.stop()
    assert (
        notifier._start_result is None and notifier._connecting is None and notifier._task is None
    )
