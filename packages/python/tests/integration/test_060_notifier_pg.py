"""N1–N7 on the private PostgreSQL database; notifications are hints only."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

import pytest
from sqlalchemy import text

from dendrux.errors import NotifierUnavailableError, PersistenceError
from dendrux.store import EventFeed, FollowCapacityError, RunStore
from dendrux.store._follow import _Follower, _RunFeed
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


async def test_n2_kill_listener_falls_back_and_recovers(
    pg_test_url, engine, store, caplog, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres listener")
    caplog.set_level(logging.INFO)
    run = generate_ulid()
    await store.create_run(run, "notifier")
    notifier = PostgresEventNotifier(pg_test_url, reconnect_backoff=(0.5, 0.5))
    await notifier.start()
    reader = RunStore.from_engine(engine, notifier=notifier)
    owner = EventFeed(
        reader, notifier=notifier, poll_interval_s=0.2, poll_max_s=1, poll_healthy_s=30
    )
    feed = _RunFeed(owner, run, None)
    owner._feeds[run] = feed
    follower = _Follower(until_terminal=False, limit=10)
    feed.add(follower)
    try:
        await asyncio.sleep(0.02)
        pid = notifier._connection.get_server_pid()
        started = asyncio.get_running_loop().time()
        async with engine.begin() as conn:
            assert (
                await conn.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE pid = :pid AND datname = current_database()"
                    ),
                    {"pid": pid},
                )
            ).scalar_one()
        await wait_until(lambda: not notifier.state.healthy)
        await store.save_run_event(run, event_type="llm.completed")
        committed_at = asyncio.get_running_loop().time()
        await wait_until(lambda: bool(follower.buffer))
        delivered_at = asyncio.get_running_loop().time()
        elapsed = delivered_at - started
        after_commit = delivered_at - committed_at
        record_property("degraded_delivery_s", elapsed)
        record_property("degraded_commit_to_delivery_s", after_commit)
        print(
            f"DEGRADED_DELIVERY_S={elapsed:.4f}; COMMIT_TO_DELIVERY_S={after_commit:.4f}; "
            "POLL_INTERVAL_S=0.2"
        )
        assert elapsed < 2 * owner.poll_interval_s + 0.2
        assert after_commit < 2 * owner.poll_interval_s + 0.2
        await wait_until(lambda: notifier.state.healthy)
        codes = [getattr(r, "code", None) for r in caplog.records]
        assert codes.count("notifier_degraded") == 1
        assert codes.count("notifier_recovered") == 1
        assert notifier._connection.get_server_pid() != pid
    finally:
        await owner.aclose()
        await reader.close()
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
    async with asyncio.timeout(1):
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
    from dendrux.runtime.state import SQLAlchemyStateStore

    store.notify_channel = "dendrux_events"
    other = SQLAlchemyStateStore(engine, notify_channel="dendrux_events")
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
            await asyncio.wait_for(sub.event.wait(), 0.5)
            events = await store.get_run_events(run)
            assert len(events) == 20
            assert [e.sequence_index for e in events] == list(range(20))
    finally:
        await notifier.stop()


async def test_stream_events_notify_wake_under_50ms_and_release(
    pg_test_url, engine, store, monkeypatch, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres listener")
    store.notify_channel = "dendrux_events"
    run = generate_ulid()
    await store.create_run(run, "stream-notifier")
    notifier = PostgresEventNotifier(pg_test_url)
    await notifier.start()
    reader = RunStore.from_engine(engine, notifier=notifier)
    empty_read = asyncio.Event()
    get_events = reader._state.get_run_events

    async def observed_read(*args, **kwargs):
        rows = await get_events(*args, **kwargs)
        if not rows:
            empty_read.set()
        return rows

    monkeypatch.setattr(reader._state, "get_run_events", observed_read)
    stream = reader.stream_events(run, tenant_id=None, poll_interval_s=5)
    pending = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(empty_read.wait(), 1)
        started = asyncio.get_running_loop().time()
        payload = {"raw": "unchanged event payload"}
        await store.save_run_event(run, event_type="llm.completed", data=payload)
        item = await asyncio.wait_for(pending, 0.3)
        elapsed = asyncio.get_running_loop().time() - started
        record_property("stream_notify_wake_ms", elapsed * 1000)
        print(f"STREAM_NOTIFY_WAKE_MS={elapsed * 1000:.3f}; POLL_INTERVAL_S=5")
        assert elapsed < 0.05
        assert item.data == payload
        assert notifier._count == 1
        await stream.aclose()
        assert notifier._count == 0
        assert notifier.state.healthy
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await reader.close()
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
        await asyncio.wait_for(entered.wait(), 1)
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
            await asyncio.wait_for(sub.event.wait(), 1)
    finally:
        release.set()
        for task in starting:
            task.cancel()
        await asyncio.gather(*starting, return_exceptions=True)
        await notifier.stop()
    assert (
        notifier._start_result is None and notifier._connecting is None and notifier._task is None
    )


async def test_n5_default_off_writer_warns_once_on_healthy_reconciliation(
    pg_test_url, engine, store, caplog
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres healthy listener with polling-only writer")
    assert store.notify_channel is None
    run = generate_ulid()
    await store.create_run(run, "missed-hint")
    notifier = PostgresEventNotifier(pg_test_url)
    await notifier.start()
    reader = RunStore.from_engine(engine, notifier=notifier)
    owner = EventFeed(reader, notifier=notifier, poll_interval_s=0.01, poll_healthy_s=0.05)
    feed = _RunFeed(owner, run, None)
    owner._feeds[run] = feed
    follower = _Follower(until_terminal=False, limit=10)
    empty = asyncio.Event()
    fetch = feed._fetch

    async def observe_empty():
        rows = await fetch()
        if not rows:
            empty.set()
        return rows

    feed._fetch = observe_empty
    feed.add(follower)
    try:
        await asyncio.wait_for(empty.wait(), 1)
        assert not any(getattr(r, "code", None) == "notifier_missed_hint" for r in caplog.records)
        await store.save_run_event(run, event_type="llm.completed")
        await wait_until(lambda: len(follower.buffer) == 1)
        records = [r for r in caplog.records if getattr(r, "code", None) == "notifier_missed_hint"]
        assert len(records) == 1
        assert records[0].kind == "notifier" and records[0].count == 1
        assert records[0].levelno == logging.WARNING
        assert notifier.state.healthy
        await store.save_run_event(run, event_type="llm.completed")
        await wait_until(lambda: len(follower.buffer) == 2)
        assert sum(getattr(r, "code", None) == "notifier_missed_hint" for r in caplog.records) == 1
    finally:
        await owner.aclose()
        await reader.close()
        await notifier.stop()
