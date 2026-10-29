"""Notifier lifecycle, limits, health fanout and bounded cleanup."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

import pytest

from dendrux.errors import NotifierUnavailableError
from dendrux.store import FollowCapacityError
from dendrux.store._notifier_pg import PostgresEventNotifier

asyncpg = pytest.importorskip("asyncpg")
URL = "postgresql+asyncpg://postgres:root@localhost:5432/dendrux_060_p3b_test"


class Connection:
    def __init__(self, *, deliver=True, hang=None):
        self.deliver = deliver
        self.hang = hang
        self.listener = None
        self.termination = None
        self.closed = False
        self.listening = False
        self.calls = []

    async def add_listener(self, channel, callback):
        self.calls.append("listen")
        if self.hang == "listen":
            await asyncio.Event().wait()
        self.listener = callback
        self.listening = True

    def add_termination_listener(self, callback):
        self.termination = callback

    async def execute(self, sql, channel, payload):
        self.calls.append("notify")
        if self.hang == "notify":
            await asyncio.Event().wait()
        if self.deliver:
            self.listener(self, 1, channel, payload)

    async def close(self, **kwargs):
        self.calls.append("close")
        if self.hang == "close":
            await asyncio.Event().wait()
        self.terminate()

    def terminate(self):
        self.closed = True
        if self.termination:
            self.termination(self)

    def is_closed(self):
        return self.closed

    def notify(self, run_id):
        self.listener(self, 1, "dendrux_events", run_id)


@pytest.fixture
def connections(monkeypatch):
    created = []

    async def connect(*args, **kwargs):
        conn = Connection()
        created.append(conn)
        return conn

    monkeypatch.setattr(asyncpg, "connect", connect)
    return created


async def settle(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


async def test_n1_shared_state_notifications_and_health_fanout(connections, caplog):
    caplog.set_level(logging.INFO)
    notifier = PostgresEventNotifier(URL, reconnect_backoff=(0.001, 0.002))
    async with AsyncExitStack() as stack:
        a = await stack.enter_async_context(notifier.subscribe("a"))
        b = await stack.enter_async_context(notifier.subscribe("b"))
        await notifier.start()
        try:
            assert a.state is b.state
            assert a.state.healthy
            assert a.state.generation == 1
            assert a.event.is_set() and b.event.is_set()
            a.event.clear()
            b.event.clear()
            connections[0].notify("a")
            assert a.event.is_set() and not b.event.is_set()
            a.event.clear()
            connections[0].terminate()
            assert not a.state.healthy
            assert a.state.generation == 2
            assert a.event.is_set() and b.event.is_set()
            await settle(lambda: a.state.healthy)
            assert len(connections) == 2
            assert connections[1].listening
            assert a.state.generation == 3
            assert [r.code for r in caplog.records] == ["notifier_degraded", "notifier_recovered"]
        finally:
            await notifier.stop()


async def test_n3_capacity_counts_subscriptions_and_releases(connections):
    notifier = PostgresEventNotifier(URL, max_subscriptions=2)
    async with notifier.subscribe("same"), notifier.subscribe("same"):
        with pytest.raises(FollowCapacityError) as error:
            async with notifier.subscribe("other"):
                pass
        assert error.value.limit == 2
    async with notifier.subscribe("other"):
        pass


async def test_thousand_subscriptions_use_one_connection(connections):
    notifier = PostgresEventNotifier(URL)
    await notifier.start()
    try:
        async with AsyncExitStack() as stack:
            subs = [
                await stack.enter_async_context(notifier.subscribe(str(i))) for i in range(1000)
            ]
            assert len(connections) == 1
            assert len({id(s.state) for s in subs}) == 1
    finally:
        await notifier.stop()


@pytest.mark.parametrize("hang", ["connect", "listen", "notify", "self_test"])
async def test_n7_start_deadlines_leave_no_connection_or_task(monkeypatch, caplog, hang):
    conn = Connection(deliver=hang != "self_test", hang=hang)

    async def connect(*args, **kwargs):
        if hang == "connect":
            await asyncio.Event().wait()
        return conn

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(URL, connect_timeout_s=0.01, self_test_timeout_s=0.01)
    with pytest.raises(NotifierUnavailableError):
        await asyncio.wait_for(notifier.start(), 0.5)
    assert conn.closed or hang == "connect"
    assert notifier._connection is None
    assert notifier._task is None
    assert not notifier.state.healthy
    assert [r.code for r in caplog.records] == ["notifier_unavailable"]


async def test_cancelled_start_cleans_up_and_can_restart(monkeypatch):
    conn = Connection(hang="listen")

    async def connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(URL)
    task = asyncio.create_task(notifier.start())
    await settle(lambda: conn.calls)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert conn.closed
    assert notifier._connection is None and notifier._task is None
    conn.hang = None
    conn.closed = False
    await notifier.start()
    await notifier.start()
    await notifier.stop()
    await notifier.stop()


async def test_stop_deadline_forces_close_and_wakes_degraded_waiters(connections):
    notifier = PostgresEventNotifier(URL, stop_timeout_s=0.01, reconnect_backoff=(10, 10))
    await notifier.start()
    connections[0].hang = "close"
    async with notifier.subscribe("a") as sub:
        await asyncio.wait_for(notifier.stop(), 0.5)
        assert connections[0].closed
        assert sub.event.is_set() and not sub.state.healthy
        sub.event.clear()
        await notifier.stop()
        assert sub.event.is_set()
    assert notifier._connection is None and notifier._task is None


async def test_reconnect_attempt_deadline_and_resubscription_before_health(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    first = Connection()
    stalled = Connection(hang="listen")
    last = Connection()
    pool = [first, stalled, last]

    async def connect(*args, **kwargs):
        return pool.pop(0)

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(
        URL, reconnect_backoff=(0.001, 0.002), reconnect_attempt_timeout_s=0.01
    )
    await notifier.start()
    try:
        async with notifier.subscribe("a") as sub:
            first.terminate()
            await settle(lambda: sub.state.healthy)
            assert stalled.closed
            assert last.listening
            assert [r.code for r in caplog.records] == ["notifier_degraded", "notifier_recovered"]
    finally:
        await notifier.stop()


async def test_n6_all_entry_points_reject_other_loop(connections):
    notifier = PostgresEventNotifier(URL)
    await notifier.start()

    async def other_loop():
        for action in (notifier.start, notifier.stop):
            with pytest.raises(RuntimeError):
                await action()
        with pytest.raises(RuntimeError):
            async with notifier.subscribe("a"):
                pass

    try:
        await asyncio.to_thread(lambda: asyncio.run(other_loop()))
    finally:
        await notifier.stop()


@pytest.mark.parametrize(
    "options",
    [
        {"max_subscriptions": 0},
        {"max_subscriptions": True},
        {"connect_timeout_s": 0},
        {"reconnect_attempt_timeout_s": -1},
        {"stop_timeout_s": float("inf")},
        {"self_test_timeout_s": float("nan")},
        {"reconnect_backoff": (2, 1)},
        {"reconnect_backoff": (0, 1)},
        {"channel": ""},
        {"channel": "x" * 64},
    ],
)
def test_invalid_bounds(options):
    with pytest.raises(ValueError):
        PostgresEventNotifier(URL, **options)


async def test_concurrent_stop_and_restart_keep_the_new_runner_owned(connections):
    notifier = PostgresEventNotifier(URL)
    await notifier.start()
    first = connections[0]
    closing, release = asyncio.Event(), asyncio.Event()

    async def slow_close(**kwargs):
        closing.set()
        await release.wait()
        first.terminate()

    first.close = slow_close
    stopping = asyncio.create_task(notifier.stop())
    await closing.wait()
    restarting = asyncio.create_task(notifier.start())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(stopping, restarting)
    try:
        assert notifier.state.healthy
        assert notifier._task is not None and not notifier._task.done()
        assert len(connections) == 2
    finally:
        await notifier.stop()
    assert connections[1].closed


async def test_stop_during_start_leaves_no_backend_or_task(monkeypatch):
    connection = Connection(hang="listen")

    async def connect(*args, **kwargs):
        return connection

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(URL, stop_timeout_s=0.01)
    starting = asyncio.create_task(notifier.start())
    await settle(lambda: connection.calls)
    await asyncio.wait_for(notifier.stop(), 0.5)
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert connection.closed
    assert notifier._connection is None and notifier._task is None


async def test_failure_logs_and_exception_never_include_driver_text(monkeypatch, caplog):
    sentinel = "SECRET_SENTINEL_driver_message"

    async def connect(*args, **kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(URL)
    with pytest.raises(NotifierUnavailableError) as error:
        await notifier.start()
    assert sentinel not in str(error.value)
    assert sentinel not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


async def test_each_health_change_sets_each_live_event_exactly_once(connections, monkeypatch):
    from unittest.mock import Mock

    notifier = PostgresEventNotifier(URL)
    async with notifier.subscribe("a") as sub:
        set_event = Mock(wraps=sub.event.set)
        monkeypatch.setattr(sub.event, "set", set_event)
        await notifier.start()
        assert set_event.call_count == 1
        set_event.reset_mock()
        await notifier.stop()
        assert set_event.call_count == 1
        set_event.reset_mock()
        await notifier.stop()
        assert set_event.call_count == 1  # Explicit stop also wakes degraded waiters.


async def test_stop_deadline_is_not_extended_by_queued_start(monkeypatch):
    connections = []

    async def connect(*args, **kwargs):
        connection = Connection(hang="listen")
        connections.append(connection)
        return connection

    monkeypatch.setattr(asyncpg, "connect", connect)
    notifier = PostgresEventNotifier(URL, stop_timeout_s=0.01)
    first = asyncio.create_task(notifier.start())
    await settle(lambda: connections and connections[0].calls)
    second = asyncio.create_task(notifier.start())
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(notifier.stop(), 0.2)
        for task in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(connections) == 1 and connections[0].closed
        assert notifier._connection is None and notifier._task is None
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await notifier.stop()


async def test_close_failure_forces_release_without_driver_text(connections, caplog):
    notifier = PostgresEventNotifier(URL)
    await notifier.start()

    async def fail_close(**kwargs):
        raise RuntimeError("SECRET_SENTINEL_close_error")

    connections[0].close = fail_close
    try:
        await notifier.stop()
        assert connections[0].closed
        assert notifier._connection is None and notifier._task is None
        assert "SECRET_SENTINEL" not in caplog.text
    finally:
        connections[0].terminate()
