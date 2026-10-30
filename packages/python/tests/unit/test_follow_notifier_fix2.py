"""Task-local feed readiness, shared startup cancellation and missed hints."""

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import NotifierUnavailableError
from dendrux.store import EventFeed, RunStore
from dendrux.store._follow import _Follower, _RunFeed
from dendrux.store._notifier_pg import PostgresEventNotifier
from dendrux.types import generate_ulid
from tests.unit.test_follow_notifier import Notifier
from tests.unit.test_follow_notifier_fix1 import State
from tests.unit.test_follow_wake import Clock, Store, ticks
from tests.unit.test_notifier_pg import URL, settle

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore


@pytest.fixture
async def clock(monkeypatch):
    import random

    loop = asyncio.get_running_loop()
    clock = Clock(loop.time())
    monkeypatch.setattr(loop, "time", lambda: clock.now)
    monkeypatch.setattr(random, "uniform", lambda low, high: 0)
    return clock


async def test_n2_completed_poller_cannot_fail_replacement_ready():
    owner = EventFeed(cast("RunStore", Store()), notifier=Notifier())
    feed = _RunFeed(owner, generate_ulid(), None)
    owner._feeds[feed.run_id] = feed
    feed.terminal = True
    feed.add(_Follower(until_terminal=True, limit=10))
    await asyncio.sleep(0)  # Old task finishes; its completion callback is queued.
    previous = feed.task
    assert previous.done()
    feed.add(_Follower(until_terminal=False, limit=10))
    replacement, ready = feed.task, feed.ready
    try:
        await asyncio.wait_for(asyncio.shield(ready), 1)
        assert replacement is not previous and not replacement.done()
    finally:
        await owner.aclose()


async def test_n3_cancelling_one_owned_start_preserves_other_follows(monkeypatch):
    import asyncpg

    attempts = 0
    release = asyncio.Event()

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        await release.wait()
        raise OSError("test listener unavailable")

    monkeypatch.setattr(asyncpg, "connect", connect)
    reader = RunStore(cast("StateStore", State()))
    notifier = PostgresEventNotifier(URL, connect_timeout_s=1)
    reader._notifier = reader._owned_notifier = notifier
    follows = [reader.follow("run", tenant_id=None, until_terminal=False) for _ in range(3)]
    starting = [asyncio.create_task(follow.start()) for follow in follows]
    try:
        await settle(lambda: attempts == 1)
        await ticks(3)
        starting[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting[0]
        release.set()
        results = await asyncio.gather(*starting[1:], return_exceptions=True)
        assert results == [None, None], results
        assert attempts == 1
        assert reader._event_feed.followers == 2
    finally:
        release.set()
        for task in starting:
            task.cancel()
        await asyncio.gather(*starting, return_exceptions=True)
        for follow in follows:
            await follow.aclose()
        await reader.close()


async def test_n4_owned_retry_backoff_grows_caps_and_resets(clock):
    async def unavailable():
        raise NotifierUnavailableError("start_failed")

    notifier = SimpleNamespace(
        reconnect_backoff=(2, 5), start=AsyncMock(side_effect=unavailable), stop=AsyncMock()
    )
    reader = RunStore(cast("StateStore", State()))
    reader._owned_notifier = notifier
    try:
        for attempt, delay in enumerate((2, 4, 5, 5), start=1):
            await reader._start_owned_notifier()
            assert notifier.start.await_count == attempt
            clock.advance(delay - 0.01)
            await reader._start_owned_notifier()
            assert notifier.start.await_count == attempt
            clock.advance(0.02)
        notifier.start.side_effect = None
        await reader._start_owned_notifier()
        assert reader._notifier_started
        # Simulate a subsequent owned-listener lifecycle after a successful start.
        reader._notifier_started = False
        notifier.start.side_effect = unavailable
        await reader._start_owned_notifier()
        count = notifier.start.await_count
        clock.advance(1.99)
        await reader._start_owned_notifier()
        assert notifier.start.await_count == count
        clock.advance(0.02)
        await reader._start_owned_notifier()
        assert notifier.start.await_count == count + 1
    finally:
        await reader.close()


def make_feed(notifier):
    store = Store()
    owner = EventFeed(cast("RunStore", store), notifier=notifier, poll_healthy_s=30)
    feed = _RunFeed(owner, generate_ulid(), None)
    owner._feeds[feed.run_id] = feed
    follower = _Follower(until_terminal=False, limit=100)
    feed.add(follower)
    return store, owner, feed, follower


def missed_hints(caplog):
    return [
        record
        for record in caplog.records
        if getattr(record, "code", None) == "notifier_missed_hint"
    ]


async def test_n5_healthy_reconciliation_logs_count_once_per_feed(clock, caplog):
    store, owner, feed, follower = make_feed(Notifier())
    store.write()  # Initial backlog is not evidence of a missed hint.
    try:
        await ticks(4)
        assert not missed_hints(caplog)
        for _ in range(2):
            store.write().data["secret"] = "SECRET_SENTINEL_event_payload"
        clock.advance(30.001)
        await ticks(5)
        assert len(follower.buffer) == 3
        records = missed_hints(caplog)
        assert len(records) == 1
        record = records[0]
        assert (record.levelno, record.kind, record.count) == (logging.WARNING, "notifier", 2)
        assert record.run_id == feed.run_id
        store.write()
        clock.advance(30.001)
        await ticks(5)
        assert len(follower.buffer) == 4
        assert len(missed_hints(caplog)) == 1
        assert "SECRET_SENTINEL" not in caplog.text
    finally:
        await owner.aclose()


@pytest.mark.parametrize(
    "cause", ["notification", "join", "degraded", "none", "short_retry", "health_change"]
)
async def test_n5_non_reconciliation_reads_do_not_warn(clock, caplog, cause):
    notifier = None if cause == "none" else Notifier()
    if cause in ("degraded", "health_change"):
        notifier.state.healthy = False
    store, owner, feed, follower = make_feed(notifier)
    try:
        await ticks(4)
        if cause == "short_retry":
            notifier.event.set()
            await ticks(4)
        store.write()
        if cause == "notification":
            notifier.event.set()
        elif cause == "join":
            feed.add(_Follower(until_terminal=False, limit=10))
        elif cause == "health_change":
            notifier.health(True)
        else:
            clock.advance(0.501)
        await ticks(5)
        assert len(follower.buffer) == 1
        assert not missed_hints(caplog)
    finally:
        await owner.aclose()
