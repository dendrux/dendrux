"""Follow cancellation wins over a simultaneous heartbeat wake."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from dendrux.store import EventFeed, FollowHeartbeat, RunStore
from tests._helpers.lost_cancel import simulate_lost_cancel
from tests.unit.test_follow_notifier import Notifier
from tests.unit.test_follow_wake import Store


@pytest.mark.parametrize("lose_cancel", [False, True])
async def test_follow_cancel_survives_simultaneous_wake(
    monkeypatch: pytest.MonkeyPatch, lose_cancel: bool
) -> None:
    if lose_cancel:
        simulate_lost_cancel(monkeypatch)
    owner = EventFeed(cast("RunStore", Store()))
    follow = owner.follow("run", tenant_id=None, max_duration_s=None)
    task = asyncio.create_task(anext(follow))
    try:
        async with asyncio.timeout(1):
            while follow._follower is None:
                await asyncio.sleep(0)
        follow._follower.wake.set()
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "follow consumer ignored cancellation after a wake"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.followers == 0
        assert owner.watched_runs == 0
        assert follow._follower is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await follow.aclose()
        await owner.aclose()


@pytest.mark.parametrize("lose_cancel", [False, True])
async def test_notified_follow_cancel_after_heartbeat(
    monkeypatch: pytest.MonkeyPatch, lose_cancel: bool
) -> None:
    if lose_cancel:
        simulate_lost_cancel(monkeypatch)
    owner = EventFeed(cast("RunStore", Store()), notifier=Notifier(), heartbeat_s=0.01)
    follow = owner.follow("run", tenant_id=None, max_duration_s=None)
    entered = asyncio.Event()

    async def consume_next():
        entered.set()
        return await anext(follow)

    try:
        async with asyncio.timeout(1):
            assert isinstance(await anext(follow), FollowHeartbeat)
        task = asyncio.create_task(consume_next())
        try:
            await entered.wait()
            assert follow._follower is not None
            follow._follower.wake.set()
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=1)
            assert task in done, "follow ignored cancellation after a heartbeat"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert owner.followers == 0
            assert owner.watched_runs == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        await follow.aclose()
        await owner.aclose()
