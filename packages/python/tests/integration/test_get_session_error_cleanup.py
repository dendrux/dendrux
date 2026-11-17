"""``get_session`` cleans up an ordinary error with an ORM rollback, then one close.

The rollback restores ORM state (a flushed new object becomes transient again, so
retrying the same object writes it) and fires the rollback events. The rollback and
the close share the bounded post-cancel wait: a rollback stuck in the driver cannot
keep a cancelled caller waiting past ``CLOSE_AFTER_CANCEL_S``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db import _pool_guard
from dendrux.db.models import DendruxSetting
from dendrux.db.session import get_session

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

BOUND_S = 0.2
_BACKENDS = ["sqlite", "sqlite_file", "postgres"]


async def _count(engine, key: str) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                select(func.count()).select_from(DendruxSetting).where(DendruxSetting.key == key)
            )
        ).scalar_one()


@pytest.mark.parametrize("engine", _BACKENDS, indirect=True)
async def test_retrying_the_same_object_after_an_error_writes_it(engine) -> None:
    setting = DendruxSetting(key="g1-retry", value="v")
    with pytest.raises(LookupError):
        async with get_session(engine) as session:
            session.add(setting)
            await session.flush()
            raise LookupError("retry this operation")
    assert await _count(engine, "g1-retry") == 0

    async with get_session(engine) as session:
        session.add(setting)

    assert await _count(engine, "g1-retry") == 1


@pytest.mark.parametrize("engine", _BACKENDS, indirect=True)
async def test_an_error_fires_the_rollback_events(engine) -> None:
    fired: list[str] = []
    with pytest.raises(LookupError):
        async with get_session(engine) as session:
            sync = session.sync_session
            event.listen(sync, "after_rollback", lambda _s: fired.append("after_rollback"))
            event.listen(
                sync, "after_soft_rollback", lambda _s, _t: fired.append("after_soft_rollback")
            )
            session.add(DendruxSetting(key="g1-events", value="v"))
            await session.flush()
            raise LookupError("roll back")

    assert "after_rollback" in fired
    assert "after_soft_rollback" in fired


@pytest.mark.parametrize("engine", ["sqlite", "sqlite_file"], indirect=True)
async def test_a_stuck_rollback_does_not_hold_a_cancelled_caller(engine, monkeypatch) -> None:
    monkeypatch.setattr(_pool_guard, "CLOSE_AFTER_CANCEL_S", BOUND_S)
    gate = asyncio.Event()
    rollback_entered = asyncio.Event()
    closes: list[Session] = []
    real_rollback = AsyncSession.rollback
    real_close = AsyncSession.close

    async def stuck_rollback(self: AsyncSession) -> None:
        # Like a rollback stuck in the driver: it ignores cancellation until released.
        rollback_entered.set()
        while not gate.is_set():
            try:
                await gate.wait()
            except asyncio.CancelledError:
                continue
        await real_rollback(self)

    async def counted_close(self: AsyncSession) -> None:
        closes.append(self.sync_session)
        await real_close(self)

    monkeypatch.setattr(AsyncSession, "rollback", stuck_rollback)
    monkeypatch.setattr(AsyncSession, "close", counted_close)

    async def caller() -> None:
        async with get_session(engine) as session:
            session.add(DendruxSetting(key="g1-stuck", value="v"))
            await session.flush()
            raise LookupError("roll back")

    task = asyncio.create_task(caller())
    await asyncio.wait_for(rollback_entered.wait(), timeout=5)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=BOUND_S * 10)
    if not done:
        gate.set()  # release the rollback so the caller and teardown can finish
        await asyncio.wait({task}, timeout=5)
        pytest.fail("a cancelled caller waited on a stuck rollback past the bound")

    assert task.cancelled()
    assert closes == []  # the close comes after the rollback, in the same cleanup
    gate.set()
    for _ in range(100):
        if closes:
            break
        await asyncio.sleep(0.01)
    assert len(closes) == 1
    assert await _count(engine, "g1-stuck") == 0
