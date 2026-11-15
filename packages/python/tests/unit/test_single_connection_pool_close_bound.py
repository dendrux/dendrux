"""``close_fully`` stops waiting for a stuck close a bounded time after cancellation.

A close that never returns (aiosqlite 0.22.1 ``close()`` after ``stop()`` waits on a
dead worker thread) used to keep a cancelled caller alive forever, absorbing every
later cancellation. Once the caller is cancelled, the wait is now bounded; the close
is left running in the background and the caller's cancellation is re-raised.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import TYPE_CHECKING

import aiosqlite
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux.db import _pool_guard
from dendrux.db._pool_guard import close_fully
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.db.models import AgentRun
from dendrux.runtime.stores._factory import store_for_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

BOUND_S = 0.2
FINISH_WITHIN_S = 3.0
SENTINEL = "CLOSE_SECRET_SENTINEL"
_GUARD_LOGGER = "dendrux.db._pool_guard"


@pytest.fixture
def short_bound(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(_pool_guard, "CLOSE_AFTER_CANCEL_S", BOUND_S, raising=False)
    return BOUND_S


@pytest.fixture
async def memory_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    await create_sqlite_schema(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


_OPEN_GATES: list[_Gate] = []


@pytest.fixture(autouse=True)
async def _open_gates_at_teardown() -> AsyncIterator[None]:
    yield
    while _OPEN_GATES:
        _OPEN_GATES.pop().release.set()
    await _settle()


class _Gate:
    """A close that runs until the test opens it, recording whether it was cancelled."""

    def __init__(self, *, fail_with: str | None = None) -> None:
        _OPEN_GATES.append(self)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.finished = False
        self._fail_with = fail_with

    async def close(self) -> None:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        if self._fail_with is not None:
            raise RuntimeError(self._fail_with)


async def _cancel_twice(task: asyncio.Task[object]) -> None:
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_twice_cancelled_caller_is_freed_from_a_close_that_never_finishes(
    short_bound: float, caplog: pytest.LogCaptureFixture
) -> None:
    gate = _Gate()
    caller = asyncio.create_task(close_fully(gate.close))
    await gate.started.wait()

    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        await _cancel_twice(caller)
        done, _ = await asyncio.wait({caller}, timeout=FINISH_WITHIN_S)

    assert caller in done, "cancelled caller is still waiting for the stuck close"
    assert caller.cancelled()
    # The close is abandoned, not cancelled: it keeps running in the background.
    assert not gate.cancelled and not gate.finished
    gate.release.set()
    await _settle()
    assert gate.finished


async def test_caller_already_cancelling_on_entry_waits_only_for_the_bound(
    short_bound: float,
) -> None:
    gate = _Gate()
    entered = asyncio.Event()

    async def session_exit() -> None:
        try:
            entered.set()
            await asyncio.sleep(3600)
        finally:
            # Cleanup reached while the cancellation unwinds, as in a session's finally.
            await close_fully(gate.close)

    caller = asyncio.create_task(session_exit())
    await entered.wait()
    caller.cancel()
    done, _ = await asyncio.wait({caller}, timeout=FINISH_WITHIN_S)

    assert caller in done, "cleanup during cancellation waited for the stuck close forever"
    assert caller.cancelled()
    assert not gate.cancelled
    gate.release.set()
    await _settle()


async def test_real_aiosqlite_close_after_stop_no_longer_traps_the_caller(
    short_bound: float, tmp_path
) -> None:
    conn = await aiosqlite.connect(tmp_path / "stuck.db")
    queued = asyncio.Event()

    async def graceful_close() -> None:
        queued.set()  # set in the same step that queues conn.close on the worker thread
        await conn.close()

    caller = asyncio.create_task(close_fully(graceful_close))
    await queued.wait()
    # SQLAlchemy's forced terminate after a second cancellation: the worker thread
    # exits and close()'s own stop() then waits on a queue nobody reads.
    conn.stop()
    await asyncio.sleep(0.05)
    await _cancel_twice(caller)
    done, _ = await asyncio.wait({caller}, timeout=FINISH_WITHIN_S)

    assert caller in done, "aiosqlite close after stop() trapped the cancelled caller"
    assert caller.cancelled()


async def test_bound_expiry_logs_one_stable_code_without_exception_text(
    short_bound: float, caplog: pytest.LogCaptureFixture
) -> None:
    gate = _Gate(fail_with=SENTINEL)
    caller = asyncio.create_task(close_fully(gate.close))
    await gate.started.wait()

    with caplog.at_level(logging.DEBUG):
        await _cancel_twice(caller)
        await asyncio.wait({caller}, timeout=FINISH_WITHIN_S)
        # The abandoned close later fails; nobody may surface its text.
        gate.release.set()
        await _settle()
        gc.collect()
        await _settle()

    assert caller.cancelled()
    coded = [r for r in caplog.records if getattr(r, "code", None) == "db_close_abandoned"]
    assert len(coded) == 1
    record = coded[0]
    assert record.name == _GUARD_LOGGER
    assert record.levelno == logging.WARNING
    assert record.kind == "store"
    assert record.timeout_s == short_bound
    assert record.exc_info is None
    assert SENTINEL not in caplog.text
    assert SENTINEL not in repr([r.__dict__ for r in caplog.records])
    assert not [r for r in caplog.records if "never retrieved" in r.getMessage()]


async def test_close_finishing_within_the_bound_is_waited_for(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(_pool_guard, "CLOSE_AFTER_CANCEL_S", 2.0, raising=False)
    gate = _Gate()
    caller = asyncio.create_task(close_fully(gate.close))
    await gate.started.wait()

    with caplog.at_level(logging.DEBUG, logger=_GUARD_LOGGER):
        await _cancel_twice(caller)
        await asyncio.sleep(0.2)  # ten times a realistic post-cancel ROLLBACK
        assert not caller.done()
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

    assert gate.finished and not gate.cancelled
    assert not [r for r in caplog.records if getattr(r, "code", None) == "db_close_abandoned"]


async def test_slow_close_without_cancellation_is_never_abandoned(
    short_bound: float, caplog: pytest.LogCaptureFixture
) -> None:
    gate = _Gate()
    caller = asyncio.create_task(close_fully(gate.close))
    await gate.started.wait()

    with caplog.at_level(logging.DEBUG, logger=_GUARD_LOGGER):
        await asyncio.sleep(short_bound * 3)
        assert not caller.done()
        gate.release.set()
        await caller

    assert gate.finished
    assert not [r for r in caplog.records if getattr(r, "code", None) == "db_close_abandoned"]


async def test_close_error_without_cancellation_still_reaches_the_caller(
    short_bound: float,
) -> None:
    gate = _Gate(fail_with="close failed")
    gate.release.set()
    with pytest.raises(RuntimeError, match="close failed"):
        await close_fully(gate.close)


def _guards_for(engine: AsyncEngine) -> list[_pool_guard._Guard]:
    pool = engine.sync_engine.pool
    return [g for (p, _loop), g in _pool_guard._GUARDS.items() if p is pool]


async def test_abandoned_close_keeps_the_single_connection_lock_until_it_finishes(
    memory_engine: AsyncEngine, short_bound: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_for_engine(memory_engine)
    await store.create_run("run-1", "A")
    original_close = AsyncSession.close
    gate = _Gate()
    stuck: list[bool] = []
    in_body = asyncio.Event()

    async def stuck_close(self: AsyncSession) -> None:
        if not stuck:
            stuck.append(True)
            await gate.close()
        await original_close(self)

    monkeypatch.setattr(AsyncSession, "close", stuck_close)

    async def holder() -> None:
        async with store._session_factory() as session:
            await session.execute(select(AgentRun))
            in_body.set()
            await asyncio.sleep(3600)

    held = asyncio.create_task(holder())
    await in_body.wait()
    await _cancel_twice(held)
    done, _ = await asyncio.wait({held}, timeout=FINISH_WITHIN_S)
    assert held in done, "cancelled store session is still waiting for the stuck close"
    assert held.cancelled()

    # The shared connection is still being closed: no other session may use it yet.
    reader = asyncio.create_task(store.get_run("run-1"))
    await asyncio.sleep(0.1)
    assert not reader.done()
    assert _guards_for(memory_engine)[0].lock.locked()

    gate.release.set()
    assert (await asyncio.wait_for(reader, timeout=FINISH_WITHIN_S)) is not None
    await _settle()
    assert not _guards_for(memory_engine)
    await asyncio.wait_for(store.create_run("run-2", "A"), timeout=FINISH_WITHIN_S)
