"""Writes on a file SQLite database cancelled twice all finish within a bound.

File SQLite uses a multi-connection pool. A write cancelled while it touches the
database, then cancelled again during its cleanup, can leave the session close
waiting forever in the driver (aiosqlite 0.22.1 close after stop). Store sessions
and ``get_session`` sessions close through ``close_fully``, which waits at most
``CLOSE_AFTER_CANCEL_S`` once the caller is cancelled, so every task finishes
(the bound is shortened here).

Each round starts 16 writers, cancels every one twice 10 ms apart, and requires
every task to finish within the close bound plus a margin.

The same driver defect has effects beyond the caller, which this test keeps out
of its way rather than asserting on:

- A close that never returns keeps its pool slot for good (SQLAlchemy checks a
  connection back in only after its terminate returns), so the engine gets one
  spare slot per writer.
- A cancellation the driver converts into a database error is retried as a
  transient failure, so a writer may carry on after both cancels. A short SQLite
  busy timeout keeps those retries short under the writers' lock contention.
- After a burst, a connection left behind by the driver can hold the database
  lock for a while, so new writes are not asserted here.
- The driver's own close tasks can need two cancellations, and ``dispose()`` can
  hang on a pooled connection the driver already stopped, so teardown bounds
  dispose, cancels the tasks the test started (up to three rounds) and fails if
  any is still pending.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db import _pool_guard
from dendrux.db.models import Base, DendruxSetting
from dendrux.db.session import get_session
from dendrux.runtime.stores._factory import store_for_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

WRITERS = 16
CANCEL_AT_S = (0.03, 0.04, 0.05, 0.06, 0.04, 0.05)
CLOSE_BOUND_S = 0.5  # the post-cancel close bound, shortened to keep the test fast
FINISH_WITHIN_S = CLOSE_BOUND_S + 15.0
BUSY_TIMEOUT_S = 1.0


@pytest.fixture
async def file_engine(tmp_path, monkeypatch) -> AsyncIterator[AsyncEngine]:
    monkeypatch.setattr(_pool_guard, "CLOSE_AFTER_CANCEL_S", CLOSE_BOUND_S)
    before = asyncio.all_tasks()
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'cancel.db'}",
        connect_args={"check_same_thread": False, "timeout": BUSY_TIMEOUT_S},
        max_overflow=WRITERS * len(CANCEL_AT_S),
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        # dispose() can hang closing a pooled connection the driver already stopped.
        dispose = asyncio.ensure_future(engine.dispose())
        await asyncio.wait({dispose}, timeout=5)
        left = await _cancel_tasks_started_since(before)
        if dispose.done() and not dispose.cancelled():
            dispose.result()
        assert not left, f"{left} tasks were still pending after teardown"


async def _cancel_tasks_started_since(before: set[asyncio.Task[object]]) -> int:
    """Cancel tasks this fixture's test started (abandoned closes, a stuck dispose).

    Returns how many are still pending after three cancel rounds.
    """
    current = asyncio.current_task()
    left: list[asyncio.Task[object]] = []
    for _ in range(3):
        left = [t for t in asyncio.all_tasks() if t is not current and t not in before]
        if not left:
            return 0
        for task in left:
            task.cancel()
        _, pending = await asyncio.wait(left, timeout=5)
        left = list(pending)
    return len(left)


async def _stuck_after_double_cancels(write: Callable[[int, int], Awaitable[object]]) -> int:
    """Run every round; return how many twice-cancelled writers never finished."""
    stuck = 0
    for round_index, cancel_at_s in enumerate(CANCEL_AT_S):
        writers = [asyncio.create_task(write(round_index, w)) for w in range(WRITERS)]
        await asyncio.sleep(cancel_at_s)
        for _ in range(2):
            for writer in writers:
                writer.cancel()
            await asyncio.sleep(0.01)
        done, pending = await asyncio.wait(writers, timeout=FINISH_WITHIN_S)
        stuck += len(pending)
        for writer in pending:
            writer.cancel()  # a third cancel frees a stuck close; keeps teardown clean
        if pending:
            await asyncio.wait(pending, timeout=5)
        for writer in done:
            if not writer.cancelled():
                writer.exception()  # finished either way; the outcome is not under test
    return stuck


def _run_id(round_index: int, writer: int) -> str:
    return f"01GAP{round_index:02d}{writer:02d}".ljust(26, "0")


async def test_store_writes_cancelled_twice_all_finish(file_engine) -> None:
    store = store_for_engine(file_engine)

    stuck = await _stuck_after_double_cancels(lambda r, w: store.create_run(_run_id(r, w), "agent"))

    assert stuck == 0, f"{stuck} cancelled store writes never finished"


async def test_get_session_writes_cancelled_twice_all_finish(file_engine) -> None:
    async def write(round_index: int, writer: int) -> None:
        async with get_session(file_engine) as session:
            session.add(DendruxSetting(key=f"gap-{round_index}-{writer}", value="x"))
            await session.flush()

    stuck = await _stuck_after_double_cancels(write)

    assert stuck == 0, f"{stuck} cancelled get_session writes never finished"
