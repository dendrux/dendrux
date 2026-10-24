"""Single-connection pools (in-memory SQLite / StaticPool) serialise sessions.

On a StaticPool every session shares one DBAPI connection and therefore one
SQLite transaction; without serialisation a session's reset-on-return
ROLLBACK discarded another session's executed-but-uncommitted INSERT, and
parallel tool calls silently lost ``tool_calls`` / ``run_events`` rows (seen on
SQLAlchemy 2.0, deterministic on 2.1). File SQLite and Postgres use
independent connections and must take no lock.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import random
import weakref
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dendrux import Agent, tool
from dendrux.db import _pool_guard
from dendrux.db._pool_guard import serialized, shares_one_connection
from dendrux.db._sqlite_sync import create_sqlite_schema
from dendrux.db.models import AgentRun, RunEvent, ToolCallRecord
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import LLMResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

_GUARD_LOGGER = "dendrux.db._pool_guard"


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


@tool()
async def hop(n: int) -> int:
    """Return n after n event-loop hops."""
    for _ in range(n):
        await asyncio.sleep(0)
    return n


@tool()
async def jittered(n: int) -> int:
    """Return n after a random number of event-loop hops."""
    for _ in range(random.randint(0, 3)):
        await asyncio.sleep(0)
    if random.random() < 0.5:
        await asyncio.sleep(random.random() * 0.002)
    return n


def _guards_for(engine: AsyncEngine) -> list[_pool_guard._Guard]:
    pool = engine.sync_engine.pool
    return [g for (p, _loop), g in _pool_guard._GUARDS.items() if p is pool]


async def _held_lock(engine: AsyncEngine) -> bool:
    guards = _guards_for(engine)
    return bool(guards) and guards[0].lock.locked()


async def _evidence_counts(engine: AsyncEngine, run_id: str) -> tuple[int, int]:
    async with engine.connect() as conn:
        calls = (
            await conn.execute(
                select(func.count())
                .select_from(ToolCallRecord)
                .where(ToolCallRecord.agent_run_id == run_id)
            )
        ).scalar_one()
        completed = (
            await conn.execute(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "tool.completed")
            )
        ).scalar_one()
    return calls, completed


# (0, 0, 0) is the v0.3.0 oracle's shape (loses a row on SQLAlchemy 2.1 without the
# guard); the staggered ones lose a row on SQLAlchemy 2.0 without it.
@pytest.mark.parametrize("hops", [(0, 0, 0), (0, 1, 5), (0, 4, 4), (3, 0, 5)])
async def test_three_parallel_tools_persist_every_row(memory_engine, hops) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    calls = [ToolCall("hop", {"n": n}, provider_tool_call_id=f"p{i}") for i, n in enumerate(hops)]
    llm = MockLLM([LLMResponse(text="Checking", tool_calls=calls), LLMResponse(text="done")])
    agent = Agent(provider=llm, prompt="p", tools=[hop], state_store=store)
    result = await agent.run("Do three things")
    assert result.status.value == "success"
    assert await _evidence_counts(memory_engine, result.run_id) == (3, 3)


async def test_jittered_parallel_tools_lose_no_rows(memory_engine) -> None:
    random.seed(1234)
    store = SQLAlchemyStateStore(memory_engine)
    width = 5
    for _ in range(20):
        calls = [
            ToolCall("jittered", {"n": i}, provider_tool_call_id=f"p{i}") for i in range(width)
        ]
        llm = MockLLM([LLMResponse(text="go", tool_calls=calls), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="p", tools=[jittered], state_store=store)
        result = await agent.run("go")
        assert await _evidence_counts(memory_engine, result.run_id) == (width, width)


async def _give_way_to(reader: Callable[[], Awaitable[object]]) -> None:
    """Start ``reader`` in another task and let it run (if it can) before returning.

    Without the guard the reader finishes, and its close (pool reset ROLLBACK)
    lands between the caller's write and commit. With the guard it waits.
    """
    task = asyncio.create_task(reader())
    await asyncio.wait({task}, timeout=0.05)


async def test_reader_session_does_not_undo_pending_writes(memory_engine) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")

    async def write(i: int) -> None:
        async with store._session_factory() as session:
            session.add(
                RunEvent(
                    id=f"ev-{i}",
                    agent_run_id="run-1",
                    event_type="e",
                    sequence_index=i,
                    iteration_index=0,
                )
            )
            await session.flush()
            await _give_way_to(lambda: store.get_run_events("run-1"))
            await session.commit()

    await asyncio.gather(*[write(i) for i in range(5)])
    await asyncio.sleep(0.01)
    assert len(await store.get_run_events("run-1")) == 5


async def test_nested_session_in_same_task_does_not_deadlock(memory_engine) -> None:
    """A store call made while the same task holds a session (cf. test_cancellation)."""
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")

    async def nested() -> None:
        async with store._session_factory() as session:
            row = (await session.execute(select(AgentRun).limit(1))).scalar_one()
            await store.request_cancel(row.id)

    await asyncio.wait_for(nested(), timeout=5)
    record = await store.get_run("run-1")
    assert record is not None


async def test_sessions_from_different_tasks_do_not_overlap(memory_engine) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    inside = 0
    peak = 0

    async def hold() -> None:
        nonlocal inside, peak
        async with store._session_factory():
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.001)
            inside -= 1

    await asyncio.gather(*[hold() for _ in range(5)])
    assert peak == 1


async def test_single_connection_store_warns_once_per_pool(memory_engine, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        SQLAlchemyStateStore(memory_engine)
        SQLAlchemyStateStore(memory_engine)
        SQLAlchemyStateStore(memory_engine.execution_options(isolation_level="SERIALIZABLE"))
    warnings = [r for r in caplog.records if r.name == _GUARD_LOGGER]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "StaticPool" in message
    assert "serialised" in message
    assert "not durable" in message
    assert "':memory:'" in message
    assert "%3A" not in message


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_independent_connection_pools_take_no_lock(engine, caplog) -> None:
    assert not shares_one_connection(engine)
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        store = SQLAlchemyStateStore(engine)
    assert not [r for r in caplog.records if r.name == _GUARD_LOGGER]
    assert isinstance(store._session_factory, sessionmaker)

    async with serialized(engine):
        assert not _guards_for(engine)

    await store.create_run("run-1", "A")
    assert not _guards_for(engine)


async def test_get_session_writes_survive_concurrent_store_reads(memory_engine) -> None:
    from dendrux.db.session import get_session

    store = SQLAlchemyStateStore(memory_engine)

    async def write(i: int) -> None:
        async with get_session(memory_engine) as session:
            session.add(AgentRun(id=f"run-{i}", agent_name="A", status="running"))
            await session.flush()
            await _give_way_to(lambda: store.get_run("missing"))

    await asyncio.gather(*[write(i) for i in range(5)])
    await asyncio.sleep(0.01)
    async with memory_engine.connect() as conn:
        count = (await conn.execute(select(func.count()).select_from(AgentRun))).scalar_one()
    assert count == 5


async def test_execution_options_engine_shares_the_lock(memory_engine) -> None:
    """Engines sharing one pool (execution_options views) must share one lock."""
    base = SQLAlchemyStateStore(memory_engine)
    view = SQLAlchemyStateStore(memory_engine.execution_options(isolation_level="SERIALIZABLE"))
    await base.create_run("run-1", "A")

    async def write(store: SQLAlchemyStateStore, i: int) -> None:
        async with store._session_factory() as session:
            session.add(
                RunEvent(
                    id=f"ev-{i}",
                    agent_run_id="run-1",
                    event_type="e",
                    sequence_index=i,
                    iteration_index=0,
                )
            )
            await session.flush()
            for _ in range(3):
                await asyncio.sleep(0)
            await session.commit()

    async def read(store: SQLAlchemyStateStore) -> None:
        await store.get_run_events("run-1")

    jobs = []
    for i in range(20):
        jobs.append(write(base if i % 2 else view, i))
        jobs.append(read(view if i % 2 else base))
    await asyncio.gather(*jobs)
    assert len(await base.get_run_events("run-1")) == 20


async def test_lock_state_is_dropped_when_idle(memory_engine) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")
    await asyncio.gather(*[store.get_run("run-1") for _ in range(5)])
    assert not _guards_for(memory_engine)


def test_closed_event_loop_is_not_kept_alive() -> None:
    """A loop that saw contention must be collectable once it is closed."""
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def contended() -> None:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        await create_sqlite_schema(engine)
        store = SQLAlchemyStateStore(engine)
        await store.create_run("run-1", "A")

        async def hold() -> None:
            async with store._session_factory():
                await asyncio.sleep(0.001)

        await asyncio.gather(*[hold() for _ in range(4)])
        await engine.dispose()

    for _ in range(3):
        asyncio.run(contended())
    gc.collect()
    assert all(ref() is None for ref in loops)
    assert not _pool_guard._GUARDS


async def test_exception_in_session_releases_the_lock(memory_engine) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    with pytest.raises(ValueError, match="boom"):
        async with store._session_factory():
            raise ValueError("boom")
    assert not _guards_for(memory_engine)
    await asyncio.wait_for(store.create_run("run-1", "A"), timeout=5)


async def test_cancelling_the_holder_and_waiters_keeps_the_lock_consistent(
    memory_engine,
) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")
    entered = asyncio.Event()

    async def holder() -> None:
        async with store._session_factory() as session:
            session.add(
                RunEvent(
                    id="ev-held",
                    agent_run_id="run-1",
                    event_type="e",
                    sequence_index=0,
                    iteration_index=0,
                )
            )
            await session.flush()
            entered.set()
            await asyncio.sleep(3600)

    async def writer(i: int) -> None:
        await store.save_run_event("run-1", event_type=f"w{i}")

    held = asyncio.create_task(holder())
    await entered.wait()
    waiters = [asyncio.create_task(writer(i)) for i in range(1, 5)]
    await asyncio.sleep(0.01)
    assert not any(w.done() for w in waiters)
    waiters[0].cancel()
    waiters[2].cancel()
    held.cancel()
    await asyncio.wait_for(asyncio.gather(*waiters, held, return_exceptions=True), timeout=5)

    assert held.cancelled() and waiters[0].cancelled() and waiters[2].cancelled()
    assert waiters[1].exception() is None and waiters[3].exception() is None
    events = await store.get_run_events("run-1")
    # The held row rolled back; the two surviving writers got the first two indexes.
    assert sorted(e.event_type for e in events) == ["w2", "w4"]
    assert sorted(e.sequence_index for e in events) == [0, 1]
    assert not _guards_for(memory_engine)


async def test_lock_is_released_only_after_close_finishes(memory_engine, monkeypatch) -> None:
    """Cancelled during close: ROLLBACK and pool reset still run under the lock."""
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")
    original_close = AsyncSession.close
    in_close = asyncio.Event()
    locked_at_close_end: list[bool] = []

    async def slow_close(self: AsyncSession) -> None:
        in_close.set()
        await asyncio.sleep(0.05)
        await original_close(self)
        locked_at_close_end.append(await _held_lock(memory_engine))

    monkeypatch.setattr(AsyncSession, "close", slow_close)

    async def holder() -> None:
        async with store._session_factory() as session:
            await session.execute(select(AgentRun))

    task = asyncio.create_task(holder())
    await in_close.wait()
    task.cancel()
    waiter = asyncio.create_task(store.get_run("run-1"))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert locked_at_close_end == [True]
    assert (await asyncio.wait_for(waiter, timeout=5)) is not None
    assert not _guards_for(memory_engine)


async def test_long_wait_logs_a_warning_naming_the_holder(
    memory_engine, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(_pool_guard, "WAIT_WARNING_S", 0.02)
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")

    async def holder() -> None:
        async with store._session_factory():
            await asyncio.sleep(0.1)

    task = asyncio.create_task(holder(), name="slow-holder")
    await asyncio.sleep(0)
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        assert (await store.get_run("run-1")) is not None
    await task
    messages = [r.getMessage() for r in caplog.records if r.name == _GUARD_LOGGER]
    assert any("'slow-holder'" in m and "held across a call" in m for m in messages)
