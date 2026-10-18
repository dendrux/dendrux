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
import logging
import random
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
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
    from collections.abc import AsyncIterator

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
async def lookup(q: str) -> str:
    """Look up q."""
    await asyncio.sleep(0)
    return f"result for {q}"


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool()
async def jittered(n: int) -> int:
    """Return n after a random number of event-loop hops."""
    for _ in range(random.randint(0, 3)):
        await asyncio.sleep(0)
    if random.random() < 0.5:
        await asyncio.sleep(random.random() * 0.002)
    return n


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


async def test_three_parallel_tools_persist_every_row(memory_engine) -> None:
    store = SQLAlchemyStateStore(memory_engine)
    for _ in range(5):
        llm = MockLLM(
            [
                LLMResponse(
                    text="Checking",
                    tool_calls=[
                        ToolCall("lookup", {"q": "alpha"}, provider_tool_call_id="p1"),
                        ToolCall("add", {"a": 7, "b": 8}, provider_tool_call_id="p2"),
                        ToolCall("lookup", {"q": "beta"}, provider_tool_call_id="p3"),
                    ],
                ),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(provider=llm, prompt="p", tools=[lookup, add], state_store=store)
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


async def test_concurrent_writer_and_reader_sessions_keep_pending_writes(memory_engine) -> None:
    """A reader's rollback must not undo another session's pending write."""
    store = SQLAlchemyStateStore(memory_engine)
    await store.create_run("run-1", "A")

    async def write(i: int) -> None:
        await store.save_run_event("run-1", event_type="e", sequence_index=i, data={"i": i})

    async def read() -> None:
        await store.get_run_events("run-1")

    await asyncio.gather(*[write(i) if i % 2 else read() for i in range(40)])
    assert len(await store.get_run_events("run-1")) == 20


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


async def test_single_connection_store_warns_once_per_engine(memory_engine, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        SQLAlchemyStateStore(memory_engine)
        SQLAlchemyStateStore(memory_engine)
    warnings = [r for r in caplog.records if r.name == _GUARD_LOGGER]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "StaticPool" in message
    assert "serialised" in message
    assert "not durable" in message


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_independent_connection_pools_take_no_lock(engine, caplog) -> None:
    assert not shares_one_connection(engine)
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        store = SQLAlchemyStateStore(engine)
    assert not [r for r in caplog.records if r.name == _GUARD_LOGGER]
    assert isinstance(store._session_factory, sessionmaker)

    async with serialized(engine):
        pass
    assert engine.sync_engine not in _pool_guard._GUARDS

    await store.create_run("run-1", "A")
    assert engine.sync_engine not in _pool_guard._GUARDS


async def test_get_session_writers_are_serialised(memory_engine) -> None:
    from dendrux.db.session import get_session

    async def write(i: int) -> None:
        async with get_session(memory_engine) as session:
            session.add(AgentRun(id=f"run-{i}", agent_name="A", status="running"))
            await asyncio.sleep(random.random() * 0.002)

    await asyncio.gather(*[write(i) for i in range(10)])
    async with memory_engine.connect() as conn:
        count = (await conn.execute(select(func.count()).select_from(AgentRun))).scalar_one()
    assert count == 10
