"""Commit visibility, enqueue outcomes and store operation flush boundaries."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, StoreOutcomeUnknown
from dendrux.runtime.durability import _CommitNotAppliedError
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import generate_ulid
from tests.unit.test_live_updates_store_lifecycle import RecordingUpdates


@pytest.fixture
async def live_store(engine):
    bus = RecordingUpdates()
    store = store_for_engine(engine, live_updates=bus)
    yield store, bus
    await store.close()


def sent(bus):
    return [run_id for message in bus.messages for run_id in json.loads(message)]


async def test_create_enqueues_only_after_commit_and_flushes_after_session(live_store, engine):
    store, bus = live_store
    run_id = generate_ulid()
    original_send = bus.send

    async def send(messages):
        async with store._session_factory() as session:
            assert await session.scalar(select(AgentRun.id).where(AgentRun.id == run_id)) == run_id
        await original_send(messages)

    bus.send = send
    async with asyncio.timeout(5):
        await store.create_run(run_id, "live")
    assert sent(bus) == [run_id]


@pytest.mark.parametrize("outcome", ["unknown", "cancel", "committed", "absent", "precommit"])
async def test_commit_outcomes_keep_original_exception_and_pending(
    live_store, monkeypatch, outcome
):
    store, bus = live_store
    run_id = generate_ulid()
    error = {
        "unknown": StoreOutcomeUnknown(),
        "cancel": asyncio.CancelledError(),
        "committed": RuntimeError("commit-raised"),
        "absent": _CommitNotAppliedError(),
        "precommit": RuntimeError("precommit"),
    }[outcome]
    commit = AsyncSession.commit

    async def faulty(session):
        if outcome == "committed":
            await commit(session)
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "flush" if outcome == "precommit" else "commit", faulty)
        with pytest.raises(PersistenceError if outcome == "absent" else type(error)) as raised:
            await store.create_run(run_id, "live")
        if outcome != "absent":
            assert raised.value is error
    assert sent(bus) == []
    assert bus.stats().pending == (0 if outcome in ("precommit", "absent") else 1)
    await store.get_run_status(run_id)
    assert sent(bus) == ([] if outcome in ("precommit", "absent") else [run_id])


@pytest.mark.parametrize("spawn", ["direct", "copy_context", "task_group"])
async def test_nested_same_store_two_stores_and_spawned_task_flush_independently(
    live_store, engine, spawn
):
    from contextvars import copy_context

    from dendrux.runtime.durability import _safe_database_call

    store, bus = live_store
    peer_bus = RecordingUpdates()
    peer = store_for_engine(engine, live_updates=peer_bus)
    ids = [generate_ulid() for _ in range(4)]
    flush = AsyncMock(wraps=bus.flush)
    bus.flush = flush

    @_safe_database_call
    async def outer(self):
        await self.create_run(ids[0], "nested")
        await self.update_run_meta(ids[0], {"test": True})
        assert sent(bus) == []
        await peer.create_run(ids[1], "peer")
        assert sent(peer_bus) == [ids[1]]
        if spawn == "task_group":
            async with asyncio.TaskGroup() as group:
                group.create_task(self.create_run(ids[2], "child"))
        elif spawn == "copy_context":
            await asyncio.create_task(self.create_run(ids[2], "child"), context=copy_context())
        else:
            await asyncio.create_task(self.create_run(ids[2], "child"))
        assert ids[2] in sent(bus)
        await self.create_run(ids[3], "nested")

    try:
        await outer(store)
        assert set(sent(bus)) == {ids[0], ids[2], ids[3]}
        assert flush.await_count == 2
    finally:
        await peer.close()


async def test_failed_outer_call_does_not_flush_nested_commit(live_store):
    from dendrux.runtime.durability import _safe_database_call

    store, bus = live_store
    run_id = generate_ulid()
    error = ValueError("original")

    @_safe_database_call
    async def outer(self):
        await self.create_run(run_id, "nested")
        raise error

    with pytest.raises(ValueError) as raised:
        await outer(store)
    assert raised.value is error
    assert sent(bus) == []
    assert bus.stats().pending == 1
    await store.close()
    assert sent(bus) == [run_id]


@pytest.mark.parametrize("applied", [False, True])
async def test_real_readback_and_retry_publish_only_commit_issued_attempt(
    live_store, monkeypatch, applied
):
    from sqlalchemy.exc import OperationalError

    from tests._helpers.db_errors import LostConnectionError

    store, bus = live_store
    run_id = generate_ulid()
    commit = AsyncSession.commit
    calls = 0
    enqueued = []
    enqueue = bus.enqueue

    def capture(ids):
        enqueued.append(set(ids))
        enqueue(ids)

    monkeypatch.setattr(bus, "enqueue", capture)

    async def lost_ack(session):
        nonlocal calls
        calls += 1
        assert enqueued == []
        if calls == 1:
            if applied:
                await commit(session)
            raise OperationalError("lost", {}, LostConnectionError())
        await commit(session)

    monkeypatch.setattr(AsyncSession, "commit", lost_ack)
    await store.create_run(run_id, "readback")
    assert calls == (1 if applied else 2)
    assert enqueued == [{run_id}]
    assert sent(bus) == [run_id]


async def test_real_commit_unknown_has_pending_then_next_flush(live_store, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from tests._helpers.db_errors import LostConnectionError

    store, bus = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "unknown")
    bus.messages.clear()
    commit = AsyncSession.commit

    async def lost_ack(session):
        await commit(session)
        raise OperationalError("lost", {}, LostConnectionError())

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", lost_ack)
        with pytest.raises(StoreOutcomeUnknown):
            await store.request_cancel(run_id)
    assert sent(bus) == []
    assert bus.stats().pending == 1
    assert await store.is_cancel_requested(run_id)
    assert sent(bus) == [run_id]


async def test_caller_cancellation_during_real_commit_preserves_pending(live_store, monkeypatch):
    store, bus = live_store
    run_id = generate_ulid()
    entered = asyncio.Event()
    commit = AsyncSession.commit

    async def gate(session):
        await commit(session)
        entered.set()
        await asyncio.Event().wait()

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", gate)
        task = asyncio.create_task(store.create_run(run_id, "cancel"))
        async with asyncio.timeout(5):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert sent(bus) == []
    assert bus.stats().pending == 1
    assert await store.get_run_status(run_id) == "running"
    assert sent(bus) == [run_id]


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_two_stores_publish_after_second_connection_visibility(live_store, engine):
    from sqlalchemy import event

    store, bus = live_store
    other_bus = RecordingUpdates()
    other = store_for_engine(engine, live_updates=other_bus)
    ids = [generate_ulid() for _ in range(20)]
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)

    async def verify(messages):
        async with engine.connect() as conn:
            for message in messages:
                for run_id in json.loads(message):
                    assert (
                        await conn.scalar(select(AgentRun.id).where(AgentRun.id == run_id))
                        == run_id
                    )
        bus.messages.extend(messages)

    bus.send = verify
    try:
        await asyncio.gather(
            *(
                store.create_run(rid, "peer") if i % 2 == 0 else other.create_run(rid, "peer")
                for i, rid in enumerate(ids)
            )
        )
        await bus.flush()
        await other_bus.flush()
        assert set(sent(bus)) == set(ids[::2])
        assert set(sent(other_bus)) == set(ids[1::2])
        assert len(sent(bus)) + len(sent(other_bus)) == 20
        assert not any("pg_notify" in sql for sql in statements)
    finally:
        await other.close()


async def test_borrowed_session_hooks_do_not_flush_inside_the_owner_session(live_store):
    store, bus = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "borrowed")
    bus.messages.clear()
    bus.enqueue([run_id])
    flush = AsyncMock(wraps=bus.flush)
    bus.flush = flush
    async with store._session_factory() as session:
        await store.reserve_parent_writer(session, run_id)
        flush.assert_not_awaited()
    await store.get_run_status(run_id)
    assert sent(bus) == [run_id]


async def test_enabled_bus_preserves_save_statement_budget_and_records_sends_separately(engine):
    from sqlalchemy import event

    groups = {False: [], True: []}
    active = False

    def capture(conn, cursor, statement, parameters, context, executemany):
        groups[active].append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    bus = RecordingUpdates()
    for enabled in (False, True):
        store = store_for_engine(engine, live_updates=bus if enabled else None)
        active = enabled
        run_id = generate_ulid()
        await store.get_run_status(run_id)
        groups[enabled].clear()
        await store.create_run(run_id, "budget")
        await store.save_run_event(run_id, event_type="run.started")
        await store.close()
    assert groups[False] == groups[True]
    assert len(bus.messages) == 2
    assert bus.stats().sent == 2
    assert not any("pg_notify" in sql for sql in groups[True])
