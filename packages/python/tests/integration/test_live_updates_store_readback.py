"""Focused ambiguous-COMMIT decisions and implicit event announcement marks."""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, update
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from dendrux._meta_keys import (
    ANNOUNCE_MARKS_SESSION_KEY,
    COMMIT_PROOFS_SESSION_KEY,
    TOUCHED_RUNS_SESSION_KEY,
)
from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, StoreOutcomeUnknown
from dendrux.runtime.durability import _CommitNotAppliedError
from dendrux.runtime.stores.sql._commit import _CommitReadback
from dendrux.types import generate_ulid
from tests._helpers.db_errors import LostConnectionError
from tests.integration.test_live_updates_store_transactions import live_store as live_store
from tests.integration.test_live_updates_store_transactions import sent


async def test_event_shim_alone_marks_its_committed_run(live_store):
    from dendrux.runtime.state import _insert_event

    store, bus = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "event-only")
    bus.messages.clear()
    async with store._session_factory() as session:
        assert not session.info.get(ANNOUNCE_MARKS_SESSION_KEY)
        await _insert_event(session, run_id, "test.event", {})
        assert sent(bus) == []
        await store._write_tx(session)
    assert bus.stats().pending == 1
    await bus.flush()
    assert sent(bus) == [run_id]
    events = await store.get_run_events(run_id)
    assert [event.event_type for event in events] == ["test.event"]


@pytest.mark.parametrize("operation", ["try_pause_run", "pause_with_event"])
@pytest.mark.parametrize(
    "status,cancelled,retry",
    [("running", False, True), ("success", False, False), ("running", True, False)],
)
async def test_pause_readback_retries_only_running_uncancelled_row(
    live_store, engine, operation, status, cancelled, retry
):
    store, _ = live_store
    run_id = generate_ulid()
    await store.create_run(run_id, "pause-proof")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(status=status, cancel_requested=cancelled)
        )
    async with store._session_factory() as session:
        proof = _CommitReadback(
            store, session, (store, operation, {"run_id": run_id, "claim_epoch": 0})
        )
        with pytest.raises(_CommitNotAppliedError if retry else StoreOutcomeUnknown):
            await proof._lost()


@pytest.mark.parametrize("connection_lost", [False, True])
async def test_failed_rollback_preserves_unknown_connection_outcome(
    live_store, monkeypatch, connection_lost
):
    store, _ = live_store
    async with store._session_factory() as session:
        rollback = AsyncMock(side_effect=SQLAlchemyError("rollback-fault"))
        with monkeypatch.context() as patch:
            patch.setattr(session, "rollback", rollback)
            proof = _CommitReadback(store, session, None)
            error = (
                OperationalError("commit-fault", {}, LostConnectionError())
                if connection_lost
                else SQLAlchemyError("commit-fault")
            )
            with pytest.raises(
                StoreOutcomeUnknown if connection_lost else PersistenceError
            ) as raised:
                await proof.resolve(error)
            assert type(raised.value) is (
                StoreOutcomeUnknown if connection_lost else PersistenceError
            )
            rollback.assert_awaited_once()


async def test_readback_locks_all_touched_parents_before_proofs(live_store, monkeypatch):
    store, _ = live_store
    ids = sorted(generate_ulid() for _ in range(2))
    for run_id in ids:
        await store.create_run(run_id, "lock-proof")
    seen = []
    reserve = store.reserve_parent_writer
    lock = store.lock_clause

    async def reserve_parent(session, run_id):
        seen.append(("reserve", run_id))
        await reserve(session, run_id)

    def locking(statement, mode):
        assert mode == "share"
        kind = "parent" if list(statement.selected_columns.keys()) == ["id"] else "proof"
        seen.append((kind, next(iter(statement.compile().params.values()))))
        return lock(statement, mode)

    monkeypatch.setattr(store, "reserve_parent_writer", reserve_parent)
    monkeypatch.setattr(store, "lock_clause", locking)
    async with store._session_factory() as session:
        session.info[TOUCHED_RUNS_SESSION_KEY] = set(reversed(ids))
        session.info[COMMIT_PROOFS_SESSION_KEY] = [
            (AgentRun.__table__, AgentRun.id == ids[0], {"agent_name": "lock-proof"})
        ]
        proof = _CommitReadback(store, session, None)
        await proof._prove()
    expected = [("parent", run_id) for run_id in ids]
    if not store.supports_row_locks:
        expected = [item for run_id in ids for item in (("reserve", run_id), ("parent", run_id))]
    assert seen == [*expected, ("proof", ids[0])]


@pytest.mark.parametrize(
    "actual,expected,matches",
    [
        (Decimal("1.234567"), 1.2345674, True),
        (datetime(2025, 1, 1), datetime(2025, 1, 1, tzinfo=UTC), True),
        ({"value": True}, {"value": 1}, False),
    ],
)
async def test_readback_compares_database_normalized_values(actual, expected, matches):
    table = Table("proof", MetaData(), Column("value", Integer))
    result = SimpleNamespace(
        mappings=lambda: SimpleNamespace(one_or_none=lambda: {"value": actual})
    )
    session = SimpleNamespace(info={}, execute=AsyncMock(return_value=result))
    store = SimpleNamespace(lock_clause=lambda statement, mode: statement)
    proof = _CommitReadback(store, session, None)
    failure = StoreOutcomeUnknown()
    proof._lost = AsyncMock(side_effect=failure)
    if matches:
        await proof._check_proof(table, table.c.value == 1, {"value": expected})
        proof._lost.assert_not_awaited()
    else:
        with pytest.raises(StoreOutcomeUnknown) as raised:
            await proof._check_proof(table, table.c.value == 1, {"value": expected})
        assert raised.value is failure


async def test_created_run_with_false_absent_readback_relies_on_polling(live_store, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    store, bus = live_store
    run_id = generate_ulid()
    commit = AsyncSession.commit

    async def committed_without_ack(session):
        await commit(session)
        raise OperationalError("commit-fault", {}, LostConnectionError())

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", committed_without_ack)
        absent = AsyncMock(side_effect=_CommitNotAppliedError())
        patch.setattr(_CommitReadback, "_prove", absent)
        result = await store.create_run(run_id, "false-absent")
    absent.assert_awaited_once()
    assert result.outcome == "created"
    assert sent(bus) == []
    assert bus.stats().pending == 0
    assert await store.get_run_status(run_id) == "running"
    assert sent(bus) == []
