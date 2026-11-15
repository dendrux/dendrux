"""Fenced lifecycle writes keep their predicates, locks, CAS and atomic event order."""

import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession

import dendrux.runtime.state as state
from dendrux._meta_keys import BLOCKED_META_KEY
from dendrux.db.models import AgentRun, ReactTrace, RunEvent
from dendrux.errors import RunFencedError
from dendrux.runtime._blocked_content import _dispatch_write
from dendrux.runtime.state import EventWrite, TraceWrite
from tests.integration.test_sql_managed_characterization import _transaction_trace as _sql_trace

RUN = "00000000000000000000000001"
TRACE = "00000000000000000000000002"
EVENT = "00000000000000000000000003"
STAMP = datetime(2026, 11, 1, tzinfo=UTC)


@contextmanager
def _write_trace(engine, record_property):
    parameters = []

    def stable(value):
        if isinstance(value, datetime):
            return "<timestamp>"
        if (
            isinstance(value, str)
            and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", value)
            and value not in (RUN, TRACE, EVENT)
        ):
            return "<generated-id>"
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [stable(v) for v in value]
        return value

    def observe(conn, cursor, statement, values, context, many):
        parameters.append(stable(context.compiled_parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", observe)
    try:
        with _sql_trace(engine) as trace:
            yield trace
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", observe)
        record_property("parameters", json.dumps(parameters, default=str, sort_keys=True))


async def _seed(store, engine):
    await store.create_run(RUN, "lifecycle", tenant_id="t1")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(
                storage_format=2,
                claim_epoch=7,
                loop_state={"phase": "model", "keep": 42},
                pause_data={"old": True},
                cancel_requested=True,
                pause_requested=True,
                reclaim_attempts=2,
                last_reclaimed_at=STAMP,
                meta={"keep": 1},
            )
        )


def _pin(record_property, statements, boundaries, connections, *, commit=True):
    assert len(connections) == 1
    assert boundaries == ["begin", "commit" if commit else "rollback"]
    record_property("sql", json.dumps(statements))
    record_property("boundaries", json.dumps(boundaries))


async def _snapshot(engine):
    async with AsyncSession(engine) as session:
        run = (await session.execute(select(AgentRun.__table__).where(AgentRun.id == RUN))).one()
        traces = (
            await session.scalars(select(ReactTrace.id).where(ReactTrace.agent_run_id == RUN))
        ).all()
        events = (
            await session.scalars(select(RunEvent.id).where(RunEvent.agent_run_id == RUN))
        ).all()
        return dict(run._mapping), list(traces), list(events)


@pytest.mark.parametrize("miss", [None, "epoch", "status"])
async def test_direct_finalize_keeps_fenced_predicates_and_conditional_error_metadata(
    store, engine, record_property, miss
):
    await _seed(store, engine)
    before = await _snapshot(engine)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        won = await store.finalize_run(
            RUN,
            status="success",
            expected_current_status="paused" if miss == "status" else "running",
            claim_epoch=6 if miss == "epoch" else 7,
            error_info={"kind": "safe"},
        )
    assert won is (miss is None)
    assert len(sql) == (3 if won else 1)
    assert sql[0].startswith("UPDATE agent_runs")
    assert "agent_runs.claim_epoch =" in sql[0].split(" WHERE ")[-1]
    assert "agent_runs.status =" in sql[0].split(" WHERE ")[-1]
    _pin(record_property, sql, boundaries, connections)
    after = await _snapshot(engine)
    if won:
        row = after[0]
        assert row["status"] == "success" and row["pause_data"] is None
        assert not row["cancel_requested"] and not row["pause_requested"]
        assert row["meta"] == {"keep": 1, "dendrux.error_info": {"kind": "safe"}}
        assert row["claim_epoch"] == 7 and row["loop_state"] == before[0]["loop_state"]
        assert after[1:] == ([], [])
    else:
        assert after == before


@pytest.mark.parametrize("fallback", [False, True], ids=["native", "locked-fallback"])
@pytest.mark.parametrize("miss", [None, "epoch", "status"])
async def test_cas_keeps_status_epoch_predicates_and_fallback_lock_order(
    store, engine, monkeypatch, record_property, fallback, miss
):
    await _seed(store, engine)
    before = await _snapshot(engine)
    locks = []
    original = store.lock_clause

    def lock(statement, intent):
        locks.append((intent, list(statement.compile().params.values())))
        return original(statement, intent)

    monkeypatch.setattr(store, "lock_clause", lock)
    if fallback:
        monkeypatch.setattr(store, "json_patch_expr", lambda column, patch: None)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        won = await store.finalize_run_if_status_in(
            RUN,
            status="cancelled",
            allowed_current_statuses=[] if miss == "status" else ["running", "paused"],
            claim_epoch=6 if miss == "epoch" else 7,
        )
    assert won is (miss is None)
    expected_count = (3 if engine.dialect.name == "sqlite" else 2) if fallback else 1
    assert len(sql) == expected_count
    assert locks == ([("update", [RUN])] if fallback else [])
    if fallback:
        assert sql[-2].startswith("SELECT agent_runs.loop_state")
        assert ("FOR UPDATE" in sql[-2]) is store.supports_row_locks
        if not store.supports_row_locks:
            assert sql[0].startswith("UPDATE agent_runs SET id=agent_runs.id")
    assert sql[-1].startswith("UPDATE agent_runs")
    assert "agent_runs.status IN" in sql[-1] and "agent_runs.claim_epoch =" in sql[-1]
    _pin(record_property, sql, boundaries, connections)
    after = await _snapshot(engine)
    if won:
        assert after[0]["loop_state"] == {"phase": "finalizing", "keep": 42}
        assert after[0]["pause_data"] is None and not after[0]["cancel_requested"]
        assert not after[0]["pause_requested"] and after[0]["claim_epoch"] == 7
        assert after[1:] == ([], [])
    else:
        assert after == before


async def test_cas_readback_capture_uses_the_fenced_returning_statement(
    store, engine, record_property
):
    await _seed(store, engine)
    captured = []
    token = state._finalized_run_read.set((store, RUN, captured))
    try:
        with _write_trace(engine, record_property) as (sql, boundaries, connections):
            assert await store.finalize_run_if_status_in(
                RUN, status="success", allowed_current_statuses=["running"], claim_epoch=7
            )
    finally:
        state._finalized_run_read.reset(token)
    assert len(sql) == 1 and "RETURNING" in sql[0]
    assert "claim_epoch" in sql[0].split(" WHERE ")[-1] and "status IN" in sql[0]
    assert len(captured) == 1
    assert (captured[0].id, captured[0].tenant_id, captured[0].reclaim_attempts) == (
        RUN,
        "t1",
        2,
    )
    assert captured[0].last_reclaimed_at == STAMP
    _pin(record_property, sql, boundaries, connections)


@pytest.mark.parametrize("identity", [False, True], ids=["locked-event", "identity-event"])
@pytest.mark.parametrize("fault", [None, "trace", "event", "commit"])
async def test_finalize_event_keeps_cas_metadata_trace_event_commit_order_and_rollback(
    store, engine, monkeypatch, record_property, identity, fault
):
    await _seed(store, engine)
    before = await _snapshot(engine)
    phases, sessions = [], []

    def observe(target, name, label):
        original = getattr(target, name)

        async def called(session, *args, **kwargs):
            phases.append(label)
            sessions.append(session)
            if fault == "commit" and label == "commit":
                raise RuntimeError("cut after event")
            result = await original(session, *args, **kwargs)
            if fault == label:
                raise RuntimeError("cut after " + label)
            return result

        monkeypatch.setattr(target, name, called)

    for name, label in [
        ("_cas_finalize", "cas"),
        ("_finalize_error_info", "meta"),
        ("_insert_trace", "trace"),
        ("_write_tx", "commit"),
    ]:
        observe(store, name, label)
    seam = "_write_event" if identity else "_insert_event_locked"
    observe(state, seam, "event")
    kwargs = dict(
        status="success",
        allowed_current_statuses=["running"],
        claim_epoch=7,
        event_type="run.completed",
        event_data={"error_info": {"kind": "safe"}},
        trace=TraceWrite(TRACE, "assistant", "done", 0),
        event=EventWrite(EVENT, "run.completed", 0, {"done": True}, None, None)
        if identity
        else None,
    )
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        if fault:
            with pytest.raises(RuntimeError, match="cut after"):
                await store.finalize_with_event(RUN, **kwargs)
        else:
            assert await store.finalize_with_event(RUN, **kwargs) == 0
    expected = ["cas", "meta", "trace", "event", "commit"]
    assert phases == (expected[: expected.index(fault) + 1] if fault else expected)
    assert all(session is sessions[0] for session in sessions)
    assert len(sql) == (4 if fault == "trace" else 5)
    assert sql[0].startswith("UPDATE agent_runs")
    assert sql[1].startswith("SELECT agent_runs.meta") and sql[2].startswith("UPDATE agent_runs")
    assert sql[3].startswith("INSERT INTO react_traces")
    if fault != "trace":
        assert sql[4].startswith("INSERT INTO run_events")
    _pin(record_property, sql, boundaries, connections, commit=not fault)
    after = await _snapshot(engine)
    if fault:
        assert after == before
    else:
        assert after[0]["status"] == "success" and after[1] == [TRACE] and len(after[2]) == 1
        assert after[0]["meta"]["dendrux.error_info"] == {"kind": "safe"}


async def test_lost_finalize_cas_rolls_back_before_metadata_trace_or_event(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine)
    before = await _snapshot(engine)

    async def forbidden(*args, **kwargs):
        raise AssertionError("CAS loser reached a write")

    for name in ["_finalize_error_info", "_insert_trace", "_write_tx"]:
        monkeypatch.setattr(store, name, forbidden)
    monkeypatch.setattr(state, "_insert_event_locked", forbidden)
    monkeypatch.setattr(state, "_write_event", forbidden)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        assert (
            await store.finalize_with_event(
                RUN,
                status="success",
                allowed_current_statuses=["running"],
                claim_epoch=6,
                event_type="run.completed",
                trace=TraceWrite(TRACE, "assistant", "done", 0),
            )
            is None
        )
    assert len(sql) == 1 and sql[0].startswith("UPDATE agent_runs")
    _pin(record_property, sql, boundaries, connections, commit=False)
    assert await _snapshot(engine) == before


@pytest.mark.parametrize("epoch", [None, 7, 6])
async def test_metadata_fences_before_locking_and_merging_in_one_transaction(
    store, engine, monkeypatch, record_property, epoch
):
    await _seed(store, engine)
    before = await _snapshot(engine)
    locks = []
    original = store.lock_clause

    def lock(statement, intent):
        locks.append(intent)
        return original(statement, intent)

    monkeypatch.setattr(store, "lock_clause", lock)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        if epoch == 6:
            with pytest.raises(RunFencedError):
                await store.update_run_meta(RUN, {"new": 2}, claim_epoch=epoch)
        else:
            await store.update_run_meta(RUN, {"new": 2, "dendrux.blocked": True}, claim_epoch=epoch)
    count = 1 if epoch == 6 else 3 if epoch == 7 or engine.dialect.name == "sqlite" else 2
    assert len(sql) == count
    assert locks == ([] if epoch == 6 else ["update"])
    if epoch is not None:
        assert sql[0].startswith("UPDATE agent_runs") and "agent_runs.claim_epoch =" in sql[0]
    if epoch != 6:
        assert sql[-2].startswith("SELECT agent_runs.") and sql[-1].startswith("UPDATE agent_runs")
        assert ("FOR UPDATE" in sql[-2]) is store.supports_row_locks
    _pin(record_property, sql, boundaries, connections, commit=epoch != 6)
    after = await _snapshot(engine)
    if epoch == 6:
        assert after == before
    else:
        assert after[0]["meta"] == {"keep": 1, "new": 2}


@pytest.mark.parametrize(
    "operation,terminal,epoch",
    [
        ("request_cancel", False, None),
        ("request_cancel", True, None),
        ("touch_progress", False, 7),
        ("touch_progress", False, 6),
        ("touch_progress", True, 7),
    ],
)
async def test_cancel_and_heartbeat_keep_their_single_statement_terminal_and_epoch_guards(
    store, engine, record_property, operation, terminal, epoch
):
    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(status="success" if terminal else "running", cancel_requested=False)
        )
    before = await _snapshot(engine)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        result = await getattr(store, operation)(RUN, **({"claim_epoch": epoch} if epoch else {}))
    assert result is (not terminal and epoch != 6)
    assert len(sql) == 1 and sql[0].startswith("UPDATE agent_runs")
    where = sql[0].split(" WHERE ")[-1]
    assert "agent_runs.status" in where
    assert ("agent_runs.claim_epoch" in where) is (epoch is not None)
    _pin(record_property, sql, boundaries, connections)
    if not result:
        assert await _snapshot(engine) == before


async def test_frontier_cancel_is_one_tenant_scoped_terminal_guarded_write(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine)
    for key, tenant in [(TRACE, "t1"), (EVENT, "t2"), ("00000000000000000000000004", None)]:
        await store.create_run(key, "frontier", tenant_id=tenant)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == TRACE).values(status="success"))
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        assert await store.mark_cancel_requested([RUN, TRACE, EVENT], tenant_id="t1") == 1
    assert len(sql) == 1 and sql[0].startswith("UPDATE agent_runs")
    where = sql[0].split(" WHERE ")[-1]
    assert "tenant_id" in where and "status NOT IN" in where
    _pin(record_property, sql, boundaries, connections)
    assert not (await store.get_run(TRACE)).cancel_requested
    assert not (await store.get_run(EVENT)).cancel_requested
    with _write_trace(engine, record_property) as (empty, _, _):
        assert await store.mark_cancel_requested([], tenant_id="t1") == 0
    assert empty == []


@pytest.mark.parametrize("operation", ["is_cancel_requested", "get_run_control"])
async def test_control_reads_keep_one_projected_select(store, engine, record_property, operation):
    await _seed(store, engine)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        result = await getattr(store, operation)(RUN)
    assert len(sql) == 1 and sql[0].startswith("SELECT agent_runs.cancel_requested")
    assert "loop_state" not in sql[0] and "pause_data" not in sql[0]
    if operation == "get_run_control":
        assert "EXISTS" in sql[0]
        assert result.cancel_requested and result.pause_requested and result.claim_epoch == 7
        assert not result.has_pending_notes and result.status == "running"
        assert await store.get_run_control("missing") == state.RunControl()
    else:
        assert result is True and await store.is_cancel_requested("missing") is False
    _pin(record_property, sql, boundaries, connections, commit=False)


async def _last_progress(engine):
    async with AsyncSession(engine) as session:
        return await session.scalar(select(AgentRun.last_progress_at).where(AgentRun.id == RUN))


async def test_touch_progress_is_scoped_by_dispatch_to_an_unblocked_run(store, engine):
    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == RUN).values(status="running"))
    with _dispatch_write(store, RUN):
        assert await store.touch_progress(RUN, claim_epoch=7) is True
    assert await _last_progress(engine) is not None


async def test_touch_progress_in_dispatch_refuses_a_blocked_run_and_leaves_it_untouched(
    store, engine
):
    from dendrux.loops._helpers import _BlockedObservationError

    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(status="running", meta={"keep": 1, BLOCKED_META_KEY: True})
        )
    before = await _last_progress(engine)
    with _dispatch_write(store, RUN), pytest.raises(_BlockedObservationError):
        await store.touch_progress(RUN, claim_epoch=7)
    assert await _last_progress(engine) == before


async def test_touch_progress_outside_dispatch_still_renews_a_blocked_run(store, engine):
    await _seed(store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(status="running", meta={BLOCKED_META_KEY: True})
        )
    assert await store.touch_progress(RUN, claim_epoch=7) is True
    assert await _last_progress(engine) is not None
