"""Run events and notes keep sequence allocation, fences, scope and statement budgets."""

import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, insert, literal, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

import dendrux.runtime.durability as durability
from dendrux._meta_keys import COMMIT_PROOFS_SESSION_KEY, FRESH_EVIDENCE_SESSION_KEY
from dendrux.db.models import AgentRun, RunEvent, RunInbox
from dendrux.errors import NoteLimitExceededError, PersistenceError, RunFencedError
from dendrux.runtime._bounds import MAX_PAGE, PAUSE_PAGE_MAX
from dendrux.runtime._digest import event_digest
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.stores.sql.base import SQLStateStore

RUN = "00000000000000000000000001"
OTHER = "00000000000000000000000002"
EVENT = "00000000000000000000000003"
MISSING = "00000000000000000000000009"
NOTES = [f"{index:026d}" for index in range(20, 24)]
CORRELATION = "0000000000000000000000000C"
STAMP = datetime(2026, 11, 1, tzinfo=UTC)
STORE_LOGGER = "dendrux.runtime.stores.sql.base"
FIXED = {RUN, OTHER, EVENT, MISSING, CORRELATION, *NOTES}


def _stable(value):
    if isinstance(value, datetime):
        return "<timestamp>"
    if isinstance(value, str) and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", value):
        return value if value in FIXED else "<generated-id>"
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stable(v) for v in value]
    return value


@contextmanager
def _trace(engine, record_property):
    """Capture statements, compiled parameters, transaction boundaries and connections."""
    # Observed connections stay alive: id() is unique only among live objects.
    statements, parameters, boundaries, connections = [], [], [], {}

    def statement(conn, cursor, sql, values, context, many):
        connections[id(conn)] = conn
        statements.append(" ".join(sql.split()))
        parameters.append(_stable(context.compiled_parameters))

    def boundary(kind):
        def observe(conn):
            connections[id(conn)] = conn
            boundaries.append(kind)

        return observe

    listeners = [("before_cursor_execute", statement)] + [
        (kind, boundary(kind)) for kind in ("begin", "commit", "rollback")
    ]
    for kind, listener in listeners:
        event.listen(engine.sync_engine, kind, listener)
    try:
        yield statements, parameters, boundaries, connections
    finally:
        for kind, listener in listeners:
            event.remove(engine.sync_engine, kind, listener)
        record_property("sql", json.dumps(statements))
        record_property("parameters", json.dumps(parameters, default=str, sort_keys=True))
        record_property("boundaries", json.dumps(boundaries))


def _values(parameters):
    return [value for group in parameters for value in group[0].values()]


async def _seed(store, engine, *, status="running"):
    await store.create_run(RUN, "events", tenant_id="t1")
    await store.create_run(OTHER, "events", tenant_id="t2")
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).values(claim_epoch=7, status=status))


async def _events(engine, run_id=RUN):
    async with AsyncSession(engine) as session:
        rows = await session.execute(
            select(RunEvent.__table__)
            .where(RunEvent.agent_run_id == run_id)
            .order_by(RunEvent.sequence_index)
        )
        return [dict(row._mapping) for row in rows]


async def _inbox(engine):
    async with AsyncSession(engine) as session:
        rows = await session.execute(select(RunInbox.__table__).order_by(RunInbox.id))
        return [dict(row._mapping) for row in rows]


async def _insert_events(engine, run_id, specs):
    async with engine.begin() as conn:
        for sequence, kind, offset in specs:
            await conn.execute(
                insert(RunEvent).values(
                    id=f"{run_id[-2:]}{sequence:024d}",
                    agent_run_id=run_id,
                    event_type=kind,
                    sequence_index=sequence,
                    iteration_index=sequence // 2,
                    data={"sequence": sequence},
                    created_at=STAMP + timedelta(seconds=offset),
                )
            )


def _observe_writes(store, monkeypatch):
    """Record the fence, announce and commit order plus the session each one used."""
    phases, sessions, fresh, proofs = [], [], [], []
    fence, announce, commit = store._fence, store._mark_announce, store._write_tx

    async def fenced(session, *args, **kwargs):
        phases.append("fence")
        sessions.append(session)
        return await fence(session, *args, **kwargs)

    def announced(session, run_ids):
        phases.append(("announce", tuple(run_ids)))
        sessions.append(session)
        return announce(session, run_ids)

    async def committed(session):
        phases.append("commit")
        sessions.append(session)
        fresh.append(session.info.get(FRESH_EVIDENCE_SESSION_KEY))
        proofs.extend(
            (table.name, dict(expected))
            for table, _, expected in session.info.get(COMMIT_PROOFS_SESSION_KEY, [])
            if table.name != "agent_runs"
        )
        return await commit(session)

    monkeypatch.setattr(store, "_fence", fenced)
    monkeypatch.setattr(store, "_mark_announce", announced)
    monkeypatch.setattr(store, "_write_tx", committed)
    return phases, sessions, fresh, proofs


@pytest.mark.parametrize("epoch", [7, None], ids=["claimed", "administrative"])
async def test_save_fences_then_allocates_dense_sequence_in_one_transaction(
    store, engine, monkeypatch, record_property, epoch
):
    await _seed(store, engine)
    phases, sessions, fresh, proofs = _observe_writes(store, monkeypatch)
    with _trace(engine, record_property) as (sql, params, boundaries, connections):
        first = await store.save_run_event(
            RUN,
            event_type="run.started",
            iteration_index=2,
            correlation_id=CORRELATION,
            data={"a": 1},
            claim_epoch=epoch,
        )
    assert first == 0
    assert len(sql) == 2 and len(connections) == 1
    assert boundaries == ["begin", "commit"]
    assert sql[0].startswith("UPDATE agent_runs SET updated_at=")
    where = sql[0].split(" WHERE ")[-1]
    assert ("agent_runs.claim_epoch =" in where) is (epoch is not None)
    assert ("last_progress_at=CASE" in sql[0]) is (epoch is not None)
    assert sql[1].startswith("INSERT INTO run_events (id, agent_run_id, event_type, sequence_index")
    assert "coalesce(max(run_events.sequence_index)" in sql[1]
    assert re.search(
        r"ON CONFLICT \(id\) DO NOTHING RETURNING (run_events\.)?sequence_index$", sql[1]
    )
    assert phases == ["fence", ("announce", (RUN,)), ("announce", (RUN,)), "commit"]
    assert all(session is sessions[0] for session in sessions)
    assert fresh == [True]
    digest = event_digest("run.started", 2, CORRELATION, {"a": 1})
    assert proofs == [("run_events", {"agent_run_id": RUN, "content_digest": digest})]
    assert -1 in _values(params[1:]) and 1 in _values(params[1:])
    later = [
        await store.save_run_event(RUN, event_type="tool.completed", data=None),
        await store.save_run_event(RUN, event_type="run.paused", claim_epoch=epoch),
    ]
    assert later == [1, 2]
    rows = await _events(engine)
    assert [row["sequence_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["event_type"] == "run.started" and rows[0]["iteration_index"] == 2
    assert rows[0]["correlation_id"] == CORRELATION and rows[0]["data"] == {"a": 1}
    assert rows[0]["content_digest"] == event_digest("run.started", 2, CORRELATION, {"a": 1})
    assert rows[1]["data"] is None and rows[1]["iteration_index"] == 0
    assert await _events(engine, OTHER) == []


@pytest.mark.parametrize(
    "case", ["replay", "foreign-content", "foreign-run", "digest-mismatch", "fenced"]
)
async def test_event_identity_replays_or_refuses_without_partial_writes(
    store, engine, monkeypatch, record_property, caplog, case
):
    await _seed(store, engine)
    digest = event_digest("run.started", 2, CORRELATION, {"a": 1})
    original = dict(
        event_type="run.started",
        iteration_index=2,
        correlation_id=CORRELATION,
        data={"a": 1},
        claim_epoch=7,
        event_id=EVENT,
    )
    assert await store.save_run_event(RUN, **original) == 0
    before_run, before_other = await _events(engine), await _events(engine, OTHER)
    phases, _, fresh, proofs = _observe_writes(store, monkeypatch)
    run_id, options = RUN, dict(original)
    if case == "replay":
        options["content_digest"] = digest
    elif case == "foreign-content":
        options["data"] = {"a": 2}
    elif case == "foreign-run":
        run_id = OTHER
    elif case == "digest-mismatch":
        options.update(event_id=None, content_digest="f" * 64)
    else:
        options["claim_epoch"] = 6
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    with _trace(engine, record_property) as (sql, _, boundaries, connections):
        if case == "replay":
            assert await store.save_run_event(run_id, **options) == 0
        elif case == "fenced":
            with pytest.raises(RunFencedError):
                await store.save_run_event(run_id, **options)
        else:
            with pytest.raises(PersistenceError) as raised:
                await store.save_run_event(run_id, **options)
            assert raised.value.code == "write_conflict"
    assert len(connections) == 1
    assert sql[0].startswith("UPDATE agent_runs SET updated_at=")
    conflicts = [r.name for r in caplog.records if getattr(r, "code", None) == "write_conflict"]
    if case in ("replay", "foreign-content", "foreign-run"):
        assert len(sql) == 3
        assert sql[1].startswith("INSERT INTO run_events")
        assert sql[2].startswith("SELECT run_events.id, run_events.agent_run_id")
        assert sql[2].endswith("WHERE run_events.id = ?") or re.search(
            r"WHERE run_events\.id = \$1::VARCHAR$", sql[2]
        )
    elif case == "digest-mismatch":
        assert len(sql) == 1
    else:
        assert not any(s.startswith("INSERT") for s in sql)
    if case == "replay":
        assert boundaries == ["begin", "commit"]
        assert phases == ["fence", ("announce", (RUN,)), "commit"]
        assert fresh == [None]
        assert proofs == [("run_events", {"agent_run_id": RUN, "content_digest": digest})]
        assert conflicts == []
    else:
        assert boundaries == ["begin", "rollback"]
        assert "commit" not in phases
        assert conflicts == ([] if case == "fenced" else [STORE_LOGGER])
    assert await _events(engine) == before_run
    assert await _events(engine, OTHER) == before_other


@pytest.mark.parametrize("case", ["fresh", "replay", "certified"])
async def test_insert_without_returning_reads_back_by_primary_key(
    store, engine, monkeypatch, record_property, case
):
    await _seed(store, engine)
    await store.save_run_event(RUN, event_type="run.started", event_id=EVENT)
    monkeypatch.setattr(store, "supports_insert_returning", False)
    certified = []
    if case == "certified":

        def inserted(result):
            certified.append(result.rowcount)
            return 41

        monkeypatch.setattr(store, "inserted_event_sequence", inserted)
    event_id = EVENT if case == "replay" else None
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        seq = await store.save_run_event(RUN, event_type="run.started", event_id=event_id)
    assert boundaries == ["begin", "commit"]
    assert "RETURNING" not in sql[1] and sql[1].endswith("ON CONFLICT (id) DO NOTHING")
    if case == "fresh":
        assert seq == 1 and len(sql) == 3
        assert sql[2].startswith("SELECT run_events.sequence_index FROM run_events WHERE")
    elif case == "replay":
        assert seq == 0 and len(sql) == 3
        assert sql[2].startswith("SELECT run_events.id, run_events.agent_run_id")
    else:
        assert seq == 41 and len(sql) == 2 and certified == [1]
    assert [row["sequence_index"] for row in await _events(engine)] == (
        [0] if case == "replay" else [0, 1]
    )


async def test_sequence_collision_maps_to_write_conflict_and_rolls_back(
    store, engine, monkeypatch, record_property, caplog
):
    await _seed(store, engine)
    await store.save_run_event(RUN, event_type="run.started", event_id=EVENT)
    seen = []

    def colliding(next_index):
        seen.append(str(next_index.compile()))
        return literal(0)

    monkeypatch.setattr(store, "event_sequence_expr", colliding)
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    with (
        _trace(engine, record_property) as (sql, _, boundaries, _connections),
        pytest.raises(PersistenceError) as raised,
    ):
        await store.save_run_event(RUN, event_type="tool.completed")
    assert raised.value.code == "write_conflict"
    assert len(seen) == 1 and "max(run_events.sequence_index)" in seen[0]
    assert len(sql) == 2 and sql[1].startswith("INSERT INTO run_events")
    assert boundaries == ["begin", "rollback"]
    assert [r.name for r in caplog.records if getattr(r, "code", None) == "write_conflict"] == [
        STORE_LOGGER
    ]
    assert [row["id"] for row in await _events(engine)] == [EVENT]


async def test_next_sequence_follows_the_maximum_in_one_read(store, engine, record_property):
    await _seed(store, engine)
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        assert await store.get_next_event_sequence(RUN) == 0
    await _insert_events(engine, RUN, [(0, "run.started", 0), (5, "run.paused", 1)])
    with _trace(engine, record_property) as (more, _, more_boundaries, _connections):
        assert await store.get_next_event_sequence(RUN) == 6
    for statements, kinds in ((sql, boundaries), (more, more_boundaries)):
        assert len(statements) == 1 and kinds == ["begin", "rollback"]
        assert statements[0].startswith("SELECT max(run_events.sequence_index)")
    assert await store.save_run_event(RUN, event_type="tool.completed") == 6


@pytest.mark.parametrize("busier", [OTHER, RUN], ids=["other-busier", "run-busier"])
async def test_next_sequence_and_allocation_ignore_other_runs(
    store, engine, record_property, busier
):
    """Each run reads and allocates from its own maximum, never a global one."""
    await _seed(store, engine)
    quieter = RUN if busier == OTHER else OTHER
    await _insert_events(engine, busier, [(i, "run.started", i) for i in range(8)])
    await _insert_events(engine, quieter, [(0, "run.started", 0), (1, "run.paused", 1)])
    with _trace(engine, record_property) as (sql, _, _boundaries, _connections):
        assert await store.get_next_event_sequence(quieter) == 2
        assert await store.get_next_event_sequence(busier) == 8
    assert all("run_events.agent_run_id =" in s.split(" WHERE ")[-1] for s in sql)
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == quieter).values(claim_epoch=7))
    with _trace(engine, record_property) as (inserts, _, _boundaries, _connections):
        assert await store.save_run_event(quieter, event_type="tool.completed") == 2
    assert "WHERE run_events.agent_run_id =" in inserts[1]
    assert await store.save_run_event(busier, event_type="tool.completed") == 8
    assert [row["sequence_index"] for row in await _events(engine, quieter)] == [0, 1, 2]
    assert [row["sequence_index"] for row in await _events(engine, busier)] == list(range(9))


async def test_first_event_of_a_run_starts_at_zero_beside_a_busy_run(store, engine):
    await _seed(store, engine)
    await _insert_events(engine, OTHER, [(i, "run.started", i) for i in range(5)])
    assert await store.get_next_event_sequence(RUN) == 0
    assert await store.save_run_event(RUN, event_type="run.started") == 0
    assert await store.get_next_event_sequence(RUN) == 1


@pytest.mark.parametrize("tenant", ["t1", "t2", None, UNSCOPED])
async def test_run_events_are_scoped_ordered_and_read_once(store, engine, record_property, tenant):
    await _seed(store, engine)
    await _insert_events(
        engine, RUN, [(2, "tool.completed", 2), (0, "run.started", 0), (1, "llm.completed", 1)]
    )
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        rows = await store.get_run_events(RUN, tenant_id=tenant)
    visible = tenant in ("t1", UNSCOPED)
    assert [row.sequence_index for row in rows] == ([0, 1, 2] if visible else [])
    assert len(sql) == 1 and boundaries == ["begin", "rollback"]
    assert sql[0].startswith("SELECT run_events.id, run_events.agent_run_id")
    assert "ORDER BY run_events.sequence_index" in sql[0] and "LIMIT" not in sql[0]
    assert ("EXISTS" in sql[0]) is (tenant is not UNSCOPED)
    if visible:
        assert rows[1].event_type == "llm.completed" and rows[1].data == {"sequence": 1}
        assert rows[1].created_at == STAMP + timedelta(seconds=1)


@pytest.mark.parametrize(
    ("limit", "pause_page", "expected"),
    [(0, False, 1), (10_000, False, MAX_PAGE), (10_000, True, PAUSE_PAGE_MAX), (2, True, 2)],
)
async def test_run_event_filters_and_limit_clamps(
    store, engine, record_property, limit, pause_page, expected
):
    await _seed(store, engine)
    await _insert_events(
        engine,
        RUN,
        [(0, "run.started", 0)] + [(i, "run.paused", i) for i in range(1, 5)] + [(5, "x", 5)],
    )
    with _trace(engine, record_property) as (sql, params, _, _connections):
        rows = await store.get_run_events(
            RUN,
            after_sequence_index=1,
            limit=limit,
            event_types=["run.paused", "run.started"],
            before=STAMP + timedelta(seconds=4),
            tenant_id="t1",
            _pause_page=pause_page,
        )
    assert [row.sequence_index for row in rows] == [2, 3][: max(1, min(limit, 2))]
    assert len(sql) == 1 and "LIMIT" in sql[0]
    assert expected in _values(params)
    where = sql[0].split(" WHERE ")[-1]
    assert "run_events.sequence_index >" in where and "run_events.created_at <" in where
    assert "run_events.event_type IN" in where


async def test_run_event_empty_type_list_and_naive_cutoff(store, engine, record_property):
    await _seed(store, engine)
    await _insert_events(engine, RUN, [(0, "run.started", 0)])
    with _trace(engine, record_property) as (sql, _, _boundaries, _connections):
        assert await store.get_run_events(RUN, event_types=[]) == []
    assert len(sql) == 1 and "event_type" not in sql[0].split(" WHERE ")[-1]
    with (
        _trace(engine, record_property) as (none, _, _boundaries, _connections),
        pytest.raises(ValueError),
    ):
        await store.get_run_events(RUN, before=datetime(2026, 11, 1))
    assert none == []


async def test_event_batches_follow_heads_limits_and_omit_missing_runs(
    store, engine, record_property
):
    await _seed(store, engine)
    await _insert_events(engine, RUN, [(i, "run.started", i) for i in range(4)])
    await _insert_events(engine, OTHER, [(i, "tool.completed", i) for i in range(3)])
    heads = {RUN: -1, OTHER: 0, MISSING: 5}
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        grouped = await store.get_events_for_runs(heads, limit_per_run=2)
    assert {run: [e.sequence_index for e in events] for run, events in grouped.items()} == {
        RUN: [0, 1],
        OTHER: [1, 2],
    }
    assert len(sql) == 1 and boundaries == ["begin", "rollback"]
    async with AsyncSession(engine) as session:
        rows = (
            await session.execute(
                SQLStateStore.events_for_runs_query(store, heads, limit_per_run=2)
            )
        ).all()
    assert sorted((row.agent_run_id, row.sequence_index) for row in rows) == [
        (RUN, 0),
        (RUN, 1),
        (OTHER, 1),
        (OTHER, 2),
    ]
    query = str(SQLStateStore.events_for_runs_query(store, heads, limit_per_run=2))
    assert query.count("UNION ALL") == 2 and query.count("ORDER BY") == 3
    with _trace(engine, record_property) as (empty, _, _boundaries, _connections):
        assert await store.get_events_for_runs({}, limit_per_run=2) == {}
        for bad in (0, 101):
            with pytest.raises(ValueError):
                await store.get_events_for_runs(heads, limit_per_run=bad)
    assert empty == []


def _observe_note(store, monkeypatch):
    phases = []
    serialize, fence, lock = store.serialize_in_tx, store._fence, store.lock_clause
    refuse, commit = store._refuse_note_insert, store._write_tx

    async def serialized(session, *, owner):
        phases.append(("serialize", owner.run_id, owner.tenant_scope))
        return await serialize(session, owner=owner)

    async def fenced(session, run_id, epoch, **values):
        phases.append(("fence", epoch))
        return await fence(session, run_id, epoch, **values)

    def locked(statement, intent):
        phases.append(("lock", intent))
        return lock(statement, intent)

    async def refused(session, run_id, tenant_id, max_notes, count_query):
        phases.append(("refuse", max_notes))
        return await refuse(session, run_id, tenant_id, max_notes, count_query)

    async def committed(session):
        phases.append("commit")
        return await commit(session)

    for name, value in (
        ("serialize_in_tx", serialized),
        ("_fence", fenced),
        ("lock_clause", locked),
        ("_refuse_note_insert", refused),
        ("_write_tx", committed),
    ):
        monkeypatch.setattr(store, name, value)
    return phases


@pytest.mark.parametrize(
    "mode",
    [
        "plain",
        "limited",
        "full",
        "terminal",
        "terminal-limited",
        "foreign",
        "foreign-limited",
        "foreign-full",
        "fenced",
    ],
)
async def test_add_note_guards_lock_order_and_single_transaction(
    store, engine, monkeypatch, record_property, caplog, mode
):
    await _seed(store, engine, status="success" if mode.startswith("terminal") else "running")
    if mode in ("full", "foreign-full"):
        async with engine.begin() as conn:
            for note in NOTES[1:3]:
                await conn.execute(
                    insert(RunInbox).values(id=note, run_id=RUN, text="old", created_at=STAMP)
                )
    before = await _inbox(engine)
    limited = mode in ("limited", "full", "terminal-limited", "foreign-limited", "foreign-full")
    tenant = "t2" if mode.startswith("foreign") else "t1"
    epoch = 6 if mode == "fenced" else 7 if mode == "limited" else None
    phases = _observe_note(store, monkeypatch)
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    call = store.add_note(
        RUN,
        note_id=NOTES[0],
        text="note",
        metadata={"k": 1} if mode == "plain" else None,
        tenant_id=tenant,
        max_notes=2 if limited else None,
        claim_epoch=epoch,
    )
    with _trace(engine, record_property) as (sql, _, boundaries, connections):
        if mode == "full":
            with pytest.raises(NoteLimitExceededError) as raised:
                await call
            assert raised.value.limit == 2
        elif mode == "fenced":
            with pytest.raises(RunFencedError):
                await call
        else:
            assert await call is (mode in ("plain", "limited"))
    assert len(connections) == 1
    postgres = engine.dialect.name == "postgresql"
    scoped_precheck = limited and postgres
    serialized = [("serialize", RUN, tenant)] if limited else []
    if mode in ("foreign-limited", "foreign-full") and postgres:
        assert phases == [] and len(sql) == 1
        assert sql[0].startswith("SELECT agent_runs.id FROM agent_runs WHERE")
        assert boundaries == ["begin", "rollback"]
        assert await _inbox(engine) == before
        return
    fence = [("fence", epoch)] if epoch is not None else []
    if mode == "fenced":
        assert phases == fence
        assert boundaries == ["begin", "rollback"]
        assert await _inbox(engine) == before
        return
    inserted = mode in ("plain", "limited")
    expected = serialized + fence + [("lock", "share")]
    expected += ["commit"] if inserted else [("refuse", 2 if limited else None)]
    assert phases == expected
    offset = 2 if scoped_precheck else 0
    if scoped_precheck:
        assert sql[0].startswith("SELECT agent_runs.id FROM agent_runs WHERE")
        assert "pg_advisory_xact_lock" in sql[1]
    insert_sql = sql[offset + len(fence)]
    assert insert_sql.startswith("INSERT INTO run_inbox (id, run_id, text, metadata, created_at)")
    assert ("count(*)" in insert_sql) is limited
    assert insert_sql.endswith(" FOR SHARE") is postgres
    refusal = sql[offset + len(fence) + 1 :]
    if inserted:
        assert refusal == [] and boundaries == ["begin", "commit"]
        row = (await _inbox(engine))[0]
        assert (row["id"], row["run_id"], row["text"]) == (NOTES[0], RUN, "note")
        assert row["metadata"] == ({"k": 1} if mode == "plain" else None)
        assert row["delivered_at"] is None and row["delivered_iteration"] is None
    else:
        assert len(refusal) == (2 if limited else 0)
        if limited:
            assert refusal[0].startswith("SELECT agent_runs.status FROM agent_runs WHERE")
            assert refusal[1].startswith("SELECT count(*)")
        assert boundaries == ["begin", "rollback"]
        assert await _inbox(engine) == before
    codes = [r.name for r in caplog.records if getattr(r, "code", None) == "note_limit"]
    assert codes == ([STORE_LOGGER] if mode == "full" else [])


async def test_add_note_refuses_invalid_limits_before_sql(store, engine, record_property):
    await _seed(store, engine)
    with _trace(engine, record_property) as (sql, _, _boundaries, _connections):
        for bad in (0, -1, True):
            with pytest.raises(ValueError):
                await store.add_note(RUN, note_id=NOTES[0], text="n", max_notes=bad)
    assert sql == [] and await _inbox(engine) == []


async def _seed_notes(engine):
    async with engine.begin() as conn:
        for note, run_id, offset, delivered in (
            (NOTES[2], RUN, 0, None),
            (NOTES[0], RUN, 1, None),
            (NOTES[1], RUN, 1, 3),
            (NOTES[3], OTHER, 0, None),
        ):
            await conn.execute(
                insert(RunInbox).values(
                    id=note,
                    run_id=run_id,
                    text=f"text-{note[-1]}",
                    metadata_={"n": note[-1]},
                    created_at=STAMP + timedelta(seconds=offset),
                    delivered_at=None if delivered is None else STAMP,
                    delivered_iteration=delivered,
                )
            )


@pytest.mark.parametrize("tenant", ["t1", "t2", None, UNSCOPED])
@pytest.mark.parametrize("undelivered", [False, True])
async def test_notes_read_in_creation_order_within_scope(
    store, engine, record_property, tenant, undelivered
):
    await _seed(store, engine)
    await _seed_notes(engine)
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        notes = await store.get_notes(RUN, undelivered_only=undelivered, tenant_id=tenant)
    expected = [NOTES[2], NOTES[0]] if undelivered else [NOTES[2], NOTES[0], NOTES[1]]
    assert [note.id for note in notes] == (expected if tenant in ("t1", UNSCOPED) else [])
    assert len(sql) == 1 and boundaries == ["begin", "rollback"]
    assert sql[0].endswith("ORDER BY run_inbox.created_at, run_inbox.id")
    assert ("EXISTS" in sql[0]) is (tenant is not UNSCOPED)
    assert ("delivered_at IS NULL" in sql[0]) is undelivered
    if notes:
        delivered = {note.id: note for note in notes}.get(NOTES[1])
        assert notes[0].metadata == {"n": NOTES[2][-1]} and notes[0].created_at == STAMP
        if delivered:
            assert delivered.delivered_iteration == 3 and delivered.delivered_at == STAMP


@pytest.mark.parametrize("epoch", [7, None, 6], ids=["claimed", "unclaimed", "fenced"])
async def test_mark_delivered_touches_only_undelivered_notes_of_the_run(
    store, engine, monkeypatch, record_property, epoch
):
    await _seed(store, engine)
    await _seed_notes(engine)
    before = await _inbox(engine)
    phases, _, _, proofs = _observe_writes(store, monkeypatch)
    with _trace(engine, record_property) as (sql, _, boundaries, _connections):
        if epoch == 6:
            with pytest.raises(RunFencedError):
                await store.mark_notes_delivered(RUN, NOTES, iteration=4, claim_epoch=epoch)
        else:
            assert await store.mark_notes_delivered(RUN, NOTES, iteration=4, claim_epoch=epoch) == 2
    after = {row["id"]: row for row in await _inbox(engine)}
    if epoch == 6:
        assert phases == ["fence"] and boundaries == ["begin", "rollback"]
        assert not any(s.startswith("UPDATE run_inbox") for s in sql)
        assert list(after.values()) == before
        return
    assert phases == (["fence"] if epoch else []) + ["commit"] and proofs == []
    assert boundaries == ["begin", "commit"]
    assert len(sql) == (2 if epoch else 1)
    assert sql[-1].startswith("UPDATE run_inbox SET delivered_at=")
    assert "run_inbox.delivered_at IS NULL" in sql[-1].split(" WHERE ")[-1]
    assert [after[n]["delivered_iteration"] for n in NOTES] == [4, 3, 4, None]
    assert after[NOTES[3]]["delivered_at"] is None


@pytest.mark.parametrize(
    ("operation", "label"),
    [
        ("add_note", "add_note"),
        ("mark_notes_delivered", "mark_notes_delivered"),
        ("get_next_event_sequence", "get_next_event_sequence"),
        ("save_run_event", None),
    ],
)
async def test_transient_session_failure_retries_only_wrapped_operations(
    store, engine, monkeypatch, record_property, operation, label
):
    await _seed(store, engine)
    retries, opened = [], []
    factory = store._session_factory

    def flaky():
        opened.append(len(opened))
        if len(opened) == 1:
            raise OperationalError("SELECT 1", {}, ConnectionError("transient"))
        return factory()

    async def backoff(attempt, retry_label, run_id, exc):
        retries.append((attempt, retry_label, run_id, type(exc).__name__))

    monkeypatch.setattr(store, "_session_factory", flaky)
    monkeypatch.setattr(durability, "_backoff", backoff)
    call = {
        "add_note": lambda: store.add_note(RUN, note_id=NOTES[0], text="n", claim_epoch=7),
        "mark_notes_delivered": lambda: store.mark_notes_delivered(RUN, [], iteration=1),
        "get_next_event_sequence": lambda: store.get_next_event_sequence(RUN),
        "save_run_event": lambda: store.save_run_event(RUN, event_type="run.started"),
    }[operation]
    with _trace(engine, record_property) as (_sql, _, boundaries, _connections):
        if label is None:
            with pytest.raises(PersistenceError) as raised:
                await call()
            assert raised.value.retryable
        else:
            await call()
    if label is None:
        assert retries == [] and opened == [0] and boundaries == []
        assert await _events(engine) == []
    else:
        assert retries == [(0, label, RUN, "OperationalError")] and opened == [0, 1]
        assert boundaries[0] == "begin" and len(boundaries) == 2


@pytest.mark.parametrize("operation", ["save_run_event", "add_note", "mark_notes_delivered"])
async def test_commit_failure_rolls_back_every_journal_write(
    store, engine, monkeypatch, record_property, operation
):
    await _seed(store, engine)
    await _seed_notes(engine)
    before = (await _events(engine), await _inbox(engine))

    async def refuse(session):
        raise PersistenceError(code="commit_refused")

    monkeypatch.setattr(store, "_write_tx", refuse)
    call = {
        "save_run_event": lambda: store.save_run_event(RUN, event_type="run.started"),
        "add_note": lambda: store.add_note(RUN, note_id=MISSING, text="n"),
        "mark_notes_delivered": lambda: store.mark_notes_delivered(RUN, NOTES, iteration=4),
    }[operation]
    with (
        _trace(engine, record_property) as (sql, _, boundaries, connections),
        pytest.raises(PersistenceError) as raised,
    ):
        await call()
    assert raised.value.code == "commit_refused"
    assert boundaries == ["begin", "rollback"] and len(connections) == 1
    assert len(sql) == {"save_run_event": 2, "add_note": 1, "mark_notes_delivered": 1}[operation]
    assert (await _events(engine), await _inbox(engine)) == before
