"""Pause claims keep their epoch CAS, run scope, submission identity and statement budgets."""

import asyncio
import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

import dendrux.runtime._blob_catalog as blob_catalog
import dendrux.runtime.durability as durability
import dendrux.runtime.state as state
from dendrux._lifecycle import CancelReason, StoredEventType
from dendrux._meta_keys import COMMIT_PROOFS_SESSION_KEY
from dendrux.db.models import AgentRun, ReactTrace, RunEvent
from dendrux.errors import (
    BlobNotFoundError,
    PauseStatusMismatchError,
    PersistenceError,
    RunFencedError,
)
from dendrux.media import BlobEdge
from dendrux.runtime.state import ClaimResult, EventWrite, LoopStatePatch, TraceWrite
from dendrux.runtime.stores.sql import base as sql_base
from dendrux.types import InputValidationError

RUN = "00000000000000000000000001"
SIBLING = "00000000000000000000000002"
OTHER = "00000000000000000000000003"
MISSING = "00000000000000000000000009"
RUNS = (RUN, SIBLING, OTHER)
TRACE = "0000000000000000000000000T"
RELEASE_EVENT = "0000000000000000000000000E"
STORE_LOGGER = "dendrux.runtime.stores.sql.base"
FIXED = {*RUNS, MISSING, TRACE, RELEASE_EVENT}
WAITING = "waiting_client_tool"
SUB = {"submission_id": "s1", "kind": "tool_results", "pause_id": "p1", "call_ids": ["c1"]}
LOOP = {
    "v": 1,
    "loop": "react",
    "phase": "await_tools",
    "pending": [{"call_id": "c1", "phase": "waiting"}, {"call_id": "c2", "phase": "waiting"}],
    "submission": SUB,
}
PAUSE = {"pause_id": "p1", "submission": {**SUB, "payload": {"c1": "old"}}}
COLUMNS = (
    "status",
    "claim_epoch",
    "storage_format",
    "pause_requested",
    "cancel_requested",
    "pause_data",
    "loop_state",
    "reclaim_attempts",
    "pii_mapping",
    "last_progress_at",
    "status_changed_at",
    "updated_at",
    "tenant_id",
)


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
    statements, parameters, boundaries, connections = [], [], [], set()

    def statement(conn, cursor, sql, values, context, many):
        connections.add(id(conn))
        statements.append(" ".join(sql.split()))
        parameters.append(_stable(context.compiled_parameters))

    def boundary(kind):
        def observe(conn):
            connections.add(id(conn))
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


def _pg(engine):
    return engine.dialect.name == "postgresql"


def _where(sql):
    return sql.split(" WHERE ", 1)[1]


async def _seed(
    store,
    engine,
    *,
    status=WAITING,
    epoch=3,
    fmt=2,
    loop_state=LOOP,
    pause_data=PAUSE,
    cancel=False,
):
    """Three identical runs: the target, a same-tenant sibling and a foreign-tenant run."""
    await store.create_run(RUN, "claims", tenant_id="t1")
    await store.create_run(SIBLING, "claims", tenant_id="t1")
    await store.create_run(OTHER, "claims", tenant_id="t2")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).values(
                status=status,
                claim_epoch=epoch,
                storage_format=fmt,
                loop_state=loop_state,
                pause_data=pause_data,
                pause_requested=True,
                cancel_requested=cancel,
                reclaim_attempts=2,
                last_progress_at=None,
                status_changed_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )


async def _rows(engine):
    async with AsyncSession(engine) as session:
        rows = await session.execute(
            select(AgentRun.id, *(getattr(AgentRun, name) for name in COLUMNS)).order_by(
                AgentRun.id
            )
        )
        return {row.id: {name: getattr(row, name) for name in COLUMNS} for row in rows}


async def _events(engine, run_id=None):
    async with AsyncSession(engine) as session:
        stmt = select(RunEvent.__table__).order_by(RunEvent.agent_run_id, RunEvent.sequence_index)
        if run_id is not None:
            stmt = stmt.where(RunEvent.agent_run_id == run_id)
        return [dict(row._mapping) for row in await session.execute(stmt)]


async def _traces(engine):
    async with AsyncSession(engine) as session:
        rows = await session.execute(select(ReactTrace.__table__).order_by(ReactTrace.id))
        return [dict(row._mapping) for row in rows]


def _set(engine, run_id, **values):
    async def apply():
        async with engine.begin() as conn:
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))

    return apply()


def _observe(store, monkeypatch):
    """Record the fence, trace, finalize, announce and commit order, sessions and proofs."""
    phases, sessions = [], []
    proofs = []
    fence, trace, finalize = store._fence, store._insert_trace, store._cas_finalize
    announce, commit = store._mark_announce, store._write_tx

    async def fenced(session, run_id, claim_epoch, **values):
        phases.append(("fence", run_id, claim_epoch, sorted(values)))
        sessions.append(session)
        return await fence(session, run_id, claim_epoch, **values)

    async def traced(session, run_id, write):
        phases.append(("trace", run_id, write.trace_id))
        sessions.append(session)
        return await trace(session, run_id, write)

    async def finalized(session, run_id, **kwargs):
        phases.append(
            (
                "finalize",
                run_id,
                kwargs["status"],
                tuple(kwargs["allowed_current_statuses"]),
                kwargs["claim_epoch"],
            )
        )
        sessions.append(session)
        return await finalize(session, run_id, **kwargs)

    def announced(session, run_ids):
        phases.append(("announce", tuple(run_ids)))
        sessions.append(session)
        return announce(session, run_ids)

    async def committed(session):
        phases.append("commit")
        sessions.append(session)
        proofs.extend(
            (table.name, _stable(expected))
            for table, _, expected in session.info.get(COMMIT_PROOFS_SESSION_KEY, [])
        )
        return await commit(session)

    monkeypatch.setattr(store, "_fence", fenced)
    monkeypatch.setattr(store, "_insert_trace", traced)
    monkeypatch.setattr(store, "_cas_finalize", finalized)
    monkeypatch.setattr(store, "_mark_announce", announced)
    monkeypatch.setattr(store, "_write_tx", committed)
    return phases, sessions, proofs


def _untouched(before, after, *run_ids):
    for run_id in run_ids:
        assert after[run_id] == before[run_id], run_id


def _codes(caplog, code):
    return [r.run_id for r in caplog.records if getattr(r, "code", None) == code]


# ---------------------------------------------------------------------------
# claim_paused_run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["returning", "locked"])
@pytest.mark.parametrize("check", ["plain", "pause_id", "validate"])
async def test_claim_wins_once_increments_epoch_and_touches_only_its_run(
    store, engine, monkeypatch, record_property, path, check
):
    await _seed(store, engine)
    if path == "locked":
        store.supports_update_returning = False
    seen = []
    phases, sessions, proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, boundaries, connections):
        token = await store.claim_paused_run(
            RUN,
            expected_status=WAITING,
            expected_pause_id="p1" if check == "pause_id" else None,
            validate=seen.append if check == "validate" else None,
        )
    assert token == ClaimResult(4, 2, LOOP, SUB)
    assert seen == ([PAUSE] if check == "validate" else [])
    after = await _rows(engine)
    row = after[RUN]
    assert (row["status"], row["claim_epoch"], row["pause_requested"]) == ("running", 4, False)
    assert row["last_progress_at"] is not None
    assert row["status_changed_at"] != before[RUN]["status_changed_at"]
    assert (row["pause_data"], row["loop_state"]) == (PAUSE, LOOP)
    _untouched(before, after, SIBLING, OTHER)
    assert phases == [("announce", (RUN,)), "commit"]
    assert all(session is sessions[0] for session in sessions)
    assert boundaries == ["begin", "commit"] and len(connections) == 1
    validating = check != "plain"
    if path == "returning":
        assert len(sql) == 2 + validating
        assert sql[0].startswith("UPDATE agent_runs SET ")
        assert "claim_epoch=(agent_runs.claim_epoch + " in sql[0]
        assert "pause_requested=" in sql[0] and "last_progress_at=" in sql[0]
        assert "agent_runs.id = " in _where(sql[0]) and "agent_runs.status = " in _where(sql[0])
        assert sql[1].startswith(
            "SELECT agent_runs.claim_epoch, agent_runs.storage_format, agent_runs.loop_state"
        )
        assert "agent_runs.id = " in _where(sql[1])
        if validating:
            assert sql[2].startswith("SELECT agent_runs.pause_data FROM agent_runs WHERE")
            assert "agent_runs.id = " in _where(sql[2])
    else:
        reserve = 0 if _pg(engine) else 1
        assert len(sql) == 3 + reserve
        if reserve:
            assert sql[0].startswith("UPDATE agent_runs SET id=agent_runs.id")
            assert "agent_runs.status = " in _where(sql[0])
        lock = sql[reserve]
        projection = "agent_runs.id, agent_runs.pause_data" if validating else "agent_runs.id"
        assert lock.startswith(f"SELECT {projection} FROM agent_runs WHERE")
        assert "agent_runs.id = " in _where(lock) and "agent_runs.status = " in _where(lock)
        assert lock.endswith(" FOR UPDATE") is _pg(engine)
        assert "claim_epoch=(agent_runs.claim_epoch + " in sql[reserve + 1]
        assert "agent_runs.status" not in _where(sql[reserve + 1])
        assert sql[reserve + 2].startswith("SELECT agent_runs.claim_epoch")

    # The run is no longer paused: a second claim loses, but still announces and commits.
    phases.clear()
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        assert await store.claim_paused_run(RUN, expected_status=WAITING) is False
    assert await _rows(engine) == after
    assert phases == [("announce", (RUN,)), "commit"]
    assert boundaries == ["begin", "commit"]
    assert len(sql) == (1 if path == "returning" else 1 + (not _pg(engine)))


@pytest.mark.parametrize("path", ["returning", "locked"])
@pytest.mark.parametrize(
    "case", ["wrong-status", "terminal", "missing-run", "pause-id-mismatch", "validate-raises"]
)
async def test_claim_refusals_change_nothing(
    store, engine, monkeypatch, record_property, caplog, path, case
):
    await _seed(store, engine)
    if case == "terminal":
        await _set(engine, RUN, status="success")
    if path == "locked":
        store.supports_update_returning = False
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    phases, _sessions, _proofs = _observe(store, monkeypatch)
    before = await _rows(engine)

    def refuse(pause):
        raise ValueError("refused")

    kwargs = {
        "wrong-status": {"expected_status": "waiting_approval"},
        "terminal": {"expected_status": WAITING},
        "missing-run": {"expected_status": WAITING},
        "pause-id-mismatch": {"expected_status": WAITING, "expected_pause_id": "p0"},
        "validate-raises": {"expected_status": WAITING, "validate": refuse},
    }[case]
    target = MISSING if case == "missing-run" else RUN
    with _trace(engine, record_property) as (_sql, _params, boundaries, _connections):
        if case == "pause-id-mismatch":
            with pytest.raises(PauseStatusMismatchError) as raised:
                await store.claim_paused_run(target, **kwargs)
            assert raised.value.reason == "pause_id"
            assert (raised.value.expected_pause_id, raised.value.current_pause_id) == ("p0", "p1")
        elif case == "validate-raises":
            with pytest.raises(ValueError, match="refused"):
                await store.claim_paused_run(target, **kwargs)
        else:
            assert await store.claim_paused_run(target, **kwargs) is False
    assert await _rows(engine) == before
    raised_case = case in ("pause-id-mismatch", "validate-raises")
    assert boundaries == (["begin", "rollback"] if raised_case else ["begin", "commit"])
    assert phases == ([] if raised_case else [("announce", (target,)), "commit"])
    assert _codes(caplog, "pause_id_mismatch") == ([RUN] if case == "pause-id-mismatch" else [])


async def test_claim_release_cycle_fences_every_stale_epoch(store, engine, record_property):
    await _seed(store, engine)
    first = await store.claim_paused_run(RUN, expected_status=WAITING)
    sibling = await store.claim_paused_run(SIBLING, expected_status=WAITING)
    assert (first.claim_epoch, sibling.claim_epoch) == (4, 4)
    assert await store.release_run_claim(RUN, status=WAITING, claim_epoch=4) is True
    second = await store.claim_paused_run(RUN, expected_status=WAITING)
    assert second.claim_epoch == 5
    before = await _rows(engine)
    assert await store.release_run_claim(RUN, status=WAITING, claim_epoch=4) is False
    with pytest.raises(RunFencedError):
        await store.consume_submission(
            RUN, claim_epoch=4, submission_id="s1", traces=[], patch=LoopStatePatch()
        )
    assert await _rows(engine) == before
    assert await store.release_run_claim(RUN, status=WAITING, claim_epoch=5) is True
    rows = await _rows(engine)
    assert (rows[RUN]["status"], rows[RUN]["claim_epoch"]) == (WAITING, 5)
    assert (rows[SIBLING]["status"], rows[SIBLING]["claim_epoch"]) == ("running", 4)
    assert (rows[OTHER]["status"], rows[OTHER]["claim_epoch"]) == (WAITING, 3)


# ---------------------------------------------------------------------------
# submit_and_claim
# ---------------------------------------------------------------------------


async def test_submit_and_claim_checkpointed_saves_submission_with_the_claim(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine)
    seen = []
    phases, sessions, proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    submission = {**SUB, "submission_id": "s2", "extra": 1, "payload": "dropped"}
    with _trace(engine, record_property) as (sql, _params, boundaries, connections):
        token = await store.submit_and_claim(
            RUN,
            expected_status=WAITING,
            submitted_data={"c1": "new"},
            submission=submission,
            validate=seen.append,
        )
    loop_submission = {**SUB, "submission_id": "s2", "extra": 1}
    assert token == ClaimResult(4, 2, {**LOOP, "submission": loop_submission}, loop_submission)
    assert seen == [PAUSE]
    after = await _rows(engine)
    row = after[RUN]
    assert (row["status"], row["claim_epoch"], row["pause_requested"]) == ("running", 4, False)
    assert row["last_progress_at"] is not None
    assert row["pause_data"] == {
        "pause_id": "p1",
        "submission": {**SUB, "submission_id": "s2", "payload": {"c1": "new"}},
    }
    assert row["loop_state"] == {**LOOP, "submission": loop_submission}
    _untouched(before, after, SIBLING, OTHER)
    assert phases == [("announce", (RUN,)), "commit"]
    assert all(session is sessions[0] for session in sessions)
    assert boundaries == ["begin", "commit"] and len(connections) == 1
    assert len(sql) == 4
    assert "claim_epoch=(agent_runs.claim_epoch + " in sql[0]
    assert "agent_runs.id = " in _where(sql[0]) and "agent_runs.status = " in _where(sql[0])
    assert sql[1].startswith(
        "SELECT agent_runs.pause_data, agent_runs.storage_format, agent_runs.loop_state"
    )
    assert sql[2].startswith("UPDATE agent_runs SET pause_data=")
    assert "loop_state=" in sql[2] and "status" not in sql[2]
    for statement in sql[1:]:
        assert _where(statement).startswith("agent_runs.id = ")
    assert sql[3].startswith("SELECT agent_runs.claim_epoch")


async def test_submit_and_claim_legacy_merges_payload_into_the_pause(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine, fmt=1, loop_state=None, pause_data={"pause_id": "p1", "a": 1})
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        token = await store.submit_and_claim(
            RUN, expected_status=WAITING, submitted_data={"a": 2, "b": 3}, submission=SUB
        )
    assert token == ClaimResult(4, 1, None, None)
    after = await _rows(engine)
    assert after[RUN]["pause_data"] == {"pause_id": "p1", "a": 2, "b": 3}
    assert after[RUN]["loop_state"] is None
    _untouched(before, after, SIBLING, OTHER)
    assert boundaries == ["begin", "commit"] and len(sql) == 4
    assert sql[2].startswith("UPDATE agent_runs SET pause_data=") and "loop_state" not in sql[2]


@pytest.mark.parametrize(
    "case",
    [
        "already-claimed",
        "wrong-status",
        "terminal",
        "missing-run",
        "no-pause",
        "validate-raises",
        "submission-missing",
        "foreign-blob-scope",
    ],
)
async def test_submit_and_claim_refusals_roll_back_the_claim(
    store, engine, monkeypatch, record_property, caplog, case
):
    await _seed(store, engine)
    if case == "already-claimed":
        await _set(engine, RUN, status="running", claim_epoch=4)
    elif case == "terminal":
        await _set(engine, RUN, status="success")
    elif case == "no-pause":
        await _set(engine, RUN, pause_data=None)
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    phases, _sessions, _proofs = _observe(store, monkeypatch)
    before = await _rows(engine)

    def refuse(pause):
        raise ValueError("refused")

    edge = BlobEdge(
        tenant_scope="t:t2", sha256="a" * 64, owner_kind="pause", owner_id="x", ordinal=0
    )
    kwargs = {
        "expected_status": "waiting_approval" if case == "wrong-status" else WAITING,
        "submitted_data": {"c1": "new"},
        "submission": None if case == "submission-missing" else SUB,
        "validate": refuse if case == "validate-raises" else None,
        "blob_refs": [edge] if case == "foreign-blob-scope" else None,
        "blob_refs_scope": "t:t2" if case == "foreign-blob-scope" else None,
    }
    target = MISSING if case == "missing-run" else RUN
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        if case == "validate-raises":
            with pytest.raises(ValueError, match="refused"):
                await store.submit_and_claim(target, **kwargs)
        elif case == "submission-missing":
            with pytest.raises(PersistenceError) as raised:
                await store.submit_and_claim(target, **kwargs)
            assert raised.value.code == "submission_missing"
        elif case == "foreign-blob-scope":
            with pytest.raises(BlobNotFoundError):
                await store.submit_and_claim(target, **kwargs)
        else:
            assert await store.submit_and_claim(target, **kwargs) is False
    assert await _rows(engine) == before
    assert phases == []
    assert boundaries == ["begin", "rollback"]
    lost = case in ("already-claimed", "wrong-status", "terminal", "missing-run")
    assert (
        len(sql)
        == {
            True: 1,
            False: {"no-pause": 2, "validate-raises": 2, "submission-missing": 2}.get(case, 4),
        }[lost]
    )
    assert _codes(caplog, "submission_missing") == ([RUN] if case == "submission-missing" else [])


@pytest.mark.parametrize("refs", ["edges", "empty"])
async def test_submit_and_claim_writes_blob_refs_inside_the_claim(
    store, engine, monkeypatch, record_property, refs
):
    await _seed(store, engine)
    calls, statements = [], []
    phases, sessions, proofs = _observe(store, monkeypatch)

    async def write_blob_refs(session, **kwargs):
        calls.append((session, kwargs, len(statements)))

    monkeypatch.setattr(blob_catalog, "write_blob_refs", write_blob_refs)
    edge = BlobEdge(
        tenant_scope="t:t1", sha256="b" * 64, owner_kind="pause", owner_id="x", ordinal=0
    )
    edges = [edge] if refs == "edges" else []
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        statements = sql
        token = await store.submit_and_claim(
            RUN,
            expected_status=WAITING,
            submitted_data={"c1": "new"},
            submission=SUB,
            blob_refs=edges,
            blob_refs_scope="t:t1",
        )
    assert token.claim_epoch == 4
    if refs == "edges":
        assert len(calls) == 1
        session, kwargs, position = calls[0]
        assert kwargs == {"run_id": RUN, "blob_refs": edges, "blob_refs_scope": "t:t1"}
        assert session is sessions[-1]
        assert position == 3  # after the claim, the pause read and the pause update
    else:
        assert calls == []
    assert boundaries == ["begin", "commit"] and len(sql) == 4


async def test_submit_and_claim_validates_persisted_payload_before_sql(
    store, engine, record_property
):
    await _seed(store, engine)
    before = await _rows(engine)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(InputValidationError),
    ):
        await store.submit_and_claim(
            RUN, expected_status=WAITING, submitted_data={"c1": "bad\x00"}, submission=SUB
        )
    assert sql == [] and boundaries == []
    assert await _rows(engine) == before


# ---------------------------------------------------------------------------
# consume_submission
# ---------------------------------------------------------------------------


def _turn(advancing=True):
    return TraceWrite(TRACE, "tool", "result", 7, advances_progress=advancing)


@pytest.mark.parametrize("progress", ["advancing", "replayed"])
@pytest.mark.parametrize("pii", [None, {"<P1>": "x"}])
async def test_consume_submission_once_under_the_claim_fence(
    store, engine, monkeypatch, record_property, progress, pii
):
    await _seed(store, engine, status="running", epoch=5)
    phases, sessions, proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    patch = LoopStatePatch(remove_pending=("c1",), set_phase="dispatch", set_fields={"k": 1})
    with _trace(engine, record_property) as (sql, _params, boundaries, connections):
        consumed = await store.consume_submission(
            RUN,
            claim_epoch=5,
            submission_id="s1",
            traces=[_turn(progress == "advancing")],
            patch=patch,
            pii_mapping=pii,
        )
    assert consumed is True
    after = await _rows(engine)
    row = after[RUN]
    assert row["loop_state"] == {
        **LOOP,
        "pending": [{"call_id": "c2", "phase": "waiting"}],
        "phase": "dispatch",
        "k": 1,
        "submission": None,
    }
    assert row["pause_data"] == {"pause_id": "p1"}
    assert row["reclaim_attempts"] == (0 if progress == "advancing" else 2)
    assert row["pii_mapping"] == pii
    assert (row["status"], row["claim_epoch"]) == ("running", 5)
    assert row["last_progress_at"] is not None
    _untouched(before, after, SIBLING, OTHER)
    traces = await _traces(engine)
    assert [(t["id"], t["agent_run_id"], t["order_index"]) for t in traces] == [(TRACE, RUN, 7)]
    fence_values = ["pii_mapping"] if pii is not None else []
    remembered = [(table, sorted(values)) for table, values in proofs]
    progress_keys = ["reclaim_attempts"] if progress == "advancing" else []
    assert remembered == [
        *([("agent_runs", ["pii_mapping"])] if pii is not None else []),
        ("agent_runs", sorted(["loop_state", *progress_keys])),
        ("react_traces", ["agent_run_id", "content_digest"]),
    ]
    assert proofs[-2][1]["loop_state"] == row["loop_state"]
    assert proofs[-1][1]["agent_run_id"] == RUN
    assert phases == [
        ("fence", RUN, 5, fence_values),
        ("trace", RUN, TRACE),
        ("announce", (RUN,)),
        "commit",
    ]
    assert all(session is sessions[0] for session in sessions)
    assert boundaries == ["begin", "commit"] and len(connections) == 1
    assert len(sql) == 4
    assert sql[0].startswith("UPDATE agent_runs SET updated_at=")
    assert "agent_runs.claim_epoch = " in _where(sql[0]) and "agent_runs.id = " in _where(sql[0])
    assert ("pii_mapping=" in sql[0]) is (pii is not None)
    assert sql[1].startswith("SELECT agent_runs.loop_state, agent_runs.pause_data FROM")
    assert _where(sql[1]).startswith("agent_runs.id = ")
    assert sql[2].startswith("INSERT INTO react_traces")
    assert sql[3].startswith("UPDATE agent_runs SET pause_data=")
    assert ("reclaim_attempts=" in sql[3]) is (progress == "advancing")
    assert _where(sql[3]).startswith("agent_runs.id = ") and "claim_epoch" not in _where(sql[3])

    # Idempotent resubmission after a lost acknowledgement: the submission is gone.
    phases.clear()
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        again = await store.consume_submission(
            RUN, claim_epoch=5, submission_id="s1", traces=[_turn()], patch=patch
        )
    assert again is False
    assert await _rows(engine) == after and len(await _traces(engine)) == 1
    assert phases == [("fence", RUN, 5, [])]
    assert boundaries == ["begin", "rollback"] and len(sql) == 2


@pytest.mark.parametrize(
    "case", ["stale-epoch", "wrong-submission", "no-submission", "missing-run", "oversize"]
)
async def test_consume_submission_refusals_change_nothing(
    store, engine, monkeypatch, record_property, caplog, case
):
    await _seed(store, engine, status="running", epoch=5)
    if case == "no-submission":
        await _set(engine, RUN, loop_state={**LOOP, "submission": None})
    caplog.set_level("WARNING", logger=STORE_LOGGER)
    before = await _rows(engine)
    patch = LoopStatePatch(set_fields={"blob": "x" * 70_000} if case == "oversize" else {"k": 1})
    kwargs = {
        "claim_epoch": 4 if case == "stale-epoch" else 5,
        "submission_id": "s2" if case == "wrong-submission" else "s1",
        "traces": [_turn()],
        "patch": patch,
    }
    target = MISSING if case == "missing-run" else RUN
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        if case in ("stale-epoch", "missing-run"):
            with pytest.raises(RunFencedError):
                await store.consume_submission(target, **kwargs)
        elif case == "oversize":
            with pytest.raises(PersistenceError):
                await store.consume_submission(target, **kwargs)
        else:
            assert await store.consume_submission(target, **kwargs) is False
    assert await _rows(engine) == before and await _traces(engine) == []
    assert boundaries == ["begin", "rollback"]
    assert len(sql) == (1 if case in ("stale-epoch", "missing-run") else 2)
    fenced = case in ("stale-epoch", "missing-run")
    assert _codes(caplog, "run_fenced") == ([target] if fenced else [])


async def test_consume_submission_on_a_terminal_run_with_the_same_epoch(store, engine):
    """Characterisation: the fence is the epoch, not the status (pins current behaviour)."""
    await _seed(store, engine, status="cancelled", epoch=5)
    before = await _rows(engine)
    assert await store.consume_submission(
        RUN, claim_epoch=5, submission_id="s1", traces=[], patch=LoopStatePatch()
    )
    after = await _rows(engine)
    assert after[RUN]["status"] == "cancelled" and after[RUN]["loop_state"]["submission"] is None
    _untouched(before, after, SIBLING, OTHER)


# ---------------------------------------------------------------------------
# release_run_claim and the cancel-requested release
# ---------------------------------------------------------------------------


def _release_event():
    return EventWrite(RELEASE_EVENT, "run.paused", 3, {"why": "invalid"}, "corr", None)


@pytest.mark.parametrize("with_event", [True, False], ids=["event", "no-event"])
async def test_release_returns_an_owned_claim_to_its_pause(
    store, engine, monkeypatch, record_property, with_event
):
    await _seed(store, engine, status="running", epoch=4)
    phases, sessions, proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, boundaries, connections):
        released = await store.release_run_claim(
            RUN, status=WAITING, claim_epoch=4, event=_release_event() if with_event else None
        )
    assert released is True
    after = await _rows(engine)
    row = after[RUN]
    assert (row["status"], row["claim_epoch"]) == (WAITING, 4)
    assert row["status_changed_at"] != before[RUN]["status_changed_at"]
    assert (row["pause_data"], row["loop_state"], row["pause_requested"]) == (PAUSE, LOOP, True)
    _untouched(before, after, SIBLING, OTHER)
    events = await _events(engine)
    if with_event:
        assert [
            (e["id"], e["agent_run_id"], e["event_type"], e["iteration_index"], e["data"])
            for e in events
        ] == [(RELEASE_EVENT, RUN, "run.paused", 3, {"why": "invalid"})]
    else:
        assert events == []
    # The event insert announces the run too.
    assert phases == [("announce", (RUN,))] * (1 + with_event) + ["commit"]
    assert all(session is sessions[0] for session in sessions)
    assert boundaries == ["begin", "commit"] and len(connections) == 1
    assert len(sql) == 1 + with_event
    where = _where(sql[0])
    for predicate in (
        "agent_runs.id = ",
        "agent_runs.status = ",
        "agent_runs.claim_epoch = ",
        "agent_runs.cancel_requested IS ",
    ):
        assert predicate in where, predicate
    if with_event:
        assert sql[1].startswith("INSERT INTO run_events")

    # Releasing again loses: one probe of the cancel request, then commit; no event.
    phases.clear()
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        assert (
            await store.release_run_claim(
                RUN, status=WAITING, claim_epoch=4, event=_release_event()
            )
            is False
        )
    assert await _rows(engine) == after and len(await _events(engine)) == with_event
    assert phases == [("announce", (RUN,)), "commit"]
    assert boundaries == ["begin", "commit"] and len(sql) == 2
    assert sql[1].startswith("SELECT agent_runs.cancel_requested FROM agent_runs WHERE")


@pytest.mark.parametrize("case", ["stale-epoch", "not-running", "cancel-stale-epoch", "missing"])
async def test_release_refusals_change_nothing(store, engine, monkeypatch, record_property, case):
    await _seed(
        store,
        engine,
        status=WAITING if case == "not-running" else "running",
        epoch=4,
        cancel=case == "cancel-stale-epoch",
    )
    phases, _sessions, _proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    target = MISSING if case == "missing" else RUN
    epoch = 3 if case in ("stale-epoch", "cancel-stale-epoch") else 4
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        assert (
            await store.release_run_claim(
                target, status=WAITING, claim_epoch=epoch, event=_release_event()
            )
            is False
        )
    assert await _rows(engine) == before and await _events(engine) == []
    assert phases == [("announce", (target,)), "commit"]
    assert boundaries == ["begin", "commit"] and len(sql) == 2


async def test_release_with_cancel_requested_finalizes_the_owned_claim(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine, status="running", epoch=4, cancel=True)
    phases, sessions, proofs = _observe(store, monkeypatch)
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, boundaries, connections):
        released = await store.release_run_claim(
            RUN, status=WAITING, claim_epoch=4, event=_release_event()
        )
    assert released is True
    after = await _rows(engine)
    row = after[RUN]
    assert (row["status"], row["claim_epoch"]) == ("cancelled", 4)
    assert (row["cancel_requested"], row["pause_requested"], row["pause_data"]) == (
        False,
        False,
        None,
    )
    assert row["loop_state"] == {**LOOP, "phase": "finalizing"}
    _untouched(before, after, SIBLING, OTHER)
    events = await _events(engine)
    assert [(e["agent_run_id"], e["event_type"], e["data"]) for e in events] == [
        (RUN, StoredEventType.RUN_CANCELLED.value, {"reason": CancelReason.CANCEL_REQUESTED.value})
    ]
    assert phases == [
        ("finalize", RUN, "cancelled", ("running",), 4),
        ("announce", (RUN,)),
        ("announce", (RUN,)),
        "commit",
    ]
    assert all(session is sessions[0] for session in sessions)
    assert boundaries == ["begin", "commit"] and len(connections) == 1
    assert len(sql) == 4
    probe = sql[1]
    assert probe.startswith("SELECT agent_runs.cancel_requested FROM agent_runs WHERE")
    for predicate in (
        "agent_runs.id = ",
        "agent_runs.status = ",
        "agent_runs.claim_epoch = ",
        "agent_runs.cancel_requested IS ",
    ):
        assert predicate in _where(probe), predicate
    assert probe.endswith(" FOR UPDATE") is _pg(engine)
    assert sql[2].startswith("UPDATE agent_runs SET ") and "agent_runs.claim_epoch = " in sql[2]
    assert sql[3].startswith("INSERT INTO run_events")


@pytest.mark.parametrize(
    ("won", "with_event", "writes"), [(True, True, 1), (True, False, 0), (False, True, 0)]
)
async def test_release_event_helper_writes_only_for_a_won_release(
    monkeypatch, won, with_event, writes
):
    calls = []

    async def write_event(session, run_id, event):
        calls.append((session, run_id, event))

    monkeypatch.setattr(state, "_write_event", write_event)
    marker, release = object(), _release_event() if with_event else None
    assert await sql_base._write_release_event(marker, RUN, won, release) is None
    assert calls == [(marker, RUN, release)] * writes


# ---------------------------------------------------------------------------
# Retries and commit failures
# ---------------------------------------------------------------------------


def _calls(store):
    return {
        "claim_paused_run": lambda: store.claim_paused_run(RUN, expected_status=WAITING),
        "submit_and_claim": lambda: store.submit_and_claim(
            RUN, expected_status=WAITING, submitted_data={"c1": "new"}, submission=SUB
        ),
        "release_run_claim": lambda: store.release_run_claim(
            RUN, status=WAITING, claim_epoch=3, event=_release_event()
        ),
        "consume_submission": lambda: store.consume_submission(
            RUN, claim_epoch=3, submission_id="s1", traces=[_turn()], patch=LoopStatePatch()
        ),
    }


@pytest.mark.parametrize(
    ("operation", "label"),
    [
        ("claim_paused_run", "claim_paused_run"),
        ("submit_and_claim", "submit_and_claim"),
        ("release_run_claim", "release_run_claim"),
        ("consume_submission", None),
    ],
)
async def test_transient_session_failure_retries_only_wrapped_operations(
    store, engine, monkeypatch, record_property, operation, label
):
    status = "running" if operation in ("release_run_claim", "consume_submission") else WAITING
    await _seed(store, engine, status=status)
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
    before = await _rows(engine)
    with _trace(engine, record_property) as (_sql, _params, boundaries, _connections):
        if label is None:
            with pytest.raises(PersistenceError) as raised:
                await _calls(store)[operation]()
            assert raised.value.retryable
        else:
            assert await _calls(store)[operation]()
    if label is None:
        assert retries == [] and opened == [0] and boundaries == []
        assert await _rows(engine) == before
    else:
        assert retries == [(0, label, RUN, "OperationalError")] and opened == [0, 1]
        assert boundaries == ["begin", "commit"]


@pytest.mark.parametrize(
    "operation",
    ["claim_paused_run", "submit_and_claim", "release_run_claim", "consume_submission", "cancel"],
)
async def test_commit_failure_rolls_back_every_claim_write(
    store, engine, monkeypatch, record_property, operation
):
    running = operation in ("release_run_claim", "consume_submission", "cancel")
    await _seed(
        store, engine, status="running" if running else WAITING, cancel=operation == "cancel"
    )
    before = (await _rows(engine), await _events(engine), await _traces(engine))

    async def refuse(session):
        raise PersistenceError(code="commit_refused")

    monkeypatch.setattr(store, "_write_tx", refuse)
    call = _calls(store)["release_run_claim" if operation == "cancel" else operation]
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, connections),
        pytest.raises(PersistenceError) as raised,
    ):
        await call()
    assert raised.value.code == "commit_refused"
    assert boundaries == ["begin", "rollback"] and len(connections) == 1
    assert (
        len(sql)
        == {
            "claim_paused_run": 2,
            "submit_and_claim": 4,
            "release_run_claim": 2,
            "consume_submission": 4,
            "cancel": 4,
        }[operation]
    )
    assert (await _rows(engine), await _events(engine), await _traces(engine)) == before


# ---------------------------------------------------------------------------
# Real concurrency (Postgres): exactly one winner, no lost update
# ---------------------------------------------------------------------------

PARALLEL = 8


def _require_pg(engine):
    if not _pg(engine):
        pytest.skip("interleaved sessions need a real server (in-memory SQLite shares one)")


@pytest.mark.parametrize("operation", ["claim", "claim-locked", "submit", "mixed"])
async def test_parallel_claimers_have_exactly_one_winner_per_run(store, engine, operation):
    _require_pg(engine)
    await _seed(store, engine)
    if operation == "claim-locked":
        store.supports_update_returning = False

    def attempt(run_id, index):
        submit = operation == "submit" or (operation == "mixed" and index % 2)
        if submit:
            return store.submit_and_claim(
                run_id,
                expected_status=WAITING,
                submitted_data={"c1": f"{run_id}-{index}"},
                submission={**SUB, "submission_id": f"s-{index}"},
            )
        return store.claim_paused_run(run_id, expected_status=WAITING, expected_pause_id="p1")

    calls = [attempt(run_id, index) for index in range(PARALLEL) for run_id in (RUN, SIBLING)]
    results = await asyncio.gather(*calls, return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    rows = await _rows(engine)
    for offset, run_id in enumerate((RUN, SIBLING)):
        mine = results[offset::2]
        winners = [(index, r) for index, r in enumerate(mine) if r is not False]
        assert len(winners) == 1, (run_id, mine)
        index, token = winners[0]
        assert token.claim_epoch == 4
        assert (rows[run_id]["status"], rows[run_id]["claim_epoch"]) == ("running", 4)
        submitted = rows[run_id]["pause_data"]["submission"]
        if operation == "submit" or (operation == "mixed" and index % 2):
            assert submitted["payload"] == {"c1": f"{run_id}-{index}"}
            assert submitted["submission_id"] == f"s-{index}"
            assert rows[run_id]["loop_state"]["submission"]["submission_id"] == f"s-{index}"
        else:
            assert rows[run_id]["pause_data"] == PAUSE
    assert (rows[OTHER]["status"], rows[OTHER]["claim_epoch"]) == (WAITING, 3)


@pytest.mark.parametrize("cancel", [False, True], ids=["release", "cancel-release"])
async def test_parallel_releasers_have_exactly_one_winner(store, engine, cancel):
    _require_pg(engine)
    await _seed(store, engine, status="running", epoch=4, cancel=cancel)
    results = await asyncio.gather(
        *[
            store.release_run_claim(RUN, status=WAITING, claim_epoch=4, event=_release_event())
            for _ in range(PARALLEL)
        ],
        return_exceptions=True,
    )
    assert sorted(map(repr, results)) == ["False"] * (PARALLEL - 1) + ["True"]
    rows = await _rows(engine)
    assert rows[RUN]["status"] == ("cancelled" if cancel else WAITING)
    assert rows[RUN]["claim_epoch"] == 4
    events = await _events(engine)
    expected = StoredEventType.RUN_CANCELLED.value if cancel else "run.paused"
    assert [(e["agent_run_id"], e["event_type"]) for e in events] == [(RUN, expected)]
    for run_id in (SIBLING, OTHER):
        assert (rows[run_id]["status"], rows[run_id]["cancel_requested"]) == ("running", cancel)


async def test_parallel_consumers_consume_a_submission_once(store, engine):
    _require_pg(engine)
    await _seed(store, engine, status="running", epoch=5)
    patch = LoopStatePatch(remove_pending=("c1",), set_phase="model_next")
    results = await asyncio.gather(
        *[
            store.consume_submission(
                RUN, claim_epoch=5, submission_id="s1", traces=[_turn()], patch=patch
            )
            for _ in range(PARALLEL)
        ],
        return_exceptions=True,
    )
    assert sorted(map(repr, results)) == ["False"] * (PARALLEL - 1) + ["True"]
    rows = await _rows(engine)
    assert rows[RUN]["loop_state"]["submission"] is None
    assert rows[RUN]["loop_state"]["pending"] == [{"call_id": "c2", "phase": "waiting"}]
    assert [t["agent_run_id"] for t in await _traces(engine)] == [RUN]
    assert rows[SIBLING]["loop_state"] == LOOP and rows[OTHER]["loop_state"] == LOOP
