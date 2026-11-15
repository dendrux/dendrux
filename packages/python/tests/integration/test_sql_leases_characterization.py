"""Lease expiry, rediscovery and reclaim keep their epoch fences, run scope and bounds."""

import asyncio
import json
import logging
import re
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

import dendrux.runtime.durability as durability
from dendrux._lifecycle import StoredEventType
from dendrux._meta_keys import GOVERNANCE_KEY, STORE_SESSION_KEY
from dendrux.db.models import AgentRun, RunEvent
from dendrux.errors import (
    ManagedChildControlError,
    PersistenceError,
    RunNotFoundError,
    RunNotReclaimableError,
)
from dendrux.runtime.state import (
    MANAGED_HEADER_KEY,
    UNSCOPED,
    ClaimResult,
    ReclaimableRun,
)

RUN = "00000000000000000000000001"
SIBLING = "00000000000000000000000002"
OTHER = "00000000000000000000000003"
MISSING = "00000000000000000000000009"
RUNS = (RUN, SIBLING, OTHER)
TENANTS = {RUN: "t1", SIBLING: "t1", OTHER: "t2"}
STORE_LOGGER = "dendrux.runtime.stores.sql.base"
SUB = {"submission_id": "s1", "kind": "tool_results", "pause_id": "p1", "call_ids": ["c1"]}
LOOP = {"v": 1, "loop": "react", "phase": "model_next", "skills": {"pins": {"s": "1"}}}
GOVERNANCE = {"digest": "g1"}
COLUMNS = (
    "status",
    "claim_epoch",
    "storage_format",
    "lease_expired_at",
    "reclaim_attempts",
    "reclaim_limit",
    "last_reclaimed_at",
    "last_progress_at",
    "status_changed_at",
    "failure_reason",
    "error",
    "pause_data",
    "pause_requested",
    "cancel_requested",
    "loop_state",
    "iteration_count",
    "tenant_id",
    "updated_at",
)
CODES = {
    "legacy_run_swept",
    "stale_running",
    "never_started",
    "reclaim_limit",
    "cancel_requested",
    "loop_unknown",
    "recovery_scan_limit",
    "sweep_cap_reached",
    "governance_mismatch",
}


def _ids(*numbers):
    return [f"{n:026d}" for n in numbers]


def _now():
    return datetime.now(UTC)


def _ago(**delta):
    return _now() - timedelta(**delta)


def _naive(value):
    return value.replace(tzinfo=None) if value is not None and value.tzinfo else value


def _stable(value):
    if isinstance(value, datetime):
        return "<timestamp>"
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, str) and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", value):
        return value if value.startswith("0000000000000000000000") else "<generated-id>"
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


def _page_limit(parameters):
    """The scan page LIMIT: SQLite also binds both OFFSETs, Postgres binds neither."""
    (values,) = parameters
    bound = [
        values[k]
        for k in sorted((k for k in values if k.startswith("param_")), key=lambda k: int(k[6:]))
    ]
    return bound[2] if len(bound) == 4 else bound[1]


def _pg(engine):
    return engine.dialect.name == "postgresql"


def _shape(sql):
    """Verb and table of each statement, in order."""
    shapes = []
    for text in sql:
        verb = text.split(" ", 1)[0]
        while re.search(r"\([^()]*\)", text):
            text = re.sub(r"\([^()]*\)", "", text)
        match = re.search(r"\b(?:FROM|INTO|UPDATE)\s+(\w+)", text)
        shapes.append(f"{verb} {match.group(1) if match else '?'}")
    return shapes


async def _seed(store, engine, rows):
    """Create each run, then give it the stale RUNNING checkpointed defaults plus overrides."""
    stale, old = _ago(hours=2), _ago(hours=3)
    for run_id, values in rows.items():
        await store.create_run(
            run_id, values.get("agent_name", "leases"), tenant_id=values.get("tenant", "t1")
        )
    async with engine.begin() as conn:
        for run_id, values in rows.items():
            columns = {
                "status": "running",
                "claim_epoch": 3,
                "storage_format": 2,
                "loop_state": LOOP,
                "meta": {GOVERNANCE_KEY: GOVERNANCE},
                "pause_data": {"marker": run_id},
                "pause_requested": True,
                "cancel_requested": False,
                "reclaim_attempts": 0,
                "reclaim_limit": None,
                "lease_expired_at": None,
                "last_reclaimed_at": None,
                "last_progress_at": stale,
                "created_at": old,
                "iteration_count": 2,
                "status_changed_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
            columns.update({k: v for k, v in values.items() if k not in ("tenant", "started")})
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(**columns))
    for run_id, values in rows.items():
        for event_type in values.get("started", ()):
            await store.save_run_event(run_id, event_type=event_type)


async def _rows(engine):
    async with AsyncSession(engine) as session:
        rows = await session.execute(
            select(AgentRun.id, *(getattr(AgentRun, name) for name in COLUMNS)).order_by(
                AgentRun.id
            )
        )
        return {row.id: {name: getattr(row, name) for name in COLUMNS} for row in rows}


async def _events(engine, *, after=None):
    async with AsyncSession(engine) as session:
        stmt = select(RunEvent.__table__).order_by(RunEvent.agent_run_id, RunEvent.sequence_index)
        rows = [dict(row._mapping) for row in await session.execute(stmt)]
    if after is not None:
        seen = {(e["agent_run_id"], e["sequence_index"]) for e in after}
        rows = [e for e in rows if (e["agent_run_id"], e["sequence_index"]) not in seen]
    return rows


def _event_view(events):
    return [
        (
            e["agent_run_id"],
            e["event_type"],
            {k: ("<gap>" if k == "gap_s" else v) for k, v in (e["data"] or {}).items()},
            e["iteration_index"],
        )
        for e in events
    ]


def _set(engine, run_id, **values):
    async def apply():
        async with engine.begin() as conn:
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))

    return apply()


def _observe(store, monkeypatch):
    """Record the run-row writes, announce and commit order, and their sessions."""
    phases, sessions = [], []
    one_rows, one, announce, commit = (
        store._update_one_rows,
        store.update_one,
        store._mark_announce,
        store._write_tx,
    )

    async def wrote(session, **kwargs):
        phases.append(("update_one_rows", kwargs["pk"]))
        sessions.append(session)
        return await one_rows(session, **kwargs)

    async def locked(session, **kwargs):
        phases.append(("update_one", kwargs["pk"], STORE_SESSION_KEY in session.info))
        sessions.append(session)
        return await one(session, **kwargs)

    def announced(session, run_ids):
        phases.append(("announce", tuple(run_ids)))
        sessions.append(session)
        return announce(session, run_ids)

    async def committed(session):
        phases.append("commit")
        sessions.append(session)
        return await commit(session)

    monkeypatch.setattr(store, "_update_one_rows", wrote)
    monkeypatch.setattr(store, "update_one", locked)
    monkeypatch.setattr(store, "_mark_announce", announced)
    monkeypatch.setattr(store, "_write_tx", committed)
    return phases, sessions


def _untouched(before, after, *run_ids):
    for run_id in run_ids:
        assert after[run_id] == before[run_id], run_id


def _logged(caplog):
    return [
        (r.code, getattr(r, "run_id", None), r.name)
        for r in caplog.records
        if getattr(r, "code", None) in CODES
    ]


def _mark(store, *, cutoff=None, **kwargs):
    arguments = {
        "cutoff": cutoff or _ago(hours=1),
        "batch_size": 100,
        "max_runs": 100,
        "max_returned": 100,
        "reclaim_limit": 3,
        "cursor": None,
    }
    arguments.update(kwargs)
    return store.mark_lease_expired(**arguments)


def _scan(store, *, on_stale, cutoff=None, **kwargs):
    arguments = {
        "cutoff": cutoff or _ago(hours=1),
        "batch_size": 100,
        "max_runs": 100,
        "max_returned": 100,
        "reclaim_limit": 3,
        "cursor": None,
        "on_stale": on_stale,
    }
    arguments.update(kwargs)
    return store._scan_expired_runs(**arguments)


def _reclaim(store, run_id=RUN, **kwargs):
    arguments = {
        "tenant_id": "t1",
        "agent_name": "leases",
        "expected_claim_epoch": 4,
        "governance": GOVERNANCE,
        "governance_policy": "raise",
        "skill_pins": {"s": "1"},
        "loop_type": "react",
    }
    arguments.update(kwargs)
    return store.reclaim_run(run_id, **arguments)


def _marked(**values):
    return {"lease_expired_at": _ago(minutes=30), "claim_epoch": 4, "reclaim_limit": 3, **values}


def _reclaimable(run_id, row):
    return ReclaimableRun(
        run_id,
        row["tenant_id"],
        "leases",
        _naive(row["last_progress_at"]),
        _naive(row["lease_expired_at"]),
        row["reclaim_attempts"],
        row["reclaim_limit"],
        row["claim_epoch"],
    )


def _plain(item):
    """Compare ReclaimableRun values with naive timestamps on every backend."""
    return replace(
        item,
        last_progress_at=_naive(item.last_progress_at),
        lease_expired_at=_naive(item.lease_expired_at),
    )


# ---------------------------------------------------------------------------
# mark_lease_expired: the epoch-fenced mark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("finalise", [False, True])
async def test_mark_fences_each_expired_lease_once_and_skips_the_rest(
    store, engine, monkeypatch, record_property, finalise
):
    r1, fresh, marked, waiting, created_only, renewed = _ids(1, 2, 3, 4, 5, 6)
    await _seed(
        store,
        engine,
        {
            r1: {},
            fresh: {"last_progress_at": _now()},
            marked: {"tenant": "t2", **_marked(claim_epoch=5, reclaim_limit=7, reclaim_attempts=1)},
            waiting: {"status": "waiting_client_tool"},
            created_only: {"last_progress_at": None},
            renewed: {"last_progress_at": _now(), "created_at": _ago(hours=5)},
        },
    )
    phases, sessions = _observe(store, monkeypatch)
    before, events_before = await _rows(engine), await _events(engine)
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        result = await _mark(store, finalise_unrecoverable=finalise)
    after = await _rows(engine)
    assert [_plain(item) for item in result.reclaimable] == [
        _reclaimable(r1, after[r1]),
        _reclaimable(marked, before[marked]),
        _reclaimable(created_only, after[created_only]),
    ]
    assert (result.finalised, result.legacy, result.inspected, result.cursor) == ([], [], 5, None)
    for run_id in (r1, created_only):
        row = after[run_id]
        assert (row["status"], row["claim_epoch"], row["reclaim_limit"]) == ("running", 4, 3)
        assert row["lease_expired_at"] is not None
        keep = set(COLUMNS) - {"claim_epoch", "lease_expired_at", "reclaim_limit", "updated_at"}
        assert {k: row[k] for k in keep} == {k: before[run_id][k] for k in keep}
    _untouched(before, after, fresh, marked, waiting, renewed)
    events = await _events(engine, after=events_before)
    assert _event_view(events) == [
        (r1, StoredEventType.RUN_LEASE_EXPIRED.value, {"gap_s": "<gap>"}, 0),
        (created_only, StoredEventType.RUN_LEASE_EXPIRED.value, {"gap_s": "<gap>"}, 0),
    ]
    assert 7100 < events[0]["data"]["gap_s"] < 7300
    assert 10700 < events[1]["data"]["gap_s"] < 10900
    assert phases == [
        step
        for run_id in (r1, created_only)
        for step in (
            ("update_one_rows", (run_id,)),
            ("update_one", (run_id,), True),
            ("announce", (run_id,)),
            ("announce", (run_id,)),
            "commit",
        )
    ]
    assert len({id(s) for s in sessions}) == 2
    assert _shape(sql) == [
        "SELECT agent_runs",
        *(["UPDATE agent_runs", "INSERT run_events"] * 2),
        "SELECT agent_runs",
    ]
    assert boundaries == ["begin", "rollback"] + ["begin", "commit"] * 2 + ["begin", "rollback"]
    cas = sql[1]
    assert "agent_runs.status = " in cas and "agent_runs.claim_epoch = " in cas
    assert "agent_runs.storage_format = " in cas and "agent_runs.lease_expired_at IS NULL" in cas
    assert "coalesce(agent_runs.last_progress_at, agent_runs.created_at) < " in cas
    assert "claim_epoch=(agent_runs.claim_epoch + " in cas.replace(" = ", "=")


A, B, C, D, E, F, G, H, W = _ids(1, 2, 3, 4, 5, 6, 7, 8, 9)
CATEGORY_SEED = {
    A: {"storage_format": 1, "loop_state": None, "started": ("run.started",)},
    B: {"loop_state": {**LOOP, "loop": "custom"}},
    C: {"tenant": "t2", "reclaim_attempts": 3},
    D: {"loop_state": {**LOOP, "loop": "single_call"}, "started": ("run.started",)},
    E: {"tenant": "t2", "loop_state": {**LOOP, "loop": "single_call"}, "started": ("run.paused",)},
    F: {"cancel_requested": True},
    G: {"started": ("run.started",)},
    H: {"tenant": "t2", **_marked(claim_epoch=6, reclaim_limit=5), "last_progress_at": None},
    W: {"status": "waiting_client_tool"},
}
# run -> (kind, failure_reason or None, status written) for each on_stale mode.
CATEGORY_OUTCOMES = {
    None: {
        A: ("finalised", "stale_running_legacy", "error"),
        B: ("skipped", None, None),
        C: ("finalised", "reclaim_limit", "error"),
        D: ("finalised", "stale_running", "error"),
        E: ("finalised", "stale_running", "error"),
        F: ("finalised", None, "cancelled"),
        G: ("marked", None, None),
        H: ("rediscovered", None, None),
    },
    "reclaim": {
        A: ("finalised", "stale_running_legacy", "error"),
        B: ("finalised", "never_started", "error"),
        C: ("finalised", "reclaim_limit", "error"),
        D: ("finalised", "stale_running", "error"),
        E: ("finalised", "never_started", "error"),
        F: ("finalised", None, "cancelled"),
        G: ("marked", None, None),
        H: ("rediscovered", None, None),
    },
    "error": {
        A: ("finalised", "stale_running", "error"),
        B: ("finalised", "never_started", "error"),
        C: ("finalised", "never_started", "error"),
        D: ("finalised", "stale_running", "error"),
        E: ("finalised", "never_started", "error"),
        F: ("finalised", "never_started", "cancelled"),
        G: ("finalised", "stale_running", "error"),
        H: ("finalised", "never_started", "error"),
    },
}


@pytest.mark.parametrize("on_stale", [None, "reclaim", "error"])
async def test_scan_categories_follow_storage_format_loop_limit_and_cancel(
    store, engine, monkeypatch, record_property, caplog, on_stale
):
    caplog.set_level(logging.DEBUG)
    await _seed(store, engine, CATEGORY_SEED)
    before, events_before = await _rows(engine), await _events(engine)
    phases, _sessions = _observe(store, monkeypatch)
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        result = await _scan(store, on_stale=on_stale)
    after = await _rows(engine)
    outcomes = CATEGORY_OUTCOMES[on_stale]
    finalised = [run for run, (kind, *_rest) in outcomes.items() if kind == "finalised"]
    reclaimable = [run for run, (kind, *_rest) in outcomes.items() if kind != "finalised"]
    reclaimable = [run for run in reclaimable if outcomes[run][0] != "skipped"]
    assert [item.run_id for item in result.reclaimable] == reclaimable
    assert [
        (s.run_id, s.agent_name, s.previous_status, s.failure_reason, s.status)
        for s in result.finalised
    ] == [
        (
            run,
            "leases",
            "running",
            outcomes[run][1] or "cancel_requested",
            outcomes[run][2],
        )
        for run in finalised
    ]
    assert [s.run_id for s in result.legacy] == [
        run for run in finalised if outcomes[run][1] == "stale_running_legacy"
    ]
    assert (result.inspected, result.cursor) == (8, None)
    for swept in result.finalised:
        assert swept.swept_at.tzinfo is not None
        assert _naive(swept.last_progress_at) == _naive(before[swept.run_id]["last_progress_at"])
    for run, (kind, reason, status) in outcomes.items():
        row, old = after[run], before[run]
        if kind == "finalised":
            assert row["status"] == status
            assert row["claim_epoch"] == old["claim_epoch"] + 1
            assert row["failure_reason"] == reason
            assert row["error"] == (None if status == "cancelled" else f"Run swept as {reason}")
            assert (row["pause_data"], row["pause_requested"], row["cancel_requested"]) == (
                None,
                False,
                False,
            )
            assert row["status_changed_at"] != old["status_changed_at"]
            assert (row["lease_expired_at"], row["reclaim_limit"]) == (
                old["lease_expired_at"],
                old["reclaim_limit"],
            )
        elif kind == "marked":
            assert (row["claim_epoch"], row["reclaim_limit"]) == (old["claim_epoch"] + 1, 3)
            assert row["lease_expired_at"] is not None and row["status"] == "running"
        else:
            _untouched(before, after, run)
    _untouched(before, after, W)
    expected_events = []
    for run, (kind, reason, status) in outcomes.items():
        if kind == "finalised":
            event_type = "run.cancelled" if status == "cancelled" else "run.interrupted"
            data = {"failure_reason": reason}
            if status == "cancelled":
                data["reason"] = "cancel_requested"
            expected_events.append((run, event_type, data, 2))
        elif kind == "marked":
            expected_events.append((run, "run.lease_expired", {"gap_s": "<gap>"}, 0))
    assert _event_view(await _events(engine, after=events_before)) == expected_events
    expected_logs = []
    for run, (kind, reason, _status) in outcomes.items():
        if run == B and on_stale != "error":
            expected_logs.append(("loop_unknown", B, STORE_LOGGER))
        if kind == "finalised":
            code = (
                "legacy_run_swept"
                if reason == "stale_running_legacy"
                else reason or "cancel_requested"
            )
            expected_logs.append((code, run, STORE_LOGGER))
    assert _logged(caplog) == expected_logs
    writes = [run for run, (kind, *_rest) in outcomes.items() if kind in ("finalised", "marked")]
    assert phases == [
        step
        for run in writes
        for step in (
            ("update_one_rows", (run,)),
            ("update_one", (run,), True),
            ("announce", (run,)),
            ("announce", (run,)),
            "commit",
        )
    ]
    assert boundaries == ["begin", "rollback"] + ["begin", "commit"] * len(writes) + [
        "begin",
        "rollback",
    ]
    assert _shape(sql) == [
        "SELECT agent_runs",
        *(["UPDATE agent_runs", "INSERT run_events"] * len(writes)),
        "SELECT agent_runs",
    ]
    started = sql[0]
    assert "run_events.agent_run_id = agent_runs.id" in started
    assert "run_events.event_type = 'run.started'" in started


async def test_scan_error_mode_skips_a_fresh_mark_but_reclaim_finalises_it_at_its_limit(
    store, engine, record_property
):
    marked_fresh = _ids(1)[0]
    seed = {marked_fresh: {**_marked(reclaim_attempts=3), "last_progress_at": _now()}}
    await _seed(store, engine, seed)
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, _boundaries, _connections):
        error = await _scan(store, on_stale="error")
    assert (error.reclaimable, error.finalised, error.inspected, len(sql)) == ([], [], 1, 2)
    assert await _rows(engine) == before
    result = await _scan(store, on_stale="reclaim")
    assert [(s.run_id, s.failure_reason) for s in result.finalised] == [
        (marked_fresh, "reclaim_limit")
    ]
    assert (await _rows(engine))[marked_fresh]["claim_epoch"] == 5


async def test_mark_keeps_a_stored_reclaim_limit_and_uses_the_argument_only_for_new_marks(
    store, engine
):
    stored, new = _ids(1, 2)
    await _seed(store, engine, {stored: _marked(reclaim_limit=5, reclaim_attempts=3), new: {}})
    result = await _mark(store, reclaim_limit=3, finalise_unrecoverable=True)
    assert [(i.run_id, i.reclaim_limit, i.reclaim_attempts) for i in result.reclaimable] == [
        (stored, 5, 3),
        (new, 3, 0),
    ]
    assert result.finalised == []
    blocked = await _mark(store, reclaim_limit=9, cursor=None, finalise_unrecoverable=True)
    assert [(i.run_id, i.reclaim_limit) for i in blocked.reclaimable] == [(stored, 5), (new, 3)]
    await _set(engine, new, reclaim_attempts=3)
    final = await _mark(store, reclaim_limit=9, finalise_unrecoverable=True)
    assert [(s.run_id, s.failure_reason) for s in final.finalised] == [(new, "reclaim_limit")]


async def test_mark_pages_by_id_stops_at_its_caps_and_continues_from_the_cursor(
    store, engine, record_property, caplog
):
    caplog.set_level(logging.DEBUG)
    ids = _ids(1, 2, 3, 4, 5, 6)
    await _seed(
        store,
        engine,
        {
            run_id: ({"tenant": "t2"} if index % 2 else {})
            | ({"last_progress_at": _now()} if run_id == ids[1] else {})
            for index, run_id in enumerate(ids)
        },
    )
    with _trace(engine, record_property) as (sql, params, _boundaries, _connections):
        first = await _mark(store, batch_size=2, max_runs=3)
    assert [i.run_id for i in first.reclaimable] == [ids[0], ids[2]]
    assert (first.inspected, first.cursor) == (3, ids[2])
    pages = [p for s, p in zip(sql, params, strict=True) if s.startswith("SELECT agent_runs.id")]
    assert [_page_limit(page) for page in pages] == [2, 1]
    assert (
        "agent_runs.id > " not in sql[0]
        and "agent_runs.id > " in [s for s in sql if s.startswith("SELECT agent_runs.id")][1]
    )
    assert "ORDER BY agent_runs.status, agent_runs.id" in sql[0]
    assert _logged(caplog) == [("sweep_cap_reached", None, STORE_LOGGER)]
    (cap,) = [r for r in caplog.records if getattr(r, "code", None) == "sweep_cap_reached"]
    assert (cap.category, cap.cap, cap.inspected) == ("stale_running", 3, 3)
    caplog.clear()
    second = await _mark(store, batch_size=2, max_runs=10, max_returned=2, cursor=first.cursor)
    assert [i.run_id for i in second.reclaimable] == [ids[3], ids[4]]
    assert (second.inspected, second.cursor) == (2, ids[4])
    assert _logged(caplog) == []
    mid_page = await _scan(store, on_stale=None, batch_size=10, max_returned=1, cursor=ids[3])
    assert [(i.run_id, i.claim_epoch) for i in mid_page.reclaimable] == [(ids[4], 4)]
    assert (mid_page.inspected, mid_page.cursor) == (1, ids[4])
    assert (await _rows(engine))[ids[5]]["claim_epoch"] == 3
    last = await _mark(store, batch_size=2, max_runs=10, cursor=second.cursor)
    assert [i.run_id for i in last.reclaimable] == [ids[5]]
    assert (last.inspected, last.cursor) == (1, None)
    rows = await _rows(engine)
    assert [rows[run_id]["claim_epoch"] for run_id in ids] == [4, 3, 4, 4, 4, 4]
    again = await _mark(store, batch_size=2, max_runs=10, cursor=ids[2])
    assert [(i.run_id, i.claim_epoch) for i in again.reclaimable] == [
        (ids[3], 4),
        (ids[4], 4),
        (ids[5], 4),
    ]
    assert (await _rows(engine)) == rows


@pytest.mark.parametrize("finalise", [False, True])
async def test_mark_finalise_flag_selects_the_reclaim_category_mode(
    store, engine, caplog, finalise
):
    caplog.set_level(logging.DEBUG)
    unknown, single = _ids(1, 2)
    await _seed(
        store,
        engine,
        {
            unknown: {"loop_state": {**LOOP, "loop": "custom"}},
            single: {"tenant": "t2", "loop_state": {**LOOP, "loop": "single_call"}},
        },
    )
    before = await _rows(engine)
    result = await _mark(store, finalise_unrecoverable=finalise)
    expected = [(unknown, "never_started"), (single, "never_started")]
    if not finalise:
        expected = [(single, "stale_running")]
    assert [(s.run_id, s.failure_reason) for s in result.finalised] == expected
    assert result.reclaimable == [] and result.inspected == 2
    if not finalise:
        _untouched(before, await _rows(engine), unknown)
    assert [c for c, run_id, _name in _logged(caplog) if run_id == unknown][0] == "loop_unknown"


async def test_mark_treats_a_run_renewed_exactly_at_the_cutoff_as_fresh(
    store, engine, record_property
):
    at_cutoff, just_before = _ids(1, 2)
    cutoff = _ago(hours=1).replace(microsecond=0)
    await _seed(
        store,
        engine,
        {
            at_cutoff: {"last_progress_at": cutoff},
            just_before: {"last_progress_at": cutoff - timedelta(seconds=1)},
        },
    )
    before = await _rows(engine)
    with _trace(engine, record_property) as (sql, _params, _boundaries, _connections):
        result = await _mark(store, cutoff=cutoff)
    assert [i.run_id for i in result.reclaimable] == [just_before]
    assert result.inspected == 2
    assert _shape(sql) == [
        "SELECT agent_runs",
        "UPDATE agent_runs",
        "INSERT run_events",
        "SELECT agent_runs",
    ]
    _untouched(before, await _rows(engine), at_cutoff)


@pytest.mark.parametrize(
    ("arguments", "match", "code"),
    [
        ({"cutoff": datetime(2026, 1, 1)}, "timezone", None),
        ({"batch_size": 0}, "positive", None),
        ({"batch_size": True}, "positive", None),
        ({"batch_size": 1001}, "batch_size", "recovery_scan_limit"),
        ({"max_runs": 0}, "positive", None),
        ({"max_returned": 0}, "positive", None),
        ({"reclaim_limit": 0}, "positive", None),
        ({"max_runs": 100_001}, "100000", "recovery_scan_limit"),
        ({"max_returned": 100_001}, "100000", "recovery_scan_limit"),
        ({"reclaim_limit": 32_768}, "SMALLINT", None),
    ],
)
async def test_mark_refuses_out_of_bound_arguments_before_sql(
    store, engine, record_property, caplog, arguments, match, code
):
    caplog.set_level(logging.DEBUG)
    await _seed(store, engine, {RUN: {}})
    before = await _rows(engine)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(ValueError, match=match),
    ):
        await _mark(store, **arguments)
    assert (sql, boundaries) == ([], [])
    assert await _rows(engine) == before
    assert _logged(caplog) == ([(code, None, STORE_LOGGER)] if code else [])


async def test_scan_accepts_its_largest_bounds(store, engine, record_property):
    await _seed(store, engine, {RUN: {}})
    with _trace(engine, record_property) as (sql, params, _boundaries, _connections):
        result = await _scan(
            store,
            on_stale=None,
            batch_size=100_000,
            max_runs=100_000,
            max_returned=100_000,
            reclaim_limit=32_767,
        )
    assert [i.reclaim_limit for i in result.reclaimable] == [32_767]
    assert _page_limit(params[0]) == 100_000


# ---------------------------------------------------------------------------
# A concurrent change between the scan read and the run's CAS
# ---------------------------------------------------------------------------

RACES = {
    # case: (on_stale, seed overrides, concurrent change, outcome)
    "mark-heartbeat": (None, {}, {"last_progress_at": "now"}, "none"),
    "mark-claimed": (None, {}, {"claim_epoch": 4}, "none"),
    "mark-elsewhere": (
        None,
        {},
        {"lease_expired_at": "now", "claim_epoch": 4, "reclaim_limit": 9},
        "current",
    ),
    "mark-same-epoch": (None, {}, {"lease_expired_at": "now"}, "current"),
    "mark-finalised": (None, {}, {"status": "error"}, "none"),
    "mark-finalised-marked": (None, {}, {"status": "error", "lease_expired_at": "now"}, "none"),
    "mark-legacy": (None, {}, {"storage_format": 1}, "none"),
    "error-heartbeat": ("error", {}, {"last_progress_at": "now"}, "none"),
    "error-heartbeat-marked": (
        "error",
        {"lease_expired_at": "old", "reclaim_limit": 3},
        {"last_progress_at": "now"},
        "none",
    ),
    "error-claimed": ("error", {}, {"claim_epoch": 4}, "none"),
    "error-paused": ("error", {}, {"status": "waiting_client_tool"}, "none"),
    "limit-claimed": ("reclaim", {"reclaim_attempts": 3}, {"claim_epoch": 4}, "none"),
    "limit-heartbeat": ("reclaim", {"reclaim_attempts": 3}, {"last_progress_at": "now"}, "none"),
    "limit-marked-heartbeat": (
        "reclaim",
        {"reclaim_attempts": 3, "lease_expired_at": "old", "reclaim_limit": 3},
        {"last_progress_at": "now"},
        "finalised",
    ),
}


def _when(values):
    times = {"now": _now(), "old": _ago(minutes=30)}
    return {k: times.get(v, v) if isinstance(v, str) else v for k, v in values.items()}


@pytest.mark.parametrize("case", list(RACES))
async def test_cas_loses_to_a_change_between_the_scan_and_the_write(
    store, engine, monkeypatch, record_property, case
):
    on_stale, seed, change, outcome = RACES[case]
    # The marked sibling sorts first and is skipped by the cursor; the foreign run
    # sorts after the target and is outside max_runs. Neither may be touched.
    marked_sibling, target = _ids(1, 2)
    await _seed(store, engine, {marked_sibling: {}, target: _when(seed), OTHER: {"tenant": "t2"}})
    await _set(engine, marked_sibling, **_marked(claim_epoch=7))
    before, events_before = await _rows(engine), await _events(engine)
    factory, opened = store._session_factory, []
    applied = {}

    async def change_then_open():
        applied.update(_when(change))
        await _set(engine, target, **applied)

    def hooked():
        opened.append(len(opened))
        return _Deferred(factory, change_then_open if len(opened) == 2 else None)

    monkeypatch.setattr(store, "_session_factory", hooked)
    with _trace(engine, record_property) as (sql, _params, boundaries, _connections):
        result = await _scan(store, on_stale=on_stale, cursor=marked_sibling, max_runs=1)
    assert opened == [0, 1]
    after = await _rows(engine)
    if outcome == "current":
        assert [_plain(i) for i in result.reclaimable] == [_reclaimable(target, after[target])]
        assert result.finalised == []
    elif outcome == "finalised":
        assert result.reclaimable == []
        assert [(s.run_id, s.failure_reason) for s in result.finalised] == [
            (target, "reclaim_limit")
        ]
    else:
        assert (result.reclaimable, result.finalised) == ([], [])
    assert (result.inspected, result.cursor) == (1, target)
    if outcome == "finalised":
        assert (after[target]["status"], after[target]["claim_epoch"]) == ("error", 4)
        assert len(await _events(engine, after=events_before)) == 1
    else:
        assert _same(after[target], {**before[target], **applied})
        assert await _events(engine, after=events_before) == []
    # The concurrent change is the second statement (its own committed transaction).
    assert _shape(sql) == [
        "SELECT agent_runs",
        "UPDATE agent_runs",
        "UPDATE agent_runs",
        *(["SELECT agent_runs"] if on_stale is None else []),
        *(["INSERT run_events"] if outcome == "finalised" else []),
    ]
    _untouched(before, after, marked_sibling, OTHER)
    assert boundaries == [
        "begin",
        "rollback",
        "begin",
        "commit",
        "begin",
        "commit" if outcome == "finalised" else "rollback",
    ]


def _same(row, expected):
    for key, value in expected.items():
        if key == "updated_at":
            continue
        if isinstance(value, datetime):
            assert _naive(row[key]) == _naive(value), key
        else:
            assert row[key] == value, key
    return True


class _Deferred:
    """Session context that runs a concurrent change before the session opens."""

    def __init__(self, factory, change):
        self._factory, self._change, self._session = factory, change, None

    async def __aenter__(self):
        if self._change is not None:
            await self._change()
        self._session = self._factory()
        return await self._session.__aenter__()

    async def __aexit__(self, *exc):
        return await self._session.__aexit__(*exc)


# ---------------------------------------------------------------------------
# get_lease_expired: rediscovery without writes
# ---------------------------------------------------------------------------


async def test_get_lease_expired_rediscovers_marks_in_id_order_without_writes(
    store, engine, record_property
):
    ids = _ids(1, 2, 3, 4, 5, 6)
    await _seed(
        store,
        engine,
        {
            ids[0]: _marked(),
            ids[1]: {**_marked(), "storage_format": 1},
            ids[2]: {**_marked(), "status": "error"},
            ids[3]: {},
            ids[4]: {"tenant": "t2", **_marked(claim_epoch=8, reclaim_attempts=2)},
            ids[5]: {**_marked(), "last_progress_at": _now()},
        },
    )
    before, events_before = await _rows(engine), await _events(engine)
    with _trace(engine, record_property) as (sql, params, boundaries, _connections):
        everything = await store.get_lease_expired(limit=10, cursor=None)
        page = await store.get_lease_expired(limit=2, cursor=None)
        after_first = await store.get_lease_expired(limit=2, cursor=ids[0])
        tail = await store.get_lease_expired(limit=2, cursor=ids[4])
        clamped = await store.get_lease_expired(limit=1001, cursor=None)
    assert [_plain(i) for i in everything] == [
        _reclaimable(run_id, before[run_id]) for run_id in (ids[0], ids[4], ids[5])
    ]
    assert [i.run_id for i in page] == [ids[0], ids[4]]
    assert [i.run_id for i in after_first] == [ids[4], ids[5]]
    assert [i.run_id for i in tail] == [ids[5]]
    assert [i.run_id for i in clamped] == [ids[0], ids[4], ids[5]]
    assert len(sql) == 5 and all(s.startswith("SELECT agent_runs.id") for s in sql)
    assert [p[0].get("param_1") for p in params] == [10, 2, 2, 2, 1000]
    assert "ORDER BY agent_runs.id" in sql[0] and "agent_runs.id > " in sql[2]
    assert boundaries == ["begin", "rollback"] * 5
    assert await _rows(engine) == before
    assert await _events(engine, after=events_before) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.0])
async def test_get_lease_expired_refuses_a_bad_limit_before_sql(
    store, engine, record_property, limit
):
    with (
        _trace(engine, record_property) as (sql, _params, _boundaries, _connections),
        pytest.raises(ValueError, match="positive"),
    ):
        await store.get_lease_expired(limit=limit, cursor=None)
    assert sql == []


# ---------------------------------------------------------------------------
# reclaim_run: the validated epoch-bumping claim of a marked run
# ---------------------------------------------------------------------------


def _reclaim_seed(**target):
    return {
        RUN: {**_marked(reclaim_attempts=1), "loop_state": {**LOOP, "submission": SUB}, **target},
        SIBLING: {**_marked(reclaim_attempts=1), "loop_state": {**LOOP, "submission": SUB}},
        OTHER: {
            "tenant": "t2",
            **_marked(reclaim_attempts=1),
            "loop_state": {**LOOP, "submission": SUB},
        },
    }


@pytest.mark.parametrize("path", ["returning", "plain"])
@pytest.mark.parametrize("scope", ["tenant", "unscoped"])
@pytest.mark.parametrize("expected_epoch", [4, None])
async def test_reclaim_bumps_the_epoch_once_and_touches_only_its_run(
    store, engine, monkeypatch, record_property, path, scope, expected_epoch
):
    await _seed(store, engine, _reclaim_seed())
    if path == "plain":
        store.supports_update_returning = False
    phases, sessions = _observe(store, monkeypatch)
    before, events_before = await _rows(engine), await _events(engine)
    with _trace(engine, record_property) as (sql, params, boundaries, connections):
        token = await _reclaim(
            store,
            tenant_id="t1" if scope == "tenant" else UNSCOPED,
            expected_claim_epoch=expected_epoch,
        )
    after = await _rows(engine)
    assert token == ClaimResult(5, 2, {**LOOP, "submission": SUB}, SUB)
    row, old = after[RUN], before[RUN]
    assert (row["status"], row["claim_epoch"], row["reclaim_attempts"]) == ("running", 5, 2)
    assert row["lease_expired_at"] is None
    assert row["last_progress_at"] == row["last_reclaimed_at"]
    assert _naive(row["last_reclaimed_at"]) > _naive(_ago(minutes=1))
    assert row["status_changed_at"] == old["status_changed_at"]
    keep = ("pause_data", "pause_requested", "cancel_requested", "loop_state", "reclaim_limit")
    assert {k: row[k] for k in keep} == {k: old[k] for k in keep}
    _untouched(before, after, SIBLING, OTHER)
    events = await _events(engine, after=events_before)
    assert _event_view(events) == [
        (RUN, "run.reclaimed", {"attempt": 2, "previous_epoch": 4, "gap_s": "<gap>"}, 0)
    ]
    assert 7100 < events[0]["data"]["gap_s"] < 7300
    assert phases == [
        ("update_one", (RUN,), True),
        ("announce", (RUN,)),
        ("announce", (RUN,)),
        "commit",
    ]
    assert len({id(s) for s in sessions}) == 1 and len(connections) == 1
    assert boundaries == ["begin", "commit"]
    lock_read = "SELECT agent_runs" if _pg(engine) else "UPDATE agent_runs"
    if _pg(engine):
        assert sql[0].endswith("FOR UPDATE")
    else:
        # SQLite reserves the writer with a self-assignment before validating.
        assert sql[0].startswith("UPDATE agent_runs SET id=agent_runs.id, updated_at=")
    expected = {
        ("postgresql", "returning"): [lock_read, "UPDATE agent_runs"],
        ("postgresql", "plain"): [lock_read, "UPDATE agent_runs", "SELECT agent_runs"],
        ("sqlite", "returning"): [lock_read, "UPDATE agent_runs"],
        ("sqlite", "plain"): [
            lock_read,
            "SELECT agent_runs",
            "UPDATE agent_runs",
            "SELECT agent_runs",
        ],
    }[engine.dialect.name, path]
    assert _shape(sql) == [*expected, "INSERT run_events"]
    where = sql[0].split(" WHERE ", 1)[1].split(" RETURNING ")[0]
    assert where.startswith("agent_runs.id = ")
    if scope == "tenant":
        spelled = "IS NOT DISTINCT FROM" if _pg(engine) else "IS ?"
        assert f"agent_runs.tenant_id {spelled}" in where
        assert "t1" in params[0][0].values()
    else:
        assert "tenant_id" not in where


async def test_reclaim_scope_matches_untenanted_runs_only_with_none(store, engine):
    await _seed(store, engine, {RUN: {**_marked(), "tenant": None}, SIBLING: _marked()})
    await _set(engine, RUN, tenant_id=None)
    before = await _rows(engine)
    with pytest.raises(RunNotFoundError):
        await _reclaim(store, tenant_id="t1")
    assert await _rows(engine) == before
    token = await _reclaim(store, tenant_id=None)
    assert token.claim_epoch == 5
    after = await _rows(engine)
    _untouched(before, after, SIBLING)


@pytest.mark.parametrize("policy", ["warn", "raise"])
async def test_reclaim_governance_change_warns_or_refuses(
    store, engine, caplog, record_property, policy
):
    caplog.set_level(logging.DEBUG)
    await _seed(store, engine, _reclaim_seed())
    before, events_before = await _rows(engine), await _events(engine)
    with _trace(engine, record_property) as (_sql, _params, boundaries, _connections):
        if policy == "raise":
            with pytest.raises(RunNotReclaimableError) as raised:
                await _reclaim(store, governance={"digest": "g2"}, governance_policy=policy)
            assert raised.value.reason == "governance_changed"
        else:
            token = await _reclaim(store, governance={"digest": "g2"}, governance_policy=policy)
            assert token.claim_epoch == 5
    assert _logged(caplog) == [("governance_mismatch", RUN, STORE_LOGGER)]
    events = _event_view(await _events(engine, after=events_before))
    if policy == "raise":
        assert await _rows(engine) == before and events == []
        assert boundaries == ["begin", "rollback"]
    else:
        assert events == [
            (
                RUN,
                "run.reclaimed",
                {"attempt": 2, "previous_epoch": 4, "gap_s": "<gap>", "governance_changed": True},
                0,
            )
        ]
        _untouched(before, await _rows(engine), SIBLING, OTHER)


REFUSALS = {
    # case: (seed overrides for the target, call overrides, expected reason or error type)
    "legacy_run": ({"storage_format": 1}, {}, "legacy_run"),
    "store_capability": ({}, {}, "store_capability"),
    "not_running": ({"status": "waiting_client_tool"}, {}, "not_running"),
    "lease_fresh": ({"lease_expired_at": None}, {}, "lease_fresh"),
    "already_reclaimed": ({}, {"expected_claim_epoch": 3}, "already_reclaimed"),
    "reclaim_limit": ({"reclaim_attempts": 3}, {}, "reclaim_limit"),
    "reclaim_limit_below": ({"reclaim_attempts": 2, "reclaim_limit": 2}, {}, "reclaim_limit"),
    "reclaim_limit_missing": ({"reclaim_limit": None}, {}, "reclaim_limit"),
    "agent_mismatch": ({}, {"agent_name": "other"}, "agent_mismatch"),
    "skill_mismatch": ({}, {"skill_pins": {"s": "2"}}, "skill_mismatch"),
    "loop_unknown": ({"loop_state": {**LOOP, "loop": "custom"}}, {}, "loop_unknown"),
    "loop_missing": ({"loop_state": {"skills": {"pins": {"s": "1"}}}}, {}, "loop_unknown"),
    "saved_single_call": (
        {"loop_state": {**LOOP, "loop": "single_call"}},
        {},
        "loop_unsupported",
    ),
    "requested_single_call": ({}, {"loop_type": "single_call"}, "loop_unsupported"),
    "legacy_before_status": (
        {"storage_format": 1, "status": "waiting_client_tool"},
        {},
        "legacy_run",
    ),
    "lease_before_epoch": ({"lease_expired_at": None}, {"expected_claim_epoch": 9}, "lease_fresh"),
    "status_before_lease": ({"status": "error", "lease_expired_at": None}, {}, "not_running"),
    "epoch_before_limit": (
        {"reclaim_attempts": 3},
        {"expected_claim_epoch": 9},
        "already_reclaimed",
    ),
    "limit_before_agent": ({"reclaim_attempts": 3}, {"agent_name": "other"}, "reclaim_limit"),
    "foreign_tenant": ({}, {"tenant_id": "t2"}, RunNotFoundError),
    "untenanted_scope": ({}, {"tenant_id": None}, RunNotFoundError),
    "missing": ({}, {"run_id": MISSING}, RunNotFoundError),
    "managed_child": (
        {"loop_state": {**LOOP, MANAGED_HEADER_KEY: {"invocation": {"id": "i"}}}},
        {},
        ManagedChildControlError,
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
async def test_reclaim_refusals_change_nothing(store, engine, monkeypatch, record_property, case):
    seed, call, expected = REFUSALS[case]
    await _seed(store, engine, _reclaim_seed(**seed))
    if case == "store_capability":
        store.capabilities = replace(store.capabilities, recovery=False)
    phases, _sessions = _observe(store, monkeypatch)
    before, events_before = await _rows(engine), await _events(engine)
    call = dict(call)
    run_id = call.pop("run_id", RUN)
    refused = (RunNotReclaimableError, RunNotFoundError, ManagedChildControlError)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(refused) as raised,
    ):
        await _reclaim(store, run_id, **call)
    if isinstance(expected, str):
        assert type(raised.value) is RunNotReclaimableError
        assert (raised.value.run_id, raised.value.reason) == (run_id, expected)
    else:
        assert type(raised.value) is expected
    assert await _rows(engine) == before
    assert await _events(engine, after=events_before) == []
    assert phases == [("update_one", (run_id,), True)]
    assert boundaries == ["begin", "rollback"]
    assert len(sql) == 1


@pytest.mark.parametrize(
    ("arguments", "code"),
    [({"governance_policy": "ignore"}, "governance_mismatch"), ({"tenant_id": ""}, None)],
)
async def test_reclaim_refuses_bad_arguments_before_sql(
    store, engine, record_property, caplog, arguments, code
):
    caplog.set_level(logging.DEBUG)
    await _seed(store, engine, _reclaim_seed())
    before = await _rows(engine)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(ValueError),
    ):
        await _reclaim(store, **arguments)
    assert (sql, boundaries) == ([], [])
    assert await _rows(engine) == before
    assert _logged(caplog) == ([(code, RUN, STORE_LOGGER)] if code else [])


@pytest.mark.parametrize("agent_name", ["leases", "other"])
async def test_reclaim_of_a_cancel_requested_run_commits_the_cancellation(
    store, engine, monkeypatch, record_property, agent_name
):
    await _seed(store, engine, _reclaim_seed(cancel_requested=True))
    await _set(engine, SIBLING, cancel_requested=True)
    phases, _sessions = _observe(store, monkeypatch)
    before, events_before = await _rows(engine), await _events(engine)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(RunNotReclaimableError) as raised,
    ):
        await _reclaim(store, agent_name=agent_name)
    assert raised.value.reason == "cancelled"
    row, old = (await _rows(engine))[RUN], before[RUN]
    assert (row["status"], row["claim_epoch"], row["pause_data"], row["cancel_requested"]) == (
        "cancelled",
        5,
        None,
        False,
    )
    assert row["status_changed_at"] != old["status_changed_at"]
    keep = ("reclaim_attempts", "lease_expired_at", "last_progress_at", "last_reclaimed_at")
    assert {k: row[k] for k in keep} == {k: old[k] for k in keep}
    _untouched(before, await _rows(engine), SIBLING, OTHER)
    assert _event_view(await _events(engine, after=events_before)) == [
        (RUN, "run.cancelled", {"reason": "cancel_requested"}, 0)
    ]
    assert phases == [
        ("update_one", (RUN,), True),
        ("announce", (RUN,)),
        ("announce", (RUN,)),
        "commit",
    ]
    assert boundaries == ["begin", "commit"]
    assert len(sql) == 3


# ---------------------------------------------------------------------------
# Retry and commit boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ["mark", "reclaim", "mark-scan", "rediscover"])
async def test_transient_session_failure_retries_only_the_per_run_attempts(
    store, engine, monkeypatch, record_property, operation
):
    await _seed(store, engine, _reclaim_seed() if operation == "reclaim" else {RUN: {}})
    retries, opened = [], []
    factory = store._session_factory
    failing = 0 if operation in ("reclaim", "mark-scan", "rediscover") else 1

    def flaky():
        opened.append(len(opened))
        if len(opened) == failing + 1:
            raise OperationalError("SELECT 1", {}, ConnectionError("transient"))
        return factory()

    async def backoff(attempt, retry_label, run_id, exc):
        retries.append((attempt, retry_label, run_id, type(exc).__name__))

    monkeypatch.setattr(store, "_session_factory", flaky)
    monkeypatch.setattr(durability, "_backoff", backoff)
    before = await _rows(engine)
    with _trace(engine, record_property) as (_sql, _params, boundaries, _connections):
        if operation == "mark":
            result = await _mark(store)
            assert [i.run_id for i in result.reclaimable] == [RUN]
        elif operation == "reclaim":
            assert (await _reclaim(store)).claim_epoch == 5
        else:
            with pytest.raises(PersistenceError) as raised:
                if operation == "mark-scan":
                    await _mark(store)
                else:
                    await store.get_lease_expired(limit=5, cursor=None)
            assert raised.value.retryable
    if operation in ("mark-scan", "rediscover"):
        assert retries == [] and opened == [0] and boundaries == []
        assert await _rows(engine) == before
    else:
        label = "mark_lease_expired" if operation == "mark" else "reclaim_run"
        assert retries == [(0, label, RUN, "OperationalError")]


@pytest.mark.parametrize("operation", ["mark", "finalise", "reclaim", "reclaim-cancelled"])
async def test_commit_failure_rolls_back_every_lease_write(
    store, engine, monkeypatch, record_property, operation
):
    seed = {
        "mark": {RUN: {}},
        "finalise": {RUN: {"reclaim_attempts": 3}},
        "reclaim": _reclaim_seed(),
        "reclaim-cancelled": _reclaim_seed(cancel_requested=True),
    }[operation]
    await _seed(store, engine, seed)
    before, events_before = await _rows(engine), await _events(engine)

    async def refuse(session):
        raise PersistenceError(code="commit_refused")

    monkeypatch.setattr(store, "_write_tx", refuse)
    with (
        _trace(engine, record_property) as (sql, _params, boundaries, _connections),
        pytest.raises(PersistenceError) as raised,
    ):
        if operation.startswith("reclaim"):
            await _reclaim(store)
        else:
            await _mark(store, finalise_unrecoverable=operation == "finalise")
    assert raised.value.code == "commit_refused"
    assert await _rows(engine) == before
    assert await _events(engine, after=events_before) == []
    tail = boundaries[-2:]
    assert tail == ["begin", "rollback"]
    assert len(sql) == 3


# ---------------------------------------------------------------------------
# Owner heartbeat, crash and reclaim (both backends)
# ---------------------------------------------------------------------------


async def test_only_the_current_epoch_renews_and_a_stopped_owner_loses_the_lease(store, engine):
    await _seed(store, engine, {RUN: {}, SIBLING: {}, OTHER: {"tenant": "t2"}})
    for run_id in RUNS:
        assert await store.touch_progress(run_id, claim_epoch=3)
    assert (await _mark(store)).reclaimable == []
    for run_id in RUNS:
        await _set(engine, run_id, last_progress_at=_ago(hours=2))
    assert await store.touch_progress(OTHER, claim_epoch=3)
    marked = await _mark(store)
    assert [(i.run_id, i.claim_epoch) for i in marked.reclaimable] == [(RUN, 4), (SIBLING, 4)]
    before = await _rows(engine)
    assert not await store.touch_progress(RUN, claim_epoch=3)
    assert await _rows(engine) == before
    token = await _reclaim(store, expected_claim_epoch=4)
    assert token.claim_epoch == 5
    assert not await store.touch_progress(RUN, claim_epoch=3)
    assert not await store.touch_progress(RUN, claim_epoch=4)
    assert await store.touch_progress(RUN, claim_epoch=5)
    rows = await _rows(engine)
    assert (rows[SIBLING]["claim_epoch"], rows[OTHER]["claim_epoch"]) == (4, 3)
    assert (await _mark(store)).reclaimable[0].run_id == SIBLING


# ---------------------------------------------------------------------------
# Real concurrency (Postgres): exactly one owner, no lost update
# ---------------------------------------------------------------------------

PARALLEL = 8


def _require_pg(engine):
    if not _pg(engine):
        pytest.skip("interleaved sessions need a real server (in-memory SQLite shares one)")


async def test_parallel_markers_fence_each_run_exactly_once(store, engine):
    _require_pg(engine)
    await _seed(
        store, engine, {RUN: {}, SIBLING: {}, OTHER: {"tenant": "t2", "last_progress_at": _now()}}
    )
    events_before = await _events(engine)
    results = await asyncio.gather(*[_mark(store) for _ in range(PARALLEL)], return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    for result in results:
        assert [(i.run_id, i.claim_epoch) for i in result.reclaimable] == [(RUN, 4), (SIBLING, 4)]
        assert result.finalised == []
    rows = await _rows(engine)
    assert [rows[run_id]["claim_epoch"] for run_id in RUNS] == [4, 4, 3]
    assert sorted(
        (e["agent_run_id"], e["event_type"]) for e in await _events(engine, after=events_before)
    ) == [(RUN, "run.lease_expired"), (SIBLING, "run.lease_expired")]


@pytest.mark.parametrize("on_stale", ["error", "reclaim"])
async def test_parallel_finalisers_finalise_each_run_exactly_once(store, engine, on_stale):
    _require_pg(engine)
    await _seed(
        store,
        engine,
        {RUN: {"reclaim_attempts": 3}, SIBLING: {"reclaim_attempts": 3}, OTHER: {"tenant": "t2"}},
    )
    await _set(engine, OTHER, last_progress_at=_now())
    events_before = await _events(engine)
    results = await asyncio.gather(
        *[_scan(store, on_stale=on_stale) for _ in range(PARALLEL)], return_exceptions=True
    )
    assert not [r for r in results if isinstance(r, BaseException)], results
    finalised = sorted(s.run_id for r in results for s in r.finalised)
    assert finalised == [RUN, SIBLING]
    rows = await _rows(engine)
    assert [(rows[r]["status"], rows[r]["claim_epoch"]) for r in RUNS] == [
        ("error", 4),
        ("error", 4),
        ("running", 3),
    ]
    assert sorted(
        (e["agent_run_id"], e["event_type"]) for e in await _events(engine, after=events_before)
    ) == [(RUN, "run.interrupted"), (SIBLING, "run.interrupted")]


@pytest.mark.parametrize("expected_epoch", [4, None])
async def test_parallel_reclaimers_have_exactly_one_owner_per_run(store, engine, expected_epoch):
    _require_pg(engine)
    await _seed(store, engine, _reclaim_seed())
    events_before = await _events(engine)
    calls = [
        _reclaim(store, run_id, expected_claim_epoch=expected_epoch)
        for _ in range(PARALLEL)
        for run_id in (RUN, SIBLING)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    rows = await _rows(engine)
    for offset, run_id in enumerate((RUN, SIBLING)):
        mine = results[offset::2]
        winners = [r for r in mine if isinstance(r, ClaimResult)]
        losers = [r for r in mine if not isinstance(r, ClaimResult)]
        assert [w.claim_epoch for w in winners] == [5], mine
        assert all(type(r) is RunNotReclaimableError for r in losers), losers
        assert {r.reason for r in losers} <= {"lease_fresh", "already_reclaimed"}
        if expected_epoch is None:
            assert {r.reason for r in losers} == {"lease_fresh"}
        assert (rows[run_id]["claim_epoch"], rows[run_id]["reclaim_attempts"]) == (5, 2)
        assert rows[run_id]["lease_expired_at"] is None
    assert (rows[OTHER]["claim_epoch"], rows[OTHER]["reclaim_attempts"]) == (4, 1)
    assert sorted(
        (e["agent_run_id"], e["event_type"], e["data"]["attempt"], e["data"]["previous_epoch"])
        for e in await _events(engine, after=events_before)
    ) == [(RUN, "run.reclaimed", 2, 4), (SIBLING, "run.reclaimed", 2, 4)]


@pytest.mark.parametrize("round_", range(3))
async def test_crashed_owner_is_marked_once_and_exactly_one_reclaimer_wins(store, engine, round_):
    _require_pg(engine)
    await _seed(store, engine, {RUN: {}, SIBLING: {}, OTHER: {"tenant": "t2"}})
    for run_id in RUNS:
        await _set(engine, run_id, last_progress_at=_now())
        assert await store.touch_progress(run_id, claim_epoch=3)
    # The owners of RUN and SIBLING crash: their progress stops and ages out.
    for run_id in (RUN, SIBLING):
        await _set(engine, run_id, last_progress_at=_ago(hours=2))
    marks = await asyncio.gather(*[_mark(store) for _ in range(PARALLEL)])
    assert {tuple((i.run_id, i.claim_epoch) for i in m.reclaimable) for m in marks} == {
        ((RUN, 4), (SIBLING, 4))
    }
    assert not await store.touch_progress(RUN, claim_epoch=3)
    calls = [
        _reclaim(store, run_id, expected_claim_epoch=4)
        for _ in range(PARALLEL)
        for run_id in (RUN, SIBLING)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    for offset, run_id in enumerate((RUN, SIBLING)):
        winners = [r for r in results[offset::2] if isinstance(r, ClaimResult)]
        assert [w.claim_epoch for w in winners] == [5]
        assert not await store.touch_progress(run_id, claim_epoch=3)
        assert not await store.touch_progress(run_id, claim_epoch=4)
        assert await store.touch_progress(run_id, claim_epoch=5)
    rows = await _rows(engine)
    assert [rows[r]["claim_epoch"] for r in RUNS] == [5, 5, 3]
    events = await _events(engine)
    for run_id in (RUN, SIBLING):
        kinds = [e["event_type"] for e in events if e["agent_run_id"] == run_id]
        assert kinds == ["run.lease_expired", "run.reclaimed"]
    assert [e for e in events if e["agent_run_id"] == OTHER] == []


@pytest.mark.parametrize("on_stale", [None, "error"])
@pytest.mark.parametrize("round_", range(5))
async def test_owner_heartbeat_and_expiry_race_has_exactly_one_winner(
    store, engine, on_stale, round_
):
    _require_pg(engine)
    await _seed(store, engine, {RUN: {}, SIBLING: {"last_progress_at": _now()}})
    heartbeat, scan = await asyncio.gather(
        store.touch_progress(RUN, claim_epoch=3),
        _scan(store, on_stale=on_stale),
    )
    row = (await _rows(engine))[RUN]
    expired = bool(scan.reclaimable or scan.finalised)
    assert heartbeat is not expired, (heartbeat, scan)
    assert row["claim_epoch"] == (4 if expired else 3)
    assert row["status"] == ("error" if expired and on_stale == "error" else "running")
    assert (await _rows(engine))[SIBLING]["claim_epoch"] == 3
