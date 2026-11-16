"""Stale and abandoned sweeps keep their categories, cutoffs, caps, cursors and row CAS."""

import asyncio
import json
import logging
import re
import warnings
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import sweep
from dendrux.db.models import AgentRun, RunEvent
from dendrux.errors import PersistenceError
from dendrux.runtime.state import SweptRun
from dendrux.runtime.stores.sql import _sql_sweep_scan

AGE = timedelta(hours=1)
STORE_LOGGER = "dendrux.runtime.stores.sql.base"
WAITING = ("waiting_client_tool", "waiting_human_input", "waiting_approval", "paused")
MANAGED = {
    "schema_version": 3,
    "pause_reason": "managed_wait",
    "managed_wait": {"child_run_id": "c"},
}
LOOP = {"v": 1, "loop": "react", "phase": "model_next"}
COLUMNS = (
    "status",
    "claim_epoch",
    "storage_format",
    "failure_reason",
    "error",
    "pause_data",
    "pause_requested",
    "cancel_requested",
    "last_progress_at",
    "status_changed_at",
    "updated_at",
    "created_at",
    "iteration_count",
    "tenant_id",
    "lease_expired_at",
    "reclaim_attempts",
    "loop_state",
)
CODES = {"sweep_cap_reached", "managed_state_invalid"}
PARALLEL = 8


def _ids(*numbers):
    return [f"{n:026d}" for n in numbers]


RUN, SIBLING, OTHER = _ids(1, 2, 3)
OTHER_TENANT = _ids(99)[0]


def _naive(value):
    return value.replace(tzinfo=None) if isinstance(value, datetime) and value.tzinfo else value


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


def _pg(engine):
    return engine.dialect.name == "postgresql"


@pytest.fixture
def clock(monkeypatch):
    """Fix the scan clock and record each scan's arguments and pages."""
    now = datetime.now(UTC).replace(microsecond=0)
    state = SimpleNamespace(now=now, cutoff=now - AGE, scans=[], pages=[])
    original = _sql_sweep_scan._SweepScan

    class FixedClockScan(original):
        def __init__(self, store, *, waiting, older_than):
            super().__init__(store, waiting=waiting, older_than=older_than)
            self.now = now
            self.cutoff = now - older_than
            state.scans.append((waiting, older_than))

        def page(self, status_filter, after, limit):
            state.pages.append((status_filter, after, limit))
            return super().page(status_filter, after, limit)

    monkeypatch.setattr(_sql_sweep_scan, "_SweepScan", FixedClockScan)
    state.scan_class = FixedClockScan
    return state


@contextmanager
def _trace(engine, record_property):
    """Capture statements, compiled parameters and transaction boundaries."""
    statements, parameters, boundaries = [], [], []

    def statement(conn, cursor, sql, values, context, many):
        statements.append(" ".join(sql.split()))
        parameters.append(_stable(context.compiled_parameters))

    def boundary(kind):
        return lambda conn: boundaries.append(kind)

    listeners = [("before_cursor_execute", statement)] + [
        (kind, boundary(kind)) for kind in ("begin", "commit", "rollback")
    ]
    for kind, listener in listeners:
        event.listen(engine.sync_engine, kind, listener)
    try:
        yield statements, parameters, boundaries
    finally:
        for kind, listener in listeners:
            event.remove(engine.sync_engine, kind, listener)
        record_property("sql", json.dumps(statements))
        record_property("parameters", json.dumps(parameters, default=str, sort_keys=True))
        record_property("boundaries", json.dumps(boundaries))


def _shape(sql):
    """Verb and table of each statement, in order."""
    shapes = []
    for statement in sql:
        verb = statement.split(" ", 1)[0]
        while re.search(r"\([^()]*\)", statement):
            statement = re.sub(r"\([^()]*\)", "", statement)
        match = re.search(r"\b(?:FROM|INTO|UPDATE)\s+(\w+)", statement)
        shapes.append(f"{verb} {match.group(1) if match else '?'}")
    return shapes


def _expected_io(steps):
    """Statement shapes and transaction cuts for page reads and row outcomes, in order."""
    shapes, boundaries = [], []
    for step in steps:
        if step == "page":
            shapes.append("SELECT agent_runs")
            boundaries += ["begin", "rollback"]
        elif step == "swept":
            shapes += ["UPDATE agent_runs", "INSERT run_events"]
            boundaries += ["begin", "commit"]
        elif step == "lost":
            shapes.append("UPDATE agent_runs")
            boundaries += ["begin", "rollback"]
        else:
            raise AssertionError(step)
    return shapes, boundaries


async def _seed(store, engine, clock, rows):
    """Create each run (its events first), then set stale RUNNING defaults plus overrides."""
    stale, old = clock.now - 2 * AGE, clock.now - 3 * AGE
    for run_id, values in rows.items():
        await store.create_run(
            run_id, values.get("agent_name", "sweeper"), tenant_id=values.get("tenant", "t1")
        )
        for event_type in values.get("events", ("run.started",)):
            await store.save_run_event(run_id, event_type=event_type)
    async with engine.begin() as conn:
        for run_id, values in rows.items():
            columns = {
                "status": "running",
                "claim_epoch": 3,
                "storage_format": 2,
                "loop_state": LOOP,
                "pause_data": {"marker": run_id},
                "pause_requested": True,
                "cancel_requested": False,
                "last_progress_at": stale,
                "created_at": old,
                "status_changed_at": stale,
                "updated_at": stale,
                "iteration_count": 2,
            }
            columns.update(
                {k: v for k, v in values.items() if k not in ("tenant", "events", "agent_name")}
            )
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(**columns))


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
    return [(e["agent_run_id"], e["event_type"], e["data"], e["iteration_index"]) for e in events]


def _set(engine, run_id, **values):
    async def apply():
        async with engine.begin() as conn:
            await conn.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))

    return apply()


def _logged(caplog):
    out = []
    for record in caplog.records:
        code = getattr(record, "code", None)
        if code in CODES:
            fields = {
                k: getattr(record, k)
                for k in ("run_id", "category", "cap", "inspected", "managed_reason")
                if hasattr(record, k)
            }
            out.append((code, record.name, fields))
    return out


def _swept(clock, run_id, row, reason, *, previous="running", status="error", agent="sweeper"):
    return SweptRun(
        run_id=run_id,
        agent_name=agent,
        previous_status=previous,
        failure_reason=reason,
        last_progress_at=_naive(row["last_progress_at"]),
        swept_at=_naive(clock.now),
        status=status,
    )


def _plain(items):
    """Compare SweptRun values with naive timestamps on every backend."""
    return [
        replace(
            item, last_progress_at=_naive(item.last_progress_at), swept_at=_naive(item.swept_at)
        )
        for item in items
    ]


def _finalised(before, after, run_id, *, status, reason, error):
    """A swept row: the CAS values written, the rest unchanged."""
    expected = {
        **before,
        "status": status,
        "claim_epoch": before["claim_epoch"] + 1,
        "failure_reason": reason,
        "error": error,
        "pause_data": None,
        "pause_requested": False,
        "cancel_requested": False,
    }
    for key in ("status_changed_at", "updated_at"):
        assert _naive(after[key]) > _naive(before[key]), (run_id, key)
        expected[key] = after[key]
    assert after == expected, run_id


def _observe(store, monkeypatch):
    """Record each row's CAS, announce and commit, with the session each one used.

    The phases hold the session objects themselves: an ``id()`` is unique only among
    live objects, and each row's session is closed (and may be freed) before the next.
    """
    phases = []
    one_rows, announce, commit = store._update_one_rows, store._mark_announce, store._write_tx

    async def wrote(session, **kwargs):
        phases.append(("update", kwargs["pk"], session))
        return await one_rows(session, **kwargs)

    def announced(session, run_ids):
        phases.append(("announce", tuple(run_ids), session))
        return announce(session, run_ids)

    async def committed(session, **kwargs):
        phases.append(("commit", kwargs, session))
        return await commit(session, **kwargs)

    monkeypatch.setattr(store, "_update_one_rows", wrote)
    monkeypatch.setattr(store, "_mark_announce", announced)
    monkeypatch.setattr(store, "_write_tx", committed)
    return phases


def _expected_phases(phases, outcomes):
    """Each swept row in one session: CAS, the event insert's announce, the row's own
    announce, commit. A lost row: the CAS only."""
    expected, sessions = [], []
    position = 0
    for run_id, outcome in outcomes:
        session = phases[position][2]
        sessions.append(session)
        steps = [("update", (run_id,), session)]
        if outcome == "swept":
            steps += [("announce", (run_id,), session)] * 2 + [("commit", {}, session)]
        expected += steps
        position += len(steps)
    assert phases == expected
    return sessions


def _untouched(before, after, *run_ids):
    for run_id in run_ids:
        assert after[run_id] == before[run_id], run_id


# ---------------------------------------------------------------------------
# Stale RUNNING categories
# ---------------------------------------------------------------------------


async def test_stale_sweep_finalises_each_stale_running_category(
    store, engine, clock, caplog, monkeypatch, record_property
):
    ids = _ids(*range(1, 14))
    now, cutoff = clock.now, clock.cutoff
    rows = {
        ids[0]: {},  # started -> stale_running
        ids[1]: {"events": ("run.resumed",), "iteration_count": 5},  # never_started
        ids[2]: {"cancel_requested": True},  # started + cancel -> cancelled
        ids[3]: {"events": (), "cancel_requested": True},  # never started + cancel
        ids[4]: {"last_progress_at": now},  # fresh
        ids[5]: {"last_progress_at": None},  # ages from an old created_at
        ids[6]: {"last_progress_at": None, "created_at": now},  # fresh by created_at
        ids[7]: {"tenant": "t2"},  # every tenant is swept
        ids[8]: {"status": "waiting_client_tool"},
        ids[9]: {"status": "error"},
        ids[10]: {"last_progress_at": cutoff},  # exactly at the cutoff: fresh
        ids[11]: {"last_progress_at": cutoff - timedelta(microseconds=1)},
        ids[12]: {"last_progress_at": None, "created_at": cutoff},
    }
    await _seed(store, engine, clock, rows)
    before, events_before = await _rows(engine), await _events(engine)
    caplog.set_level(logging.DEBUG)
    phases = _observe(store, monkeypatch)
    with _trace(engine, record_property) as (sql, _params, boundaries):
        result = await store.sweep_stale_runs(AGE)
    after = await _rows(engine)
    expected = {
        ids[0]: ("stale_running", "error"),
        ids[1]: ("never_started", "error"),
        ids[2]: ("stale_running", "cancelled"),
        ids[3]: ("never_started", "cancelled"),
        ids[5]: ("stale_running", "error"),
        ids[7]: ("stale_running", "error"),
        ids[11]: ("stale_running", "error"),
    }
    assert _plain(result) == [
        _swept(clock, run_id, before[run_id], reason, status=status)
        for run_id, (reason, status) in expected.items()
    ]
    for run_id, (reason, status) in expected.items():
        error = None if status == "cancelled" else f"Run swept as {reason}"
        _finalised(before[run_id], after[run_id], run_id, status=status, reason=reason, error=error)
    _untouched(before, after, *(i for i in ids if i not in expected))
    assert _event_view(await _events(engine, after=events_before)) == [
        (
            run_id,
            "run.cancelled" if status == "cancelled" else "run.interrupted",
            {
                "failure_reason": reason,
                **({"reason": "cancel_requested"} if status == "cancelled" else {}),
            },
            before[run_id]["iteration_count"],
        )
        for run_id, (reason, status) in expected.items()
    ]
    assert clock.scans == [(False, AGE)]
    assert clock.pages == [("running", None, 100)]
    assert _logged(caplog) == []
    sessions = _expected_phases(phases, [(run_id, "swept") for run_id in expected])
    # One session per row: no two rows share a session object.
    assert all(a is not b for i, a in enumerate(sessions) for b in sessions[i + 1 :])
    shapes, cuts = _expected_io(["page"] + ["swept"] * len(expected))
    assert (_shape(sql), boundaries) == (shapes, cuts)
    page, cas = sql[0], sql[1]
    assert "agent_runs.status BETWEEN" in page
    assert "ORDER BY agent_runs.status, agent_runs.id" in page
    assert "run_events.agent_run_id = agent_runs.id" in page
    assert "run_events.event_type = 'run.started'" in page
    for predicate in (
        "agent_runs.id = ",
        "agent_runs.status = ",
        "agent_runs.claim_epoch = ",
        "coalesce(agent_runs.last_progress_at, agent_runs.created_at) < ",
    ):
        assert predicate in cas, predicate
    assert "agent_runs.updated_at <" not in cas


# ---------------------------------------------------------------------------
# Abandoned WAITING categories and managed-wait exemptions
# ---------------------------------------------------------------------------


async def test_abandoned_sweep_finalises_each_waiting_status_and_exempts_managed_waits(
    store, engine, clock, caplog, monkeypatch, record_property
):
    ids = _ids(*range(1, 21))
    now, cutoff = clock.now, clock.cutoff
    wct = "waiting_client_tool"
    rows = {
        ids[0]: {"status": wct},
        ids[1]: {"status": "waiting_human_input"},
        ids[2]: {"status": "waiting_approval"},
        ids[3]: {"status": "paused", "iteration_count": 7},
        ids[4]: {"status": wct, "cancel_requested": True},
        ids[5]: {"status": wct, "status_changed_at": now},  # fresh: outside the page
        ids[6]: {"status": wct, "updated_at": now},  # counted; the CAS loses
        ids[7]: {"status": wct, "pause_data": MANAGED},  # exempt, valid marks
        ids[8]: {"status": wct, "pause_data": {"schema_version": 3}},
        ids[9]: {"status": wct, "pause_data": {"pause_reason": "managed_wait"}},
        ids[10]: {"status": wct, "pause_data": {**MANAGED, "managed_wait": {}}},
        ids[11]: {"status": wct, "pause_data": {"schema_version": 2, "pause_reason": "other"}},
        ids[12]: {"status": wct, "tenant": "t2"},  # every tenant is swept
        ids[13]: {},  # running
        ids[14]: {"status": "error"},
        ids[15]: {"status": wct, "status_changed_at": cutoff},  # at the cutoff: outside
        ids[16]: {"status": wct, "status_changed_at": cutoff - timedelta(microseconds=1)},
        ids[17]: {"status": wct, "updated_at": cutoff},  # CAS loses at the cutoff
        ids[18]: {"status": wct, "updated_at": cutoff - timedelta(microseconds=1)},
        ids[19]: {"status": wct, "pause_data": None},
    }
    await _seed(store, engine, clock, rows)
    before, events_before = await _rows(engine), await _events(engine)
    caplog.set_level(logging.DEBUG)
    phases = _observe(store, monkeypatch)
    with _trace(engine, record_property) as (sql, _params, boundaries):
        result = await store.sweep_abandoned_runs(AGE)
    after = await _rows(engine)
    order = sorted(
        (
            run_id
            for run_id, row in before.items()
            if row["status"] in WAITING and _naive(row["status_changed_at"]) < _naive(cutoff)
        ),
        key=lambda r: (
            WAITING.index(before[r]["status"]),
            _naive(before[r]["status_changed_at"]),
            r,
        ),
    )
    exempt = {ids[7], ids[8], ids[9], ids[10]}
    lost = {ids[6], ids[17]}
    swept = [r for r in order if r not in exempt | lost]
    assert swept == [ids[0], ids[4], ids[11], ids[12], ids[18], ids[19], ids[16]] + ids[1:4]
    assert _plain(result) == [
        _swept(clock, r, before[r], "abandoned_waiting", previous=before[r]["status"])
        for r in swept
    ]
    for run_id in swept:
        _finalised(
            before[run_id],
            after[run_id],
            run_id,
            status="error",
            reason="abandoned_waiting",
            error="Run swept as abandoned_waiting",
        )
    _untouched(before, after, *(r for r in ids if r not in swept))
    assert _event_view(await _events(engine, after=events_before)) == [
        (r, "run.abandoned", {"failure_reason": "abandoned_waiting"}, before[r]["iteration_count"])
        for r in sorted(swept)
    ]
    assert _logged(caplog) == [
        ("managed_state_invalid", STORE_LOGGER, {"run_id": r, "managed_reason": "wait_invalid"})
        for r in (ids[8], ids[9], ids[10])
    ]
    assert clock.scans == [(True, AGE)]
    assert clock.pages == [(status, None, 100) for status in WAITING]
    steps = ["page"]
    for run_id in order[:13]:
        steps += [] if run_id in exempt else ["lost" if run_id in lost else "swept"]
    steps += ["page", "swept", "page", "swept", "page", "swept"]
    inspected = [r for r in order if r not in exempt]
    _expected_phases(phases, [(r, "lost" if r in lost else "swept") for r in inspected])
    shapes, cuts = _expected_io(steps)
    assert (_shape(sql), boundaries) == (shapes, cuts)
    page, cas = sql[0], sql[1]
    assert "agent_runs.status = " in page
    assert "agent_runs.status_changed_at < " in page
    assert "ORDER BY agent_runs.status_changed_at, agent_runs.id" in page
    for predicate in (
        "agent_runs.id = ",
        "agent_runs.status = ",
        "agent_runs.claim_epoch = ",
        "agent_runs.status_changed_at < ",
        "agent_runs.updated_at < ",
    ):
        assert predicate in cas, predicate


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

ENTRY_POINTS = ("stale", "abandoned", "scan-abandoned", "sweep-runs")


def _entry(store, entry, older_than=AGE, **limits):
    if entry == "stale":
        return store.sweep_stale_runs(older_than, **limits)
    if entry == "abandoned":
        return store.sweep_abandoned_runs(older_than, **limits)
    arguments = {"batch_size": 100, "max_runs": 100, **limits}
    if entry == "scan-abandoned":
        return store._scan_abandoned_runs(older_than, cursor=None, **arguments)
    return store._sweep_runs(older_than, waiting=False, **arguments)


BAD_BOUNDS = [
    ("batch_size", 0, 1000),
    ("batch_size", -1, 1000),
    ("batch_size", True, 1000),
    ("batch_size", 1.0, 1000),
    ("batch_size", "5", 1000),
    ("batch_size", 1001, 1000),
    ("max_runs", 0, 100000),
    ("max_runs", False, 100000),
    ("max_runs", 2.5, 100000),
    ("max_runs", 100001, 100000),
]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(("name", "value", "ceiling"), BAD_BOUNDS)
async def test_bounds_are_refused_before_any_sql(
    store, engine, clock, record_property, entry, name, value, ceiling
):
    with (
        _trace(engine, record_property) as (sql, _params, boundaries),
        pytest.raises(ValueError) as caught,
    ):
        await _entry(store, entry, **{name: value})
    assert str(caught.value) == f"{name} must be between 1 and {ceiling}"
    assert (sql, boundaries, clock.scans) == ([], [], [])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("older_than", [timedelta(seconds=-1), 3600, None])
async def test_a_bad_age_is_refused_after_the_bounds_and_before_any_sql(
    store, engine, clock, record_property, entry, older_than
):
    with _trace(engine, record_property) as (sql, _params, boundaries):
        with pytest.raises(ValueError) as caught:
            await _entry(store, entry, older_than=older_than)
        assert str(caught.value) == "sweep age must be a non-negative timedelta"
        # Bounds are checked first.
        with pytest.raises(ValueError) as caught:
            await _entry(store, entry, older_than=older_than, batch_size=0)
        assert str(caught.value) == "batch_size must be between 1 and 1000"
    assert (sql, boundaries, clock.scans) == ([], [], [])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_the_largest_bounds_and_a_zero_age_are_accepted(store, engine, clock, entry):
    await _seed(store, engine, clock, {RUN: {"last_progress_at": clock.now}})
    await _entry(store, entry, batch_size=1000, max_runs=100_000)
    await _entry(store, entry, older_than=timedelta(0), batch_size=1, max_runs=1)
    waiting = entry in ("abandoned", "scan-abandoned")
    statuses = WAITING if waiting else ("running",)
    pages = [(s, None, 1000) for s in statuses]
    if waiting:
        pages += [(s, None, 1) for s in statuses]
    else:
        pages += [("running", None, 1)]
    assert clock.scans == [(waiting, AGE), (waiting, timedelta(0))]
    assert clock.pages == pages


# ---------------------------------------------------------------------------
# Paging, the work cap and continuation
# ---------------------------------------------------------------------------

RUNNING_PAGES = {
    # case: (row ages, batch, max_runs, expected pages as (after index, limit), swept, capped)
    "pages": (["stale"] * 5, 2, 100, [(None, 2), (1, 2), (3, 2)], 5, False),
    "cap-mid-page": (["stale"] * 5, 2, 3, [(None, 2), (1, 1)], 3, True),
    "fresh-count": (["fresh", "fresh", "stale"], 10, 2, [(None, 2)], 0, True),
    "cap-at-end": (["stale"] * 2, 2, 2, [(None, 2)], 2, True),
    "short-page": (["stale"] * 2, 2, 3, [(None, 2), (1, 1)], 2, False),
    # Fresh rows stay RUNNING: only the (status, id) position moves the page past them.
    "fresh-pages": (
        ["fresh", "fresh", "stale", "stale", "stale"],
        2,
        100,
        [(None, 2), (1, 2), (3, 2)],
        3,
        False,
    ),
}


@pytest.mark.parametrize("entry", ["sweep-runs", "public"])
@pytest.mark.parametrize("case", list(RUNNING_PAGES))
async def test_running_pages_follow_the_batch_and_stop_at_the_cap(
    store, engine, clock, caplog, record_property, case, entry
):
    ages, batch, max_runs, pages, swept_count, capped = RUNNING_PAGES[case]
    ids = _ids(*range(1, len(ages) + 1))
    rows = {
        run_id: {"last_progress_at": clock.now if age == "fresh" else clock.now - 2 * AGE}
        for run_id, age in zip(ids, ages, strict=True)
    }
    rows[OTHER_TENANT] = {"tenant": "t2", "status": "waiting_client_tool"}
    await _seed(store, engine, clock, rows)
    before = await _rows(engine)
    caplog.set_level(logging.DEBUG)
    with _trace(engine, record_property):
        if entry == "public":
            result, cursor = (
                await store.sweep_stale_runs(AGE, batch_size=batch, max_runs=max_runs),
                None,
            )
        else:
            result, cursor = await store._sweep_runs(
                AGE, waiting=False, batch_size=batch, max_runs=max_runs
            )
    # A running scan never returns a continuation, even at the cap.
    assert cursor is None
    assert clock.scans == [(False, AGE)]
    assert [s.run_id for s in result] == [
        r for r, a in zip(ids, ages, strict=True) if a == "stale"
    ][:swept_count]
    assert clock.pages == [
        ("running", None if index is None else ids[index], limit) for index, limit in pages
    ]
    after = await _rows(engine)
    _untouched(before, after, *(r for r in before if r not in {s.run_id for s in result}))
    assert _logged(caplog) == (
        [
            (
                "sweep_cap_reached",
                STORE_LOGGER,
                {"category": "stale_running", "cap": max_runs, "inspected": max_runs},
            )
        ]
        if capped
        else []
    )


async def test_waiting_cursor_continues_across_statuses_and_resets_the_position(
    store, engine, clock, caplog, record_property
):
    a, b, c, d, e = _ids(1, 2, 3, 4, 5)
    now = clock.now
    rows = {
        a: {"status": "waiting_client_tool", "status_changed_at": now - 5 * AGE},
        b: {"status": "waiting_client_tool", "status_changed_at": now - 4 * AGE},
        c: {"status": "waiting_client_tool", "status_changed_at": now - 3 * AGE},
        d: {"status": "waiting_human_input", "status_changed_at": now - 6 * AGE},
        e: {"status": "paused", "status_changed_at": now - 7 * AGE},
    }
    await _seed(store, engine, clock, rows)
    before = await _rows(engine)
    caplog.set_level(logging.DEBUG)
    with _trace(engine, record_property):
        first, cursor = await store._scan_abandoned_runs(
            AGE, batch_size=10, max_runs=2, cursor=None
        )
    assert [s.run_id for s in first] == [a, b]
    assert (cursor[0], _naive(cursor[1]), cursor[2]) == (
        "waiting_client_tool",
        _naive(before[b]["status_changed_at"]),
        b,
    )
    assert clock.pages == [("waiting_client_tool", None, 2)]
    clock.pages.clear()
    second, cursor2 = await store._scan_abandoned_runs(
        AGE, batch_size=10, max_runs=2, cursor=cursor
    )
    # The human-input row is older than the cursor position: the position resets per status.
    assert [s.run_id for s in second] == [c, d]
    assert (cursor2[0], _naive(cursor2[1]), cursor2[2]) == (
        "waiting_human_input",
        _naive(before[d]["status_changed_at"]),
        d,
    )
    assert clock.pages == [("waiting_client_tool", cursor[1:], 2), ("waiting_human_input", None, 1)]
    clock.pages.clear()
    third, cursor3 = await store._scan_abandoned_runs(
        AGE, batch_size=10, max_runs=2, cursor=cursor2
    )
    assert ([s.run_id for s in third], cursor3) == ([e], None)
    assert clock.pages == [
        ("waiting_human_input", cursor2[1:], 2),
        ("waiting_approval", None, 2),
        ("paused", None, 2),
    ]
    cap = {"category": "abandoned_waiting", "cap": 2, "inspected": 2}
    assert _logged(caplog) == [("sweep_cap_reached", STORE_LOGGER, cap)] * 2
    assert {r: v["status"] for r, v in (await _rows(engine)).items()} == dict.fromkeys(
        (a, b, c, d, e), "error"
    )


async def test_waiting_cursor_passes_exempt_waits_and_breaks_age_ties_by_id(
    store, engine, clock, caplog, record_property
):
    m1, m2, m3, s = _ids(1, 2, 3, 4)
    age = clock.now - 2 * AGE
    managed = {"status": "waiting_client_tool", "pause_data": MANAGED, "status_changed_at": age}
    rows = {m1: managed, m2: managed, m3: managed, s: {**managed, "pause_data": None}}
    await _seed(store, engine, clock, rows)
    before = await _rows(engine)
    caplog.set_level(logging.DEBUG)
    # Without a cursor a repeated sweep restarts at the oldest row and stalls behind
    # max_runs exempt waits.
    for _ in range(2):
        assert await store.sweep_abandoned_runs(AGE, batch_size=2, max_runs=3) == []
    assert [(p[0], p[2]) for p in clock.pages] == [
        ("waiting_client_tool", 2),
        ("waiting_client_tool", 1),
    ] * 2
    assert [(p[1] is None) or (_naive(p[1][0]), p[1][1]) for p in clock.pages] == [
        True,
        (_naive(age), m2),
    ] * 2
    clock.pages.clear()
    swept, cursor = await store._scan_abandoned_runs(AGE, batch_size=10, max_runs=3, cursor=None)
    assert (swept, cursor[0], _naive(cursor[1]), cursor[2]) == (
        [],
        "waiting_client_tool",
        _naive(age),
        m3,
    )
    with _trace(engine, record_property):
        swept, cursor = await store._scan_abandoned_runs(
            AGE, batch_size=10, max_runs=3, cursor=cursor
        )
    assert ([r.run_id for r in swept], cursor) == ([s], None)
    assert [(p[0], p[2]) for p in clock.pages] == [
        ("waiting_client_tool", 3),
        ("waiting_client_tool", 3),
        ("waiting_human_input", 2),
        ("waiting_approval", 2),
        ("paused", 2),
    ]
    assert [p[1] is None for p in clock.pages] == [True, False, True, True, True]
    assert (_naive(clock.pages[1][1][0]), clock.pages[1][1][1]) == (_naive(age), m3)
    after = await _rows(engine)
    _untouched(before, after, m1, m2, m3)
    assert after[s]["status"] == "error"
    cap = {"category": "abandoned_waiting", "cap": 3, "inspected": 3}
    assert _logged(caplog) == [("sweep_cap_reached", STORE_LOGGER, cap)] * 3


# ---------------------------------------------------------------------------
# The public sweep(): composite cursor and dispatch through the identity properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("on_stale", [None, "error", "reclaim"])
async def test_sweep_continues_waiting_progress_in_the_composite_cursor(
    store, engine, clock, on_stale
):
    running, a, b, c = _ids(1, 2, 3, 4)
    now = clock.now
    rows = {
        running: {"last_progress_at": now - 2 * AGE},
        a: {"status": "waiting_client_tool", "status_changed_at": now - 4 * AGE},
        b: {"status": "waiting_client_tool", "status_changed_at": now - 3 * AGE},
        c: {"status": "waiting_client_tool", "status_changed_at": now - 2 * AGE},
    }
    await _seed(store, engine, clock, rows)
    before = await _rows(engine)
    arguments = {"state_store": store, "stale_running": AGE, "abandoned_waiting": AGE}
    with pytest.warns(FutureWarning) if on_stale is None else _no_warning():
        first = await sweep(**arguments, max_runs=2, on_stale=on_stale)
    if on_stale == "reclaim":
        assert ([r.run_id for r in first.reclaimable], first.stale_running) == ([running], [])
    else:
        assert ([r.run_id for r in first.stale_running], first.reclaimable) == ([running], [])
    assert [r.run_id for r in first.abandoned_waiting] == [a, b]
    assert first.cursor.startswith("sweep:v1:")
    running_cursor, waiting, running_done, waiting_done = json.loads(first.cursor[9:])
    assert (running_cursor, waiting[0], waiting[2], running_done, waiting_done) == (
        None,
        "waiting_client_tool",
        b,
        True,
        False,
    )
    assert _naive(datetime.fromisoformat(waiting[1])) == _naive(before[b]["status_changed_at"])
    with _no_warning():
        second = await sweep(**arguments, max_runs=2, on_stale=on_stale, cursor=first.cursor)
    assert (second.stale_running, second.reclaimable, second.cursor) == ([], [], None)
    assert [r.run_id for r in second.abandoned_waiting] == [c]
    assert clock.scans == [(True, AGE), (True, AGE)]
    # Without a waiting threshold the cursor is the plain running cursor.
    later, last = _ids(5, 6)
    await _seed(store, engine, clock, {later: {}, last: {}})
    only_running = await sweep(state_store=store, stale_running=AGE, on_stale="error", max_runs=1)
    # A run marked by the reclaim sweep is still running and sorts first.
    first_running = running if on_stale == "reclaim" else later
    assert ([r.run_id for r in only_running.stale_running], only_running.cursor) == (
        [first_running],
        first_running,
    )
    assert only_running.abandoned_waiting == []


@contextmanager
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


async def test_sweep_dispatch_follows_the_identity_properties(store, engine, clock):
    a, b, running = _ids(1, 2, 3)
    calls = []

    class OwnAbandoned(type(store)):
        async def sweep_abandoned_runs(self, older_than, **limits):
            calls.append(("sweep_abandoned_runs", older_than, limits))
            return await super().sweep_abandoned_runs(older_than, **limits)

    class OwnStale(type(store)):
        async def sweep_stale_runs(self, older_than, **limits):
            calls.append(("sweep_stale_runs", older_than, limits))
            return await super().sweep_stale_runs(older_than, **limits)

    assert (store._uses_waiting_sweep_cursor, store._uses_combined_sweep) == (True, True)
    own_abandoned, own_stale = OwnAbandoned(engine), OwnStale(engine)
    assert (own_abandoned._uses_waiting_sweep_cursor, own_abandoned._uses_combined_sweep) == (
        False,
        True,
    )
    assert (own_stale._uses_waiting_sweep_cursor, own_stale._uses_combined_sweep) == (True, False)
    waiting = {"status": "waiting_client_tool"}
    await _seed(store, engine, clock, {a: waiting, b: waiting, running: {}})
    # An overriding store gets no waiting continuation: one bounded call, no cursor.
    result = await sweep(state_store=own_abandoned, abandoned_waiting=AGE, max_runs=1)
    assert ([r.run_id for r in result.abandoned_waiting], result.cursor) == ([a], None)
    assert calls == [("sweep_abandoned_runs", AGE, {"max_runs": 1})]
    calls.clear()
    result = await sweep(state_store=own_stale, stale_running=AGE, on_stale="error")
    assert ([r.run_id for r in result.stale_running], result.cursor) == ([running], None)
    assert calls == [("sweep_stale_runs", AGE, {})]
    assert clock.scans == [(True, AGE), (False, AGE)]


# ---------------------------------------------------------------------------
# A concurrent change between the page read and the row's CAS
# ---------------------------------------------------------------------------

RACES = {
    # case: (waiting, concurrent change, outcome). Waiting changes restate a stale
    # updated_at (a Core UPDATE otherwise stamps it), isolating each CAS predicate.
    "heartbeat": (False, {"last_progress_at": "now"}, "lost"),
    "heartbeat-at-cutoff": (False, {"last_progress_at": "cutoff"}, "lost"),
    "heartbeat-below-cutoff": (False, {"last_progress_at": "below"}, "swept"),
    "claimed": (False, {"claim_epoch": 4}, "lost"),
    "paused": (False, {"status": "waiting_client_tool"}, "lost"),
    "finalised": (False, {"status": "error"}, "lost"),
    "cancel-requested": (False, {"cancel_requested": True}, "cancelled"),
    "touched": (True, {"updated_at": "now"}, "lost"),
    "touched-at-cutoff": (True, {"updated_at": "cutoff"}, "lost"),
    "touched-below-cutoff": (True, {"updated_at": "below"}, "swept"),
    "re-paused": (True, {"status_changed_at": "now", "updated_at": "stale"}, "lost"),
    "re-paused-at-cutoff": (True, {"status_changed_at": "cutoff", "updated_at": "stale"}, "lost"),
    "wait-claimed": (True, {"claim_epoch": 4, "updated_at": "stale"}, "lost"),
    "resumed": (True, {"status": "running", "updated_at": "stale"}, "lost"),
    "other-wait": (True, {"status": "waiting_human_input", "updated_at": "stale"}, "lost"),
    "managed": (True, {"pause_data": MANAGED, "updated_at": "stale"}, "lost"),
    "managed-version": (True, {"pause_data": {"schema_version": 3}, "updated_at": "stale"}, "lost"),
    "managed-reason": (
        True,
        {"pause_data": {"pause_reason": "managed_wait"}, "updated_at": "stale"},
        "lost",
    ),
    "unmanaged-marks": (
        True,
        {"pause_data": {"schema_version": 2, "pause_reason": "x"}, "updated_at": "stale"},
        "swept",
    ),
    "cleared-marks": (True, {"pause_data": None, "updated_at": "stale"}, "swept"),
}


def _when(values, clock):
    times = {
        "now": clock.now,
        "cutoff": clock.cutoff,
        "below": clock.cutoff - timedelta(microseconds=1),
        "stale": clock.now - 2 * AGE,
    }
    return {k: times.get(v, v) if isinstance(v, str) else v for k, v in values.items()}


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


@pytest.mark.parametrize("case", list(RACES))
async def test_cas_loses_to_a_change_between_the_page_and_the_write(
    store, engine, clock, monkeypatch, caplog, record_property, case
):
    waiting, change, outcome = RACES[case]
    target, later = _ids(1, 2)
    status = {"status": "waiting_client_tool"} if waiting else {}
    await _seed(store, engine, clock, {target: status, later: {**status, "tenant": "t2"}})
    before, events_before = await _rows(engine), await _events(engine)
    factory, opened, applied = store._session_factory, [], {}

    async def change_then_open():
        applied.update(_when(change, clock))
        await _set(engine, target, **applied)

    def hooked():
        opened.append(len(opened))
        return _Deferred(factory, change_then_open if len(opened) == 2 else None)

    monkeypatch.setattr(store, "_session_factory", hooked)
    caplog.set_level(logging.DEBUG)
    with _trace(engine, record_property) as (sql, _params, boundaries):
        result, cursor = await store._sweep_runs(AGE, waiting=waiting, batch_size=10, max_runs=1)
    assert opened == [0, 1]
    after = await _rows(engine)
    changed = {**before[target], **applied}
    if outcome == "lost":
        assert result == []
        for key, value in changed.items():
            if key != "updated_at":
                assert _naive(after[target][key]) == _naive(value), key
        assert await _events(engine, after=events_before) == []
    else:
        reason = "abandoned_waiting" if waiting else "stale_running"
        status_after = "cancelled" if outcome == "cancelled" else "error"
        assert [(s.run_id, s.failure_reason, s.status) for s in result] == [
            (target, reason, status_after)
        ]
        assert (after[target]["status"], after[target]["claim_epoch"]) == (status_after, 4)
        assert len(await _events(engine, after=events_before)) == 1
    _untouched(before, after, later)
    category = "abandoned_waiting" if waiting else "stale_running"
    assert cursor == ((("waiting_client_tool", cursor[1], target)) if waiting else None)
    assert _logged(caplog) == [
        ("sweep_cap_reached", STORE_LOGGER, {"category": category, "cap": 1, "inspected": 1})
    ]
    # The concurrent change is its own committed transaction before the row's CAS.
    shapes, cuts = _expected_io(["page", "swept" if outcome != "lost" else "lost"])
    assert _shape(sql) == [shapes[0], "UPDATE agent_runs", *shapes[1:]]
    assert boundaries == [*cuts[:2], "begin", "commit", *cuts[2:]]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ENTRY_POINTS)
async def test_a_failed_page_read_raises_a_typed_error_and_writes_nothing(
    store, engine, clock, monkeypatch, entry
):
    waiting = entry in ("abandoned", "scan-abandoned")
    await _seed(store, engine, clock, {RUN: {"status": "waiting_client_tool"} if waiting else {}})
    before = await _rows(engine)
    factory, opened = store._session_factory, []

    class FailingRead:
        async def __aenter__(self):
            self._context = factory()
            session = await self._context.__aenter__()

            async def lost(*args, **kwargs):
                raise OperationalError("SELECT", {}, Exception("connection reset"))

            session.execute = lost
            return session

        async def __aexit__(self, *exc):
            return await self._context.__aexit__(*exc)

    def hooked():
        opened.append(len(opened))
        return FailingRead()

    monkeypatch.setattr(store, "_session_factory", hooked)
    with pytest.raises(PersistenceError) as caught:
        await _entry(store, entry)
    assert type(caught.value) is PersistenceError
    assert (caught.value.__cause__, caught.value.__context__) == (None, None)
    assert opened == [0]
    assert await _rows(engine) == before


@pytest.mark.parametrize("waiting", [False, True])
async def test_a_failed_commit_keeps_earlier_rows_and_rolls_back_its_own(
    store, engine, clock, monkeypatch, waiting
):
    ids = _ids(1, 2, 3)
    status = {"status": "waiting_client_tool"} if waiting else {}
    await _seed(store, engine, clock, dict.fromkeys(ids, status))
    before, events_before = await _rows(engine), await _events(engine)
    commit, commits = store._write_tx, []

    async def failing(session, **kwargs):
        commits.append(len(commits))
        if len(commits) == 2:
            raise OperationalError("COMMIT", {}, Exception("commit lost"))
        return await commit(session, **kwargs)

    monkeypatch.setattr(store, "_write_tx", failing)
    with pytest.raises(PersistenceError):
        await store._sweep_runs(AGE, waiting=waiting, batch_size=10, max_runs=10)
    after = await _rows(engine)
    assert after[ids[0]]["status"] in ("error",)
    _untouched(before, after, ids[1], ids[2])
    assert [e["agent_run_id"] for e in await _events(engine, after=events_before)] == [ids[0]]


# ---------------------------------------------------------------------------
# Postgres real concurrency
# ---------------------------------------------------------------------------


def _require_pg(engine):
    if not _pg(engine):
        pytest.skip("interleaved sessions need a real server (in-memory SQLite shares one)")


@pytest.mark.parametrize("waiting", [False, True])
async def test_parallel_sweepers_finalise_each_row_exactly_once(store, engine, clock, waiting):
    _require_pg(engine)
    status = {"status": "waiting_client_tool"} if waiting else {}
    fresh = {"status_changed_at": clock.now} if waiting else {"last_progress_at": clock.now}
    await _seed(
        store,
        engine,
        clock,
        {RUN: status, SIBLING: status, OTHER: {**status, **fresh, "tenant": "t2"}},
    )
    events_before = await _events(engine)
    method = store.sweep_abandoned_runs if waiting else store.sweep_stale_runs
    results = await asyncio.gather(*[method(AGE) for _ in range(PARALLEL)], return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    assert sorted(s.run_id for r in results for s in r) == [RUN, SIBLING]
    rows = await _rows(engine)
    assert [(rows[r]["status"], rows[r]["claim_epoch"]) for r in (RUN, SIBLING, OTHER)] == [
        ("error", 4),
        ("error", 4),
        (rows[OTHER]["status"], 3),
    ]
    kind = "run.abandoned" if waiting else "run.interrupted"
    assert sorted(
        (e["agent_run_id"], e["event_type"]) for e in await _events(engine, after=events_before)
    ) == [(RUN, kind), (SIBLING, kind)]


async def test_parallel_public_sweeps_finalise_each_row_exactly_once(store, engine, clock):
    _require_pg(engine)
    run_a, run_b, wait_a, wait_b = _ids(1, 2, 3, 4)
    wait = {"status": "waiting_client_tool"}
    await _seed(store, engine, clock, {run_a: {}, run_b: {}, wait_a: wait, wait_b: wait})
    results = await asyncio.gather(
        *[
            sweep(state_store=store, stale_running=AGE, abandoned_waiting=AGE, on_stale="error")
            for _ in range(PARALLEL)
        ],
        return_exceptions=True,
    )
    assert not [r for r in results if isinstance(r, BaseException)], results
    assert sorted(s.run_id for r in results for s in r.stale_running) == [run_a, run_b]
    assert sorted(s.run_id for r in results for s in r.abandoned_waiting) == [wait_a, wait_b]
    rows = await _rows(engine)
    assert {r: (v["status"], v["claim_epoch"]) for r, v in rows.items()} == dict.fromkeys(
        (run_a, run_b, wait_a, wait_b), ("error", 4)
    )
    assert len(await _events(engine)) == 4 + 4


@pytest.mark.parametrize("round_", range(5))
async def test_heartbeat_and_stale_sweep_race_has_exactly_one_winner(store, engine, clock, round_):
    _require_pg(engine)
    await _seed(
        store, engine, clock, {RUN: {}, OTHER: {"tenant": "t2", "last_progress_at": clock.now}}
    )
    beat, swept = await asyncio.gather(
        store.touch_progress(RUN, claim_epoch=3), store.sweep_stale_runs(AGE)
    )
    rows = await _rows(engine)
    if beat:
        assert (swept, rows[RUN]["status"], rows[RUN]["claim_epoch"]) == ([], "running", 3)
    else:
        assert ([s.run_id for s in swept], rows[RUN]["status"], rows[RUN]["claim_epoch"]) == (
            [RUN],
            "error",
            4,
        )
    assert (rows[OTHER]["status"], rows[OTHER]["claim_epoch"]) == ("running", 3)


@pytest.mark.parametrize("round_", range(5))
async def test_resume_and_abandoned_sweep_race_has_exactly_one_winner(store, engine, clock, round_):
    _require_pg(engine)
    await _seed(store, engine, clock, {RUN: {"status": "waiting_client_tool"}})

    async def resume():
        async with engine.begin() as conn:
            result = await conn.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == RUN,
                    AgentRun.status == "waiting_client_tool",
                    AgentRun.claim_epoch == 3,
                )
                .values(status="running", claim_epoch=4)
            )
            return result.rowcount

    resumed, swept = await asyncio.gather(resume(), store.sweep_abandoned_runs(AGE))
    row = (await _rows(engine))[RUN]
    assert resumed + len(swept) == 1
    assert (row["status"], row["claim_epoch"]) == (("running", 4) if resumed else ("error", 4))
