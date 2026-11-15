"""Pause generations retain CAS predicates, transaction cuts and bounded reads."""

import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import DBAPIError

import dendrux.runtime.state as state
from dendrux.db.models import AgentRun
from dendrux.errors import PauseStatusMismatchError
from dendrux.types import UsageStats
from tests.integration.test_sql_lifecycle_characterization import (
    RUN,
    _pin,
    _snapshot,
    _write_trace,
)

PAUSE = {"pause_id": "generation-one", "pause_reason": "requested", "pending": ["c1"]}


async def _seed(store, engine, **values):
    await store.create_run(RUN, "pause", tenant_id="t1")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == RUN)
            .values(**dict(dict(claim_epoch=7, pause_requested=True, meta={"keep": 1}), **values))
        )


@pytest.mark.parametrize("mode", ["reason", "clear", "absent", "not-running"])
async def test_request_locks_before_reason_and_preserves_metadata(
    store, engine, record_property, mode
):
    await _seed(
        store,
        engine,
        status="paused" if mode == "not-running" else "running",
        meta={
            "keep": 1,
            **({state.PAUSE_REQUEST_META_KEY: {"reason": "old"}} if mode == "clear" else {}),
        },
    )
    before = await _snapshot(engine)
    reason = "operator" if mode == "reason" else None
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        won = await store.request_pause(RUN, reason=reason)
    assert won is (mode != "not-running")
    assert len(sql) == (1 if not won else 2 if mode == "absent" else 3)
    assert sql[0].startswith("UPDATE agent_runs SET pause_requested=")
    assert "agent_runs.status =" in sql[0].split(" WHERE ")[-1]
    if won:
        assert sql[1].startswith("SELECT agent_runs.meta")
        if mode != "absent":
            assert sql[2].startswith("UPDATE agent_runs SET meta=")
    _pin(record_property, sql, boundaries, connections, commit=won)
    after = await _snapshot(engine)
    if won:
        assert after[0]["pause_requested"]
        assert after[0]["meta"] == {
            "keep": 1,
            **({state.PAUSE_REQUEST_META_KEY: {"reason": reason}} if mode != "absent" else {}),
        }
    else:
        assert before == after


@pytest.mark.parametrize("operation", ["try_pause_run", "pause_with_event"])
@pytest.mark.parametrize("miss", [None, "epoch", "status", "cancel"])
async def test_pause_cas_preserves_generation_cancel_guards_and_event_order(
    store, engine, monkeypatch, record_property, operation, miss
):
    await _seed(
        store,
        engine,
        status="paused" if miss == "status" else "running",
        cancel_requested=miss == "cancel",
    )
    before = await _snapshot(engine)
    phases, sessions = [], []
    original_event, original_commit = state._insert_event_locked, store._write_tx

    async def insert(session, *args, **kwargs):
        phases.append("event")
        sessions.append(session)
        return await original_event(session, *args, **kwargs)

    async def commit(session):
        phases.append("commit")
        sessions.append(session)
        return await original_commit(session)

    monkeypatch.setattr(state, "_insert_event_locked", insert)
    monkeypatch.setattr(store, "_write_tx", commit)
    options = dict(
        status="paused",
        pause_data=PAUSE,
        iteration_count=4,
        pii_mapping={"p": "v"},
        total_usage=UsageStats(input_tokens=5, output_tokens=3),
        claim_epoch=6 if miss == "epoch" else 7,
    )
    if operation == "pause_with_event":
        options["event_data"] = {"pause_id": "generation-one"}
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        result = await getattr(store, operation)(RUN, **options)
    won = miss is None
    if operation == "pause_with_event":
        assert result == (0 if won else None)
    else:
        assert result is won
    assert len(sql) == (2 if won and operation == "pause_with_event" else 1)
    assert sql[0].startswith("UPDATE agent_runs")
    where = sql[0].split(" WHERE ")[-1]
    assert all(
        token in where
        for token in [
            "agent_runs.claim_epoch =",
            "agent_runs.status =",
            "agent_runs.cancel_requested IS",
        ]
    )
    if won and operation == "pause_with_event":
        assert sql[1].startswith("INSERT INTO run_events")
    _pin(record_property, sql, boundaries, connections, commit=operation == "try_pause_run" or won)
    assert phases == (
        ["event", "commit"]
        if won and operation == "pause_with_event"
        else ["commit"]
        if operation == "try_pause_run"
        else []
    )
    assert not sessions or all(s is sessions[0] for s in sessions)
    after = await _snapshot(engine)
    if won:
        row = after[0]
        assert row["status"] == "paused" and row["pause_data"] == PAUSE
        assert not row["pause_requested"] and row["claim_epoch"] == 7
        assert row["iteration_count"] == 4 and row["pii_mapping"] == {"p": "v"}
        assert (row["total_input_tokens"], row["total_output_tokens"]) == (5, 3)
        events = await store.get_run_events(RUN)
        assert [e.event_type for e in events] == (
            ["run.paused"] if operation == "pause_with_event" else []
        )
    else:
        assert before == after


@pytest.mark.parametrize("mode", ["unfenced", "current", "stale", "not-running"])
async def test_legacy_pause_keeps_optional_epoch_guard_and_pending_document(
    store, engine, record_property, mode
):
    await _seed(
        store,
        engine,
        status="success" if mode in ("unfenced", "not-running") else "running",
        cancel_requested=True,
    )
    before = await _snapshot(engine)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        await store.pause_run(
            RUN,
            status="paused",
            pause_data=PAUSE,
            claim_epoch=None if mode == "unfenced" else 6 if mode == "stale" else 7,
            iteration_count=4,
            pii_mapping={"p": "v"},
            total_usage=UsageStats(input_tokens=5, output_tokens=3),
        )
    assert len(sql) == 1
    where = sql[0].split(" WHERE ")[-1]
    assert ("claim_epoch" in where) is (mode != "unfenced")
    assert ("agent_runs.status =" in where) is (mode != "unfenced")
    assert "cancel_requested" not in where
    _pin(record_property, sql, boundaries, connections)
    after = await _snapshot(engine)
    if mode in ("unfenced", "current"):
        assert after[0]["pause_data"] == PAUSE and after[0]["status"] == "paused"
        assert after[0]["pause_requested"] and after[0]["cancel_requested"]
        assert after[0]["iteration_count"] == 4 and after[0]["pii_mapping"] == {"p": "v"}
        assert (after[0]["total_input_tokens"], after[0]["total_output_tokens"]) == (5, 3)
    else:
        assert before == after


@pytest.mark.parametrize("fault", ["event", "commit"])
async def test_pause_fault_rolls_back_cas_and_real_event_insert(
    store, engine, monkeypatch, record_property, fault
):
    await _seed(store, engine)
    before = await _snapshot(engine)
    fired = []
    original = state._insert_event_locked

    async def insert(session, *args, **kwargs):
        result = await original(session, *args, **kwargs)
        fired.append("event")
        if fault == "event":
            raise ValueError("pause cut")
        return result

    async def commit(session):
        fired.append("commit")
        raise ValueError("pause cut")

    monkeypatch.setattr(state, "_insert_event_locked", insert)
    if fault == "commit":
        monkeypatch.setattr(store, "_write_tx", commit)
    with (
        _write_trace(engine, record_property) as (sql, boundaries, connections),
        pytest.raises(ValueError, match="pause cut"),
    ):
        await store.pause_with_event(RUN, status="paused", pause_data=PAUSE, claim_epoch=7)
    assert fired == (["event", "commit"] if fault == "commit" else ["event"])
    assert len(sql) == 2 and sql[1].startswith("INSERT INTO run_events")
    _pin(record_property, sql, boundaries, connections, commit=False)
    assert await _snapshot(engine) == before


@pytest.mark.parametrize(
    "suffix,active",
    [
        ((), True),
        (("run.resumed",), False),
        (("run.resumed", "run.resume_aborted"), True),
        (("run.resumed", "run.resume_aborted", "run.resumed"), False),
        (("run.cancelled",), False),
        (("run.paused",), True),
    ],
)
async def test_active_generation_uses_abort_pairing_and_one_scoped_query(
    store, engine, record_property, suffix, active
):
    await _seed(store, engine, status="paused")
    for i, kind in enumerate(("run.paused", *suffix)):
        await store.save_run_event(RUN, event_type=kind, event_id=f"{i + 10:026d}", data={})
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        sequence = await store.get_active_pause_sequence(RUN, tenant_id="t1")
    assert (
        sequence == (len(suffix) if suffix == ("run.paused",) else 0)
        if active
        else sequence is None
    )
    assert len(sql) == 1 and "max(run_events.sequence_index)" in sql[0]
    assert "agent_runs.tenant_id =" in sql[0] and "agent_runs.status IN" in sql[0]
    assert "FOR UPDATE" not in sql[0]
    _pin(record_property, sql, boundaries, connections, commit=False)
    assert await store.get_active_pause_sequence(RUN, tenant_id="t2") is None


async def test_resume_generation_rejects_stale_owner_and_preserves_notes(store, engine):
    await _seed(store, engine)
    assert await store.add_note(RUN, note_id="note", text="remember", tenant_id="t1", claim_epoch=7)
    assert await store.pause_with_event(RUN, status="paused", pause_data=PAUSE, claim_epoch=7) == 0
    with pytest.raises(PauseStatusMismatchError):
        await store.claim_paused_run(RUN, expected_status="paused", expected_pause_id="wrong")
    assert (await store.get_run(RUN)).status == "paused"
    claim = await store.claim_paused_run(
        RUN, expected_status="paused", expected_pause_id="generation-one"
    )
    assert claim.claim_epoch == 8
    second = dict(PAUSE, pause_id="generation-two", pending=["c2"])
    assert (
        await store.pause_with_event(RUN, status="paused", pause_data=second, claim_epoch=7) is None
    )
    assert await store.pause_with_event(RUN, status="paused", pause_data=second, claim_epoch=8) == 1
    assert await store.get_pause_state(RUN) == second
    assert await store.get_active_pause_sequence(RUN, tenant_id="t1") == 1
    assert [n.text for n in await store.get_notes(RUN, tenant_id="t1")] == ["remember"]
    assert (await store.get_run_control(RUN)).has_pending_notes


@pytest.mark.parametrize("pending", [False, True])
async def test_summaries_keep_projection_scope_and_pending_payload(
    store, engine, record_property, pending
):
    await _seed(
        store,
        engine,
        status="paused",
        pause_data={**PAUSE, "messages": ["large-history"], "last_step": {"x": 1}},
    )
    await store.save_run_event(RUN, event_type="run.paused", data={})
    read = store.get_pending_pause_summary if pending else store.get_pause_summary
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        summary = await read(RUN, tenant_id="t1")
    assert summary.status == "paused" and summary.paused_event_sequence == 0
    assert (
        summary.pause["pause_id"] == "generation-one"
        and summary.pause["pause_reason"] == "requested"
    )
    assert "messages" not in summary.pause
    assert len(sql) == 1 and "agent_runs.tenant_id =" in sql[0]
    _pin(record_property, sql, boundaries, connections, commit=False)
    assert await read(RUN, tenant_id="t2") is None
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        assert (await store.get_pause_state(RUN))["pending"] == ["c1"]
    assert len(sql) == 1 and sql[0].startswith("SELECT agent_runs.pause_data")


async def test_approval_interval_keeps_latest_args_and_query_bound(store, engine, record_property):
    await _seed(store, engine, status="waiting_approval")
    events = [
        ("approval.requested", {"call_id": "old", "params": {"x": 0}}),
        ("run.paused", {}),
        ("run.resumed", {}),
        ("approval.requested", {"call_id": "c", "params": {"x": 1}}),
        ("approval.requested", {"call_id": "c", "params": {"x": 2}}),
        ("run.paused", {}),
        ("approval.requested", {"call_id": "future", "params": {"x": 3}}),
    ]
    for kind, data in events:
        await store.save_run_event(RUN, event_type=kind, data=data)
    with _write_trace(engine, record_property) as (sql, boundaries, connections):
        assert await store.get_approval_params(RUN, 5, tenant_id="t1") == {"c": {"x": 2}}
    assert (
        len(sql) == 1 and "ORDER BY run_events.sequence_index DESC" in sql[0] and "LIMIT" in sql[0]
    )
    assert "run_events.sequence_index >" in sql[0] and "run_events.sequence_index <" in sql[0]
    _pin(record_property, sql, boundaries, connections, commit=False)
    assert await store.get_approval_params(RUN, 5, tenant_id="t2") == {}


async def test_summary_propagates_database_fault_outside_json_fallback(
    store, engine, monkeypatch, record_property
):
    await _seed(store, engine)
    classified = []
    original = store.json_conversion_error

    def classify(error):
        classified.append(error)
        return original(error)

    monkeypatch.setattr(store, "json_conversion_error", classify)
    monkeypatch.setattr(
        store,
        "pause_summary_query",
        lambda *args, **kwargs: text(
            "SELECT pause_characterization_missing_column FROM agent_runs"
        ),
    )
    with (
        _write_trace(engine, record_property) as (sql, boundaries, connections),
        pytest.raises(DBAPIError) as caught,
    ):
        await store._read_pause_summary(RUN, "t1", pending=True)
    assert classified == [caught.value] and not original(caught.value)
    assert len(sql) == 1 and sql[0].startswith("SELECT pause_characterization_missing_column")
    _pin(record_property, sql, boundaries, connections, commit=False)
