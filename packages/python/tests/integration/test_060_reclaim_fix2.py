"""Final recovery review regressions and bounded resume query measurements."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import event, update

from dendrux import sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotReclaimableError, StoreCapabilityError
from dendrux.runtime._reclaim import needs_resolution
from dendrux.runtime.state import StoreCapabilities
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.stores.sql.base import SQLStateStore
from dendrux.store import RunStore
from dendrux.types import Limits, PauseState, ToolCall, UsageStats
from tests.integration.test_060_reclaim_agent import checkpoint, make_agent
from tests.integration.test_060_reclaim_sweep import candidate


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("policy", [None, "error", "reclaim"])
@pytest.mark.parametrize("has_threshold", [False, True])
async def test_n1_sweep_validates_recovery_before_calling_store(
    engine, native, policy, has_threshold
):
    calls = []

    class OldMarker(type(store_for_engine(engine)) if native else object):
        capabilities = SQLStateStore.capabilities if native else StoreCapabilities(recovery=True)

        async def mark_lease_expired(
            self, *, cutoff, batch_size, max_runs, max_returned, reclaim_limit, cursor
        ):
            calls.append("mark")
            raise AssertionError("invalid marker reached")

        async def get_lease_expired(self, *, limit, cursor):
            raise AssertionError("unexpected I/O")

        async def reclaim_run(self, **kwargs):
            raise AssertionError("unexpected I/O")

        async def sweep_stale_runs(self, *, older_than):
            calls.append("error")
            return []

    store = OldMarker(engine) if native else OldMarker()
    kwargs = {"stale_running": timedelta(minutes=1)} if has_threshold else {}
    if policy == "reclaim":
        with pytest.raises(StoreCapabilityError) as exc:
            await sweep(state_store=store, on_stale=policy, **kwargs)
        assert exc.value.capability == "recovery"
        assert exc.value.method == "mark_lease_expired"
        assert exc.value.missing == ("finalise_unrecoverable",)
        assert calls == []
    else:
        await sweep(state_store=store, on_stale=policy, **kwargs)
        assert calls == (["error"] if has_threshold else [])


@pytest.mark.parametrize("finalise", [False, True])
async def test_n2_raw_marker_logs_poisoned_row(store, engine, caplog, finalise):
    from datetime import UTC, datetime

    run = await candidate(store, engine, marker=None)
    page = await store.mark_lease_expired(
        cutoff=datetime.now(UTC) - timedelta(minutes=1),
        batch_size=10,
        max_runs=10,
        max_returned=10,
        reclaim_limit=3,
        cursor=None,
        finalise_unrecoverable=finalise,
    )
    assert any(
        getattr(r, "code", None) == "loop_unknown" and getattr(r, "run_id", None) == run
        for r in caplog.records
    )
    assert [r.run_id for r in page.finalised] == ([run] if finalise else [])
    assert (await store.get_run(run)).status == ("error" if finalise else "running")


@pytest.mark.parametrize(
    "reason",
    [
        "history_too_large",
        "recovery_call_missing",
        "recovery_state_missing",
        "recovery_state_invalid",
    ],
)
async def test_n3_redacted_rebuild_failure_exposes_only_closed_reason(store, engine, reason):
    agent = make_agent(
        store, **({"limits": Limits(max_rebuild_rows=1)} if reason == "history_too_large" else {})
    )
    item = await checkpoint(agent, store, engine)
    state = await store.get_loop_state(item.run_id)
    if reason == "recovery_call_missing":
        state["pending"] = [{"call_id": "missing"}]
    elif reason == "recovery_state_missing":
        state["image_view_ref"] = "missing"
    elif reason == "recovery_state_invalid":
        state["pending"] = [None]
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == item.run_id).values(loop_state=state)
        )
    with pytest.raises(RunNotReclaimableError) as exc:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert exc.value.reason == reason
    reader = RunStore.from_engine(engine)
    raw = next(
        e
        for e in await reader.get_events(item.run_id, tenant_id="tenant")
        if e.event_type == "run.error"
    )
    view = await reader.get_event_view(item.run_id, raw, tenant_id="tenant")
    assert view.data == {"failure_reason": reason}
    # Unknown strings and vendor payloads never become a safe failure reason.
    raw = replace(raw, data={"failure_reason": "private-sentinel", "error": "private-sentinel"})
    view = await reader.get_event_view(item.run_id, raw, tenant_id="tenant")
    assert "private-sentinel" not in json.dumps(view.data)


@pytest.mark.parametrize("outcome", ["started", "completed", "failed", "unknown"])
async def test_n4_intent_check_only_routes_started_calls(store, engine, outcome):
    call = ToolCall("work")
    agent = make_agent(store)
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=True
    )
    from dendrux.db.models import ToolCallRecord

    async with engine.begin() as conn:
        await conn.execute(
            update(ToolCallRecord)
            .where(ToolCallRecord.agent_run_id == item.run_id)
            .values(outcome=outcome)
        )
    pause = PauseState(
        agent_name=agent.name,
        pending_tool_calls=[call],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=4,
        usage=UsageStats(),
        schema_version=2,
    )
    assert await needs_resolution(store, item.run_id, pause) == (outcome == "started")


@pytest.mark.parametrize("pending", [False, True])
async def test_n4_resume_query_counts(store, engine, pending, record_property):
    statements = []
    measured = {}

    @tool()
    async def work() -> str:
        """One normal pending call."""
        return "ok"

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[call] if pending else [],
        phases=["dispatchable"] if pending else [],
    )
    state = await store.get_loop_state(item.run_id)
    pause = PauseState(
        agent_name=agent.name,
        pending_tool_calls=[call] if pending else [],
        history=[],
        steps=[],
        iteration=state["iteration"],
        trace_order_offset=state["trace_order_offset"],
        usage=UsageStats(),
        schema_version=2,
        pause_id="resume-measure",
    )
    await store.pause_run(item.run_id, status="paused", pause_data=pause.to_dict())
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == item.run_id).values(lease_expired_at=None)
        )
    original = agent.provider.complete

    async def complete(*args, **kwargs):
        measured["before_first_model"] = len(statements)
        measured["intent_selects"] = sum(s.startswith("SELECT tool_calls.") for s in statements)
        return await original(*args, **kwargs)

    agent.provider.complete = complete

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await agent.resume(
            item.run_id, tenant_id="tenant", expected_pause_id=pause.pause_id
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert result.status.value == "success"
    assert measured["intent_selects"] == int(pending)
    measured["total_resume"] = len(statements)
    record_property("resume_query_counts", json.dumps(measured, sort_keys=True))
