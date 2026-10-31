"""Regression reproductions for the independent phase-2a review."""

from datetime import timedelta
from functools import wraps
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update

from dendrux import sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotReclaimableError, TenantScopeRequiredError
from dendrux.runtime.state import EventWrite, StoreCapabilities
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import Limits, PauseState, ToolCall, generate_ulid
from tests.integration.test_060_reclaim_agent import checkpoint, make_agent
from tests.integration.test_060_reclaim_sweep import candidate


async def resume(agent, run, pause, entry, **kwargs):
    if entry == "stream":
        async with agent.resume_stream(
            run, tenant_id="tenant", expected_pause_id=pause, **kwargs
        ) as stream:
            events = [e async for e in stream]
        return stream.result, events
    return await agent.resume(run, tenant_id="tenant", expected_pause_id=pause, **kwargs), []


@pytest.mark.parametrize("entry", ["batch", "stream"])
@pytest.mark.parametrize("gate", ["static", "hook"])
async def test_p1_started_idempotent_gate_survives_pause(store, engine, entry, gate):
    from dendrux.types import ToolDecision

    effects = []

    @tool(idempotent=True)
    async def work() -> str:
        """Visible effect."""
        effects.append(1)
        return "ok"

    async def before(context):
        return ToolDecision.require_approval("review")

    agent = make_agent(
        store,
        tools=[work],
        **({"require_approval": ["work"]} if gate == "static" else {"before_tool": before}),
    )
    call = ToolCall("work")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=True, idempotent=True
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "waiting_approval"
    raw = await store.get_pause_state(item.run_id)
    assert raw["approval_gates"][call.id]["source"] == "recovery"
    result, _ = await resume(agent, item.run_id, paused.pause_id, entry)
    assert result.status.value == "success"
    assert effects == [1]
    rows = await store.get_tool_calls(item.run_id)
    assert len(rows) == 1 and rows[0].replayed_at is not None


@pytest.mark.parametrize("entry", ["batch", "stream"])
@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.parametrize("strip_gate", [False, True])
async def test_p1_barrier_and_old_pause_without_unique_constraint(
    store, engine, monkeypatch, entry, idempotent, strip_gate
):
    effects, duplicate_intents = [], []

    @tool(idempotent=idempotent)
    async def work() -> str:
        """Visible effect."""
        effects.append(1)
        return "ok"

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[call],
        phases=["dispatchable"],
        started=True,
        idempotent=idempotent,
    )
    original = store.reclaim_run

    @wraps(original)
    async def claim(*a, **kw):
        result = await original(*a, **kw)
        await store.request_pause(item.run_id)
        return result

    monkeypatch.setattr(store, "reclaim_run", claim)
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status.value == "paused"
    raw = await store.get_pause_state(item.run_id)
    if not strip_gate:
        assert raw["approval_gates"][call.id]["source"] == "recovery"
    else:
        raw["approval_gates"] = None
        async with engine.begin() as conn:
            await conn.execute(
                update(AgentRun).where(AgentRun.id == item.run_id).values(pause_data=raw)
            )

    # This double accepts another intent: uniqueness must not provide safety.
    original_start = store.start_tool_call

    @wraps(original_start)
    async def permit_duplicate(*a, **kw):
        duplicate_intents.append(kw["tool_call_id"])
        return kw["row_id"]

    monkeypatch.setattr(store, "start_tool_call", permit_duplicate)
    result, _ = await resume(agent, item.run_id, paused.pause_id, entry)
    assert not duplicate_intents
    assert result.status.value == ("success" if idempotent else "waiting_approval")
    assert effects == ([1] if idempotent else [])


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("entry", ["submit", "stream"])
async def test_p1_mixed_recovery_decision_audit_and_reason(store, engine, approved, entry):
    from dendrux.runtime.submit import build_rejection_results

    @tool()
    async def work() -> str:
        """Interrupted effect."""
        return "ok"

    @tool()
    async def gated() -> str:
        """New gated effect."""
        return "ok"

    agent = make_agent(store, tools=[work, gated], require_approval=["gated"])
    a, b = ToolCall("work"), ToolCall("gated")
    item = await checkpoint(agent, store, engine, calls=[a, b], phases=["dispatchable", "gated"])
    await store.start_tool_call(
        item.run_id,
        claim_epoch=item.claim_epoch,
        row_id=generate_ulid(),
        tool_call_id=a.id,
        provider_tool_call_id=None,
        tool_name=a.name,
        target="server",
        params=None,
        iteration_index=1,
        idempotent=False,
        event=EventWrite(generate_ulid(), "tool.started", 1, {}, a.id, None),
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    from dendrux.loops.base import BaseNotifier

    notifier = BaseNotifier()
    notifier.on_governance_event = AsyncMock()
    if entry == "submit":
        result = await agent.submit_approval(
            item.run_id,
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            approved=approved,
            rejection_reason="Too risky",
            notifier=notifier,
        )
    else:
        raw = await store.get_pause_state(item.run_id)
        results = (
            None if approved else build_rejection_results(PauseState.from_dict(raw), "Too risky")
        )
        result, _ = await resume(
            agent, item.run_id, paused.pause_id, "stream", tool_results=results, notifier=notifier
        )
    assert result.status.value == "success"
    events = await store.get_run_events(item.run_id)
    decided = [e for e in events if e.event_type == "approval.decided"]
    assert len(decided) == 1
    assert decided[0].data["decision"] == ("approved" if approved else "rejected")
    assert (
        len(
            [
                c
                for c in notifier.on_governance_event.call_args_list
                if c.args[1] == "approval.decided"
            ]
        )
        == 1
    )
    if not approved:
        messages = agent.provider.wire_messages[0]
        from dendrux.runtime.submit import DEFAULT_DENIAL_REASON

        reason = "Too risky" if entry == "submit" else DEFAULT_DENIAL_REASON
        assert reason in next(m.content for m in messages if m.call_id == b.id)
        assert "outcome_unknown" in next(m.content for m in messages if m.call_id == a.id)


@pytest.mark.parametrize("policy", [None, "error", "reclaim"])
async def test_p2_sweep_cancel_payload_and_unknown_log(store, engine, caplog, policy):
    run = await candidate(store, engine, marker=None, cancel_requested=True)
    await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale=policy)
    event = next(e for e in await store.get_run_events(run) if e.event_type == "run.cancelled")
    assert event.data["failure_reason"] == "stale_running"
    assert any(getattr(r, "code", None) == "loop_unknown" for r in caplog.records) == (
        policy == "reclaim"
    )


@pytest.mark.parametrize("policy", ["error", "reclaim"])
async def test_p2_sweep_respects_subclass_public_hooks(engine, policy):
    called = []

    class Custom(type(store_for_engine(engine))):
        async def sweep_stale_runs(self, **kwargs):
            called.append("error")
            return []

        async def mark_lease_expired(self, *, finalise_unrecoverable=False, **kwargs):
            called.append("reclaim")
            assert finalise_unrecoverable
            return await super().mark_lease_expired(
                finalise_unrecoverable=finalise_unrecoverable, **kwargs
            )

    store = Custom(engine)
    await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale=policy)
    assert called == [policy]


async def test_p2_custom_store_finalises_unrecoverable(store, engine):
    class Custom:
        capabilities = StoreCapabilities(recovery=True)

        async def get_lease_expired(self, **kwargs):
            return await store.get_lease_expired(**kwargs)

        async def reclaim_run(self, *args, **kwargs):
            return await store.reclaim_run(*args, **kwargs)

        async def mark_lease_expired(self, *, finalise_unrecoverable=False, **kwargs):
            assert finalise_unrecoverable
            return await store.mark_lease_expired(
                finalise_unrecoverable=finalise_unrecoverable, **kwargs
            )

    ids = [await candidate(store, engine, marker=marker) for marker in (None, "single_call")]
    page = await sweep(state_store=Custom(), stale_running=timedelta(minutes=1), on_stale="reclaim")
    assert [r.run_id for r in page.stale_running] == ids


@pytest.mark.parametrize(
    "kind", ["history_too_large", "recovery_call_missing", "recovery_state_missing"]
)
async def test_p2_rebuild_refusal_finalises_once(store, engine, kind):
    agent = make_agent(
        store, **({"limits": Limits(max_rebuild_rows=1)} if kind == "history_too_large" else {})
    )
    item = await checkpoint(agent, store, engine)
    state = await store.get_loop_state(item.run_id)
    if kind == "recovery_call_missing":
        state["pending"] = [{"call_id": "missing"}]
    if kind == "recovery_state_missing":
        state["image_view_ref"] = "missing"
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == item.run_id).values(loop_state=state)
        )
    with pytest.raises(RunNotReclaimableError) as exc:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert exc.value.reason == kind
    row = await store.get_run(item.run_id)
    assert row.status == "error" and row.failure_reason == kind and row.reclaim_attempts == 1


async def test_p2_current_idempotency_must_allow_replay(store, engine):
    effects = []

    @tool(idempotent=False)
    async def work() -> str:
        """No longer safe to repeat."""
        effects.append(1)
        return "ok"

    agent = make_agent(store, tools=[work])
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[ToolCall("work")],
        phases=["dispatchable"],
        started=True,
        idempotent=True,
    )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_approval" and effects == []


@pytest.mark.parametrize("strict", [False, True])
async def test_p2_missing_tenant_always_requires_scope(store, strict, monkeypatch):
    agent = make_agent(store)
    agent._strict_tenancy = strict
    io = AsyncMock(side_effect=AssertionError("tenant validation must precede I/O"))
    monkeypatch.setattr(agent, "_resolve_state_store", io)
    with pytest.raises(TenantScopeRequiredError):
        await agent.reclaim("00000000000000000000000403")
    io.assert_not_called()


async def test_p2_stream_recovery_cancel_event(store, engine, monkeypatch):
    from dendrux.runtime import _reclaim
    from dendrux.types import RunEventType

    @tool()
    async def work() -> str:
        """Interrupted effect."""
        raise AssertionError("cancelled")

    agent = make_agent(store, tools=[work])
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("work")], phases=["dispatchable"], started=True
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    original = _reclaim.resolve

    async def cancel(agent, owned, run, ctx, **kwargs):
        await store.request_cancel(run)
        return await original(agent, owned, run, ctx, **kwargs)

    monkeypatch.setattr(_reclaim, "resolve", cancel)
    result, events = await resume(agent, item.run_id, paused.pause_id, "stream")
    assert result.status.value == "cancelled"
    assert events[-1].type == RunEventType.RUN_CANCELLED


@pytest.mark.parametrize("payload", ["plain é text", '{ "text": "é" }'])
async def test_p2_observe_preserves_original_text(store, engine, monkeypatch, payload):
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, claimed_store
    from dendrux.types import ToolResult

    @tool()
    async def work() -> str:
        """Must not run."""
        raise AssertionError("already resolved")

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["invalid"])
    state = await store.get_loop_state(item.run_id)
    recorder = PersistenceRecorder(
        claimed_store(store, item.run_id, ClaimResult(item.claim_epoch, 2, state, None)),
        item.run_id,
    )
    # Simulate the cut after save_tool_call, before the observation checkpoint.
    original = store.save_run_event

    @wraps(original)
    async def stop(*a, **kw):
        raise RuntimeError("cut")

    monkeypatch.setattr(store, "save_run_event", stop)
    with pytest.raises(RuntimeError):
        await recorder.on_tool_completed(
            item.run_id,
            call,
            ToolResult(name=call.name, call_id=call.id, payload=payload, success=False),
            1,
        )
    monkeypatch.setattr(store, "save_run_event", original)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    message = next(m for m in agent.provider.wire_messages[0] if m.call_id == call.id)
    assert message.content == payload


async def test_p2_reclaim_does_not_reread_pause_or_rewrite_meta(store, engine, monkeypatch):
    from sqlalchemy import event

    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert sum(s.startswith("SELECT agent_runs.pause_data") for s in statements) == 1
    assert sum(s.startswith("SELECT agent_runs.meta ") for s in statements) <= 1


@pytest.mark.parametrize("policy", [None, "error"])
async def test_p2_compatibility_scan_has_no_unknown_loop_warning(store, engine, caplog, policy):
    await candidate(store, engine, marker=None)
    await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale=policy)
    assert not any(getattr(r, "code", None) == "loop_unknown" for r in caplog.records)
