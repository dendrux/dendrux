"""Pause visibility, cancellation precedence and durable continuation boundaries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux import tool
from dendrux.errors import RunFencedError
from dendrux.runtime import runner
from dendrux.runtime._pending import _PendingReader
from dendrux.runtime.persistence import PersistenceRecorder
from dendrux.runtime.state import EventWrite, claimed_store
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, generate_ulid
from tests._fixtures.storage import pin_storage_format
from tests._helpers.gates import wait_for_gate
from tests.integration.test_060_reclaim_agent import checkpoint
from tests.integration.test_cross_path_invariants import make_agent, submission_case


@pytest.mark.parametrize("view", ["pending", "history", "follow", "custom"])
async def test_aborted_resume_keeps_approval_pause_active(store, engine, monkeypatch, view):
    _, run_id, invoke = await submission_case(store, "approve")
    reader = RunStore.from_engine(engine)
    original_sequence = await store.get_active_pause_sequence(run_id, tenant_id="tenant")
    failure = RuntimeError("setup_failure")
    try:
        # Repeated attempts must still pair the real resume with the original pause.
        for _ in range(2):
            with monkeypatch.context() as patcher:
                patcher.setattr(runner, "_start_trace", AsyncMock(side_effect=failure))
                with pytest.raises(RuntimeError) as caught:
                    await invoke()
                assert caught.value is failure
            if view == "pending":
                pending = await reader.get_pending(run_id, tenant_id="tenant")
                assert pending.tool_calls[0].params == {"order_id": 7}
                assert not pending.tool_calls[0].args_hidden
            elif view == "history":
                pairs, truncated = await reader.get_pause_page(run_id, tenant_id="tenant")
                assert not truncated and len(pairs) == 1
                assert pairs[0].resume_sequence_index is None
                assert pairs[0].pending_tool_calls[0]["params"] == {"order_id": 7}
            elif view == "custom":
                legacy = SimpleNamespace(
                    get_run_status=store.get_run_status, get_run_events=store.get_run_events
                )
                assert (
                    await _PendingReader(legacy)._active_pause_sequence(run_id, "tenant")
                    == original_sequence
                )
            else:
                feed = reader.follow(run_id, tenant_id="tenant", until_terminal=False)
                try:
                    async for event in feed:
                        if event.event_type == "run.paused":
                            assert event.data["pending_tool_calls"][0]["params"] == {"order_id": 7}
                            break
                finally:
                    await feed.aclose()
        assert (await invoke()).status == RunStatus.SUCCESS
        assert await store.get_active_pause_sequence(run_id, tenant_id="tenant") is None
        pairs = await reader.get_pauses(run_id, tenant_id="tenant")
        assert len(pairs) == 1
        resumed = [e for e in await store.get_run_events(run_id) if e.event_type == "run.resumed"]
        assert pairs[0].resume_sequence_index == resumed[-1].sequence_index
    finally:
        await reader.close()


@pytest.mark.parametrize("kind", ["input", "approve"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_cancel_during_claimed_setup_wins_release(
    store, engine, monkeypatch, kind, storage_format
):
    await pin_storage_format(engine, storage_format)
    _, run_id, invoke = await submission_case(store, kind)
    entered, fail_now = asyncio.Event(), asyncio.Event()
    original = RuntimeError("setup_failure")

    async def fail_setup(*args, **kwargs):
        entered.set()
        await fail_now.wait()
        raise original

    monkeypatch.setattr(store, "get_pii_mapping", fail_setup)
    task = asyncio.create_task(invoke())
    try:
        await wait_for_gate(entered, task=task)
        # A separate Agent has no local task to interrupt. Its cancel commits
        # through another DB session while the claimed submit is suspended.
        remote = make_agent(store, [])
        await remote.cancel_run(run_id, tenant_id="tenant")
        assert (await store.get_run(run_id)).status == "running"
        fail_now.set()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(task, 10)
        assert caught.value is original
        row = await store.get_run(run_id)
        assert row.status == "cancelled"
        assert await store.get_pause_state(run_id) is None
        control = await store.get_run_control(run_id)
        assert not control.cancel_requested
        events = await store.get_run_events(run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1
        assert not any(e.event_type == "run.resume_aborted" for e in events)
        reader = RunStore.from_engine(engine)
        try:
            assert await reader.get_pending(run_id, tenant_id="tenant") is None
        finally:
            await reader.close()
    finally:
        fail_now.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_concurrent_cancel_release_has_one_terminal_event(store, engine):
    if engine.dialect.name not in {"postgresql", "mysql"}:
        pytest.skip("independent connection race requires a native row-locking database")
    _, run_id, _ = await submission_case(store, "input")
    token = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    await store.request_cancel(run_id)
    ready = asyncio.Barrier(4)

    async def release():
        await ready.wait()
        return await store.release_run_claim(
            run_id,
            status="waiting_human_input",
            claim_epoch=token.claim_epoch,
            event=EventWrite(
                generate_ulid(),
                "run.resume_aborted",
                0,
                {"failure_kind": "setup_error"},
                None,
                None,
            ),
        )

    assert sum(await asyncio.gather(*(release() for _ in range(4)))) == 1
    assert (await store.get_run(run_id)).status == "cancelled"
    types = [e.event_type for e in await store.get_run_events(run_id)]
    assert types.count("run.cancelled") == 1 and "run.resume_aborted" not in types


@pytest.mark.parametrize("started_sibling", [False, True])
async def test_recovery_barrier_keeps_static_and_started_gates(
    store, engine, monkeypatch, started_sibling
):
    @tool()
    async def work() -> str:
        """Cannot execute without human approval."""
        raise AssertionError("executed")

    agent = make_agent(store, [LLMResponse(text="done")], tools=[work], require_approval=["work"])
    calls = [ToolCall("work"), ToolCall("work")]
    item = await checkpoint(agent, store, engine, calls=calls, phases=["dispatchable"] * 2)
    if started_sibling:
        await store.start_tool_call(
            item.run_id,
            claim_epoch=item.claim_epoch,
            row_id=generate_ulid(),
            tool_call_id=calls[1].id,
            provider_tool_call_id=None,
            tool_name="work",
            target="server",
            params=None,
            iteration_index=1,
            idempotent=False,
            event=EventWrite(
                generate_ulid(), "tool.started", 1, {"call_id": calls[1].id}, calls[1].id, None
            ),
        )
    original = PersistenceRecorder.on_governance_event

    async def hold_then_pause(self, run_id, event_type, iteration, data, **kwargs):
        await original(self, run_id, event_type, iteration, data, **kwargs)
        if event_type == "approval.requested":
            await store.request_pause(run_id)

    monkeypatch.setattr(PersistenceRecorder, "on_governance_event", hold_then_pause)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status == RunStatus.PAUSED
    raw = await store.get_pause_state(item.run_id)
    expected = {calls[0].id: {"source": "static", "reason": "requires_approval"}}
    if started_sibling:
        expected[calls[1].id] = {"source": "recovery", "reason": "replay_after_crash"}
    assert raw["approval_gates"] == expected


async def test_fenced_owned_validation_does_not_attempt_release(store, monkeypatch, caplog):
    agent, run_id, _ = await submission_case(store, "input")
    claim = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    assert await store.release_run_claim(
        run_id, status="waiting_human_input", claim_epoch=claim.claim_epoch
    )
    await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "release_run_claim", release)
    with pytest.raises(RunFencedError):
        await runner._validate_owned_pause(
            claimed_store(store, run_id, claim.claim_epoch), run_id, agent, "waiting_human_input"
        )
    release.assert_not_awaited()
    assert not any(getattr(r, "code", None) == "persistence_error" for r in caplog.records)


@pytest.mark.parametrize("branch", ["policy_denied", "approved"])
async def test_policy_denial_marks_work_before_notifying(store, monkeypatch, branch):
    from dendrux.loops.base import BaseNotifier
    from dendrux.types import PauseState

    agent, run_id, _ = await submission_case(store, "approve")
    raw = await store.get_pause_state(run_id)
    pause = PauseState.from_dict(raw)
    tc = pause.pending_tool_calls[0]
    agent._deny = {tc.name}
    if branch == "policy_denied":
        pause.pending_targets[tc.id] = "client"
    marked = []

    async def persist_governance(run_id, event_type, iteration, data, **kwargs):
        await store.save_run_event(
            run_id, event_type=event_type, iteration_index=iteration, data=data, **kwargs
        )

    recorder = SimpleNamespace(on_governance_event=AsyncMock(side_effect=persist_governance))
    notifier = BaseNotifier()
    failure = RuntimeError("after_governance_write")
    monkeypatch.setattr("dendrux.loops._helpers.notify_governance", AsyncMock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        await runner._execute_approved_tools(
            agent, pause, recorder, notifier, run_id, on_work=lambda: marked.append(True)
        )
    assert caught.value is failure
    recorder.on_governance_event.assert_awaited_once()
    assert marked == [True], "durable policy.denied must prevent claim release"
    assert (
        sum(event.event_type == "policy.denied" for event in await store.get_run_events(run_id))
        == 1
    )


async def test_active_pause_plan_is_indexed(store, engine, record_property):
    from sqlalchemy import event, insert, text

    from dendrux.db.models import AgentRun, RunEvent

    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL query plan contract")
    runs = [generate_ulid() for _ in range(1000)]
    async with engine.begin() as conn:
        await conn.execute(
            insert(AgentRun),
            [
                dict(id=r, agent_name="plan", tenant_id="tenant", status="waiting_approval")
                for r in runs
            ],
        )
        for start in range(0, len(runs), 100):
            await conn.execute(
                insert(RunEvent),
                [
                    dict(
                        id=generate_ulid(),
                        agent_run_id=r,
                        event_type="run.paused"
                        if i == 0
                        else "run.resumed"
                        if i == 98
                        else "run.resume_aborted"
                        if i == 99
                        else "tools.declared",
                        sequence_index=i,
                    )
                    for r in runs[start : start + 100]
                    for i in range(100)
                ],
            )
        await conn.execute(text("ANALYZE agent_runs, run_events"))
    captured = []

    def capture(conn, cursor, sql, params, context, many):
        if sql.lstrip().upper().startswith("SELECT"):
            captured.append((sql, params))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await store.get_active_pause_sequence(runs[0], tenant_id="tenant") == 0
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(captured) == 1
    sql, params = captured[0]
    async with engine.connect() as conn:
        plan = (
            await conn.exec_driver_sql("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
        ).scalar()

    def walk(node):
        yield node
        for child in node.get("Plans", []):
            yield from walk(child)

    nodes = list(walk(plan[0]["Plan"]))
    scans = [node for node in nodes if node.get("Relation Name") == "run_events"]
    assert scans and all(
        "Index" in node["Node Type"] or "Bitmap" in node["Node Type"] for node in scans
    ), plan
    assert "run.resume_aborted" in params
    record_property("active_pause_explain", json.dumps({"sql": sql, "plan": plan}, default=str))


async def test_cancel_release_event_failure_rolls_back_terminal_transition(store, monkeypatch):
    from dendrux.runtime.stores.sql import base

    _, run_id, _ = await submission_case(store, "input")
    claim = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    await store.request_cancel(run_id)
    event = EventWrite(
        generate_ulid(), "run.resume_aborted", 0, {"failure_kind": "setup_error"}, None, None
    )
    original = base._insert_event_locked

    async def fail_cancel(session, target, event_type, data):
        if event_type == "run.cancelled":
            raise RuntimeError("cancel_event_write_failure")
        return await original(session, target, event_type, data)

    with monkeypatch.context() as patcher:
        patcher.setattr(base, "_insert_event_locked", fail_cancel)
        with pytest.raises(RuntimeError, match="cancel_event_write_failure"):
            await store.release_run_claim(
                run_id, status="waiting_human_input", claim_epoch=claim.claim_epoch, event=event
            )
    assert (await store.get_run(run_id)).status == "running"
    assert (await store.get_run_control(run_id)).cancel_requested
    assert not any(
        e.event_type in {"run.cancelled", "run.resume_aborted"}
        for e in await store.get_run_events(run_id)
    )
    assert await store.release_run_claim(
        run_id, status="waiting_human_input", claim_epoch=claim.claim_epoch, event=event
    )
    assert (await store.get_run(run_id)).status == "cancelled"
