"""Recovery preserves sibling ownership, saved decisions and sweep budgets."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import ManagedStateError, RunFencedError, RunNotReclaimableError
from dendrux.loops import _managed_batch
from dendrux.runtime import _managed_fanout, _reclaim, _reclaim_managed, durability, runner
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.runtime.stores.sql import _sql_sweep, _sql_sweep_scan
from dendrux.types import (
    ApprovalDecision,
    Limits,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunAlreadyActiveError,
    RunStatus,
    ToolCall,
    UsageStats,
    generate_ulid,
)
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_adoption import (
    TENANT,
    Child,
    lead,
    node_of,
    observations,
    parked,
)

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


async def test_reconnect_branch_error_keeps_sibling_owned(store, monkeypatch):
    root = lead()
    fanout = await parked(store, root, [Child(status="running"), Child(status="running")])
    node = node_of(fanout, root)
    entered = asyncio.Event()
    owners, completed = [], []
    before = await store.get_run(node.children[1].run_id)
    failure = RunNotReclaimableError(node.children[0].run_id, "governance_changed")

    async def branch(store, child, **kwargs):
        owners.append(asyncio.current_task())
        if child == node.children[0]:
            await entered.wait()
            raise failure
        entered.set()
        await store.finalize_run(
            child.run_id,
            status="success",
            answer="ok",
            claim_epoch=0,
            expected_current_status="running",
        )
        completed.append(await store.get_run(child.run_id))

    monkeypatch.setattr(_managed_fanout, "drive_node", branch)
    try:
        with pytest.raises(RunNotReclaimableError) as error:
            await _managed_fanout._reconnect_children(store, node, fanout_tree=True, notifier=None)
        assert error.value is failure
        assert len(completed) == 1
        after = await store.get_run(node.children[1].run_id)
        assert after == completed[0]
        assert after.status == "success" and after.claim_epoch == before.claim_epoch
        assert all(owner.done() and not owner.cancelled() for owner in owners)
    finally:
        await root.close()


async def test_reconnect_reports_errors_in_stored_call_order(store, monkeypatch):
    root = lead()
    fanout = await parked(store, root, [Child(), Child()])
    node = node_of(fanout, root)
    second_failed = asyncio.Event()
    errors = [ValueError("first"), RuntimeError("second")]

    async def branch(store, child, **kwargs):
        if child == node.children[0]:
            await second_failed.wait()
            await store.get_run(child.run_id)
            raise errors[0]
        second_failed.set()
        raise errors[1]

    monkeypatch.setattr(_managed_fanout, "drive_node", branch)
    try:
        with pytest.raises(ValueError) as error:
            await _managed_fanout._reconnect_children(store, node, fanout_tree=True, notifier=None)
        assert error.value is errors[0]
    finally:
        await root.close()


@pytest.mark.parametrize("kind", ["not_running", "fenced", "active"])
async def test_reconnect_claim_races_are_not_owners(store, monkeypatch, kind):
    root = lead()
    fanout = await parked(store, root, [Child(status="running"), Child()])
    leaf = node_of(fanout, root).children[0]
    error = (
        RunFencedError(leaf.run_id)
        if kind == "fenced"
        else RunAlreadyActiveError(leaf.run_id, RunStatus.RUNNING)
        if kind == "active"
        else RunNotReclaimableError(leaf.run_id, "not_running")
    )

    async def refused(*args, **kwargs):
        if kind == "not_running":
            await store.finalize_run(
                leaf.run_id,
                status="success",
                answer="finished",
                claim_epoch=0,
                expected_current_status="running",
            )
        raise error

    monkeypatch.setattr(_reclaim, "reclaim", refused)
    try:
        result = await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=True)
        assert result.status == (RunStatus.SUCCESS if kind == "not_running" else RunStatus.RUNNING)
        assert (await store.get_run(leaf.run_id)).claim_epoch == 0
        if kind != "not_running":
            with pytest.raises(type(error)):
                await _managed_fanout.drive_node(store, leaf, tenant_id=TENANT, fanout_tree=False)
    finally:
        await root.close()


async def test_young_ordinary_wait_does_not_issue_update(store, engine):
    run_id = generate_ulid()
    await store.create_run(run_id, "lead", tenant_id=TENANT)
    pause = PauseState("lead", [], [], [], 0, 0, UsageStats(), pause_id=generate_ulid())
    await store.pause_run(run_id, status="paused", pause_data=pause.to_dict(), claim_epoch=0)
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                status_changed_at=datetime.now(UTC) - timedelta(minutes=10),
                updated_at=datetime.now(UTC) - timedelta(minutes=10),
            )
        )
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        swept, _ = await _sql_sweep._collect_runs(
            store,
            timedelta(hours=1),
            waiting=True,
            batch_size=1,
            max_runs=10,
            adoptable_after=timedelta(minutes=1),
            adoptable=[],
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert swept == []
    assert not [sql for sql in statements if sql.lstrip().upper().startswith("UPDATE")]
    assert (await store.get_run(run_id)).status == "paused"


async def test_adoption_collector_enters_one_sweep_boundary(store, monkeypatch):
    observed, boundaries = [], []
    original = durability._sweep_budget

    def budget(current, sweep):
        boundaries.append(sweep)
        return original(current, sweep)

    async def collect(scan, *args):
        observed.append(boundaries.copy())
        return [], 0, None

    monkeypatch.setattr(durability, "_sweep_budget", budget)
    monkeypatch.setattr(_sql_sweep_scan._SweepScan, "collect", collect)
    await _sql_sweep._collect_runs(
        store,
        None,
        waiting=True,
        batch_size=1,
        max_runs=10,
        adoptable_after=timedelta(0),
        adoptable=[],
    )
    assert observed and all(calls == [True] for calls in observed)
    boundaries.clear()
    observed.clear()
    await store._sweep_runs(timedelta(0), waiting=True, batch_size=1, max_runs=10)
    assert observed and all(calls == [True] for calls in observed)


@pytest.mark.parametrize("reason", ["exclusive", "fanout_limit"])
async def test_partial_refusal_keeps_durable_reason(store, engine, monkeypatch, reason):
    from dendrux.loops import react
    from tests.integration.test_managed_adoption import ProcessDeathError

    calls = [ToolCall("reviewer", {"task": "go"}) for _ in range(3)]
    child = Agent(name="reviewer", description="Reviews", prompt="Review", provider=ScriptedLLM([]))
    root = Agent(
        name="lead",
        prompt="Lead",
        state_store=store,
        subagents=[child],
        limits=Limits(max_parallel_subagents=4 if reason == "exclusive" else 2),
        exclusive_subagents=["reviewer"] if reason == "exclusive" else None,
        provider=ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")]),
    )
    original = react._ToolBatch.reject

    async def cut(batch, call, result):
        await original(batch, call, result)
        raise ProcessDeathError

    try:
        with monkeypatch.context() as changes:
            changes.setattr(react._ToolBatch, "reject", cut)
            with pytest.raises(ProcessDeathError):
                await root.run("go", tenant_id=TENANT)
        async with AsyncSession(engine) as session:
            run_id = (
                await session.scalars(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
            ).one()
        before = await refusal_rows(engine, run_id)
        assert len(before) == 1 and before[0].meta["reason"] == reason
        result = await root.reclaim(
            run_id, tenant_id=TENANT, item=await expire(store, engine, run_id)
        )
        assert result.status == RunStatus.SUCCESS
        rows = await refusal_rows(engine, run_id)
        assert len(rows) == 3
        assert all(
            {k: row.meta.get(k) for k in ("code", "reason", "limit", "requested")}
            == {k: before[0].meta.get(k) for k in ("code", "reason", "limit", "requested")}
            for row in rows
        )
        assert all(row.error_message == before[0].error_message for row in rows)
        assert child._require_provider().calls == 0
    finally:
        await root.close()
        await child.close()


async def refusal_rows(engine, run_id):
    async with AsyncSession(engine) as session:
        return (
            await session.scalars(
                select(ToolCallRecord).where(ToolCallRecord.agent_run_id == run_id)
            )
        ).all()


@pytest.mark.parametrize("admitted", [None, "a", ["a", "b", "c"]])
def test_survivor_admission_retains_stored_order(engine, admitted):
    calls = [ToolCall("reviewer", id=call_id) for call_id in ("a", "b", "c")]
    found = _reclaim_managed._ManagedPending.from_calls(calls, admitted=admitted, run_id="r")
    state = {
        "pending": [
            {"call_id": "c", "phase": "approved"},
            {"call_id": "b", "phase": "rejected"},
            {"call_id": "a", "phase": "approved"},
        ],
        "managed_admitted": admitted,
    }
    narrowed = _reclaim_managed._surviving_managed(found, state, "r")
    assert narrowed.call_ids == ("a", "c")
    assert narrowed.admitted is isinstance(admitted, list)
    state["pending"] = [{"call_id": "a", "phase": "approved"}]
    narrowed = _reclaim_managed._surviving_managed(found, state, "r")
    assert narrowed.call_ids == ("a",) and narrowed.admitted is (admitted is not None)
    assert _reclaim_managed._surviving_managed(found, {"pending": []}, "r") is None
    assert _reclaim_managed._surviving_managed(None, state, "r") is None


def test_empty_admission_refuses_with_run_id(engine):
    run_id = generate_ulid()
    with pytest.raises(ManagedStateError) as error:
        _reclaim_managed._ManagedPending.from_calls([], admitted=[], run_id=run_id)
    assert error.value.reason == "binding_conflict" and error.value.run_id == run_id


@pytest.mark.parametrize(
    "body,reason,limit,count",
    [
        ({"error": "old"}, None, 1, 3),
        ({"error": "old", "reason": "client_call"}, "client_call", 1, 3),
        ({"error": "old", "reason": "exclusive"}, "exclusive", 1, 3),
        (
            {"error": "old", "reason": "fanout_limit", "limit": 4, "requested": 5},
            "fanout_limit",
            4,
            5,
        ),
    ],
)
def test_saved_refusal_reuses_earliest_durable_verdict(engine, body, reason, limit, count):
    original = _managed_batch.BatchDecision("refuse", 3, "fanout_limit")
    history = [
        Message(Role.ASSISTANT, "not json"),
        Message(Role.TOOL, "not json", call_id="a", name="reviewer"),
        Message(Role.TOOL, "[]", call_id="a", name="reviewer"),
        Message(Role.TOOL, json.dumps(body), call_id="a", name="reviewer"),
        Message(Role.TOOL, '{"error":"later","reason":"exclusive"}', call_id="b", name="reviewer"),
    ]
    decision, saved_limit = _reclaim_managed._saved_refusal(history, {"a", "b"}, original, 1)
    assert (decision.reason, decision.count, saved_limit) == (reason, count, limit)
    assert _reclaim_managed._saved_refusal([], {"a"}, original, 1) == (original, 1)


@pytest.mark.parametrize("first_error", [False, True])
async def test_child_cancellation_joins_healthy_sibling(store, monkeypatch, first_error):
    root = lead()
    fanout = await parked(store, root, [Child(status="running"), Child(status="running")])
    node = node_of(fanout, root)
    entered = asyncio.Event()
    completed = []

    async def branch(store, child, **kwargs):
        if child == node.children[0]:
            await entered.wait()
            if first_error:
                raise ValueError("first")
            raise asyncio.CancelledError("branch")
        entered.set()
        await store.finalize_run(
            child.run_id,
            status="success",
            answer="ok",
            claim_epoch=0,
            expected_current_status="running",
        )
        completed.append(child.run_id)
        if first_error:
            raise asyncio.CancelledError("branch")

    monkeypatch.setattr(_managed_fanout, "drive_node", branch)
    try:
        with pytest.raises(asyncio.CancelledError) as error:
            await _managed_fanout._reconnect_children(store, node, fanout_tree=True, notifier=None)
        assert error.value.args == ("branch",)
        assert completed == [node.children[1].run_id]
        assert (await store.get_run(completed[0])).status == "success"
    finally:
        await root.close()


@pytest.mark.parametrize("survivors", [1, 2])
async def test_approval_subset_recovers_after_decision_crash(store, engine, tmp_path, survivors):
    effects, phase = tmp_path / "effects.jsonl", tmp_path / "phase.json"
    env = {
        **os.environ,
        "PYTHONPATH": "src:.",
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_EFFECTS": str(effects),
        "T6_PHASE": str(phase),
    }
    for mode, expected in (("crash", 17), ("decisions", 17), ("recover", 0)):
        result = subprocess.run(
            [sys.executable, "-m", __name__, mode, str(survivors)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stderr[-3000:]
        counts = FanoutEffects(effects).counts()
        assert counts.tool("root", "hook") == 5
        assert counts.tool("root", "fetch") == 1
        assert counts.tool("root", "authorize") == int(mode == "recover")
        assert counts.provider_calls("root") == (3 if mode == "recover" else 2)
        assert counts.provider_calls("reviewer") == (survivors if mode == "recover" else 0)
        assert counts.transform_calls() == (2 + survivors if mode == "recover" else 1)
    saved = json.loads(phase.read_text())
    assert saved["version"] == (3 if survivors == 1 else 4)
    assert saved["bound"] == saved["survivors"]
    assert saved["admitted"] is True
    assert saved["children"] == survivors and saved["rows"] == 5
    assert saved["decisions"] == 1 and saved["observations"] == 5


async def approval_agents(store, effects, mode):
    @tool()
    async def fetch() -> str:
        """Read a saved server value."""
        effects.record_tool("root", "fetch")
        return "ok"

    @tool()
    async def authorize() -> str:
        """Finish an approved server operation."""
        effects.record_tool("root", "authorize")
        return "ok"

    async def hook(ctx):
        effects.record_tool("root", "hook")

    child = Agent(
        name="reviewer",
        description="Reviews",
        prompt="Review",
        provider=effects.provider("reviewer", ScriptedLLM([LLMResponse(text="ok")] * 2)),
    )
    ids = sorted([generate_ulid() for _ in range(3)], reverse=True)
    calls = [ToolCall("reviewer", {"task": str(i)}, id=call_id) for i, call_id in enumerate(ids)]
    script = (
        [
            LLMResponse(tool_calls=[ToolCall("fetch")]),
            LLMResponse(tool_calls=[calls[0], ToolCall("authorize"), *calls[1:]]),
        ]
        if mode == "crash"
        else [LLMResponse(text="done")]
    )
    root = Agent(
        name="lead",
        prompt="Lead",
        subagents=[child],
        tools=[fetch, authorize],
        before_tool=hook,
        require_approval=["authorize"],
        limits=Limits(max_parallel_subagents=4),
        state_store=store,
        provider=effects.provider("root", ScriptedLLM(script)),
        tool_result_transform=effects.transform(),
    )
    return root, child


async def approval_phase(mode, survivors):
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    phase = Path(os.environ["T6_PHASE"])
    root, child = await approval_agents(store, effects, mode)
    if mode == "crash":
        waiting = await root.run("go", tenant_id=TENANT)
        assert waiting.status == RunStatus.WAITING_APPROVAL
        pause, ids = await seed_managed_approval(store, engine, waiting.run_id)
        approved = ids[::2][:survivors]
        phase.write_text(json.dumps({"survivors": approved}))
        original = store.submit_and_claim

        async def claimed(*args, **kwargs):
            await original(*args, **kwargs)
            os._exit(17)

        store.submit_and_claim = claimed
        await root.submit_approval(
            waiting.run_id,
            tenant_id=TENANT,
            expected_pause_id=waiting.pause_id,
            decisions={
                i: ApprovalDecision(approved=i in approved or i not in ids)
                for i in pause["approval_gates"]
            },
        )
        raise RuntimeError("approval claim cut was not reached")
    saved = json.loads(phase.read_text())
    if mode == "decisions":
        original_decision = runner._record_approval_decided

        async def decided(*args, **kwargs):
            await original_decision(*args, **kwargs)
            os._exit(17)

        runner._record_approval_decided = decided
    install_binding_capture(store, saved)
    original_handoff = _reclaim._reclaim_managed._handed_off

    def handed_off(found, held):
        if found is not None:
            saved["admitted"] = found.admitted
        return original_handoff(found, held)

    _reclaim._reclaim_managed._handed_off = handed_off
    async with AsyncSession(engine) as session:
        run_id = (
            await session.scalars(select(AgentRun.id).where(AgentRun.parent_run_id.is_(None)))
        ).one()
    result = await root.reclaim(run_id, tenant_id=TENANT, item=await expire(store, engine, run_id))
    assert result.status == RunStatus.SUCCESS, result.status.value
    async with AsyncSession(engine) as session:
        saved["children"] = len(
            (
                await session.scalars(select(AgentRun.id).where(AgentRun.parent_run_id == run_id))
            ).all()
        )
    saved["rows"] = len(await store.get_tool_calls(run_id))
    saved["decisions"] = sum(
        e.event_type == "approval.decided" for e in await store.get_run_events(run_id)
    )
    saved["observations"] = len(await observations(engine, run_id))
    phase.write_text(json.dumps(saved))
    await root.close()
    await child.close()
    await engine.dispose()


async def seed_managed_approval(store, engine, run_id):
    # The SQL fixture represents the managed approval snapshot supplied by the routing layer.
    pause = await store.get_pause_state(run_id)
    state = await store.get_loop_state(run_id)
    ids = [call["id"] for call in pause["pending_tool_calls"] if call["name"] == "reviewer"]
    pause["approval_gates"].update(
        {i: {"source": "static", "reason": "requires_approval"} for i in ids}
    )
    state["managed_admitted"] = ids
    for pending in state["pending"]:
        if pending["call_id"] in ids:
            pending.update(phase="gated", decision_id=f"{pause['pause_id']}:{pending['call_id']}")
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(pause_data=pause, loop_state=state)
        )
    return pause, ids


def install_binding_capture(store, saved):
    single, multiple = store.bind_managed_child, store.bind_managed_children

    async def bind_one(request):
        saved.update(version=3, bound=[request.tool_call_id])
        return await single(request)

    async def bind_many(request):
        saved.update(version=4, bound=[c.tool_call_id for c in request.children])
        return await multiple(request)

    store.bind_managed_child, store.bind_managed_children = bind_one, bind_many


if __name__ == "__main__":
    asyncio.run(approval_phase(sys.argv[1], int(sys.argv[2])))


@pytest.mark.parametrize("streaming", [False, True])
async def test_public_continuation_frame_is_configured_before_preparation_returns(
    store, monkeypatch, streaming
):
    from dendrux.runtime import _run_resume_prep
    from dendrux.types import ToolResult

    @tool(target="client")
    async def lookup() -> str:
        """Read a value on the client."""
        return ""

    root = Agent(
        name="root",
        prompt="Review",
        state_store=store,
        tools=[lookup],
        provider=ScriptedLLM(
            [LLMResponse(tool_calls=[ToolCall("lookup", {})]), LLMResponse(text="done")]
        ),
    )
    paused = await root.run("go", tenant_id=TENANT)
    observed = []
    original = _run_resume_prep._prepare_resume

    async def prepare(*args, **kwargs):
        ctx = await original(*args, **kwargs)
        observed.append(ctx.recorder._managed_fanout_capable)
        return ctx

    monkeypatch.setattr(_run_resume_prep, "_prepare_resume", prepare)
    call = paused.pending.tool_calls[0]
    kwargs = dict(
        tenant_id=TENANT,
        expected_pause_id=paused.pause_id,
        tool_results=[ToolResult(call_id=call.id, name=call.name, payload="value")],
    )
    try:
        if streaming:
            async with root.resume_stream(paused.run_id, **kwargs) as stream:
                events = [event async for event in stream]
            result = events[-1].run_result
        else:
            result = await root.resume(paused.run_id, **kwargs)
        assert result.status == RunStatus.SUCCESS
        assert observed == [not streaming]
    finally:
        await root.close()
