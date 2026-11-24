"""Future or malformed managed waits refuse before claims or other mutations.

Old-worker v4 behavior is covered by the coordinator's pinned 0.7.0 harness
(spec §16.4, O1–O3), rather than this process's v4-capable decoder.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import fields, make_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, select, update

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun, Base
from dendrux.errors import (
    GovernanceMismatchError,
    ManagedStateError,
    PersistenceError,
    RunControlError,
    RunNotFoundError,
    RunNotReclaimableError,
)
from dendrux.runtime import _reclaim
from dendrux.runtime.state import EventWrite, ManagedHeader
from dendrux.runtime.submit import GOVERNANCE_KEY, claim_managed_adoption, governance_snapshot
from dendrux.types import (
    Limits,
    LLMResponse,
    ManagedWait,
    PauseState,
    RunStatus,
    ToolCall,
    ToolResult,
    UsageStats,
    generate_ulid,
)
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

TENANT = "forward-compat"
AGE = timedelta(hours=1)


def _agent(store: Any, registered: bool = True) -> Agent:
    child = Agent(
        name="reviewer", description="Reviews.", prompt="Review.", provider=ScriptedLLM([])
    )
    return Agent(
        name="lead",
        prompt="Lead.",
        provider=ScriptedLLM([LLMResponse(text="done")]),
        state_store=store,
        subagents=[child] if registered else [],
    )


async def _seed(
    store: Any,
    engine: Any,
    agent: Agent,
    cut: str = "claimed",
    version: int = 5,
    *,
    marked: bool = True,
) -> tuple[str, str, str]:
    """Seed managed records through store primitives for format-guard tests."""
    root, pause_id = generate_ulid(), generate_ulid()
    calls = [ToolCall(name="reviewer", params={}, id=generate_ulid()) for _ in range(2)]
    children = [generate_ulid(), generate_ulid()]
    await store.create_run(
        root,
        agent.name,
        tenant_id=TENANT,
        meta={"dendrux.governance": governance_snapshot(agent)},
    )
    await store.save_run_event(root, event_type="run.started", claim_epoch=0)
    for call, child in zip(calls, children, strict=True):
        await store.create_run(child, "reviewer", parent_run_id=root, tenant_id=TENANT)
        await store.checkpoint_step(
            child,
            claim_epoch=0,
            pii_mapping=None,
            trace=None,
            loop_state={
                "loop": "react",
                "managed": ManagedHeader(
                    root_run_id=root,
                    path=("reviewer",),
                    parent_run_id=root,
                    tool_call_id=call.id,
                ).to_dict(),
            },
        )
        await store.finalize_run(child, status="success", answer="saved report")
        await store.start_tool_call(
            root,
            claim_epoch=0,
            row_id=generate_ulid(),
            tool_call_id=call.id,
            provider_tool_call_id=None,
            tool_name=call.name,
            target="agent",
            params=None,
            iteration_index=1,
            idempotent=False,
            event=EventWrite(
                generate_ulid(), "tool.started", 1, {"call_id": call.id}, call.id, None
            ),
            meta={"managed": {"v": 1, "child_run_id": child, "bind": pause_id}},
        )
    state = {
        "v": 1,
        "loop": "react",
        "phase": "dispatch",
        "iteration": 1,
        "trace_order_offset": 4,
        "steps_count": 0,
        "usage": {},
        "tool_counts": {},
        "provider_kwargs": {},
        "skills": {"pins": {}},
        "managed": ManagedHeader(root_run_id=root, path=()).to_dict(),
        "managed_wait": {
            "v": 1,
            "pause_id": pause_id,
            "bindings": {c.id: child for c, child in zip(calls, children, strict=True)},
        },
        "managed_admitted": [c.id for c in calls],
        "pending": [
            {
                "call_id": c.id,
                "tool": c.name,
                "phase": "dispatchable",
                "target": "agent",
                "order_index": 2 + i,
            }
            for i, c in enumerate(calls)
        ],
        "submission": None
        if cut == "parked"
        else {"managed_fanout_adoption": {"v": 1, "calls": [c.id for c in calls]}},
    }
    await store.checkpoint_step(root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None)
    raw = PauseState(
        agent_name=agent.name,
        history=[],
        steps=[],
        pending_tool_calls=calls[:1],
        pending_targets={calls[0].id: "agent"},
        iteration=1,
        trace_order_offset=4,
        usage=UsageStats(),
        schema_version=3,
        pause_reason="managed_wait",
        pause_id=pause_id,
        loop_state_ref=True,
        managed_wait=ManagedWait(tool_call_id=calls[0].id, child_run_id=children[0]),
    ).to_dict()
    raw.pop("managed_wait")
    raw.update(
        schema_version=version,
        pending_tool_calls=[
            {"id": c.id, "name": c.name, "params": {}, "provider_tool_call_id": None} for c in calls
        ],
        pending_targets={c.id: "agent" for c in calls},
        managed_fanout={
            "v": 1,
            "n": 2,
            "bindings": [
                {"tool_call_id": c.id, "child_run_id": child}
                for c, child in zip(calls, children, strict=True)
            ],
        },
    )
    await store.pause_run(root, status="paused", pause_data=raw, claim_epoch=0)
    old = datetime.now(UTC) - timedelta(hours=3)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == root)
            .values(
                status="paused" if cut == "parked" else "running",
                claim_epoch=7,
                reclaim_attempts=1,
                reclaim_limit=3,
                lease_expired_at=old if marked and cut != "parked" else None,
                last_progress_at=old,
                status_changed_at=old,
                updated_at=old,
            )
        )
    if cut == "partial":
        from dendrux.db.models import ToolCallRecord

        async with engine.begin() as conn:
            await conn.execute(
                update(ToolCallRecord)
                .where(
                    ToolCallRecord.agent_run_id == root,
                    ToolCallRecord.tool_call_id == calls[0].id,
                )
                .values(outcome="completed", result={"output": "saved report"}, success=True)
            )
    return root, calls[0].id, children[0]


async def _snapshot(engine: Any) -> dict[str, list[Any]]:
    async with engine.connect() as conn:
        return {
            table.name: (await conn.execute(select(table).order_by(*table.primary_key.columns)))
            .mappings()
            .all()
            for table in Base.metadata.sorted_tables
        }


@pytest.mark.parametrize(
    ("registered", "version", "cut"),
    [
        (registered, version, cut)
        for registered in (False, True)
        for version in (5, 6)
        for cut in ("parked", "claimed", "partial")
    ],
)
async def test_reclaim_refuses_future_wait_before_claim(
    store: Any,
    engine: Any,
    caplog: Any,
    cut: str,
    version: int,
    registered: bool,
) -> None:
    agent = _agent(store, registered)
    root, _, _ = await _seed(store, engine, agent, cut, version)
    before = await _snapshot(engine)
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises((RunControlError, PersistenceError)) as exc,
    ):
        await agent.reclaim(root, tenant_id=TENANT)
    # Check mutation first: the red run exposes the consumed epoch and attempt.
    record = await store.get_run(root)
    assert (record.claim_epoch, record.reclaim_attempts) == (7, 1)
    assert await _snapshot(engine) == before
    if cut != "parked" or not registered:
        assert isinstance(exc.value, RunNotReclaimableError)
        assert exc.value.reason == "pause_format_unsupported"
        assert any(
            getattr(r, "code", None) == "pause_format_unsupported" and r.run_id == root
            for r in caplog.records
        )
    else:
        assert isinstance(exc.value, ManagedStateError)


@pytest.mark.parametrize(
    "marker", ["checkpoint_only", "pause_only", "empty", "scalar", "extra", "missing"]
)
async def test_reclaim_checks_both_saved_formats(store: Any, engine: Any, marker: str) -> None:
    agent = _agent(store, False)
    root, call, child = await _seed(store, engine, agent, version=4)
    state = await store.get_loop_state(root)
    raw = await store.get_pause_state(root)
    valid = {"tool_call_id": call, "child_run_id": child, "pause_id": raw["pause_id"]}
    state["managed_wait"] = {
        "pause_only": valid,
        "empty": {},
        "scalar": "future",
        "extra": {**valid, "bindings": {}},
        "missing": {"pause_id": raw["pause_id"]},
    }.get(marker, state["managed_wait"])
    if marker != "pause_only":
        raw["schema_version"] = 3
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == root)
            .values(
                loop_state=state,
                pause_data=raw,
            )
        )
    before = await _snapshot(engine)
    with pytest.raises((RunControlError, PersistenceError)) as exc:
        await agent.reclaim(root, tenant_id=TENANT)
    assert await _snapshot(engine) == before
    record = await store.get_run(root)
    assert (record.claim_epoch, record.reclaim_attempts) == (7, 1)
    if marker == "pause_only":
        assert isinstance(exc.value, ManagedStateError)
        assert exc.value.reason == "binding_conflict"
    else:
        assert isinstance(exc.value, RunNotReclaimableError)
        assert exc.value.reason == "pause_format_unsupported"


@pytest.mark.parametrize(
    ("operation", "registered", "cut"),
    [
        (operation, registered, cut)
        for operation in (
            "pending",
            "resume",
            "resume_stream",
            "input",
            "approval",
            "results",
            "adoption",
        )
        for registered in (False, True)
        for cut in ("parked", "claimed", "partial")
    ],
)
async def test_other_entries_leave_future_workflow_untouched(
    store: Any,
    engine: Any,
    run_store: Any,
    cut: str,
    operation: str,
    registered: bool,
) -> None:
    agent = _agent(store, registered)
    root, call, child = await _seed(store, engine, agent, cut)
    before = await _snapshot(engine)
    if operation == "adoption":
        # The 0.7 single-child adoption never claims a fan-out wait, in any state.
        adopted = await claim_managed_adoption(
            store,
            root,
            agent=agent,
            call_id=call,
            child_run_id=child,
            result=ToolResult(call_id=call, name="reviewer", payload="report"),
        )
        assert adopted is False
    else:
        with contextlib.suppress(RunControlError, PersistenceError):
            await _enter(operation, agent, run_store, root, call, None)
    assert await _snapshot(engine) == before
    assert agent.provider.calls == 0
    assert all(child.provider.calls == 0 for child in agent.subagents)


async def _enter(
    operation: str, agent: Agent, run_store: Any, root: str, call: str, token: str | None
) -> None:
    if operation == "pending":
        await run_store.get_pending(root, tenant_id=TENANT)
    elif operation == "resume":
        await agent.resume(root, tenant_id=TENANT, expected_pause_id=token)
    elif operation == "resume_stream":
        stream = agent.resume_stream(root, tenant_id=TENANT, expected_pause_id=token)
        events = [item async for item in stream]
        last = events[-1] if events else None
        if last is not None and getattr(last.type, "value", last.type) == "run_error":
            raise RunControlError("the stream ended with run_error")
    elif operation == "input":
        await agent.submit_input(root, "answer", tenant_id=TENANT, expected_pause_id=token)
    elif operation == "approval":
        await agent.submit_approval(root, approved=True, tenant_id=TENANT, expected_pause_id=token)
    else:
        await agent.submit_tool_results(
            root,
            [ToolResult(call_id=call, name="reviewer", payload="report")],
            tenant_id=TENANT,
            expected_pause_id=token,
        )


@pytest.mark.parametrize(
    ("operation", "registered", "cut", "token"),
    [
        (operation, registered, cut, token)
        for operation in ("resume", "resume_stream", "input", "approval", "results")
        for registered in (False, True)
        for cut in ("parked", "claimed", "partial")
        for token in ("wait", "foreign")
    ],
)
async def test_token_bearing_entries_leave_future_workflow_untouched(
    store: Any,
    engine: Any,
    run_store: Any,
    cut: str,
    operation: str,
    registered: bool,
    token: str,
) -> None:
    """A token never claims a fan-out wait: neither the wait's own generation nor any other."""
    agent = _agent(store, registered)
    root, call, _ = await _seed(store, engine, agent, cut, version=4)
    pause_id = (await store.get_pause_state(root))["pause_id"] if token == "wait" else None
    before = await _snapshot(engine)
    with pytest.raises((RunControlError, PersistenceError, ValueError)):
        await _enter(operation, agent, run_store, root, call, pause_id or generate_ulid())
    assert await _snapshot(engine) == before
    assert agent.provider.calls == 0
    assert all(child.provider.calls == 0 for child in agent.subagents)


@pytest.mark.parametrize("mode", ["error", "reclaim"])
async def test_waiting_sweep_leaves_future_wait_untouched(
    store: Any, engine: Any, mode: str
) -> None:
    agent = _agent(store)
    await _seed(store, engine, agent, "parked")
    before = await _snapshot(engine)
    result = await sweep(state_store=store, abandoned_waiting=AGE, on_stale=mode)
    assert not result.abandoned_waiting
    assert await _snapshot(engine) == before


async def test_foreign_future_wait_still_looks_missing(store: Any, engine: Any) -> None:
    agent = _agent(store, False)
    root, _, _ = await _seed(store, engine, agent)
    before = await _snapshot(engine)
    with pytest.raises(RunNotFoundError):
        await agent.reclaim(root, tenant_id="other")
    assert await _snapshot(engine) == before


async def test_sweep_item_cannot_claim_future_wait(store: Any, engine: Any) -> None:
    agent = _agent(store)
    root, _, _ = await _seed(store, engine, agent, "partial", marked=False)
    marked = await sweep(state_store=store, stale_running=AGE, on_stale="reclaim")
    [item] = marked.reclaimable
    assert item.run_id == root
    before = await _snapshot(engine)
    with pytest.raises(RunNotReclaimableError, match="pause_format_unsupported"):
        await agent.reclaim(root, tenant_id=TENANT, item=item)
    assert await _snapshot(engine) == before


async def test_parallel_old_reclaimers_never_claim_future_wait(store: Any, engine: Any) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent connections require Postgres")
    agent = _agent(store)
    root, _, _ = await _seed(store, engine, agent, "partial")
    before = await _snapshot(engine)
    results = await asyncio.gather(
        *(agent.reclaim(root, tenant_id=TENANT) for _ in range(8)),
        return_exceptions=True,
    )
    assert await _snapshot(engine) == before
    assert all(
        isinstance(r, RunNotReclaimableError) and r.reason == "pause_format_unsupported"
        for r in results
    )


async def test_preclaim_reads_each_format_once_and_never_writes(store: Any, engine: Any) -> None:
    agent = _agent(store, False)
    root, _, _ = await _seed(store, engine, agent)
    statements: list[str] = []

    def record(
        _conn: Any, _cursor: Any, statement: str, _params: Any, _ctx: Any, _many: Any
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        with pytest.raises(RunNotReclaimableError, match="pause_format_unsupported"):
            await _reclaim.reclaim(
                agent, store, root, tenant_id=TENANT, item=None, notifier=None, via_root=True
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)
    assert len(statements) == 3
    assert all(s.lstrip().upper().startswith("SELECT") for s in statements)


@tool(target="client")
async def governance_read() -> str:
    """Read the value on the client."""
    raise AssertionError("client tool executed on server")


def _governance_agent(store: Any, *, limit: int = 4) -> Agent:
    return Agent(
        name="governance",
        prompt="Read the value.",
        tools=[governance_read],
        provider=ScriptedLLM([tool_call("governance_read", pid="read"), LLMResponse(text="done")]),
        state_store=store,
        limits=Limits(max_parallel_subagents=limit),
    )


async def test_pre_parallel_snapshot_resumes_with_default_limit(
    store: Any, engine: Any, caplog: Any
) -> None:
    agent = _governance_agent(store)
    paused = await agent.run("read", tenant_id=TENANT)
    current = agent._limits
    previous_fields = [f.name for f in fields(current) if f.name != "max_parallel_subagents"]
    previous_type = make_dataclass("PreParallelLimits", [(name, int) for name in previous_fields])
    try:
        agent._limits = previous_type(*(getattr(current, name) for name in previous_fields))
        previous_snapshot = governance_snapshot(agent)
    finally:
        agent._limits = current
    record = await store.get_run(paused.run_id)
    raw = await store.get_pause_state(paused.run_id)
    raw[GOVERNANCE_KEY] = previous_snapshot
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == paused.run_id)
            .values(meta={**record.meta, GOVERNANCE_KEY: previous_snapshot}, pause_data=raw)
        )
    call = paused.pending.tool_calls[0]
    with caplog.at_level(logging.WARNING):
        result = await agent.resume(
            paused.run_id,
            tool_results=[ToolResult(name=call.name, call_id=call.id, payload="value")],
            expected_pause_id=paused.pause_id,
            tenant_id=TENANT,
        )
    assert result.status == RunStatus.SUCCESS
    events = await store.get_run_events(paused.run_id)
    assert not any(event.event_type == "governance.changed" for event in events)
    assert not any(
        getattr(r, "code", None) in {"governance_changed", "governance_mismatch"}
        for r in caplog.records
    )


async def test_parallel_override_change_is_detected_before_resume_claim(
    store: Any, engine: Any
) -> None:
    agent = _governance_agent(store, limit=8)
    paused = await agent.run("read", tenant_id=TENANT)
    saved = governance_snapshot(agent)
    agent._limits = Limits(max_parallel_subagents=16)
    assert saved != governance_snapshot(agent)
    before = await _snapshot(engine)
    call = paused.pending.tool_calls[0]
    with pytest.raises(GovernanceMismatchError) as exc:
        await agent.resume(
            paused.run_id,
            tool_results=[ToolResult(name=call.name, call_id=call.id, payload="value")],
            expected_pause_id=paused.pause_id,
            tenant_id=TENANT,
        )
    assert exc.value.changed == ("limits",)
    assert await _snapshot(engine) == before


@pytest.mark.parametrize("shape", ["single_binding", "duplicate_child"])
@pytest.mark.parametrize("entry", ["reclaim", "resume"])
async def test_malformed_v4_wait_refuses_before_claim(
    store: Any, engine: Any, shape: str, entry: str
) -> None:
    agent = _agent(store)
    root, _, _ = await _seed(store, engine, agent, "parked", version=4)
    raw = await store.get_pause_state(root)
    envelope = raw["managed_fanout"]
    if shape == "single_binding":
        envelope["n"] = 1
        envelope["bindings"] = envelope["bindings"][:1]
        raw["pending_tool_calls"] = raw["pending_tool_calls"][:1]
        raw["pending_targets"] = {raw["pending_tool_calls"][0]["id"]: "agent"}
    else:
        envelope["bindings"][1]["child_run_id"] = envelope["bindings"][0]["child_run_id"]
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).where(AgentRun.id == root).values(pause_data=raw))
    before = await _snapshot(engine)
    with pytest.raises(ManagedStateError) as exc:
        if entry == "reclaim":
            await agent.reclaim(root, tenant_id=TENANT)
        else:
            await agent.resume(root, expected_pause_id=raw["pause_id"], tenant_id=TENANT)
    assert exc.value.reason == "wait_invalid"
    record = await store.get_run(root)
    assert (record.claim_epoch, record.reclaim_attempts) == (7, 1)
    assert await _snapshot(engine) == before
