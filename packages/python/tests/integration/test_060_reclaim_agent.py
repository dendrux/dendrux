"""Developer-called reclaim, continuation and interruption policy."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotFoundError, RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.state import EventWrite
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    UsageStats,
    _usage_to_dict,
    generate_ulid,
)


async def checkpoint(
    agent, store, engine, *, calls=(), phases=(), started=False, idempotent=False, **state
):
    run_id = generate_ulid()
    seeds = [
        message_trace(Message(Role.SYSTEM, agent.prompt), 0, kind="seed"),
        message_trace(Message(Role.USER, "go"), 1, kind="seed"),
    ]
    await store.create_run(
        run_id,
        agent.name,
        tenant_id="tenant",
        seed_traces=seeds,
        meta={"dendrux.governance": governance_snapshot(agent)},
    )
    pending = [
        {
            "call_id": tc.id,
            "tool": tc.name,
            "phase": phase,
            "decision_id": None,
            "order_index": 3 + i,
            "target": "client" if phase == "client" else "server",
        }
        for i, (tc, phase) in enumerate(zip(calls, phases, strict=True))
    ]
    cp = dict(
        v=1,
        loop="react",
        phase="dispatch" if pending else "model_next",
        iteration=1 if calls else 0,
        trace_order_offset=3 + len(calls) if calls else 2,
        steps_count=0,
        usage=_usage_to_dict(UsageStats()),
        tool_counts={},
        provider_kwargs={},
        pending=pending,
        submission=None,
        skills={"pins": {}},
        **state,
    )
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state=cp,
        pii_mapping=None,
        trace=message_trace(Message(Role.ASSISTANT, "", tool_calls=list(calls)), 2)
        if calls
        else None,
    )
    if started:
        for tc in calls:
            await store.start_tool_call(
                run_id,
                claim_epoch=0,
                row_id=generate_ulid(),
                tool_call_id=tc.id,
                provider_tool_call_id=None,
                tool_name=tc.name,
                target="server",
                params=tc.params or None,
                iteration_index=1,
                idempotent=idempotent,
                event=EventWrite(
                    generate_ulid(),
                    "tool.started",
                    1,
                    {"call_id": tc.id, "tool_name": tc.name, "idempotent": idempotent},
                    tc.id,
                    None,
                ),
            )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    return (
        await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    ).reclaimable[0]


def make_agent(store, **kwargs):
    return Agent(
        name="recovery",
        prompt="p",
        state_store=store,
        provider=MockLLM([LLMResponse(text="done")]),
        strict_tenancy=True,
        on_governance_change="raise",
        **kwargs,
    )


async def test_continue_seed_and_projection(store, engine):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    assert hasattr(agent, "reclaim"), "Agent must expose the explicit recovery entry"
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success" and result.answer == "done"
    assert result.recovery.reclaim_attempts == 1
    fresh = make_agent(store)
    loaded = await fresh.get_result(item.run_id, tenant_id="tenant")
    assert loaded.recovery == result.recovery


@pytest.mark.parametrize("tenant", [None, "foreign"])
async def test_tenant_refusal_has_no_write(store, engine, tenant):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    before = await store.get_run(item.run_id)
    assert hasattr(agent, "reclaim")
    with pytest.raises(RunNotFoundError):
        await agent.reclaim(item.run_id, **({} if tenant == "omitted" else {"tenant_id": tenant}))
    assert await store.get_run(item.run_id) == before


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("storage_format", 1, "legacy_run"),
        ("status", "success", "not_running"),
        ("lease_expired_at", None, "lease_fresh"),
        ("claim_epoch", 7, "already_reclaimed"),
        ("reclaim_attempts", 3, "reclaim_limit"),
        ("agent_name", "different", "agent_mismatch"),
        ("loop_state", {}, "loop_unknown"),
        ("loop_state", {"loop": "single_call"}, "loop_unsupported"),
        (
            "loop_state",
            {"loop": "react", "skills": {"pins": {"missing": "digest"}}},
            "skill_mismatch",
        ),
    ],
)
async def test_refusals(store, engine, field, value, reason, caplog):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == item.run_id).values(**{field: value})
        )
    before = await store.get_run(item.run_id)
    assert hasattr(agent, "reclaim")
    with pytest.raises(RunNotReclaimableError) as caught:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert caught.value.reason == reason
    assert await store.get_run(item.run_id) == before
    assert any(
        getattr(r, "code", None) == reason and getattr(r, "run_id", None) == item.run_id
        for r in caplog.records
    )


async def test_item_identity(store, engine):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    assert hasattr(agent, "reclaim")
    with pytest.raises(ValueError):
        await agent.reclaim(
            item.run_id, tenant_id="tenant", item=replace(item, run_id=generate_ulid())
        )


async def test_cancel_requested_finalises(store, engine):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    await store.request_cancel(item.run_id)
    assert hasattr(agent, "reclaim")
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "cancelled"


@pytest.mark.parametrize("policy", ["warn", "raise"])
async def test_governance_policy(store, engine, policy):
    @tool()
    async def changed() -> str:
        """Governed tool."""
        return "ok"

    agent = make_agent(store, tools=[changed])
    item = await checkpoint(agent, store, engine)
    other = Agent(
        name=agent.name,
        prompt="p",
        state_store=store,
        provider=MockLLM([LLMResponse(text="done")]),
        deny=["changed"],
        tools=[changed],
        strict_tenancy=True,
        on_governance_change=policy,
    )
    assert hasattr(other, "reclaim")
    if policy == "raise":
        with pytest.raises(RunNotReclaimableError) as caught:
            await other.reclaim(item.run_id, tenant_id="tenant", item=item)
        assert caught.value.reason == "governance_changed"
    else:
        assert (
            await other.reclaim(item.run_id, tenant_id="tenant", item=item)
        ).status.value == "success"
        event = next(
            e for e in await store.get_run_events(item.run_id) if e.event_type == "run.reclaimed"
        )
        assert event.data["governance_changed"] is True


@pytest.mark.parametrize(
    "started,idempotent,phase",
    [
        (False, False, "dispatchable"),
        (True, True, "dispatchable"),
        (False, False, "gated"),
        (True, False, "dispatchable"),
    ],
)
@pytest.mark.parametrize("approve", [True, False])
async def test_dispatch_and_replay_approval(store, engine, started, idempotent, phase, approve):
    effects = []

    @tool(idempotent=idempotent)
    async def work() -> str:
        """Record one effect."""
        effects.append(1)
        return "ok"

    agent = make_agent(store, tools=[work], require_approval=["work"] if phase == "gated" else [])
    call = ToolCall("work")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=[phase], started=started, idempotent=idempotent
    )
    assert hasattr(agent, "reclaim")
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    gated = phase == "gated" or started and not idempotent
    if gated:
        assert result.status.value == "waiting_approval" and effects == []
        if started:
            raw = await store.get_pause_state(item.run_id)
            assert raw["approval_gates"][call.id] == {
                "source": "recovery",
                "reason": "replay_after_crash",
            }
        submitted = await agent.submit_approval(
            item.run_id, approved=approve, tenant_id="tenant", expected_pause_id=result.pause_id
        )
        result = submitted
    assert result.status.value == "success"
    assert len(effects) == (not gated or approve)
    rows = await store.get_tool_calls(item.run_id)
    if started:
        row = next(r for r in rows if r.tool_call_id == call.id)
        assert row.outcome == ("completed" if approve or idempotent else "unknown")
        assert (row.replayed_at is not None) == (approve or idempotent)
    messages = await store.conversation_traces(item.run_id, after_position=None)
    assert len([m for m in messages.entries if m.role == "tool"]) == 1


@pytest.mark.parametrize("bound", ["max_rebuild_rows", "max_rebuild_bytes"])
async def test_a18_history_bound(store, engine, bound):
    from dendrux.types import Limits

    agent = make_agent(store, limits=Limits(**{bound: 1}))
    item = await checkpoint(agent, store, engine)
    assert hasattr(agent, "reclaim")
    with pytest.raises(RunNotReclaimableError) as caught:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert caught.value.reason == "history_too_large"


async def test_client_pause_recreated_and_consumed_once(store, engine):
    from dendrux.types import ToolResult

    @tool(target="client")
    async def client() -> str:
        """Client only."""
        raise AssertionError("client ran on server")

    agent = make_agent(store, tools=[client])
    call = ToolCall("client")
    item = await checkpoint(agent, store, engine, calls=[call], phases=["client"])
    assert hasattr(agent, "reclaim")
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_client_tool"
    assert (await store.get_loop_state(item.run_id))["pending"][0]["call_id"] == call.id
    assert not await store.get_tool_calls(item.run_id)
    result = await agent.submit_tool_results(
        item.run_id,
        [ToolResult(name="client", call_id=call.id, payload='"ok"')],
        expected_pause_id=result.pause_id,
        tenant_id="tenant",
    )
    assert result.status.value == "success"
    assert (await store.get_loop_state(item.run_id))["pending"] == []
    messages = await store.conversation_traces(item.run_id, after_position=None)
    assert len([m for m in messages.entries if m.role == "tool"]) == 1


async def test_stored_result_observed_without_execution(store, engine):
    from dendrux.db.models import ToolCallRecord

    @tool()
    async def work() -> str:
        """Already ran."""
        raise AssertionError("stored result reran")

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=True
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(ToolCallRecord)
            .where(ToolCallRecord.agent_run_id == item.run_id)
            .values(outcome="completed", result='"stored"', success=True)
        )
    assert hasattr(agent, "reclaim")
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    entries = (await store.conversation_traces(item.run_id, after_position=None)).entries
    observations = [m for m in entries if m.role == "tool"]
    assert len(observations) == 1 and "stored" in observations[0].content


@pytest.mark.parametrize("entry", ["resume", "resume_stream"])
async def test_recovery_approval_resume_entries(store, engine, entry):
    effects = []

    @tool()
    async def work() -> str:
        """Record replay after consent."""
        effects.append(1)
        return "ok"

    agent = make_agent(store, tools=[work])
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("work")], phases=["dispatchable"], started=True
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    if entry == "resume":
        result = await agent.resume(
            item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        )
    else:
        async with agent.resume_stream(
            item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        ) as stream:
            async for _ in stream:
                pass
            result = stream.result
    assert result.status.value == "success" and effects == [1]
    assert len(result.recovery.replayed_tool_calls) == 1


async def test_twenty_public_reclaimers_one_continuation(store, engine):
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from dendrux.runtime.state import SQLAlchemyStateStore

    if engine.dialect.name != "postgresql":
        pytest.skip("Fleet contention requires independent PostgreSQL connections")
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    pool = create_async_engine(engine.url, pool_size=20, max_overflow=0)
    connections = await asyncio.gather(*(pool.connect() for _ in range(20)))
    await asyncio.gather(*(c.close() for c in connections))
    barrier = asyncio.Barrier(20)
    entered = set()

    class Session(AsyncSession):
        async def execute(self, statement, *args, **kwargs):
            task = asyncio.current_task()
            if task not in entered:
                entered.add(task)
                await self.connection()
                await asyncio.wait_for(barrier.wait(), 20)
            return await super().execute(statement, *args, **kwargs)

    contenders = []
    for _ in range(20):
        contender_store = SQLAlchemyStateStore(pool)
        contender_store._session_factory = async_sessionmaker(
            pool, class_=Session, expire_on_commit=False
        )
        contenders.append(make_agent(contender_store))
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(a.reclaim(item.run_id, tenant_id="tenant", item=item) for a in contenders),
                return_exceptions=True,
            ),
            30,
        )
        assert sum(not isinstance(r, BaseException) for r in results) == 1
        assert sum(isinstance(r, RunNotReclaimableError) for r in results) == 19
        assert sum(a.provider.calls_made for a in contenders) == 1
        assert (await store.get_run(item.run_id)).reclaim_attempts == 1
    finally:
        await pool.dispose()


async def test_second_crash_during_approved_replay_requires_new_approval(
    store, engine, monkeypatch
):
    class Crash(BaseException):
        pass

    effects = []

    @tool()
    async def work() -> str:
        """A non-idempotent effect that crashes again."""
        effects.append(1)
        raise Crash()

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=True
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    # Kill immediately before a second replay effect, after its authorization
    # has been consumed. BaseException bypasses ordinary error finalization.
    with pytest.raises(Crash):
        await agent.resume(item.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == item.run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    item = (
        await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    ).reclaimable[0]
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "waiting_approval"
    assert effects == [1]
    assert result.pause_id != paused.pause_id


async def test_reclaim_obeys_cancel_after_claim_before_dispatch(store, engine, monkeypatch):
    from functools import wraps

    effects = []

    @tool()
    async def work() -> str:
        """Must not run after cancellation."""
        effects.append(1)
        return "ok"

    agent = make_agent(store, tools=[work])
    item = await checkpoint(agent, store, engine, calls=[ToolCall("work")], phases=["dispatchable"])
    original = store.reclaim_run

    @wraps(original)
    async def claim(*args, **kwargs):
        result = await original(*args, **kwargs)
        await store.request_cancel(item.run_id)
        return result

    monkeypatch.setattr(store, "reclaim_run", claim)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "cancelled" and effects == []


async def test_gated_call_is_validated_before_recreated_pause(store, engine):
    @tool()
    async def work(value: int) -> str:
        """Needs a validated integer."""
        raise AssertionError("invalid call ran")

    agent = make_agent(store, tools=[work], require_approval=["work"])
    item = await checkpoint(
        agent, store, engine, calls=[ToolCall("work", {"value": []})], phases=["gated"]
    )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    assert (await store.get_tool_calls(item.run_id))[0].outcome == "failed"


async def test_reclaim_finalisation_logs_code(store, engine, caplog):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    assert any(
        getattr(r, "code", None) == "run_finalized" and getattr(r, "run_id", None) == item.run_id
        for r in caplog.records
    )


async def test_pause_during_recovery_keeps_earlier_held_calls(store, engine, monkeypatch):
    from dendrux.runtime.persistence import PersistenceRecorder

    @tool()
    async def work() -> str:
        """Cannot execute without replay approval."""
        raise AssertionError("executed")

    agent = make_agent(store, tools=[work])
    calls = [ToolCall("work"), ToolCall("work")]
    item = await checkpoint(
        agent, store, engine, calls=calls, phases=["dispatchable"] * 2, started=True
    )
    original = PersistenceRecorder.on_governance_event

    async def gate(self, run_id, event_type, iteration, data, **kwargs):
        await original(self, run_id, event_type, iteration, data, **kwargs)
        if event_type == "approval.requested":
            await store.request_pause(run_id)

    monkeypatch.setattr(PersistenceRecorder, "on_governance_event", gate)
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "paused"
    state = await store.get_loop_state(item.run_id)
    assert [p["call_id"] for p in state["pending"]] == [tc.id for tc in calls]
    raw = await store.get_pause_state(item.run_id)
    assert raw["approval_gates"][calls[0].id]["source"] == "recovery"


async def test_completed_observation_is_not_inserted_twice(store, engine):
    from dendrux.db.models import ToolCallRecord

    @tool()
    async def work() -> str:
        """Already completed."""
        raise AssertionError("completed tool reran")

    agent = make_agent(store, tools=[work])
    call = ToolCall("work")
    item = await checkpoint(
        agent, store, engine, calls=[call], phases=["dispatchable"], started=True
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(ToolCallRecord)
            .where(ToolCallRecord.agent_run_id == item.run_id)
            .values(outcome="completed", result="stored", success=True)
        )
    await store.save_trace(
        item.run_id,
        "tool",
        '"stored"',
        order_index=3,
        meta={"call_id": call.id, "tool_name": call.name},
    )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    traces = await store.get_traces(item.run_id)
    assert len([t for t in traces if t.role == "tool"]) == 1


@pytest.mark.parametrize("producer", ["override", "different"])
async def test_reclaim_preserves_reasoning_for_saved_model_override(store, engine, producer):
    agent = make_agent(store)
    item = await checkpoint(agent, store, engine)
    state = await store.get_loop_state(item.run_id)
    state.update(iteration=1, trace_order_offset=3, provider_kwargs={"model": "override"})
    blocks = [{"type": "thinking", "thinking": "reason", "signature": "signature"}]
    await store.checkpoint_step(
        item.run_id,
        claim_epoch=item.claim_epoch,
        loop_state=state,
        pii_mapping=None,
        trace=message_trace(
            Message(
                Role.ASSISTANT, "prior", reasoning_blocks=blocks, meta={"reasoning_model": producer}
            ),
            2,
        ),
    )
    result = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    prior = next(m for m in agent.provider.wire_messages[0] if m.content == "prior")
    assert prior.reasoning_blocks == (blocks if producer == "override" else None)
