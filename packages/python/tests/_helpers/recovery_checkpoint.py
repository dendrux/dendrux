"""Seed a durable recovery checkpoint and mark its expired lease."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from dendrux import sweep
from dendrux.db.models import AgentRun
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.state import EventWrite
from dendrux.runtime.submit import governance_snapshot
from dendrux.types import Message, Role, UsageStats, _usage_to_dict, generate_ulid


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


async def expire(store, engine, run):
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    return next(
        r
        for r in (
            await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
        ).reclaimable
        if r.run_id == run
    )
