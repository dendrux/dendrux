"""Budget correctness under load and maintenance-query plan evidence."""

import asyncio
import datetime as dt
import json
import time

import pytest
from sqlalchemy import event, text

from dendrux import Agent, tool
from dendrux.db.models import AgentRun, RunEvent
from dendrux.types import Budget, RunStatus, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


async def test_two_hundred_budget_runs_with_lost_forensics(
    store, engine, monkeypatch, record_property
):
    @tool()
    async def work() -> str:
        return "ok"

    async def lost(*args, **kwargs):
        raise RuntimeError("fixture evidence outage")

    monkeypatch.setattr(store, "save_llm_interaction", lost)
    checked = peak = 0

    def checkout(*args):
        nonlocal checked, peak
        checked += 1
        peak = max(peak, checked)

    def checkin(*args):
        nonlocal checked
        checked -= 1

    durations = []

    async def run_one():
        responses = [tool_call("work", pid=str(i)) for i in range(3)]
        for r in responses:
            r.usage = UsageStats(input_tokens=300, total_tokens=300)
        llm = ScriptedLLM(responses)
        a = Agent(
            name="load",
            prompt="p",
            tools=[work],
            provider=llm,
            state_store=store,
            budget=Budget(max_tokens=500, count_cache_tokens=True, on_exceed="cancel"),
            strict_tenancy=False,
        )
        started = time.perf_counter()
        result = await a.run("go")
        durations.append((time.perf_counter() - started) * 1000)
        assert result.status == RunStatus.CANCELLED and llm.calls == 2
        assert (await store.get_run(result.run_id)).status == "cancelled"
        events = await store.get_run_events(result.run_id)
        assert len({e.sequence_index for e in events}) == len(events)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1

    event.listen(engine.sync_engine, "checkout", checkout)
    event.listen(engine.sync_engine, "checkin", checkin)
    try:
        await asyncio.wait_for(asyncio.gather(*(run_one() for _ in range(200))), 90)
    finally:
        event.remove(engine.sync_engine, "checkout", checkout)
        event.remove(engine.sync_engine, "checkin", checkin)
    durations.sort()
    metrics = {
        "p50_ms": round(durations[99], 2),
        "p95_ms": round(durations[189], 2),
        "peak_connections": peak,
    }
    record_property("budget_load", json.dumps(metrics))
    print("budget_load", engine.dialect.name, json.dumps(metrics))


async def test_sweep_projection_uses_status_index(engine, record_property):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres planner evidence")
    async with engine.begin() as connection:
        await connection.execute(
            AgentRun.__table__.insert(),
            [
                {"id": f"plan-{i:05}", "agent_name": "plan", "status": "success"}
                for i in range(10000)
            ]
            + [{"id": "stale", "agent_name": "plan", "status": "running"}],
        )
        await connection.execute(
            RunEvent.__table__.insert(),
            [
                {
                    "id": f"event-{i:05}",
                    "agent_run_id": f"plan-{i:05}",
                    "event_type": "tool.completed",
                    "sequence_index": 0,
                }
                for i in range(10000)
            ]
            + [
                {
                    "id": "event-stale",
                    "agent_run_id": "stale",
                    "event_type": "run.started",
                    "sequence_index": 0,
                }
            ],
        )
        await connection.execute(text("ANALYZE agent_runs"))
        await connection.execute(text("ANALYZE run_events"))
        plan = (
            (
                await connection.execute(
                    text(
                        "EXPLAIN SELECT id, agent_name, status, last_progress_at, iteration_count, "
                        "EXISTS (SELECT 1 FROM run_events WHERE agent_run_id = agent_runs.id "
                        "AND event_type = 'run.started') AS started "
                        "FROM agent_runs WHERE status IN ('running') AND "
                        "coalesce(last_progress_at, created_at) < :cutoff "
                        "AND id > '' ORDER BY id LIMIT 100"
                    ),
                    {"cutoff": dt.datetime.now(dt.UTC)},
                )
            )
            .scalars()
            .all()
        )
    assert any("Index" in line and "status" in line for line in plan)
    assert any("Index" in line and "run_events" in line for line in plan)
    assert not any("Seq Scan" in line for line in plan)
    record_property("sweep_plan", "\n".join(plan))
    print("sweep_plan", "\n".join(plan))
