"""Bounded resume contention and the point-lookup plan on the private test DB."""

import asyncio
import json
import time

import pytest
from sqlalchemy import event, text

from dendrux.db.models import AgentRun
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import agent, consume, result_for


async def test_fifty_resume_submit_races_on_ten_runs(store, engine, record_property):
    agents, paused, providers = [], [], []
    for _ in range(10):
        provider = ScriptedLLM(
            [tool_call("client_read", pid="a"), LLMResponse(text="done")], gate_at=1
        )
        owner = agent(store, provider)
        agents.append(owner)
        providers.append(provider)
        paused.append(await owner.run("go"))
    connections = {"active": 0, "peak": 0}

    def checkout(*args):
        connections["active"] += 1
        connections["peak"] = max(connections["peak"], connections["active"])

    def checkin(*args):
        connections["active"] -= 1

    event.listen(engine.sync_engine, "checkout", checkout)
    event.listen(engine.sync_engine, "checkin", checkin)
    durations = []

    async def race(index, variant):
        owner, pause = agents[index], paused[index]
        started = time.perf_counter()
        try:
            if variant % 3 == 0:
                return await owner.submit_tool_results(
                    pause.run_id, [result_for(pause)], expected_pause_id=pause.pause_id
                )
            if variant % 3 == 1:
                result, _ = await consume(
                    owner.resume_stream(
                        pause.run_id,
                        tool_results=[result_for(pause)],
                        expected_pause_id=pause.pause_id,
                    )
                )
                return result
            return await owner.resume(
                pause.run_id, tool_results=[result_for(pause)], expected_pause_id=pause.pause_id
            )
        finally:
            durations.append((time.perf_counter() - started) * 1000)

    tasks = [asyncio.create_task(race(i, variant)) for i in range(10) for variant in range(5)]
    try:
        await asyncio.wait_for(asyncio.gather(*(p.entered.wait() for p in providers)), 15)
        records = await asyncio.gather(*(store.get_run(pause.run_id) for pause in paused))
        assert all(record.status == "running" for record in records)
    finally:
        for provider in providers:
            provider.gate.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        event.remove(engine.sync_engine, "checkout", checkout)
        event.remove(engine.sync_engine, "checkin", checkin)
    assert sum(getattr(result, "status", None) == RunStatus.SUCCESS for result in outcomes) == 10
    assert all(provider.calls == 2 for provider in providers)
    for pause in paused:
        assert (await store.get_run(pause.run_id)).status == "success"
        assert await store.get_pause_state(pause.run_id) is None
        events = await store.get_run_events(pause.run_id)
        assert len({e.sequence_index for e in events}) == len(events)
        assert sum(e.event_type == "run.resumed" for e in events) == 1
        assert not any(e.event_type in {"run.error", "run.cancelled"} for e in events)
    durations.sort()
    metrics = {
        "p50_ms": round(durations[24], 2),
        "p95_ms": round(durations[47], 2),
        "peak_connections": connections["peak"],
    }
    record_property("resume_contention", json.dumps(metrics))
    print("resume_contention", engine.dialect.name, json.dumps(metrics))


async def test_claim_predicate_uses_run_id_index(engine, record_property):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres planner assertion")
    async with engine.begin() as connection:
        await connection.execute(
            AgentRun.__table__.insert(),
            [{"id": f"plan-{i}", "agent_name": "plan", "status": "running"} for i in range(1000)],
        )
        await connection.execute(text("ANALYZE agent_runs"))
        plan = (
            (
                await connection.execute(
                    text(
                        "EXPLAIN SELECT claim_epoch, cancel_requested FROM agent_runs "
                        "WHERE id = :run_id AND status = 'running' AND claim_epoch = :epoch"
                    ),
                    {"run_id": "plan-500", "epoch": 0},
                )
            )
            .scalars()
            .all()
        )
    assert any(
        "Index Scan" in line
        and any(name in line for name in ("agent_runs_pkey", "ix_agent_runs_sweep_running"))
        for line in plan
    ), plan
    assert any("Index Cond:" in line and "plan-500" in line for line in plan), plan
    assert not any("Seq Scan" in line for line in plan), plan
    record_property("claim_plan", "\n".join(plan))
    print("claim_plan", "\n".join(plan))
