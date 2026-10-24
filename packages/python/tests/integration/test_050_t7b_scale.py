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


async def test_sweep_projection_uses_status_index(store, engine, monkeypatch, record_property):
    """Explain the runtime's actual first/last ORM pages, including all waits."""
    from types import SimpleNamespace

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.dml import Update
    from sqlalchemy.sql.selectable import Select

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres planner evidence")
    statuses = [
        "running",
        "waiting_client_tool",
        "waiting_human_input",
        "waiting_approval",
        "paused",
        "success",
        "error",
        "cancelled",
    ]
    now = dt.datetime.now(dt.UTC)
    old = now - dt.timedelta(days=2)
    async with engine.begin() as connection:
        # Bounded seed batches: 20k runs / 200k events, 20k run.started events.
        # Every status has 2500 rows with fresh and expired ages interleaved.
        for base in range(0, 20000, 500):
            await connection.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": f"scale-{i:05}",
                        "agent_name": "plan",
                        "status": statuses[i % 8],
                        "last_progress_at": now if (i // 8) % 4 == 0 else old,
                        "created_at": old,
                        "updated_at": now if (i // 8) % 4 == 0 else old,
                    }
                    for i in range(base, base + 500)
                ],
            )
            await connection.execute(
                RunEvent.__table__.insert(),
                [
                    {
                        "id": f"event-{i:05}-{j}",
                        "agent_run_id": f"scale-{i:05}",
                        "event_type": "run.started" if j == 0 else "tool.completed",
                        "sequence_index": j,
                    }
                    for i in range(base, base + 500)
                    for j in range(10)
                ],
            )
        await connection.execute(text("ANALYZE agent_runs"))
        await connection.execute(text("ANALYZE run_events"))

    original = AsyncSession.execute
    captured = {}

    async def execute(session, statement, *args, **kwargs):
        if isinstance(statement, Update) and "failure_reason" in str(statement):
            # Observe candidate traversal without destroying the seed. A losing
            # CAS has no event insert, exactly as in the real contention path.
            return SimpleNamespace(scalar_one_or_none=lambda: None)
        if isinstance(statement, Select) and "started" in statement.selected_columns:
            params = statement.compile().params
            value = params["status_1"]
            category = tuple(value) if isinstance(value, list) else (value,)
            captured.setdefault(category, []).append(statement)
        return await original(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", execute)
    await store.sweep_stale_runs(dt.timedelta(days=1), batch_size=100, max_runs=10000)
    await store.sweep_abandoned_runs(dt.timedelta(days=1), batch_size=100, max_runs=10000)
    plans = {}
    async with engine.connect() as connection:
        for category, statements in captured.items():
            for position, statement in [("first", statements[0]), ("last", statements[-2])]:
                sql = str(
                    statement.compile(
                        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
                    )
                )
                plan = (
                    await connection.execute(text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql))
                ).scalar_one()[0]
                plans[f"{category}:{position}"] = plan
    record_property("sweep_realistic_plans", json.dumps(plans, default=str))
    for category, plan in plans.items():

        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)

        all_nodes = list(nodes(plan["Plan"]))
        assert not any(n["Node Type"] == "Sort" for n in all_nodes), (
            "sweep pages must not sort all matching rows"
        )
        scans = [n for n in all_nodes if n.get("Relation Name") == "agent_runs"]
        assert scans and all(
            n.get("Actual Rows", 0) + n.get("Rows Removed by Filter", 0) <= 101 for n in scans
        ), "each sweep page must read a bounded candidate range"
        assert any(n.get("Index Name") == "ix_run_events_run_type" for n in all_nodes), (
            "classification must use a run-and-type lookup, not all started events"
        )
        print("sweep_plan", category, json.dumps(plan, default=str))
    assert set(captured) == {(status,) for status in statuses[:5]}, (
        "explain all five sweep status categories"
    )
