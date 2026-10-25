"""Status transition timestamps, sweep paging and HOT update eligibility."""

import datetime as dt

import pytest
from sqlalchemy import event, select, text, update

from dendrux.db.models import AgentRun


async def test_status_timestamp_changes_only_on_transition(store, engine):
    await store.create_run("clock", "agent")

    async def stamp():
        async with engine.connect() as conn:
            return (
                (await conn.execute(select(AgentRun.__table__).where(AgentRun.id == "clock")))
                .one()
                ._mapping
            )

    before = await stamp()
    assert "status_changed_at" in before, "status clock must be stored independently of progress"
    await store.touch_progress("clock")
    touched = await stamp()
    assert touched["status_changed_at"] == before["status_changed_at"]
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=3)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == "clock").values(status_changed_at=old)
        )
    await store.pause_run("clock", status="paused", pause_data={})
    paused = await stamp()
    assert paused["status_changed_at"] > before["status_changed_at"]
    await store.pause_run("clock", status="paused", pause_data={"same": True})
    assert (await stamp())["status_changed_at"] == paused["status_changed_at"]
    await store.claim_paused_run("clock", expected_status="paused")
    running = await stamp()
    assert running["status_changed_at"] > paused["status_changed_at"]
    await store.finalize_run("clock", status="error")
    done = await stamp()
    assert done["status_changed_at"] > running["status_changed_at"]
    await store.finalize_run("clock", status="error")
    assert (await stamp())["status_changed_at"] == done["status_changed_at"]


async def test_progress_updates_are_hot_eligible(store, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres indexed-column dependencies")
    import asyncio

    from dendrux.runtime import context

    await store.create_run("hot", "agent")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE agent_runs"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    token = context.set_delegation_context(
        context.DelegationContext(
            run_id="hot", persisted=True, state_store=store, loop=asyncio.get_running_loop()
        )
    )
    try:
        await store.touch_progress("hot")
        context.heartbeat()
        await asyncio.gather(*tuple(context._heartbeat_tasks))
    finally:
        context.reset_delegation_context(token)
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 2
    async with engine.connect() as conn:
        indexed = (
            (
                await conn.execute(
                    text("""
            SELECT DISTINCT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid=i.indrelid
            WHERE i.indrelid='agent_runs'::regclass
            AND (a.attnum=ANY(i.indkey) OR EXISTS (
                SELECT 1 FROM pg_depend d WHERE d.objid=i.indexrelid
                AND d.refobjid=i.indrelid AND d.refobjsubid=a.attnum))
        """)
                )
            )
            .scalars()
            .all()
        )
    assert not set(indexed) & {"last_progress_at", "updated_at"}, (
        "heartbeats must remain HOT-eligible"
    )
    import re

    for statement in statements:
        set_clause = statement.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
        changed = set(re.findall(r"(\w+)=", set_clause))
        assert changed == {"last_progress_at", "updated_at"}
        assert not changed.intersection(indexed)


async def test_status_migration_backfills_and_repairs_interrupted_index(engine, tmp_path):
    import asyncio
    from contextlib import asynccontextmanager

    from sqlalchemy import inspect
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from tests.integration._migration_schema import migration_schema, test_database_url
    from tests.integration.test_migration_run_control import _alembic

    @asynccontextmanager
    async def target():
        if engine.dialect.name == "postgresql":
            async with migration_schema(test_database_url(), "t7_fix2_migration") as migrated:
                yield migrated
        else:
            migrated = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/backfill.db")
            try:
                yield migrated
            finally:
                await migrated.dispose()

    async with target() as migrated:
        old = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
        async with migrated.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                AgentRun.__table__.insert(),
                [
                    {
                        "id": "kept",
                        "agent_name": "duplicate",
                        "status": "paused",
                        "updated_at": old,
                    },
                    {
                        "id": "other",
                        "agent_name": "duplicate",
                        "status": "paused",
                        "updated_at": old,
                    },
                ],
            )
            for name in (
                "ix_agent_runs_sweep_running",
                "ix_agent_runs_sweep_waiting",
                "ix_run_events_started",
            ):
                await conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
            await conn.execute(text("ALTER TABLE agent_runs DROP COLUMN status_changed_at"))
        if migrated.dialect.name == "postgresql":
            async with migrated.connect() as conn:
                conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                with pytest.raises(IntegrityError):
                    await conn.execute(
                        text(
                            "CREATE UNIQUE INDEX CONCURRENTLY ix_agent_runs_sweep_waiting "
                            "ON agent_runs(agent_name)"
                        )
                    )
                assert (
                    await conn.execute(
                        text(
                            "SELECT indisvalid FROM pg_index "
                            "WHERE indexrelid='ix_agent_runs_sweep_waiting'::regclass"
                        )
                    )
                ).scalar_one() is False
        url = migrated.url.render_as_string(hide_password=False)
        await asyncio.to_thread(_alembic, url, "stamp", "d7e8f9a0b1c2")
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with migrated.connect() as conn:
            columns = await conn.run_sync(
                lambda c: {v["name"]: v for v in inspect(c).get_columns("agent_runs")}
            )
            assert "status_changed_at" in columns, "migration must add status clock and backfill it"
            assert not columns["status_changed_at"]["nullable"]
            rows = (
                (await conn.execute(text("SELECT status_changed_at = updated_at FROM agent_runs")))
                .scalars()
                .all()
            )
            assert rows == [True, True]
            names = await conn.run_sync(
                lambda c: {
                    i["name"]
                    for table in ("agent_runs", "run_events")
                    for i in inspect(c).get_indexes(table)
                }
            )
            if migrated.dialect.name == "sqlite":
                names.update(
                    (
                        await conn.execute(
                            text("SELECT name FROM sqlite_master WHERE type='index'")
                        )
                    ).scalars()
                )
            assert "uq_agent_runs_tenant_idempotency_key" in names
            assert {
                "ix_agent_runs_sweep_running",
                "ix_agent_runs_sweep_waiting",
                "ix_run_events_started",
            } <= names
            assert not {"ix_agent_runs_sweep_progress", "ix_run_events_run_type"} & names
        await asyncio.to_thread(_alembic, url, "downgrade", "d7e8f9a0b1c2")
        async with migrated.connect() as conn:
            names = await conn.run_sync(
                lambda c: {v["name"] for v in inspect(c).get_columns("agent_runs")}
            )
            assert "status_changed_at" not in names
            assert (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one() == 2


@pytest.mark.benchmark
async def test_hot_updates_recover_with_four_tools_per_run(store, engine, record_property):
    import asyncio
    import json
    import time

    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux import Agent, tool
    from dendrux.runtime.state import SQLAlchemyStateStore
    from dendrux.types import LLMResponse, RunStatus
    from tests._helpers.scripted_provider import ScriptedLLM, tool_call

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres HOT statistics")

    # Match the reviewer's shared pool (10 + 5), 200 runs and unique call IDs.
    engine = create_async_engine(engine.url, pool_size=10, max_overflow=5)
    store = SQLAlchemyStateStore(engine)

    @tool()
    async def work() -> str:
        return "ok"

    layouts = {
        "9391133f": [],
        "99f3ac7b": [
            "CREATE INDEX ix_agent_runs_sweep_progress "
            "ON agent_runs(status, coalesce(last_progress_at, created_at), id)",
            "CREATE INDEX ix_agent_runs_sweep_waiting ON agent_runs(status, updated_at, id)",
        ],
        "fixed": [
            "CREATE INDEX ix_agent_runs_sweep_running ON agent_runs(status, id)",
            "CREATE INDEX ix_agent_runs_sweep_waiting ON agent_runs(status, status_changed_at, id)",
        ],
    }
    metrics = {}
    phases = []

    class PhaseLLM(ScriptedLLM):
        async def complete(self, *args, **kwargs):
            # Keep the 200 live runs in the same call phase for every index
            # layout: scheduler/GC delays must not change heap packing cohorts.
            await phases[self.calls].wait()
            return await super().complete(*args, **kwargs)

    async def one_run(index):
        llm = PhaseLLM(
            [tool_call("work", pid=f"{index}-{i}") for i in range(4)] + [LLMResponse(text="done")]
        )
        agent = Agent(
            name="hot",
            prompt="p",
            tools=[work],
            provider=llm,
            state_store=store,
            strict_tenancy=False,
        )
        assert (await agent.run("go")).status == RunStatus.SUCCESS
        assert llm.calls == 5

    try:
        for label, ddl in layouts.items():
            phases = [asyncio.Barrier(200) for _ in range(5)]
            async with engine.begin() as conn:
                await conn.execute(text("TRUNCATE agent_runs CASCADE"))
                for name in (
                    "ix_agent_runs_sweep_running",
                    "ix_agent_runs_sweep_waiting",
                    "ix_agent_runs_sweep_progress",
                ):
                    await conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
                for statement in ddl:
                    await conn.execute(text(statement))
                await conn.execute(
                    text("SELECT pg_stat_reset_single_table_counters('agent_runs'::regclass)")
                )
            started = time.perf_counter()
            # Same shared pool and workload for every layout; 800 tool calls per sample.
            await asyncio.wait_for(asyncio.gather(*(one_run(i) for i in range(200))), 120)
            await engine.dispose()  # flush the owning backends' cumulative statistics
            async with engine.connect() as conn:
                row = (
                    await conn.execute(
                        text("""SELECT n_tup_upd, n_tup_hot_upd,
                    pg_indexes_size('agent_runs') FROM pg_stat_all_tables
                    WHERE relid='agent_runs'::regclass""")
                    )
                ).one()
                metrics[label] = {
                    "updates": row[0],
                    "hot_updates": row[1],
                    "hot_pct": round(100 * row[1] / row[0], 3),
                    "index_bytes": row[2],
                    "seconds": round(time.perf_counter() - started, 3),
                }
    finally:
        async with engine.begin() as conn:
            for name in (
                "ix_agent_runs_sweep_running",
                "ix_agent_runs_sweep_waiting",
                "ix_agent_runs_sweep_progress",
            ):
                await conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
            for statement in layouts["fixed"]:
                await conn.execute(text(statement))
        await engine.dispose()
    record_property("hot_probe", json.dumps(metrics))
    print("hot_probe", json.dumps(metrics))
    assert metrics["99f3ac7b"]["hot_updates"] == 0
    # Page packing and pool scheduling vary by a few rows. The catalog test
    # above enforces exact eligibility; this statistical check rejects lost HOT
    # (the old layout scores 0%) while allowing five percentage points of noise on 3000
    # updates: repeated serial runs differ from the baseline layout by up to ~1.2 points.
    assert metrics["fixed"]["hot_pct"] >= metrics["9391133f"]["hot_pct"] - 5, metrics
    assert metrics["fixed"]["index_bytes"] < metrics["99f3ac7b"]["index_bytes"], metrics


async def test_running_sweep_counts_fresh_rows_against_work_cap(store, engine):
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    for i in range(6):
        await store.create_run(f"bounded-{i}", "agent")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id >= "bounded-3")
            .values(last_progress_at=old, created_at=old)
        )
    swept = await store.sweep_stale_runs(dt.timedelta(days=1), batch_size=2, max_runs=3)
    assert swept == [], "fresh running candidates must count against the inspection cap"
    assert len(await store.sweep_stale_runs(dt.timedelta(days=1), batch_size=2, max_runs=6)) == 3


async def test_waiting_sweep_recent_metadata_delays_abandonment(store, engine):
    await store.create_run("waiting-age", "agent")
    await store.pause_run("waiting-age", status="paused", pause_data={})
    old = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == "waiting-age").values(status_changed_at=old)
        )
    await store.update_run_meta("waiting-age", {"dendrux.patch": 1})
    assert await store.sweep_abandoned_runs(dt.timedelta(days=1)) == [], (
        "recent metadata delays abandonment for mixed-version safety"
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == "waiting-age").values(updated_at=old)
        )
    assert [r.run_id for r in await store.sweep_abandoned_runs(dt.timedelta(days=1))] == [
        "waiting-age"
    ], "abandonment resumes once both clocks are stale"
