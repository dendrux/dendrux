"""Mixed-version pause safety and bounded migration lock acquisition."""

import datetime as dt

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Select

from dendrux import sweep
from dendrux.db.models import AgentRun


async def test_legacy_pause_requires_both_clocks_to_be_stale(store, engine):
    """A 0.4.1 pause writes status/updated_at, but not the 0.5 status clock."""
    now = dt.datetime.now(dt.UTC)
    old = now - dt.timedelta(hours=2)
    await store.create_run("legacy-pause", "agent")
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == "legacy-pause")
            .values(status_changed_at=old, updated_at=old)
        )
        # Preserve the clock's dialect-specific datetime representation.
        before = (await conn.execute(select(AgentRun.status_changed_at))).scalar_one()
        await conn.execute(
            text(
                "UPDATE agent_runs SET status='waiting_client_tool', updated_at=:now "
                "WHERE id='legacy-pause'"
            ),
            {"now": now},
        )
        assert (await conn.execute(select(AgentRun.status_changed_at))).scalar_one() == before
    result = await sweep(state_store=store, abandoned_waiting=dt.timedelta(hours=1))
    assert result.abandoned_waiting == [], "a fresh legacy pause must not be abandoned"
    assert (await store.get_run("legacy-pause")).status == "waiting_client_tool"
    assert not any(
        e.event_type == "run.abandoned" for e in await store.get_run_events("legacy-pause")
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == "legacy-pause").values(updated_at=old)
        )
    swept = await store.sweep_abandoned_runs(dt.timedelta(hours=1))
    assert [r.run_id for r in swept] == ["legacy-pause"], "both stale clocks must allow abandonment"


async def test_legacy_refresh_between_candidate_scan_and_cas_survives(store, engine, monkeypatch):
    old = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)
    await store.create_run("legacy-race", "agent")
    await store.pause_run("legacy-race", status="waiting_client_tool", pause_data={})
    async with engine.begin() as conn:
        await conn.execute(update(AgentRun).values(status_changed_at=old, updated_at=old))
    execute = AsyncSession.execute
    refreshed = False

    async def refresh(session, statement, *args, **kwargs):
        nonlocal refreshed
        result = await execute(session, statement, *args, **kwargs)
        if (
            isinstance(statement, Select)
            and "started" in statement.selected_columns
            and not refreshed
        ):
            refreshed = True
            # Commit a legacy writer's refresh after selection, before the CAS.
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE agent_runs SET status='waiting_client_tool', updated_at=:now "
                        "WHERE id='legacy-race'"
                    ),
                    {"now": dt.datetime.now(dt.UTC)},
                )
        return result

    monkeypatch.setattr(AsyncSession, "execute", refresh)
    assert await store.sweep_abandoned_runs(dt.timedelta(hours=1)) == [], (
        "the final CAS must recheck updated_at after a legacy refresh"
    )
    assert refreshed
    assert (await store.get_run("legacy-race")).status == "waiting_client_tool"


@pytest.mark.parametrize(
    "phase", ["add_column", "add_check", "set_not_null", "drop_check", "drop_column"]
)
async def test_exclusive_migration_lock_timeout_is_retryable(engine, phase, record_property):
    import asyncio
    import json
    import threading
    import time

    from sqlalchemy import event, inspect

    from tests.integration._migration_schema import migration_schema
    from tests.integration._migration_schema import test_database_url as pg_url
    from tests.integration.test_050_t7b_fix3 import _prior_schema
    from tests.integration.test_migration_run_control import _alembic

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres lock timeout and resumable migration")
    markers = {
        "add_column": "add column if not exists status_changed_at",
        "add_check": "add constraint ck_agent_runs_status_clock_not_null",
        "set_not_null": "alter column status_changed_at set not null",
        "drop_check": "drop constraint ck_agent_runs_status_clock_not_null",
        "drop_column": "drop column status_changed_at",
    }
    async with migration_schema(pg_url(), f"t7_fix5_{phase}") as target:
        await _prior_schema(target, 20)
        url = target.url.render_as_string(hide_password=False)
        if phase == "drop_column":
            await asyncio.to_thread(_alembic, url, "upgrade", "head")
        action, revision = (
            ("downgrade", "d7e8f9a0b1c2") if phase == "drop_column" else ("upgrade", "head")
        )
        loop = asyncio.get_running_loop()
        reached = asyncio.Event()
        release = threading.Event()
        thread_id = None
        observed = []
        initial = []
        restored = []

        def before(conn, cursor, statement, parameters, context, executemany):
            if threading.get_ident() != thread_id:
                return
            if "pg_try_advisory_lock(" in statement:
                initial.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
            if "pg_advisory_unlock(" in statement:
                restored.append(conn.exec_driver_sql("SHOW lock_timeout").scalar_one())
            if markers[phase] in statement.lower():
                timeout = conn.exec_driver_sql("SHOW lock_timeout").scalar_one()
                observed.append(timeout)
                loop.call_soon_threadsafe(reached.set)
                assert release.wait(20), "test must release the DDL after installing its blocker"
                assert timeout == "5s", "short exclusive migration steps must use a 5s lock_timeout"

        def migrate():
            nonlocal thread_id
            thread_id = threading.get_ident()
            _alembic(url, action, revision)

        async with target.connect() as blocker:
            transaction = await blocker.begin()
            event.listen(target.sync_engine.__class__, "before_cursor_execute", before)
            task = asyncio.create_task(asyncio.to_thread(migrate))
            try:
                await asyncio.wait_for(reached.wait(), 20)
                # ACCESS SHARE permits writers, but blocks the short exclusive DDL.
                await blocker.execute(text("SELECT id FROM agent_runs LIMIT 1"))
                started = time.perf_counter()
                release.set()
                with pytest.raises(RuntimeError, match="[Rr]etry") as error:
                    await asyncio.wait_for(asyncio.shield(task), 12)
                elapsed = time.perf_counter() - started
                assert elapsed < 10, "exclusive lock acquisition must remain bounded"
                assert error.value.__suppress_context__, (
                    "retry guidance must suppress raw DB chaining"
                )
                assert observed == ["5s"]
                assert restored == initial, (
                    "the direct connection must restore its prior lock_timeout"
                )
                record_property(
                    f"lock_timeout_{phase}",
                    json.dumps({"seconds": elapsed, "timeout": observed[0]}),
                )
            finally:
                release.set()
                await transaction.rollback()
                await asyncio.gather(task, return_exceptions=True)
                event.remove(target.sync_engine.__class__, "before_cursor_execute", before)
        async with target.connect() as conn:
            expected = "e8f9a0b1c2d3" if phase == "drop_column" else "d7e8f9a0b1c2"
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == expected
            assert (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one() == 20
        # Resume precisely the failed operation, including committed earlier steps.
        await asyncio.to_thread(_alembic, url, action, revision)
        if phase == "drop_column":
            await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with target.connect() as conn:
            column = await conn.run_sync(
                lambda c: next(
                    v
                    for v in inspect(c).get_columns("agent_runs")
                    if v["name"] == "status_changed_at"
                )
            )
            assert not column["nullable"]
            assert (
                await conn.execute(
                    text("SELECT count(*) FROM agent_runs WHERE status_changed_at IS NULL")
                )
            ).scalar_one() == 0
            assert (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM pg_constraint "
                        "WHERE conrelid='agent_runs'::regclass "
                        "AND conname='ck_agent_runs_status_clock_not_null'"
                    )
                )
            ).scalar_one() == 0
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == "e8f9a0b1c2d3"
            assert (
                await conn.execute(
                    text("""
                SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
                WHERE i.indrelid IN ('agent_runs'::regclass, 'run_events'::regclass)
                AND i.indisvalid AND i.indisready AND c.relname IN (
                    'ix_agent_runs_sweep_running', 'ix_agent_runs_sweep_waiting',
                    'ix_run_events_started')
            """)
                )
            ).scalar_one() == 3


async def test_later_downgrade_refusal_rolls_back_clock_removal(engine):
    import asyncio

    from sqlalchemy import inspect

    from dendrux.runtime.state import SQLAlchemyStateStore
    from tests.integration._migration_schema import migration_schema
    from tests.integration._migration_schema import test_database_url as pg_url
    from tests.integration.test_050_t1_r1_migration import PRIOR, _alembic, _seed

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres multi-revision downgrade rollback")
    async with migration_schema(pg_url(), "t7_fix5_chain_rollback") as target:
        url = target.url.render_as_string(hide_password=False)
        await asyncio.to_thread(_alembic, url, "upgrade", PRIOR)
        await _seed(target)
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        store = SQLAlchemyStateStore(target)
        await store.create_run(
            "duplicate", "a", tenant_id="B", idempotency_key="k1", idempotency_fingerprint="fp"
        )
        with pytest.raises(RuntimeError, match="more than one tenant"):
            await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        async with target.connect() as conn:
            columns = await conn.run_sync(
                lambda c: {v["name"] for v in inspect(c).get_columns("agent_runs")}
            )
            assert "status_changed_at" in columns, (
                "a later downgrade refusal must roll back status-clock removal"
            )
            assert (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one() == "e8f9a0b1c2d3"
        async with target.begin() as conn:
            await conn.execute(
                text("UPDATE agent_runs SET idempotency_key=NULL WHERE id='duplicate'")
            )
        await asyncio.to_thread(_alembic, url, "downgrade", PRIOR)
        await asyncio.to_thread(_alembic, url, "upgrade", "head")
        async with target.connect() as conn:
            assert (await conn.execute(text("SELECT count(*) FROM agent_runs"))).scalar_one() == 5
            assert (
                await conn.execute(
                    text("SELECT count(*) FROM agent_runs WHERE status_changed_at IS NULL")
                )
            ).scalar_one() == 0
