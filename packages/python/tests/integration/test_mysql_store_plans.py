"""Plans for actual store SQL and driver traffic from server instrumentation."""

# ruff: noqa: F401, F811
import asyncio
import json
import time

import pytest
from sqlalchemy import event, insert, text

from dendrux.db.models import AgentRun, RunEvent
from dendrux.runtime.stores.sql.mysql import MySQLStateStore
from dendrux.types import generate_ulid
from tests.integration._mysql import mysql_engine, store


async def test_store_hot_query_explain(store, mysql_engine, caplog):
    caplog.set_level("ERROR", logger="asyncmy")
    root = generate_ulid()
    await store.create_run(
        root, "agent", tenant_id="t", idempotency_key="root", idempotency_fingerprint="digest"
    )
    rows = [
        dict(
            id=generate_ulid(),
            agent_name="agent",
            tenant_id=f"scope-{i}",
            parent_run_id=root if i < 20 else None,
            idempotency_key=f"key-{i}",
            idempotency_fingerprint="digest",
        )
        for i in range(500)
    ]
    async with mysql_engine.begin() as conn:
        await conn.execute(insert(AgentRun), rows)
        await conn.execute(
            insert(RunEvent),
            [
                dict(id=generate_ulid(), agent_run_id=row["id"], event_type="one", sequence_index=0)
                for row in rows
            ],
        )
        await conn.execute(text("ANALYZE TABLE agent_runs, run_events"))
    captured = []

    def capture(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith(("SELECT", "UPDATE", "INSERT")):
            captured.append((statement, parameters))

    event.listen(mysql_engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store._check_idempotency("root", "digest", tenant_id="t")
        await store.get_run(root, tenant_id="t")
        await store.get_run_control(root)
        await store.get_run_events(rows[0]["id"], tenant_id="scope-0", limit=10)
        await store.get_children_page([root], after=None, limit=10, tenant_id="t")
        await store.list_runs(tenant_id="scope-0", limit=10)
        await store.add_note(
            root, note_id=generate_ulid(), text="bounded", tenant_id="t", max_notes=1
        )
        await store.save_run_event(root, event_type="one")
        await store.pause_with_event(
            root, status="waiting_client_tool", pause_data={"pause_id": "p"}
        )
        await store.claim_paused_run(
            root, expected_status="waiting_client_tool", expected_pause_id="p"
        )
        await store.finalize_with_event(
            root, status="success", allowed_current_statuses=["running"], event_type="run.completed"
        )
    finally:
        event.remove(mysql_engine.sync_engine, "before_cursor_execute", capture)
    async with mysql_engine.connect() as conn:
        for sql, params in captured:
            plan = [
                dict(row)
                for row in (await conn.exec_driver_sql("EXPLAIN " + sql, params)).mappings()
            ]
            print("EXPLAIN", json.dumps({"sql": sql, "plan": plan}, default=str))
            for row in plan:
                if (
                    row.get("table") in {"agent_runs", "run_events"}
                    and row.get("type")
                    not in {
                        None,
                        "system",
                    }
                    and row.get("select_type") != "INSERT"
                ):
                    assert row.get("key") is not None, plan
    assert any("agent_runs.tenant_scope" in sql for sql, _ in captured)
    assert not any("coalesce(agent_runs.tenant_id" in sql.lower() for sql, _ in captured)


async def test_driver_traffic_and_checkout_latency(mysql_engine):
    owned = MySQLStateStore.from_url(
        mysql_engine.url.render_as_string(hide_password=False), pool_size=1, max_overflow=0
    )
    query = text(
        "SELECT EVENT_NAME, COUNT_STAR FROM "
        "performance_schema.events_statements_summary_by_thread_by_event_name "
        "WHERE THREAD_ID = :thread AND COUNT_STAR > 0"
    )
    try:
        async with owned._engine.connect() as conn:
            enabled = await conn.scalar(text("SELECT @@performance_schema"))
            thread = (
                await conn.scalar(
                    text(
                        "SELECT THREAD_ID FROM performance_schema.threads "
                        "WHERE PROCESSLIST_ID = CONNECTION_ID()"
                    )
                )
                if enabled
                else None
            )
            cold = dict((await conn.execute(query, {"thread": thread})).all()) if enabled else {}
        timings = []
        for _ in range(10):
            started = time.perf_counter()
            async with owned._engine.connect() as conn:
                timings.append((time.perf_counter() - started) * 1000)
                await conn.execute(text("SELECT 1"))
        async with owned._engine.connect() as conn:
            warm = dict((await conn.execute(query, {"thread": thread})).all()) if enabled else {}
        delta = {
            name: count - cold.get(name, 0)
            for name, count in warm.items()
            if count != cold.get(name, 0)
        }
        held = await owned._engine.connect()
        entered = asyncio.Event()

        async def blocked_checkout():
            entered.set()
            started = time.perf_counter()
            async with owned._engine.connect():
                return (time.perf_counter() - started) * 1000

        waiter = asyncio.create_task(blocked_checkout())
        try:
            await entered.wait()
            await asyncio.sleep(0.05)
        finally:
            await held.close()
        pool_wait = await asyncio.wait_for(waiter, 5)
        assert pool_wait >= 45
        print(
            "DRIVER_TRAFFIC",
            json.dumps(
                {
                    "enabled": bool(enabled),
                    "cold": cold,
                    "warm_delta": delta,
                    "checkout_ms": timings,
                    "saturated_pool_wait_ms": pool_wait,
                },
                sort_keys=True,
            ),
        )
        if not enabled:
            pytest.skip("performance_schema disabled; driver traffic not certified")
        if not cold:
            pytest.skip("statement instrumentation disabled; driver traffic not certified")
        assert not any("set_option" in name and count for name, count in delta.items())
    finally:
        await owned._engine.dispose()
