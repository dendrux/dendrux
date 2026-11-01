"""Refuse unsafe connections and give portable, sanitized schema diagnostics."""

import asyncio

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime.stores._factory import store_for_engine


@pytest.fixture
def backend_engine(engine):
    return engine


@pytest.mark.parametrize("route", ["engine", "execution_options"])
async def test_autocommit_refused_before_twenty_claims(backend_engine, route):
    options = (
        {"isolation_level": "AUTOCOMMIT"}
        if route == "engine"
        else {"execution_options": {"isolation_level": "AUTOCOMMIT"}}
    )
    if backend_engine.dialect.name in {"mysql", "mariadb"}:
        options["connect_args"] = {"charset": "utf8mb4"}
    borrowed = create_async_engine(backend_engine.url, **options)
    store = store_for_engine(borrowed)
    claims = []

    def capture(conn, cursor, sql, params, context, many):
        if sql.startswith("UPDATE agent_runs"):
            claims.append(sql)

    event.listen(borrowed.sync_engine, "before_cursor_execute", capture)
    try:
        results = await asyncio.gather(
            *(store.claim_paused_run("absent", expected_status="paused") for _ in range(20)),
            return_exceptions=True,
        )
        assert all(
            isinstance(result, ValueError) and "AUTOCOMMIT" in str(result) for result in results
        )
        assert claims == []
    finally:
        await borrowed.dispose()


@pytest.mark.parametrize("missing", ["table", "column"])
@pytest.mark.parametrize("operation", ["read", "create", "append"])
async def test_schema_driver_errors_are_typed(backend_engine, store, missing, operation):
    from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError

    armed = True

    def change_query(conn, cursor, sql, params, context, many):
        nonlocal armed
        if armed and any(
            marker in sql
            for marker in ("FROM agent_runs", "INSERT INTO agent_runs", "UPDATE agent_runs")
        ):
            armed = False
            return (
                "SELECT * FROM rx_missing_table"
                if missing == "table"
                else "SELECT rx_missing_column FROM agent_runs",
                (),
            )
        return sql, params

    event.listen(backend_engine.sync_engine, "before_cursor_execute", change_query, retval=True)
    try:
        expected = SchemaNotInitializedError if missing == "table" else SchemaOutOfDateError
        with pytest.raises(expected, match="dendrux db migrate"):
            if operation == "read":
                await store.get_run("absent")
            elif operation == "create":
                await store.create_run("new", "agent")
            else:
                await store.save_trace("absent", "user", "ok", order_index=0, claim_epoch=0)
        assert not armed
    finally:
        event.remove(backend_engine.sync_engine, "before_cursor_execute", change_query)
