"""Isolate migration rehearsals in a schema of the explicitly selected test database."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import Engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from tests._helpers.pg_url import render, validate_test_pg_url


def test_database_url() -> str | None:
    """The validated, worker-scoped test database URL, or None when Postgres is off.

    For tests that also request the ``engine`` fixture, which has already created the
    database; tests that do not use ``engine`` take ``pg_test_url`` instead.
    """
    raw = os.environ.get("DENDRUX_TEST_PG_URL")
    return render(validate_test_pg_url(raw)) if raw else None


@asynccontextmanager
async def migration_schema(url: str, schema: str) -> Any:
    """Own ``schema`` inside the (already created) test database ``url``.

    Tests reach ``url`` through the ``pg_test_url`` fixture, which creates the database
    first; this helper only refuses unsafe URLs before it opens any connection.
    """
    validate_test_pg_url(url, append_suffix=False)
    assert schema.isidentifier()
    admin = create_async_engine(url, poolclass=NullPool)
    engine = create_async_engine(url, poolclass=NullPool)
    async with admin.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    def set_schema(connection: Any, _record: Any) -> None:
        previous = connection.autocommit
        connection.autocommit = True
        cursor = connection.cursor()
        try:
            cursor.execute(f'SET search_path TO "{schema}"')
        finally:
            cursor.close()
            connection.autocommit = previous

    event.listen(Engine, "connect", set_schema)
    try:
        yield engine
    finally:
        event.remove(Engine, "connect", set_schema)
        await engine.dispose()
        try:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            await admin.dispose()


def create_pre_activation_schema(connection: Any) -> None:
    """Build an additive model schema with the migration's pre-activation invariants.

    Only synthetic empty migration fixtures use this. Production activation is
    irreversible; these tests must not start from an active create_all database.
    """
    from dendrux.db._settings import put_setting
    from dendrux.db._storage_schema import ACTIVATION_INDEXES
    from dendrux.db.models import Base

    Base.metadata.create_all(connection)
    connection.execute(text("DELETE FROM dendrux_settings WHERE key='activate_step'"))
    put_setting(connection, "storage_format", "1")
    for name, (table, column, old) in ACTIVATION_INDEXES.items():
        connection.execute(text(f"DROP INDEX IF EXISTS {name}"))
        if old:
            connection.execute(
                text(f"CREATE INDEX IF NOT EXISTS {old} ON {table} (agent_run_id, {column})")
            )
