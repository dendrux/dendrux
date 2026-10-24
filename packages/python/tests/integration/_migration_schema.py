"""Isolate migration rehearsals in a schema of the explicitly selected test database."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import Engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


def test_database_url() -> str | None:
    url = os.environ.get("DENDRUX_TEST_PG_URL")
    if url:
        assert urlparse(url).path.endswith("_test"), "Migration tests require a test database"
    return url


@asynccontextmanager
async def migration_schema(url: str, schema: str) -> Any:
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
