"""A leaked table lock must fail the Postgres cleanup fixtures fast, never hang the worker."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from tests.integration import conftest as integration_conftest

# Outer guard only: the cleanup itself must give up long before this.
_GUARD_SECONDS = 15


@pytest.fixture
async def leaked_lock(pg_test_url, _pg_schema_setup, monkeypatch):
    """Another connection left idle in transaction holding a lock on a cleaned table."""
    monkeypatch.setattr(integration_conftest, "_PG_CLEANUP_LOCK_TIMEOUT", "1s")
    holder = create_async_engine(pg_test_url, poolclass=NullPool)
    try:
        async with holder.connect() as connection:
            await connection.execute(text("LOCK TABLE agent_runs IN ACCESS SHARE MODE"))
            yield
    finally:
        await holder.dispose()


@pytest.mark.parametrize("cleanup", ["_pg_truncate_all", "_pg_drop_all"])
async def test_cleanup_raises_when_a_table_lock_is_leaked(pg_test_url, leaked_lock, cleanup):
    cleaner = create_async_engine(pg_test_url, poolclass=NullPool)
    try:
        with pytest.raises(DBAPIError) as raised:
            async with cleaner.begin() as connection:
                await asyncio.wait_for(
                    getattr(integration_conftest, cleanup)(connection), _GUARD_SECONDS
                )
        assert getattr(raised.value.orig, "sqlstate", None) == "55P03"  # lock_not_available
    finally:
        await cleaner.dispose()
