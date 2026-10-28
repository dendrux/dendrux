"""Backend and store matrix for the conformance suite.

``backend_engine``: in-memory SQLite always; Postgres when ``DENDRUX_TEST_PG_URL`` is set
(only the exact database authorised in ``pg_guard.py``, ``dendrux_060_0c_test``; serial).
Each test sees a clean schema.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from tests._helpers import v030_oracle as o
from tests.conformance import pg_guard
from tests.integration.conftest import _PG_TRUNCATE_SQL, _ensure_test_db_exists

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest.fixture(scope="session")
def conformance_pg_url() -> str | None:
    """The authorised Postgres URL, or None when Postgres is not configured.

    Validated against the exact authorised database name before anything connects.
    """
    raw = os.environ.get(pg_guard.URL_ENV)
    if not raw or os.environ.get("PYTEST_XDIST_WORKER"):
        return None  # workers rename the database per worker: Postgres cells skip there
    pg_guard.authorize(raw)
    return raw


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _conformance_pg_schema(conformance_pg_url: str | None) -> AsyncIterator[None]:
    """Create the database if missing and the current schema once per session."""
    if not conformance_pg_url:
        yield
        return
    name = pg_guard.authorize(conformance_pg_url)  # re-checked right before DDL
    await _ensure_test_db_exists(conformance_pg_url, name)
    eng = create_async_engine(conformance_pg_url, poolclass=NullPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield
    finally:
        await eng.dispose()


def _params() -> list:
    pg = pytest.param("postgres", id="postgres")
    if os.environ.get(pg_guard.URL_ENV):
        return [pytest.param("sqlite", id="sqlite"), pg]
    skip = pytest.mark.skip(reason=f"{pg_guard.URL_ENV} not set")
    return [
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=skip),
    ]


@pytest_asyncio.fixture(params=_params(), loop_scope="function")
async def backend_engine(
    request: pytest.FixtureRequest,
    conformance_pg_url: str | None,
    _conformance_pg_schema: None,
) -> AsyncIterator[AsyncEngine]:
    """An empty database on SQLite or on the authorised Postgres database."""
    if request.param == "postgres":
        if os.environ.get("PYTEST_XDIST_WORKER"):
            pytest.skip("Postgres conformance runs serial (-n 0): workers rename the database")
        assert conformance_pg_url
        eng = create_async_engine(conformance_pg_url)
        try:
            async with eng.begin() as conn:
                await conn.execute(text(_PG_TRUNCATE_SQL))
            yield eng
        finally:
            await eng.dispose()
        return
    eng = await o.create_sqlite_engine()
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture(loop_scope="function")
async def sqlite_engine() -> AsyncIterator[AsyncEngine]:
    """In-memory SQLite only (store-fixture variants run here)."""
    eng = await o.create_sqlite_engine()
    try:
        yield eng
    finally:
        await eng.dispose()
