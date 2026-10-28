"""Backend and store matrix for the conformance suite.

``backend_engine``: in-memory SQLite always; Postgres when ``DENDRUX_TEST_PG_URL`` is set
(the URL's database is forced to a ``*_test`` name by the shared helpers; the owner of this
suite uses ``dendrux_060_0c_test``). Each test sees a clean schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from tests._helpers import v030_oracle as o
from tests.integration.conftest import (  # noqa: F401  (pg_database is a re-exported fixture)
    _PG_TRUNCATE_SQL,
    _engine_params,
    _resolve_pg_url,
    pg_database,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _conformance_pg_schema(pg_database: str | None) -> AsyncIterator[None]:  # noqa: F811
    """Create the current schema once per session on the private ``*_test`` database."""
    if not pg_database:
        yield
        return
    eng = create_async_engine(pg_database, poolclass=NullPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield
    finally:
        await eng.dispose()


@pytest_asyncio.fixture(params=_engine_params(), loop_scope="function")
async def backend_engine(
    request: pytest.FixtureRequest, _conformance_pg_schema: None
) -> AsyncIterator[AsyncEngine]:
    """An empty database on SQLite or Postgres."""
    if request.param == "postgres":
        url = _resolve_pg_url()
        if not url:
            pytest.skip("DENDRUX_TEST_PG_URL not set")
        eng = create_async_engine(url)
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
