"""Backend and store matrix for the conformance suite.

``backend_engine``: in-memory SQLite always; Postgres when ``DENDRUX_TEST_PG_URL`` is set
(only the exact database authorised in ``pg_guard.py``, ``dendrux_060_0c_test``; serial).
Each test sees a clean schema.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base
from dendrux.testing.store_kit.sql import SqlFaultInjector
from tests._fixtures.storage import pin_storage_format
from tests._helpers import v030_oracle as o
from tests.conformance import _kit_backends, pg_guard
from tests.integration.conftest import _PG_TRUNCATE_SQL, _ensure_test_db_exists

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from sqlalchemy.ext.asyncio import AsyncEngine


def pytest_configure(config: pytest.Config) -> None:
    """The published store kit runs here as a plugin, like it does for a store author."""
    config.pluginmanager.import_plugin("dendrux.testing.store_kit")
    config.pluginmanager.import_plugin("dendrux.testing.store_kit._plugin")


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
            await pin_storage_format(eng, 1)
            yield eng
        finally:
            await eng.dispose()
        return
    eng = await o.create_sqlite_engine()
    try:
        await pin_storage_format(eng, 1)
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture(loop_scope="function")
async def sqlite_engine() -> AsyncIterator[AsyncEngine]:
    """In-memory SQLite only (store-fixture variants run here)."""
    eng = await o.create_sqlite_engine()
    try:
        await pin_storage_format(eng, 1)
        yield eng
    finally:
        await eng.dispose()


# ---------------------------------------------------------------------------
# The fixtures a store author provides to the published test kit
# (``dendrux.testing.store_kit``): see tests/conformance/_kit_backends.py.
# ---------------------------------------------------------------------------


@pytest.fixture(params=_kit_backends.backend_params())
def kit_backend_kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def kit_backend_schema(kit_backend_kind: str, request: pytest.FixtureRequest) -> None:
    """Postgres: create the authorised database's schema once (resolved outside the loop)."""
    if kit_backend_kind == "postgres":
        request.getfixturevalue("_conformance_pg_schema")


@pytest_asyncio.fixture(loop_scope="function")
async def kit_backend(
    kit_backend_kind: str, kit_backend_schema: None, tmp_path: Any
) -> AsyncIterator[_kit_backends.KitBackend]:
    backend = _kit_backends.KitBackend(kit_backend_kind, tmp_path)
    try:
        yield backend
    finally:
        await backend.close()


@pytest.fixture
def dendrux_store_factory(kit_backend: _kit_backends.KitBackend) -> Any:
    return kit_backend.fresh_store


@pytest.fixture
def dendrux_store_reopen(kit_backend: _kit_backends.KitBackend) -> Any:
    return kit_backend.reopen


@pytest.fixture
def dendrux_store_second_instance(kit_backend: _kit_backends.KitBackend) -> Any:
    return kit_backend.second


@pytest.fixture
def dendrux_clock(kit_backend: _kit_backends.KitBackend) -> Any:
    return kit_backend


@pytest.fixture
def dendrux_fault_injector() -> Iterator[SqlFaultInjector]:
    injector = SqlFaultInjector()
    yield injector
    injector.close()
