"""Backend matrix for integration tests — every store-touching test runs on both engines.

Two backends are exposed via the ``engine`` fixture:

- ``sqlite``: in-memory SQLite (default; always runs).
- ``postgres``: real Postgres at ``DENDRUX_TEST_PG_URL`` (skipped if unset).

Each test sees a clean schema. The two backends are reset differently:

- SQLite is per-test (in-memory engine, cheap to create/destroy).
- Postgres uses a session-scoped DDL fixture that creates the schema once.
  Each test then gets a fresh engine bound to its own event loop (asyncpg
  connections are loop-bound; sharing pooled connections across per-test
  loops raises "another operation is in progress"), and starts by
  ``TRUNCATE``ing the tables. ``TRUNCATE`` is ~5ms; ``drop_all`` +
  ``create_all`` per test is ~80ms of DDL — the saved schema-rebuild cost
  is the optimization we wanted.

The matrix exists because the original tz-naive bug shipped specifically
because no integration test exercised Postgres — without this, the same
class of regression can re-enter unnoticed.

To run the matrix locally:

    DENDRUX_TEST_PG_URL=postgresql+asyncpg://user:pw@host:5432/dbname \\
        pytest tests/integration

Without the env var, the Postgres parametrization is skipped (so local
dev with no Postgres still gets a green suite).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from dendrux.db.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine


_PG_URL_ENV = "DENDRUX_TEST_PG_URL"


def _engine_params() -> list:
    # Evaluated at conftest import — DENDRUX_TEST_PG_URL must be set in the
    # shell environment before pytest is invoked. Setting it inside a
    # fixture or conftest hook is too late; the parametrization is already
    # decided. (CI sets it via the workflow `env:` block, locally the
    # caller exports it.)
    sqlite_param = pytest.param("sqlite", id="sqlite")
    if os.environ.get(_PG_URL_ENV):
        return [sqlite_param, pytest.param("postgres", id="postgres")]
    return [
        sqlite_param,
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skip(reason=f"{_PG_URL_ENV} not set"),
        ),
    ]


# Comma-separated table list, sorted by FK dependency. Built once at import
# from the single declarative ``Base``. If a future table is ever registered
# on a different declarative base it will silently NOT be truncated between
# tests, allowing data to leak across the matrix — keep all models on
# ``dendrux.db.models.Base`` or extend this resolution explicitly.
_PG_TRUNCATE_SQL = "TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE".format(
    tables=", ".join(t.name for t in Base.metadata.sorted_tables),
)


# ---------------------------------------------------------------------------
# Postgres: session-scoped schema lifecycle
#
# DDL fires once per session via a short-lived NullPool engine (no loop
# binding to worry about). Per-test engines are created fresh in their
# own loop and reset state via TRUNCATE — much cheaper than drop_all +
# create_all on every test.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _pg_schema_setup() -> AsyncIterator[None]:
    # Marker-style fixture: depends on its side effect (schema exists for
    # the rest of the session), not on the yielded value. The function-
    # scoped ``engine`` fixture takes it as a positional dep purely to
    # establish ordering — schema must be in place before any test
    # tries to TRUNCATE it.
    url = os.environ.get(_PG_URL_ENV)
    if not url:
        yield
        return

    # NullPool: this engine only runs DDL once and is disposed; we don't
    # want it holding pooled connections bound to the session loop.
    eng = create_async_engine(url, poolclass=NullPool)
    try:
        async with eng.begin() as conn:
            # Drop any leftover schema from a prior run (e.g. older models).
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield
    finally:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await eng.dispose()


# ---------------------------------------------------------------------------
# Function-scoped engine fixture (the one tests actually use)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(params=_engine_params(), loop_scope="function")
async def engine(request, _pg_schema_setup) -> AsyncIterator[AsyncEngine]:
    """Async engine parametrized across the backend matrix."""
    backend = request.param

    if backend == "postgres":
        url = os.environ.get(_PG_URL_ENV)
        if not url:
            pytest.skip(f"{_PG_URL_ENV} not set")
        # Fresh engine in this test's event loop — asyncpg connections
        # bind to the loop that opens them.
        eng = create_async_engine(url)
        try:
            async with eng.begin() as conn:
                await conn.execute(text(_PG_TRUNCATE_SQL))
            yield eng
        finally:
            await eng.dispose()
        return

    # SQLite: fresh in-memory engine per test (cheap).
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # SQLite needs PRAGMA foreign_keys = ON for CASCADE/SET NULL to fire.
    # PG enforces FKs natively, no listener needed.
    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _connection_record):  # noqa: ANN001
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
def store(engine):
    """StateStore backed by the matrix engine."""
    from dendrux.runtime.state import SQLAlchemyStateStore

    return SQLAlchemyStateStore(engine)


@pytest.fixture
def session_factory(engine):
    """Raw async session factory for direct DB assertions."""
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
