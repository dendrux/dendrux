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

## Test database safety (Django-style)

Tests are destructive — schema gets dropped and recreated, rows get
``TRUNCATE``-d between tests. Pointing them at a dev or prod database
deletes data. To prevent that, the URL handed to ``DENDRUX_TEST_PG_URL``
is auto-rewritten so the database name always ends in ``_test``:

    DENDRUX_TEST_PG_URL=postgresql+asyncpg://u:p@host:5432/dendrux
    # ↓ auto-rewritten to ↓
    postgresql+asyncpg://u:p@host:5432/dendrux_test

The rewrite emits a one-line warning the first time it kicks in. The
``dendrux_test`` database is created automatically on first use (via the
maintenance ``postgres`` database on the same host); subsequent runs
reuse it. A hard guardrail refuses to proceed if the final DB name does
not end in ``_test`` — there is no escape hatch on purpose.

To run the matrix locally:

    DENDRUX_TEST_PG_URL=postgresql+asyncpg://user:pw@host:5432/dbname \\
        pytest tests/integration

Without the env var, the Postgres parametrization is skipped (so local
dev with no Postgres still gets a green suite).
"""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

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
_TEST_DB_SUFFIX = "_test"
_DB_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_REWRITE_WARNED = False


def _rewrite_to_test_db(url: str) -> tuple[str, str, bool]:
    """Ensure URL points at a ``*_test`` database.

    Returns ``(rewritten_url, db_name, was_rewritten)``. Rewrites the URL's
    database name by appending ``_test`` if it doesn't already end that way.
    Validates the final name matches ``[A-Za-z0-9_]+`` so it can be safely
    interpolated into ``CREATE DATABASE`` (asyncpg/PG don't bind-parameterize
    DDL identifiers).
    """
    parsed = urlparse(url)
    db_name = parsed.path.lstrip("/")
    if not db_name:
        raise pytest.UsageError(f"{_PG_URL_ENV} has no database name in path: {url!r}")

    rewritten = False
    if not db_name.endswith(_TEST_DB_SUFFIX):
        db_name = f"{db_name}{_TEST_DB_SUFFIX}"
        parsed = parsed._replace(path=f"/{db_name}")
        url = urlunparse(parsed)
        rewritten = True

    if not _DB_NAME_RE.match(db_name):
        raise pytest.UsageError(
            f"{_PG_URL_ENV} database name {db_name!r} contains characters "
            f"outside [A-Za-z0-9_]; refuse to interpolate into DDL."
        )
    if not db_name.endswith(_TEST_DB_SUFFIX):
        # Belt + suspenders — the rewrite above guarantees this, but keep
        # the assert so a future refactor can't regress the safety property.
        raise pytest.UsageError(
            f"{_PG_URL_ENV} resolved DB name {db_name!r} does not end in "
            f"{_TEST_DB_SUFFIX!r} — refusing to run destructive tests against it."
        )
    return url, db_name, rewritten


async def _ensure_test_db_exists(url: str, db_name: str) -> None:
    """``CREATE DATABASE {db_name}`` on the maintenance DB if missing.

    Connects to ``postgres`` (the standard maintenance database) on the same
    host with AUTOCOMMIT — PG forbids ``CREATE DATABASE`` inside a transaction.
    Idempotent: if the DB already exists, this is a no-op.
    """
    parsed = urlparse(url)
    maintenance_url = urlunparse(parsed._replace(path="/postgres"))

    eng = create_async_engine(maintenance_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        async with eng.connect() as conn:
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            )
            if result.first() is None:
                # db_name is validated against _DB_NAME_RE above; safe to interpolate.
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        await eng.dispose()


def _resolve_pg_url() -> str | None:
    """Read env, rewrite to ``*_test``, validate. Returns URL or None if unset."""
    global _REWRITE_WARNED
    raw = os.environ.get(_PG_URL_ENV)
    if not raw:
        return None
    url, db_name, rewritten = _rewrite_to_test_db(raw)
    if rewritten and not _REWRITE_WARNED:
        sys.stderr.write(
            f"\n[dendrux tests] {_PG_URL_ENV} rewritten to use database "
            f"{db_name!r} (tests are destructive — never run them against a "
            f"non-{_TEST_DB_SUFFIX} database).\n"
        )
        _REWRITE_WARNED = True
    return url


def _engine_params() -> list:
    # Evaluated at conftest import — DENDRUX_TEST_PG_URL must be set in the
    # shell environment before pytest is invoked. Setting it inside a
    # fixture or conftest hook is too late; the parametrization is already
    # decided. (CI sets it via the workflow `env:` block, locally the
    # caller exports it.)
    sqlite_param = pytest.param("sqlite", id="sqlite")
    if _resolve_pg_url() is not None:
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
    url = _resolve_pg_url()
    if not url:
        yield
        return

    # Auto-create the *_test database if missing — Django-style. Connects
    # to the postgres maintenance DB on the same host. See module docstring.
    _, db_name, _ = _rewrite_to_test_db(os.environ[_PG_URL_ENV])
    await _ensure_test_db_exists(url, db_name)

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
        url = _resolve_pg_url()
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
