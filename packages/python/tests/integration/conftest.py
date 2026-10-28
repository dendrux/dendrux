"""Backend matrix for integration tests — every store-touching test runs on both engines.

Two backends are exposed via the ``engine`` fixture:

- ``sqlite``: in-memory SQLite (default; always runs).
- ``postgres``: real Postgres at ``DENDRUX_TEST_PG_URL`` (skipped if unset).

Concurrent reader/writer tests can explicitly request ``sqlite_file`` instead
of ``sqlite`` to get a temporary database with independent connections.

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

import importlib.util
import os
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from dendrux.db._settings import seed_settings
from dendrux.db.models import Base
from tests._helpers.pg_url import (
    TEST_SUFFIX,
    UnsafeTestDatabaseUrl,
    maintenance_url,
    render,
    validate_test_pg_url,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine


_PG_URL_ENV = "DENDRUX_TEST_PG_URL"
_TEST_DB_SUFFIX = TEST_SUFFIX
_REWRITE_WARNED = False


def pytest_configure(config: pytest.Config) -> None:
    """Keep optional NLP disk caches inside this test process's writable scratch space.

    Both full-suite commands collect integration tests before running any test;
    this historic hook also configures the docs/unit cases in those sessions.
    Each xdist worker gets its own cache and restores the extractor at shutdown.
    """
    if importlib.util.find_spec("tldextract") is None:
        return
    from tldextract.cache import DiskCache
    from tldextract.tldextract import TLD_EXTRACTOR

    cache = TemporaryDirectory(prefix="dendrux-test-tldextract-")
    config.add_cleanup(cache.cleanup)
    patch = pytest.MonkeyPatch()
    patch.setenv("TLDEXTRACT_CACHE", cache.name)
    patch.setattr(TLD_EXTRACTOR, "_cache", DiskCache(cache.name))
    config.add_cleanup(patch.undo)


def _rewrite_to_test_db(url: str) -> tuple[str, str, bool]:
    """Ensure URL points at a ``*_test`` database.

    Returns ``(rewritten_url, db_name, was_rewritten)``. Appends ``_test`` to the
    database name when it is missing. Validation (see ``tests/_helpers/pg_url.py``)
    refuses query options that would override the database, names outside
    ``[A-Za-z0-9_]`` (the name is interpolated into ``CREATE DATABASE``; PG does not
    bind-parameterize DDL identifiers) and identifiers over PostgreSQL's 63 bytes.
    """
    validated = validate_test_pg_url(url)
    name = validated.database or ""
    return render(validated), name, name != make_url(url).database


async def _ensure_test_db_exists(url: str, db_name: str) -> None:
    """``CREATE DATABASE {db_name}`` on the maintenance DB if missing.

    Connects to ``postgres`` (the standard maintenance database) on the same
    host with AUTOCOMMIT — PG forbids ``CREATE DATABASE`` inside a transaction.
    Idempotent: if the DB already exists, this is a no-op. The URL is validated
    before any engine exists.
    """
    validated = validate_test_pg_url(url, append_suffix=False)
    if validated.database != db_name:
        raise UnsafeTestDatabaseUrl("database to create does not match the validated URL")

    eng = create_async_engine(
        render(maintenance_url(validated)), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    try:
        async with eng.connect() as conn:
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            )
            if result.first() is None:
                try:
                    # db_name passed validation ([A-Za-z0-9_], <= 63): safe to interpolate.
                    await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                except ProgrammingError as exc:
                    if "already exists" not in str(exc.orig):  # lost a creation race: fine
                        raise
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
async def pg_database() -> AsyncIterator[str | None]:
    """Make sure the validated ``*_test`` database exists. Yields its URL, or None if unset.

    Every Postgres fixture and test must depend on this (directly or through
    ``pg_test_url`` / ``_pg_schema_setup``): under xdist each worker owns a database that
    does not exist on a fresh server, and nothing else creates it. It only creates the
    database; the public-schema lifecycle lives in ``_pg_schema_setup``.
    ``test_pg_fixture_bootstrap`` pins that no Postgres-touching fixture skips it.
    """
    url = _resolve_pg_url()
    if url:
        _, db_name, _ = _rewrite_to_test_db(os.environ[_PG_URL_ENV])
        await _ensure_test_db_exists(url, db_name)
    yield url


@pytest.fixture
def pg_test_url(pg_database: str | None) -> str:
    """The validated, existing test database URL for tests that manage their own schema."""
    if pg_database is None:
        pytest.skip(f"{_PG_URL_ENV} not set")
    return pg_database


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _pg_schema_setup(pg_database: str | None) -> AsyncIterator[None]:
    # Marker-style fixture: depends on its side effect (public schema exists for the
    # rest of the session), not on the yielded value. The function-scoped ``engine``
    # fixture takes it as a positional dep purely to establish ordering — schema must be
    # in place before any test tries to TRUNCATE it.
    url = pg_database
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
async def engine(request, _pg_schema_setup, tmp_path) -> AsyncIterator[AsyncEngine]:
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
                await conn.run_sync(lambda c: seed_settings(c, active=True))
            yield eng
        finally:
            await eng.dispose()
        return

    # Concurrent sessions require separate connections: an in-memory database
    # shares one connection and lets reader rollbacks undo pending writes.
    sqlite_url = (
        f"sqlite+aiosqlite:///{tmp_path / 'integration.db'}"
        if backend == "sqlite_file"
        else "sqlite+aiosqlite:///:memory:"
    )
    eng = create_async_engine(
        sqlite_url,
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
