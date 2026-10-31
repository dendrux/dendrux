"""The repo's own store fixtures for the published test kit (``dendrux.testing.store_kit``).

This is what a store author writes: ``tests/conformance/conftest.py`` exposes it as the
``dendrux_*`` fixtures. Backends: in-memory SQLite, file SQLite, in-memory SQLite pinned to
storage format 1, Postgres (only the database authorised in ``pg_guard.py``), and
MySQL / MariaDB when ``DENDRUX_TEST_MYSQL_URL`` / ``DENDRUX_TEST_MARIADB_URL`` are set, in a
private database named ``dendrux_p5_f_<id>_test`` made and dropped per test.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._settings import put_setting, seed_settings
from dendrux.db.models import AgentRun, Base
from dendrux.runtime.stores._factory import store_for_engine
from tests.conformance import pg_guard
from tests.integration.conftest import _PG_TRUNCATE_SQL

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

KINDS = ("sqlite-memory", "sqlite-file", "sqlite-format1", "postgres", "mysql", "mariadb")
MYSQL_ENV = {"mysql": "DENDRUX_TEST_MYSQL_URL", "mariadb": "DENDRUX_TEST_MARIADB_URL"}
_TIME_COLUMNS = (
    "last_progress_at",
    "status_changed_at",
    "updated_at",
    "created_at",
    "lease_expired_at",
    "last_reclaimed_at",
)


def skip_reason(kind: str) -> str | None:
    """Why a backend cannot run here, or ``None`` when it can."""
    if kind == "postgres":
        if not os.environ.get(pg_guard.URL_ENV):
            return f"{pg_guard.URL_ENV} not set"
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return "Postgres conformance runs serial (-n 0): workers rename the database"
    if kind in MYSQL_ENV:
        if not os.environ.get(MYSQL_ENV[kind]):
            return f"{MYSQL_ENV[kind]} not set"
        if importlib.util.find_spec("dendrux.runtime.stores.sql.mysql") is None:
            return "MySQLStateStore is not on this branch yet (track B)"
    return None


def _sqlite_engine(url: str) -> AsyncEngine:
    engine = create_async_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    return engine


class KitBackend:
    """One database the kit can build empty stores on, with the author-side fixtures' work."""

    def __init__(self, kind: str, tmp_path: Path) -> None:
        self.kind = kind
        self._tmp = tmp_path
        self._engines: list[AsyncEngine] = []
        self._url: str | None = None
        self._private_db: tuple[str, str] | None = None
        self._count = 0

    # -- the database ----------------------------------------------------------------------

    async def _prepare(self) -> AsyncEngine:
        kind = self.kind
        if kind.startswith("sqlite"):
            self._count += 1
            memory = kind != "sqlite-file"
            self._url = (
                "sqlite+aiosqlite:///:memory:"
                if memory
                else f"sqlite+aiosqlite:///{self._tmp / f'kit{self._count}.db'}"
            )
            engine = _sqlite_engine(self._url)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                await conn.run_sync(lambda c: seed_settings(c, active=kind != "sqlite-format1"))
                if kind == "sqlite-format1":
                    await conn.run_sync(lambda c: put_setting(c, "storage_format", "1"))
            return engine
        if kind == "postgres":
            # The schema is created once per session by ``_conformance_pg_schema``.
            self._url = (
                pg_guard.authorize(os.environ[pg_guard.URL_ENV]) and os.environ[pg_guard.URL_ENV]
            )
            return create_async_engine(self._url)
        return await self._prepare_mysql()

    async def _prepare_mysql(self) -> AsyncEngine:
        base = make_url(os.environ[MYSQL_ENV[self.kind]])
        name = f"dendrux_p5_f_{uuid.uuid4().hex[:10]}_test"
        server = base._replace(database=None)
        admin = create_async_engine(server, poolclass=NullPool)
        try:
            async with admin.connect() as conn:
                await conn.execute(text(f"CREATE DATABASE `{name}`"))
        finally:
            await admin.dispose()
        self._private_db = (str(server), name)
        url = base.set(database=name)
        self._url = url.render_as_string(hide_password=False)
        engine = create_async_engine(
            url, isolation_level="READ COMMITTED", pool_size=10, max_overflow=10
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda c: seed_settings(c, active=True))
        await engine.dispose()
        return create_async_engine(url, isolation_level="READ COMMITTED")

    async def _truncate(self, engine: AsyncEngine) -> None:
        if self.kind == "postgres":
            async with engine.begin() as conn:
                await conn.execute(text(_PG_TRUNCATE_SQL))
                await conn.run_sync(lambda c: seed_settings(c, active=True))

    # -- fixtures ---------------------------------------------------------------------------

    async def fresh_store(self) -> Any:
        engine = await self._prepare()
        self._engines.append(engine)
        await self._truncate(engine)
        return store_for_engine(engine)

    async def reopen(self, store: Any) -> Any:
        """A new store over the same data (same engine when the data lives in the engine)."""
        if self.kind in ("sqlite-memory", "sqlite-format1"):
            return store_for_engine(store._engine)
        return await self.second(store)

    async def second(self, store: Any) -> Any:
        """An independent store (own engine, own connections) over the same data."""
        if self.kind in ("sqlite-memory", "sqlite-format1"):
            raise RuntimeError("an in-memory database has no second connection")
        assert self._url is not None
        if self.kind.startswith("sqlite"):
            engine = _sqlite_engine(self._url)
        elif self.kind == "postgres":
            engine = create_async_engine(self._url)
        else:
            engine = create_async_engine(self._url, isolation_level="READ COMMITTED")
        self._engines.append(engine)
        return store_for_engine(engine)

    async def advance(self, delta: timedelta) -> None:
        """Make every stored run clock read ``delta`` older (the kit's controllable clock)."""
        engine = self._engines[0]
        async with engine.begin() as conn:
            rows = (
                (
                    await conn.execute(
                        select(AgentRun.id, *[getattr(AgentRun, c) for c in _TIME_COLUMNS])
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                values = {}
                for column in _TIME_COLUMNS:
                    stamp = row[column]
                    if stamp is None:
                        continue
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=UTC)
                    values[column] = stamp - delta
                if values:
                    await conn.execute(
                        update(AgentRun).where(AgentRun.id == row["id"]).values(**values)
                    )

    async def close(self) -> None:
        for engine in self._engines:
            await engine.dispose()
        self._engines.clear()
        if self._private_db is not None:
            server, name = self._private_db
            admin = create_async_engine(server, poolclass=NullPool)
            try:
                async with admin.connect() as conn:
                    await conn.execute(text(f"DROP DATABASE IF EXISTS `{name}`"))
            finally:
                await admin.dispose()


def now() -> datetime:
    return datetime.now(UTC)


def backend_params() -> list[Any]:
    return [
        pytest.param(kind, id=kind, marks=pytest.mark.skip(reason=skip_reason(kind)))
        if skip_reason(kind)
        else pytest.param(kind, id=kind)
        for kind in KINDS
    ]
