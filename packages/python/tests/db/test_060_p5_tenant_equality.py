"""T-TENANT at schema level: tenants, keys and ids compare exactly on every dialect.

``'t'``, ``'t '`` and ``'T'`` are three tenants, and composed and decomposed Unicode forms
are two. Equality filters, the tenant-scoped idempotency unique index and primary keys must
all agree. On MySQL/MariaDB this needs the NO PAD binary collation (``utf8mb4_bin`` would
make ``'t' = 't '``); Postgres and SQLite compare bytes already.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import func, insert, literal_column, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._dialects import is_mysql_family
from dendrux.db._dialects.mysql import tenant_scope_column
from dendrux.db.models import AgentRun, Base, Blob

from .conftest import mysql_database, run_sync

if TYPE_CHECKING:
    from collections.abc import Iterator

TENANTS = ["t", "t ", "T", "é", "é"]  # last two: NFC and NFD of the same letter
KEYS = ["k", "k ", "K"]
IDS = ["i", "i ", "I"]


def _pg_url() -> Iterator[str]:
    from tests._helpers.pg_url import maintenance_url, render, validate_test_pg_url

    raw = os.environ.get("DENDRUX_TEST_PG_URL")
    if not raw:
        pytest.skip("DENDRUX_TEST_PG_URL not set")
    base = validate_test_pg_url(raw)
    name = f"{(base.database or '').removesuffix('_test')[:40]}_tn_{uuid.uuid4().hex[:8]}_test"
    url = base.set(database=name)
    admin = maintenance_url(base)

    async def admin_sql(statement: str) -> None:
        engine = create_async_engine(
            render(admin), poolclass=NullPool, isolation_level="AUTOCOMMIT"
        )
        try:
            async with engine.connect() as conn:
                await conn.execute(text(statement))
        finally:
            await engine.dispose()

    run_sync(admin_sql(f'CREATE DATABASE "{name}"'))
    try:
        yield render(url)
    finally:
        run_sync(admin_sql(f'DROP DATABASE IF EXISTS "{name}"'))


@pytest.fixture(params=["sqlite", "postgresql", "mysql", "mariadb"])
def any_url(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[str]:
    backend = request.param
    if backend == "sqlite":
        yield f"sqlite+aiosqlite:///{tmp_path / 'tenancy.db'}"
    elif backend == "postgresql":
        yield from _pg_url()
    else:
        yield from mysql_database(backend)


def _scope_matches(engine: Any, tenant: str) -> Any:
    if is_mysql_family(engine.dialect):
        return tenant_scope_column(AgentRun.__table__) == tenant
    return func.coalesce(AgentRun.tenant_id, literal_column("''")) == tenant


async def _count(conn: Any, *where: Any) -> int:
    return int((await conn.execute(select(func.count()).where(*where))).scalar_one())


async def test_tenants_keys_and_ids_compare_exactly(any_url: str) -> None:
    engine = create_async_engine(any_url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for i, tenant in enumerate(TENANTS):
                await conn.execute(
                    insert(AgentRun).values(
                        id=f"run{i}", tenant_id=tenant, agent_name="a", idempotency_key="k"
                    )
                )
            for i, key in enumerate(KEYS):
                await conn.execute(
                    insert(AgentRun).values(
                        id=f"key{i}", tenant_id="keys", agent_name="a", idempotency_key=key
                    )
                )
            for ident in IDS:
                await conn.execute(insert(AgentRun).values(id=ident, agent_name="a"))
            for tenant in ("t", "t ", "T"):
                await conn.execute(
                    insert(Blob).values(
                        tenant_scope=tenant,
                        sha256="0" * 64,
                        generation="g",
                        state="live",
                        media_type="image/png",
                        byte_length=1,
                        width=1,
                        height=1,
                        storage="db",
                    )
                )
        async with engine.connect() as conn:
            for tenant in TENANTS:
                assert await _count(conn, AgentRun.tenant_id == tenant) == 1, repr(tenant)
                assert (
                    await _count(
                        conn, _scope_matches(engine, tenant), AgentRun.idempotency_key == "k"
                    )
                    == 1
                ), repr(tenant)
            for key in KEYS:
                assert (
                    await _count(
                        conn, AgentRun.tenant_id == "keys", AgentRun.idempotency_key == key
                    )
                    == 1
                ), repr(key)
            for ident in IDS:
                assert await _count(conn, AgentRun.id == ident) == 1, repr(ident)
            for tenant in ("t", "t ", "T"):
                assert await _count(conn, Blob.tenant_scope == tenant) == 1, repr(tenant)

        # The unique index still rejects the exact same (tenant, key) ...
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(
                    insert(AgentRun).values(
                        id="dup", tenant_id="t ", agent_name="a", idempotency_key="k"
                    )
                )
        # ... and an untenanted run shares the '' scope with tenant ''.
        async with engine.begin() as conn:
            await conn.execute(
                insert(AgentRun).values(id="u1", agent_name="a", idempotency_key="shared")
            )
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(
                    insert(AgentRun).values(
                        id="u2", tenant_id="", agent_name="a", idempotency_key="shared"
                    )
                )
    finally:
        await engine.dispose()
