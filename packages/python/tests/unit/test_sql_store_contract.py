"""Phase-5 relational store contracts and safe driver classification."""

import sqlite3
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from dendrux.runtime import durability, state
from dendrux.runtime._capabilities import validate_capabilities


def test_sql_store_factory_and_base_are_available():
    from importlib.util import find_spec

    assert find_spec("dendrux.runtime.stores._factory") is not None, "dialect factory is required"
    assert find_spec("dendrux.runtime.stores.sql.base") is not None, "shared SQL base is required"


@pytest.mark.parametrize("dialect,hint", [("oracle", "subclass SQLStateStore")])
def test_factory_rejects_reserved_and_unknown_dialects(dialect, hint):
    from dendrux.runtime.stores._factory import store_for_engine

    with pytest.raises(ValueError, match=hint):
        store_for_engine(SimpleNamespace(dialect=SimpleNamespace(name=dialect)))


def test_factory_returns_mysql_store():
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.runtime.stores.sql.mysql import MySQLStateStore

    engine = create_async_engine("mysql+asyncmy://root@127.0.0.1:53306/dendrux_p5_b_factory_test")
    assert isinstance(store_for_engine(engine), MySQLStateStore)


@pytest.mark.parametrize("generated", ["onupdate", "server_onupdate", "computed"])
async def test_mysql_increment_reads_back_generated_returning_columns(generated):
    from unittest.mock import AsyncMock, Mock

    from sqlalchemy import Column, Computed, Integer, MetaData, String, Table, text

    from dendrux.runtime.stores.sql.mysql import MySQLStateStore, _increment_site

    options = {generated: text("2")} if generated != "computed" else {}
    computed = [Computed("claim_epoch + 2")] if generated == "computed" else []
    table = Table(
        "agent_runs",
        MetaData(),
        Column("id", String, primary_key=True),
        Column("claim_epoch", Integer),
        Column("generated_value", Integer, *computed, **options),
    )
    locked = Mock()
    locked.mappings.return_value.first.return_value = {"id": "r", "generated_value": 1}
    readback = Mock()
    readback.mappings.return_value.one.return_value = {"generated_value": 2}
    session = Mock(info={})
    session.execute = AsyncMock(side_effect=[locked, Mock(), readback])
    store = object.__new__(MySQLStateStore)
    token = _increment_site.set((store, "claim_paused_run"))
    try:
        row = await store.update_one(
            session,
            table=table,
            pk=("r",),
            predicate=table.c.claim_epoch == 0,
            values=[(table.c.claim_epoch, table.c.claim_epoch + 1)],
            returning=[table.c.generated_value],
        )
    finally:
        _increment_site.reset(token)
    assert row["generated_value"] == 2
    assert session.execute.await_count == 3


def test_deployment_is_explicit_and_legacy_declarations_warn():
    caps = state.StoreCapabilities()
    assert hasattr(caps, "deployment"), "capabilities must describe their deployment domain"
    assert caps.deployment is None
    store = SimpleNamespace(capabilities=caps)
    with pytest.warns(DeprecationWarning, match="deployment"):
        assert validate_capabilities(store).deployment is None


def test_protocol_declares_update_run_meta():
    assert "update_run_meta" in vars(state.StateStore)


@pytest.mark.parametrize(
    "url,deployment",
    [
        ("sqlite+aiosqlite://", "single_process"),
        ("sqlite+aiosqlite:///:memory:", "single_process"),
        ("sqlite+aiosqlite:///file:shared?mode=memory&cache=shared&uri=true", "single_process"),
        ("sqlite+aiosqlite:///file::memory:?cache=shared&uri=true", "single_process"),
        ("sqlite+aiosqlite:///deployment.db", "single_host"),
    ],
)
async def test_sqlite_deployment_includes_memory_uris(url, deployment):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    engine = create_async_engine(url)
    try:
        assert SQLiteStateStore(engine).capabilities.deployment == deployment
    finally:
        await engine.dispose()


def test_parent_cursors_keep_their_own_membership_bindings():
    from sqlalchemy.dialects import postgresql

    from dendrux.db.models import AgentRun
    from dendrux.runtime.stores.sql._types import RunParentCursor
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore

    store = object.__new__(PostgresStateStore)
    stmt = store.child_page_query(
        runs=AgentRun.__table__,
        parents=[RunParentCursor("a", "a1"), RunParentCursor("b", "b2")],
        tenant_scope=None,
        page_size=10,
    )
    parameters = list(stmt.compile(dialect=postgresql.dialect()).params.values())
    assert ["a"] in parameters and ["b"] in parameters


@pytest.mark.parametrize("code", [None, "08006", "23505", "42601"])
def test_precommit_operational_errors_keep_legacy_retryability(code):
    class DriverError(Exception):
        sqlstate = code

    error = OperationalError("PRIVATE-DRIVER-SENTINEL", {}, DriverError())
    assert durability.is_transient_db_error(error)


@pytest.mark.parametrize("code", ["40P01", "40001", "55P03"])
def test_postgres_transaction_retry_codes(code):
    class DriverError(Exception):
        sqlstate = code

    assert durability.is_transient_db_error(DBAPIError(None, None, DriverError()))


@pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED])
def test_sqlite_lock_retry_codes(code):
    error = sqlite3.OperationalError("PRIVATE-DRIVER-SENTINEL")
    error.sqlite_errorcode = code
    assert durability.is_transient_db_error(OperationalError(None, None, error))


@pytest.mark.parametrize(
    "code,kind",
    [
        ("23505", "unique_violation"),
        ("40P01", "deadlock_or_serialization"),
        ("40001", "deadlock_or_serialization"),
        ("55P03", "lock_timeout"),
        ("08006", "connection_lost"),
        ("42601", "other"),
    ],
)
def test_postgres_error_metadata_has_no_driver_text(code, kind):
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore

    class DriverError(Exception):
        sqlstate = code
        constraint_name = "known_key"

    error = PostgresStateStore.classify_db_error(
        object.__new__(PostgresStateStore),
        OperationalError(None, None, DriverError("PRIVATE-DRIVER-SENTINEL")),
    )
    assert error.kind == kind and error.code == code
    assert error.constraint == ("known_key" if code == "23505" else None)
    assert "PRIVATE-DRIVER-SENTINEL" not in repr(error)


@pytest.mark.parametrize(
    "code,kind",
    [
        (sqlite3.SQLITE_CONSTRAINT_UNIQUE, "unique_violation"),
        (sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY, "unique_violation"),
        (sqlite3.SQLITE_BUSY, "lock_timeout"),
        (sqlite3.SQLITE_LOCKED, "lock_timeout"),
        (sqlite3.SQLITE_IOERR, "other"),
    ],
)
def test_sqlite_error_metadata_has_no_driver_text(code, kind):
    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    driver = sqlite3.OperationalError("PRIVATE-DRIVER-SENTINEL")
    driver.sqlite_errorcode = code
    error = SQLiteStateStore.classify_db_error(
        object.__new__(SQLiteStateStore), OperationalError(None, None, driver)
    )
    assert error.kind == kind and error.code == code and error.constraint is None
    assert "PRIVATE-DRIVER-SENTINEL" not in repr(error)
