"""No unsafe configuration or driver diagnostics cross the store boundary."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from dendrux.db._engine_options import mysql_engine_options
from dendrux.errors import PersistenceError
from dendrux.runtime.durability import _safe_database_call, retry_transient_db
from dendrux.runtime.stores.sql.mysql import MySQLStateStore


def test_mysql_execution_options_isolation_policy():
    with pytest.raises(ValueError, match="READ COMMITTED"):
        mysql_engine_options({"execution_options": {"isolation_level": "AUTOCOMMIT"}})


async def test_capacity_sanitizes_first_connection_error():
    @asynccontextmanager
    async def failing_session():
        raise OperationalError(None, None, Exception(2013, "PRIVATE_DB_USER_HOST"))
        yield

    store = object.__new__(MySQLStateStore)
    store._dialect_facts = None
    store._session_factory = failing_session
    with pytest.raises(PersistenceError) as caught:
        await store.statement_capacity()
    assert "PRIVATE_DB_USER_HOST" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "kind", ["unicode", "attribute", "sqlalchemy", "driver_value", "driver_runtime"]
)
async def test_driver_errors_have_no_value_or_suppressed_context(kind):
    @_safe_database_call
    async def operation(store):
        if kind == "unicode":
            namespace = {"__name__": "asyncmy.connection"}
            exec(
                "def fail(): raise UnicodeEncodeError('ascii', 'PRIVATE_VALUE', 0, 1, 'x')",
                namespace,
            )
            namespace["fail"]()
        if kind == "attribute":
            namespace = {"__name__": "asyncmy.connection"}
            exec("def fail(): raise AttributeError('PRIVATE_VALUE')", namespace)
            namespace["fail"]()
        if kind in {"driver_value", "driver_runtime"}:
            base = ValueError if kind == "driver_value" else RuntimeError
            driver_error = type("DriverError", (base,), {"__module__": "asyncmy.errors"})
            raise driver_error("PRIVATE_VALUE")
        raise DBAPIError(None, None, Exception("PRIVATE_VALUE"))

    with pytest.raises(PersistenceError) as caught:
        await operation(SimpleNamespace())
    assert caught.value.__context__ is None
    assert "PRIVATE_VALUE" not in str(caught.value)


@pytest.mark.parametrize("kind", [AttributeError, UnicodeError])
async def test_caller_errors_propagate_unchanged(kind):
    error = kind("caller bug")

    @_safe_database_call
    async def operation(store, validate):
        validate()

    def validate():
        raise error

    with pytest.raises(kind) as caught:
        await operation(SimpleNamespace(), validate)
    assert caught.value is error


@pytest.mark.parametrize("wrapper", [DBAPIError, OperationalError])
async def test_postgres_statement_cancel_is_not_retried(wrapper):
    class CancelledError(Exception):
        sqlstate = "57014"

    calls = 0

    async def attempt():
        nonlocal calls
        calls += 1
        raise wrapper(None, None, CancelledError("PRIVATE_VALUE"))

    with pytest.raises(PersistenceError) as caught:
        await retry_transient_db(attempt, label="cancel", run_id="run")
    assert calls == 1
    assert caught.value.retryable is False
    assert caught.value.__context__ is None


@pytest.mark.parametrize("route", ["factory", "reader", "constructor"])
def test_explicit_mariadb_engine_is_supported(route):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.store import RunStore

    engine = create_async_engine("mariadb+asyncmy://root@127.0.0.1/private_test")
    if route == "factory":
        assert isinstance(store_for_engine(engine), MySQLStateStore)
    elif route == "reader":
        RunStore.from_engine(engine)
    else:
        MySQLStateStore(engine)


def test_unsupported_dialect_names_rejected_dialect():
    from dendrux.runtime.stores._factory import store_for_engine

    with pytest.raises(ValueError, match="sentinel_dialect"):
        store_for_engine(SimpleNamespace(dialect=SimpleNamespace(name="sentinel_dialect")))


def test_removed_class_names_mysql_and_public_import():
    from dendrux.runtime import state

    with pytest.raises(ImportError, match="MySQLStateStore.*dendrux.store"):
        state.__getattr__("SQLAlchemyStateStore")


@pytest.mark.parametrize(
    "hook,args",
    [
        ("json_get", (None, (), "str")),
        ("json_contains", (None, (), 1)),
        ("json_is_null", (None, ())),
        ("upsert_setting", (None,)),
    ],
)
def test_unimplemented_hook_names_itself(hook, args):
    from dendrux.runtime.stores.sql.base import SQLStateStore

    store = object.__new__(SQLStateStore)
    kwargs = dict(key="key", value="value", updated_at=None) if hook == "upsert_setting" else {}
    with pytest.raises(NotImplementedError, match=hook):
        getattr(store, hook)(*args, **kwargs)


@pytest.mark.parametrize("package", ["asyncmy", "cryptography"])
def test_from_url_missing_driver_names_the_extra(monkeypatch, package):
    import sqlalchemy.ext.asyncio

    def missing(*args, **kwargs):
        raise ModuleNotFoundError(f"No module named {package}", name=package)

    monkeypatch.setattr(sqlalchemy.ext.asyncio, "create_async_engine", missing)
    with pytest.raises(ImportError, match=r"dendrux\[mysql\]") as caught:
        MySQLStateStore.from_url("mysql://root@127.0.0.1/private_test")
    assert caught.value.__context__ is None


async def test_missing_auth_dependency_names_the_extra():
    @_safe_database_call
    async def connect(store):
        raise RuntimeError(
            "'cryptography' package is required for "
            "sha256_password or caching_sha2_password auth methods"
        )

    with pytest.raises(ImportError, match=r"dendrux\[mysql\]") as caught:
        await connect(SimpleNamespace())
    assert caught.value.__context__ is None


def test_shared_pool_warning_omits_database_paths(caplog):
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    from dendrux.db._pool_guard import warn_if_shared_connection

    engine = create_async_engine("sqlite+aiosqlite:////PRIVATE_DATABASE_PATH", poolclass=StaticPool)
    warn_if_shared_connection(engine)
    assert "PRIVATE_DATABASE_PATH" not in caplog.text


def test_sqlite_adapter_uses_sql_scope_helper():
    import inspect

    from dendrux.runtime.stores.sql import sqlite

    source = inspect.getsource(sqlite)
    assert "_check_scope" not in source
    assert "_Unscoped" not in source


@pytest.mark.parametrize("site", ["lock", "column", "index"])
def test_sqlite_schema_failures_omit_driver_context(monkeypatch, site):
    from sqlalchemy import create_engine

    from dendrux.db import _sqlite_sync as schema
    from dendrux.db.models import Base
    from dendrux.errors import SqliteSchemaSyncError

    def fail(*args, **kwargs):
        raise OperationalError(None, None, Exception("PRIVATE_DATABASE_PATH_VALUE"))

    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        monkeypatch.setattr(conn, "exec_driver_sql", fail)
        monkeypatch.setattr(schema, "_has_work", lambda *a: True)
        if site == "lock":

            def action():
                schema._begin_immediate(conn)
        elif site == "column":

            def action():
                schema._add_column(
                    conn,
                    Base.metadata.tables["agent_runs"],
                    Base.metadata.tables["agent_runs"].c.output_data,
                )
        else:
            monkeypatch.setattr(
                schema,
                "_exact_indexes",
                lambda *a: [SimpleNamespace(name="index", ddl="CREATE INDEX")],
            )
            monkeypatch.setattr(schema, "_exact_state", lambda *a: "missing")
            monkeypatch.setattr(schema, "_index_sql", lambda *a: {})
            monkeypatch.setattr(schema, "_index_owners", lambda *a: {})

            def action():
                schema._reconcile_exact(conn, Base.metadata.tables["agent_runs"], [])

        with pytest.raises(SqliteSchemaSyncError) as caught:
            action()
        assert "PRIVATE_DATABASE_PATH_VALUE" not in str(caught.value)
        assert caught.value.__context__ is None
    engine.dispose()


async def test_retry_exhaustion_logs_only_closed_database_codes(monkeypatch, caplog):
    from dendrux.runtime import durability
    from dendrux.types import generate_ulid

    class SerializationError(Exception):
        sqlstate = "40001"

    monkeypatch.setattr(durability, "BASE_BACKOFF_S", 0)

    async def attempt():
        raise DBAPIError(None, None, SerializationError("PRIVATE_VALUE"))

    with pytest.raises(PersistenceError):
        await retry_transient_db(attempt, label="serialize", run_id=generate_ulid())
    assert any(getattr(record, "db_code", None) == "40001" for record in caplog.records)
    assert "PRIVATE_VALUE" not in caplog.text


@pytest.mark.parametrize("adapter", ["sqlite", "postgres"])
async def test_adapter_database_hooks_sanitize_errors(adapter):
    from dendrux.runtime.stores.sql._types import RunLockScope
    from dendrux.runtime.stores.sql.postgres import PostgresStateStore
    from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

    async def execute(*args, **kwargs):
        raise OperationalError(None, None, Exception("PRIVATE_DB_USER_HOST"))

    session = SimpleNamespace(execute=execute)
    with pytest.raises(PersistenceError) as caught:
        if adapter == "sqlite":
            await object.__new__(SQLiteStateStore).reserve_parent_writer(session, "run")
        else:
            await object.__new__(PostgresStateStore).serialize_in_tx(
                session, owner=RunLockScope("run", None)
            )
    assert caught.value.__context__ is None
    assert "PRIVATE_DB_USER_HOST" not in str(caught.value)


def test_recovery_declaration_requires_replay_reader():
    from dendrux import Agent
    from dendrux.errors import StoreCapabilityError
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.state import StoreCapabilities

    class MissingReplayReader:
        capabilities = StoreCapabilities(recovery=True, deployment="single_process")

        async def mark_lease_expired(self, *, finalise_unrecoverable=False):
            pass

        async def get_lease_expired(self):
            pass

        async def reclaim_run(self):
            pass

    with pytest.raises(StoreCapabilityError) as caught:
        Agent(prompt="p", provider=MockLLM([]), state_store=MissingReplayReader())
    assert caught.value.capability == "recovery"
    assert caught.value.method == "get_replayed_tool_calls"
