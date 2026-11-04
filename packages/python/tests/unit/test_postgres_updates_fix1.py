"""Implicit engines are accepted only when destination and TLS can be established."""

import asyncio
import ssl
from types import SimpleNamespace

import pytest
from sqlalchemy import URL, pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from dendrux.db._postgres_updates import PostgresUpdates

BASE = "postgresql+asyncpg://u:p@localhost:5432/db"


def bind(engine, **options):
    updates = PostgresUpdates(**options)
    updates.bind(SimpleNamespace(_engine=engine))
    return updates


@pytest.mark.parametrize(
    "transport", [{}, {"server_settings": {"application_name": "x"}}, {"ssl": True}]
)
@pytest.mark.parametrize(
    "setting",
    [
        {"host": "SECRET"},
        {"dbname": "SECRET"},
        {"hostaddr": "SECRET"},
        {"conninfo": "SECRET"},
        {"krbsrvname": "SECRET"},
        {"connection_class": object},
    ],
)
def test_transport_options_never_waive_destination_or_unknown_engine_settings(transport, setting):
    engine = create_async_engine(BASE, connect_args=setting)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url=") as caught:
        bind(engine, connect_args=transport)
    assert "SECRET" not in str(caught.value)
    assert bind(engine, url="postgresql://u@localhost/db")


@pytest.mark.parametrize("transport", [None, {}, {"server_settings": {"application_name": "x"}}])
def test_only_transport_ssl_waives_engine_tls(transport):
    engine = create_async_engine(BASE, connect_args={"ssl": ssl.create_default_context()})
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(engine, connect_args=transport)
    assert bind(engine, connect_args={"ssl": True})
    assert bind(engine, connect_args={"ssl": False})


@pytest.mark.parametrize(
    "broken", ["missing_creator", "no_pool", "no_dispatch", "creator_property", "method"]
)
def test_private_engine_inspection_failure_is_a_safe_refusal(broken):
    engine = create_async_engine(BASE)
    if broken == "missing_creator":
        engine.sync_engine.pool = SimpleNamespace()
    elif broken == "no_pool":
        engine.sync_engine.pool = None
    elif broken == "no_dispatch":
        engine.sync_engine.dialect = SimpleNamespace()
    elif broken == "creator_property":

        class BadPool:
            @property
            def _creator(self):
                raise RuntimeError("SECRET")

        engine.sync_engine.pool = BadPool()
    else:
        engine.dialect.create_connect_args = lambda url: (_ for _ in ()).throw(
            RuntimeError("SECRET")
        )
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url=") as caught:
        bind(engine)
    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_patched_dialect_connect_is_refused():
    engine = create_async_engine(BASE)
    engine.dialect.connect = lambda *a, **kw: None
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(engine)


def test_custom_dbapi_adapter_is_refused():
    normal = create_async_engine(BASE).dialect.dbapi

    class CustomModule:
        paramstyle = normal.paramstyle

        def __getattr__(self, name):
            return getattr(normal, name)

        def connect(self, *args, **kwargs):
            raise AssertionError("must not connect")

    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(create_async_engine(BASE, module=CustomModule()))


NORMAL_URLS = [
    BASE,
    "postgresql+asyncpg://u:p@localhost/db",
    URL.create(
        "postgresql+asyncpg",
        username="u",
        password="p@ss/w:rd",
        host="db",
        port=5432,
        database="db",
    ),
    "postgresql+asyncpg://u:p@/db?host=localhost",
    "postgresql+asyncpg://u:p@/db?host=localhost:5433",
    "postgresql+asyncpg://u:p@/db?host=a:5432&host=b:5433",
    "postgresql+asyncpg://u:p@/db?host=a:5432&host=b:5433&target_session_attrs=primary",
    "postgresql+asyncpg://u:p@/db?host=/var/run/postgresql",
    "postgresql+asyncpg://u:p@/db?host=/var/run/postgresql&port=5433",
    BASE + "?sslmode=require",
    BASE + "?application_name=x",
    BASE + "?options=-csearch_path%3Da",
    "postgresql+asyncpg://u:p@/db?host=%20",
    "postgresql+asyncpg://u:p@db%2Eprod/db",
    "postgresql+asyncpg://u:a%3Fb@db/db",
]


@pytest.mark.parametrize("url", NORMAL_URLS)
def test_normal_url_matrix_still_binds(url):
    assert bind(create_async_engine(url))


@pytest.mark.parametrize(
    "options",
    [
        {"poolclass": pool.NullPool},
        {"poolclass": pool.StaticPool},
        {
            "pool_size": 20,
            "max_overflow": 5,
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "echo": False,
            "isolation_level": "AUTOCOMMIT",
        },
        {"connect_args": {"statement_cache_size": 0, "prepared_statement_cache_size": 0}},
        {"connect_args": {"server_settings": {"application_name": "web", "search_path": "a"}}},
        {"connect_args": {"command_timeout": 30, "timeout": 5}},
        {"connect_args": {"host": "localhost"}},
        {"connect_args": {"server_settings": {"options": "x"}}},
    ],
)
def test_normal_options_matrix_still_binds(options):
    assert bind(create_async_engine(BASE, **options))


def test_normal_wrappers_and_standard_dbapi_reuse_still_bind():
    class Sub(AsyncEngine):
        pass

    assert bind(create_async_engine(BASE).execution_options(isolation_level="AUTOCOMMIT"))
    assert bind(Sub(create_async_engine(BASE).sync_engine))
    assert bind(create_async_engine(BASE, module=create_async_engine(BASE).dialect.dbapi))


def test_query_at_sign_is_not_mistaken_for_password():
    from dendrux.db._postgres_updates import asyncpg_dsn

    dsn = asyncpg_dsn("postgresql://u:p@host/db?application_name=alice@example.com")
    assert "application_name=alice%40example.com" in dsn


def test_unknown_engine_parameter_is_refused_even_when_none():
    engine = create_async_engine(BASE, connect_args={"future_destination": None})
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(engine)


def test_dsn_keys_are_cached_and_fall_back_without_source(monkeypatch):
    import inspect

    from dendrux.db import _postgres_updates as pg

    calls = []

    def missing_source(value):
        calls.append(value)
        raise OSError("SECRET")

    clear = getattr(pg._dsn_keys, "cache_clear", lambda: None)
    clear()
    try:
        monkeypatch.setattr(inspect, "getsource", missing_source)
        assert pg._dsn_keys() == pg._FALLBACK_DSN_KEYS
        assert pg._dsn_keys() == pg._FALLBACK_DSN_KEYS
        assert len(calls) == 1
    finally:
        clear()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_sender_close_failure_terminates_and_clears_owner(failure):
    from unittest.mock import AsyncMock, Mock

    bus = PostgresUpdates(url="postgresql://u@localhost/db")
    connection = Mock()
    connection.is_closed.return_value = False
    connection.close = AsyncMock(side_effect=failure("SECRET"))
    bus._sender = connection
    with pytest.raises(failure):
        await bus.aclose_sender()
    assert bus._sender is None
    connection.terminate.assert_called_once_with()
    await bus.aclose_sender()
    await bus.close()


def test_borrowed_standard_pool_cannot_prove_engine_destination():
    engine = create_async_engine(BASE)
    borrowed = create_async_engine(BASE)
    engine.sync_engine.pool = borrowed.pool
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(engine)
