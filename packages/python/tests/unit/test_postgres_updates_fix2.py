"""Keep standard engine use compatible without trusting changed connection adapters."""

import logging
import ssl
from types import MethodType, SimpleNamespace

import asyncpg
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from dendrux.db._postgres_updates import PostgresUpdates

BASE = "postgresql+asyncpg://u:p@localhost/db"


def bind(engine, **options):
    bus = PostgresUpdates(**options)
    bus.bind(SimpleNamespace(_engine=engine))
    return bus


def test_cached_asyncpg_error_translation_does_not_change_connection_configuration():
    engine = create_async_engine(BASE)
    assert engine.dialect.dbapi._asyncpg_error_translate
    assert bind(engine)


def test_swapped_asyncpg_module_is_refused_without_echo():
    engine = create_async_engine(BASE)
    engine.dialect.dbapi.asyncpg = SimpleNamespace(connect="SECRET")
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url=") as caught:
        bind(engine)
    assert "SECRET" not in str(caught.value)
    assert caught.value.__context__ is caught.value.__cause__ is None


def test_adapter_connect_override_is_refused_even_with_standard_constructor_attributes():
    engine = create_async_engine(BASE)
    adapter = engine.dialect.dbapi

    def connect(self, *args, **kwargs):
        raise AssertionError("SECRET: must never execute a connector while inspecting")

    adapter.connect = MethodType(connect, adapter)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url=") as caught:
        bind(engine)
    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize(
    "options",
    [
        {"max_cached_statement_lifetime": 600},
        {"max_cacheable_statement_size": 32768},
        {"record_class": asyncpg.Record},
        {"connect_timeout": 10},
        {"prepare_threshold": None},
        {"autocommit": True},
        {"application_name": "worker"},
        {"keepalives_idle": 60},
    ],
)
def test_harmless_driver_settings_do_not_change_inherited_destination(options):
    # The inheritance policy is driver-neutral; no DBAPI connection is executed here.
    assert bind(create_async_engine(BASE, connect_args=options))


def test_pgbouncer_prepared_statement_name_func_is_harmless():
    def name():
        raise AssertionError("must not invoke the statement-name callback while binding")

    engine = create_async_engine(
        BASE,
        poolclass=NullPool,
        connect_args={
            "prepared_statement_cache_size": 0,
            "statement_cache_size": 0,
            "prepared_statement_name_func": name,
        },
    )
    assert bind(engine)


@pytest.mark.parametrize("key", ["connection_class", "krbsrvname", "gsslib"])
def test_connector_and_authentication_settings_stay_refused(key):
    engine = create_async_engine(BASE, connect_args={key: "SECRET"})
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url=") as caught:
        bind(engine)
    assert "SECRET" not in str(caught.value)


def test_ssl_none_does_not_waive_engine_tls_validation():
    engine = create_async_engine(BASE, connect_args={"ssl": ssl.create_default_context()})
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        bind(engine, connect_args={"ssl": None})
    assert bind(engine, connect_args={"ssl": False})


def test_ssl_none_does_not_silence_unknown_tls(caplog):
    bus = PostgresUpdates(url="postgresql://u@remote.invalid/db", connect_args={"ssl": None})
    with caplog.at_level(logging.WARNING):
        bus._check_tls()
    assert [record.code for record in caplog.records] == ["live_updates_tls_unknown"]
