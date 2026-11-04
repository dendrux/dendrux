"""Transport destinations must be explicit when engine options cannot be inherited."""

from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._postgres_updates import PostgresUpdates, _loopback
from dendrux.store import PostgresStateStore


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://",
        "postgresql+asyncpg://u@/db",
        "postgresql+asyncpg://u@localhost",
        "postgresql+asyncpg://u@/db?host=",
    ],
)
def test_missing_engine_destination_refused(url):
    engine = create_async_engine(url)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)
    assert PostgresStateStore(engine, live_updates=None).live_updates is None
    explicit = PostgresUpdates(url="postgresql://u@localhost/db")
    assert PostgresStateStore(engine, live_updates=explicit).live_updates is explicit


@pytest.mark.parametrize(
    "options",
    [
        {"host": "destination.invalid"},
        {"port": 5433},
        {"database": "elsewhere"},
        {"ssl": True},
        {"ssl": False},
        {"direct_tls": True},
    ],
)
def test_engine_connect_settings_need_explicit_transport(options):
    engine = create_async_engine("postgresql+asyncpg://u@localhost/db", connect_args=options)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)
    assert PostgresStateStore(engine, live_updates=None).live_updates is None
    for bus in (
        PostgresUpdates(url="postgresql://u@localhost/db"),
        PostgresUpdates(connect_args=options),
    ):
        assert PostgresStateStore(engine, live_updates=bus).live_updates is bus


def test_async_creator_requires_explicit_destination():
    async def creator():
        raise AssertionError("must not connect")

    engine = create_async_engine("postgresql+asyncpg://", async_creator=creator)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)


def test_socket_destination_and_database_query_are_explicit():
    engine = create_async_engine("postgresql+asyncpg://u@/db?host=/tmp")
    assert PostgresStateStore(engine, live_updates=True).live_updates is not None
    bus = PostgresUpdates()
    bus.bind(SimpleNamespace(_engine=SimpleNamespace(url="postgresql://u@localhost?dbname=db")))


def test_empty_host_is_not_loopback():
    assert not _loopback("")


def test_sqlalchemy_ssl_error_has_working_alternative():
    with pytest.raises(ValueError, match="sslmode=require"):
        PostgresUpdates(url="postgresql://u@localhost/db?ssl=require")


def test_refusal_never_echoes_engine_settings(caplog):
    secret = "RF2-SENTINEL"
    engine = create_async_engine(
        f"postgresql+asyncpg://u:{secret}@localhost/db",
        connect_args={"host": secret, "ssl": secret},
    )
    with pytest.raises(ValueError) as caught:
        PostgresStateStore(engine, live_updates=True)
    assert secret not in repr(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert secret not in caplog.text


def test_engine_dsn_override_requires_explicit_transport():
    engine = create_async_engine(
        "postgresql+asyncpg://u@localhost/db",
        connect_args={"dsn": "postgresql:///?sslmode=require"},
    )
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)


def test_engine_connection_hook_requires_explicit_transport():
    from sqlalchemy import event

    engine = create_async_engine("postgresql+asyncpg://u@localhost/db")

    @event.listens_for(engine.sync_engine, "do_connect")
    def change_host(dialect, record, args, params):
        params["host"] = "elsewhere.invalid"

    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)


def test_custom_creator_with_cparams_closure_is_not_trusted():
    def factory():
        cparams = {"host": "localhost", "database": "db", "user": "u"}

        def creator():
            options = dict(cparams, host="elsewhere.invalid")
            raise AssertionError(f"must not call creator: {len(options)}")

        return creator

    engine = create_async_engine("postgresql+asyncpg://u@localhost/db", creator=factory())
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)
    assert PostgresStateStore(engine, live_updates=None).live_updates is None
    bus = PostgresUpdates(url="postgresql://u@localhost/db")
    assert PostgresStateStore(engine, live_updates=bus).live_updates is bus


def test_complete_url_with_async_creator_still_requires_explicit_configuration():
    async def creator():
        raise AssertionError("must not call creator")

    engine = create_async_engine("postgresql+asyncpg://u@localhost/db", async_creator=creator)
    with pytest.raises(ValueError, match=r"PostgresUpdates\(url="):
        PostgresStateStore(engine, live_updates=True)
