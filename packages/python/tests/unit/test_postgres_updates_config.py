"""PostgresUpdates configuration: DSN conversion, option precedence, TLS warning, limits.

Destination equivalence compares what asyncpg would connect to from our DSN with
what SQLAlchemy's asyncpg dialect would connect to from the original URL. All
URLs are synthetic; nothing connects.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from asyncpg import connect_utils
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.engine import make_url

from dendrux.db._postgres_updates import PostgresUpdates, asyncpg_dsn, connect_options

SECRET = "s3cr3t-pa@ss/w:rd%"


def _target(**kwargs: Any) -> tuple[Any, ...]:
    params: dict[str, Any] = dict(
        dsn=None,
        host=None,
        port=None,
        user=None,
        password=None,
        passfile=None,
        database=None,
        ssl=None,
        direct_tls=None,
        server_settings=None,
        target_session_attrs=None,
        krbsrvname=None,
        gsslib=None,
        service=None,
        servicefile=None,
    )
    params.update(kwargs)
    addrs, parsed = connect_utils._parse_connect_dsn_and_args(**params)
    return (
        tuple(addrs),
        parsed.user,
        parsed.password,
        parsed.database,
        parsed.server_settings,
        parsed.target_session_attrs,
    )


def _sqlalchemy_target(url: str) -> tuple[Any, ...]:
    _, params = PGDialect_asyncpg().create_connect_args(make_url(url))
    known = {"host", "port", "user", "password", "database", "target_session_attrs"}
    settings = {k: v for k, v in params.items() if k not in known and k not in ("sslmode",)}
    return _target(
        host=params.get("host"),
        port=params.get("port"),
        user=params.get("user"),
        password=params.get("password"),
        database=params.get("database"),
        server_settings=settings or None,
        target_session_attrs=params.get("target_session_attrs"),
    )


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://app:pw@db.internal:5433/orbit",
        "postgresql+asyncpg://us%40er:s3cr3t-pa%40ss%2Fw%3Ard%25@db.internal/orbit",
        "postgresql+asyncpg://app:pw@[::1]:5433/orbit",
        "postgresql+asyncpg://app:pw@[2001:db8::7]/orbit",
        "postgresql+asyncpg://app:pw@/orbit?host=/var/run/postgresql",
        "postgresql+asyncpg://app:pw@/orbit?host=h1:5432&host=h2:5433&target_session_attrs=read-write",
        "postgresql+asyncpg://app:pw@db.internal/d%20b",
        "postgresql+asyncpg://app:pw@db.internal/orbit?application_name=web",
    ],
)
def test_dsn_keeps_the_destination_sqlalchemy_would_use(url: str) -> None:
    assert _target(dsn=asyncpg_dsn(url)) == _sqlalchemy_target(url)


def test_dsn_escapes_credentials_and_keeps_tls_keys() -> None:
    url = make_url("postgresql+asyncpg://db.internal/orbit").set(
        username="us@er:", password=SECRET, query={"sslmode": "require"}
    )
    dsn = asyncpg_dsn(url)
    assert dsn.startswith("postgresql://")
    assert SECRET not in dsn  # escaped, never raw
    target = _target(dsn=dsn)
    assert (target[1], target[2]) == ("us@er:", SECRET)
    _, parsed = connect_utils._parse_connect_dsn_and_args(
        dsn=dsn,
        host=None,
        port=None,
        user=None,
        password=None,
        passfile=None,
        database=None,
        ssl=None,
        direct_tls=None,
        server_settings=None,
        target_session_attrs=None,
        krbsrvname=None,
        gsslib=None,
        service=None,
        servicefile=None,
    )
    assert parsed.sslmode.name == "require"


def test_every_asyncpg_dsn_key_is_kept() -> None:
    keys = {
        "passfile": "/pgpass",
        "sslmode": "require",
        "sslcert": "/c",
        "sslkey": "/k",
        "sslrootcert": "/r",
        "sslcrl": "/l",
        "ssl_min_protocol_version": "TLSv1.2",
        "target_session_attrs": "any",
        "options": "-c statement_timeout=5000",
        "application_name": "worker",
        "krbsrvname": "postgres",
        "gsslib": "gssapi",
    }
    url = make_url("postgresql+asyncpg://app:pw@db.internal/orbit").set(query=keys)
    query = make_url(asyncpg_dsn(url)).query
    assert dict(query) == keys


@pytest.mark.parametrize(
    "key", ["prepared_statement_cache_size", "ssl", "pool_size", "statement_cache_size"]
)
def test_unknown_query_key_is_rejected_naming_only_the_key(key: str) -> None:
    url = f"postgresql+asyncpg://app:{SECRET}@db.internal/orbit?{key}=1"
    with pytest.raises(ValueError) as caught:
        asyncpg_dsn(url)
    message = str(caught.value)
    assert key in message and "url=" in message
    assert SECRET not in message and "db.internal" not in message


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://app:pw@h0/orbit?host=h1",
        "postgresql+asyncpg://app:pw@h0/orbit?user=other",
        "postgresql+asyncpg://app:pw@h0/orbit?dbname=other",
        "postgresql+asyncpg://app:pw@h0/orbit?sslmode=require&sslmode=disable",
    ],
)
def test_ambiguous_destination_is_rejected(url: str) -> None:
    with pytest.raises(ValueError) as caught:
        asyncpg_dsn(url)
    assert "pw" not in str(caught.value) and "h0" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        f"mysql+aiomysql://app:{SECRET}@db.internal/orbit",
        f"sqlite+aiosqlite:///{SECRET}.db",
        f"not a url {SECRET}",
    ],
)
def test_non_postgres_or_unparseable_url_never_echoes_it(url: str) -> None:
    with pytest.raises(ValueError) as caught:
        asyncpg_dsn(url)
    assert SECRET not in str(caught.value)
    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


def test_connection_options_precedence() -> None:
    options = connect_options(
        {"ssl": "ctx", "server_settings": {"search_path": "app"}, "command_timeout": 9},
        timeout=0.4,
        url_settings=frozenset(),
    )
    assert options["ssl"] == "ctx" and options["command_timeout"] == 9
    assert options["statement_cache_size"] == 0 and options["timeout"] == 0.4
    assert options["server_settings"] == {
        "search_path": "app",
        "application_name": "dendrux-live-updates",
    }
    caller_name = connect_options(
        {"server_settings": {"application_name": "mine"}}, timeout=1, url_settings=frozenset()
    )
    assert caller_name["server_settings"] == {"application_name": "mine"}
    url_name = connect_options({}, timeout=1, url_settings=frozenset({"application_name"}))
    assert "server_settings" not in url_name  # the URL's own name wins


@pytest.mark.parametrize("key", ["statement_cache_size", "timeout"])
def test_connect_args_that_defeat_forced_options_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        PostgresUpdates(url="postgresql://a@localhost/d", connect_args={key: 1})


def test_message_cap_and_channel_are_validated() -> None:
    PostgresUpdates(url="postgresql://a@localhost/d", max_message_bytes=7_999)
    with pytest.raises(ValueError, match="max_message_bytes"):
        PostgresUpdates(url="postgresql://a@localhost/d", max_message_bytes=8_000)
    with pytest.raises(ValueError):
        PostgresUpdates(url="postgresql://a@localhost/d", channel="")
    with pytest.raises(ValueError):
        PostgresUpdates(url="postgresql://a@localhost/d", channel="x" * 64)


def test_ten_times_realistic_message_budget_still_encodes_under_the_postgres_limit() -> None:
    updates = PostgresUpdates(url="postgresql://a@localhost/d")
    ids = [f"01J{index:023d}" for index in range(5_000)]  # ten 500-id batches of ULID size
    messages = updates._encode(ids)
    assert all(len(m.encode()) <= 7_900 for m in messages)
    assert sum(len(updates._parse_message(m)) for m in messages) == 5_000


def test_url_defaults_to_the_bound_store_engine() -> None:
    class Engine:
        url = make_url(f"postgresql+asyncpg://app:{SECRET}@127.0.0.1/orbit")

    class Store:
        _engine = Engine()

    updates = PostgresUpdates()
    updates.bind(Store())
    assert _target(dsn=updates._send_dsn) == _target(dsn=asyncpg_dsn(Engine.url))
    assert updates._listen_dsn == updates._send_dsn


def test_listen_url_is_used_only_for_listening() -> None:
    updates = PostgresUpdates(
        url="postgresql://a@pooler:6432/d", listen_url="postgresql://a@direct:5432/d"
    )
    assert _target(dsn=updates._send_dsn)[0] == (("pooler", 6432),)
    assert _target(dsn=updates._listen_dsn)[0] == (("direct", 5432),)


def test_store_without_engine_needs_an_explicit_url() -> None:
    with pytest.raises(ValueError, match="url="):
        PostgresUpdates().bind(object())


def test_bad_engine_url_rejects_bind_without_echo() -> None:
    class Engine:
        url = make_url(
            f"postgresql+asyncpg://app:{SECRET}@db/orbit?prepared_statement_cache_size=0"
        )

    class Store:
        _engine = Engine()

    with pytest.raises(ValueError) as caught:
        PostgresUpdates().bind(Store())
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    ("url", "connect_args", "warns"),
    [
        ("postgresql://a@db.internal/d", None, True),
        ("postgresql://a@db.internal/d?sslmode=require", None, False),
        ("postgresql://a@db.internal/d?sslmode=disable", None, False),
        ("postgresql://a@db.internal/d", {"ssl": "require"}, False),
        ("postgresql://a@localhost/d", None, False),
        ("postgresql://a@127.0.0.1/d", None, False),
        ("postgresql://a@[::1]/d", None, False),
        ("postgresql://a@/d?host=/var/run/postgresql", None, False),
        ("postgresql://a@/d?host=localhost:5432&host=db.internal:5432", None, True),
    ],
)
def test_tls_unknown_is_logged_once_for_remote_hosts(
    url: str, connect_args: dict[str, Any] | None, warns: bool, caplog: Any
) -> None:
    caplog.set_level(logging.WARNING, logger="dendrux.db._postgres_updates")
    updates = PostgresUpdates(url=url, connect_args=connect_args)
    updates._check_tls()
    updates._check_tls()
    codes = [r for r in caplog.records if getattr(r, "code", None) == "live_updates_tls_unknown"]
    assert len(codes) == (1 if warns else 0)
    assert all("db.internal" not in r.getMessage() for r in caplog.records)
