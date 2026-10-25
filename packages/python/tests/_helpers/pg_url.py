"""Validation of the Postgres URL that destructive tests run against.

Every Postgres test entry point (worker scoping, the engine fixtures, the migration
schema helper, the maintenance connection) goes through ``validate_test_pg_url`` so that
the database the driver will connect to, not only the URL path, is a ``*_test`` database.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

TEST_SUFFIX = "_test"
MAX_IDENTIFIER_LENGTH = 63
_DB_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Query options that would move the connection to another database or another server than
# the URL's path and authority name. A destructive-test URL must not carry any of them.
_OVERRIDE_OPTIONS = frozenset(
    {
        "database",
        "dbname",
        "host",
        "hostaddr",
        "port",
        "service",
        "servicefile",
        "unix_sock",
        "unix_socket",
        "socket",
    }
)
_ENV = "DENDRUX_TEST_PG_URL"


class UnsafeTestDatabaseUrl(pytest.UsageError):
    """The configured test database URL is not safe to run destructive tests on."""


def _unsafe(reason: str) -> UnsafeTestDatabaseUrl:
    return UnsafeTestDatabaseUrl(f"{_ENV} is not safe for destructive tests: {reason}")


def _effective_database(url: URL) -> str | None:
    """The database the driver would connect to, per SQLAlchemy's dialect rules."""
    try:
        _args, kwargs = url.get_dialect()().create_connect_args(url)
    except Exception as exc:
        raise _unsafe("the driver could not translate it, so its target is unknown") from exc
    effective: Any = kwargs.get("database", kwargs.get("dbname"))
    return str(effective) if effective is not None else None


def validate_test_pg_url(raw: str, *, append_suffix: bool = True) -> URL:
    """Return ``raw`` as a URL whose effective database is a safe ``*_test`` database.

    With ``append_suffix`` a name that does not end in ``_test`` gets the suffix added
    (the integration conftest's Django-style rewrite); otherwise it is refused. Query
    options that would override the path database or the host, port or socket are refused,
    the effective database must equal the path database, and the identifier must fit
    PostgreSQL's 63 bytes.
    """
    try:
        url = make_url(raw)
    except ArgumentError as exc:
        raise _unsafe("it cannot be parsed") from exc
    if url.get_backend_name() != "postgresql":
        raise _unsafe("it is not a postgresql URL")
    name = url.database or ""
    if not name:
        raise _unsafe("it has no database name in its path")
    overrides = sorted(k for k in url.query if k.lower() in _OVERRIDE_OPTIONS)
    if overrides:
        raise _unsafe(
            f"query option(s) {overrides} would override the database or server named by the URL"
        )
    if not name.endswith(TEST_SUFFIX):
        if not append_suffix:
            raise _unsafe(f"database name must end in {TEST_SUFFIX!r}")
        name = f"{name}{TEST_SUFFIX}"
    if not _DB_NAME_RE.match(name):
        raise _unsafe(
            "database name has characters outside [A-Za-z0-9_] (refusing to use it in DDL)"
        )
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise _unsafe(f"database name is longer than {MAX_IDENTIFIER_LENGTH} characters")
    url = url.set(database=name)
    if _effective_database(url) != name:
        raise _unsafe("the driver would connect to a different database than the path names")
    return url


def maintenance_url(url: URL) -> URL:
    """URL of the ``postgres`` maintenance database on the same server, used to CREATE DATABASE.

    ``url`` must come from ``validate_test_pg_url``, so it carries no database override.
    """
    admin = url.set(database="postgres")
    if _effective_database(admin) != "postgres":
        raise _unsafe("the maintenance URL would not target the maintenance database")
    return admin


def render(url: URL) -> str:
    """The URL as a string a driver can use (the password is kept)."""
    return url.render_as_string(hide_password=False)
