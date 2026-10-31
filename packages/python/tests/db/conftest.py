"""MySQL / MariaDB test databases for the schema and maintenance tests.

``DENDRUX_TEST_MYSQL_URL`` and ``DENDRUX_TEST_MARIADB_URL`` name a server and a base
database whose name ends in ``_test`` (for example
``mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_ci_test``). Each test gets a
fresh database named after the base, the xdist worker and a random suffix, still ending in
``_test``, and drops it afterwards. Unset means the server's tests are skipped.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator

ENV = {"mysql": "DENDRUX_TEST_MYSQL_URL", "mariadb": "DENDRUX_TEST_MARIADB_URL"}
_NAME = re.compile(r"^[A-Za-z0-9_]+_test$")
_OVERRIDES = frozenset(
    {"db", "database", "host", "port", "unix_socket", "socket", "read_default_file"}
)
_MAX_NAME = 64


class UnsafeMySQLTestUrl(pytest.UsageError):
    """The configured MySQL-family test URL is not safe for destructive tests."""


def validated(raw: str, env: str) -> URL:
    """The URL, refusing anything that could reach a non-test database."""
    url = make_url(raw)
    if url.get_backend_name() not in {"mysql", "mariadb"}:
        raise UnsafeMySQLTestUrl(f"{env} is not a mysql URL")
    name = url.database or ""
    if not _NAME.match(name):
        raise UnsafeMySQLTestUrl(f"{env} must name a database matching [A-Za-z0-9_]+_test")
    if any(key.lower() in _OVERRIDES for key in url.query):
        raise UnsafeMySQLTestUrl(f"{env} carries a query option that overrides the target")
    return url


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` on a private loop in a helper thread (never touches the test's loop)."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


async def _admin(url: URL, statement: str) -> None:
    engine = create_async_engine(url._replace(database=None), poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET SESSION lock_wait_timeout = 10"))
            if statement.startswith("DROP DATABASE"):
                # Never hang teardown on a session a failing test leaked: end the sessions
                # still attached to this private test database first.
                leaked = (
                    await conn.execute(
                        text(
                            "SELECT id FROM information_schema.processlist "
                            "WHERE db = :name AND id <> CONNECTION_ID()"
                        ),
                        {"name": url.database},
                    )
                ).scalars()
                for session in list(leaked):
                    await conn.execute(text(f"KILL {int(session)}"))
            await conn.execute(text(statement))
    finally:
        await engine.dispose()


def sessions_left(url: str) -> int:
    """Other sessions still attached to this test database (leaked connections)."""

    async def go() -> int:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return int(
                    (
                        await conn.execute(
                            text(
                                "SELECT COUNT(*) FROM information_schema.processlist "
                                "WHERE db = DATABASE() AND id <> CONNECTION_ID()"
                            )
                        )
                    ).scalar_one()
                )
        finally:
            await engine.dispose()

    return int(run_sync(go()))


def fresh_database_url(base: URL) -> URL:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw")
    stem = (base.database or "").removesuffix("_test")
    name = f"{stem}_{worker}_{uuid.uuid4().hex[:8]}_test"
    if len(name) > _MAX_NAME or not _NAME.match(name):
        raise UnsafeMySQLTestUrl("per-test database name would be invalid")
    return base.set(database=name)


@pytest.fixture(params=["mysql", "mariadb"])
def mysql_backend(request: pytest.FixtureRequest) -> str:
    env = ENV[request.param]
    if not os.environ.get(env):
        pytest.skip(f"{env} not set")
    return str(request.param)


def mysql_database(backend: str) -> Iterator[str]:
    """Create a fresh, empty database on the backend's server; drop it afterwards."""
    env = ENV[backend]
    if not os.environ.get(env):
        pytest.skip(f"{env} not set")
    url = fresh_database_url(validated(os.environ[env], env))
    run_sync(_admin(url, f"CREATE DATABASE `{url.database}`"))
    try:
        yield url.render_as_string(hide_password=False)
    finally:
        run_sync(_admin(url, f"DROP DATABASE IF EXISTS `{url.database}`"))


@pytest.fixture
def mysql_url(mysql_backend: str) -> Iterator[str]:
    """A fresh, empty database on the backend's server, dropped afterwards."""
    yield from mysql_database(mysql_backend)
