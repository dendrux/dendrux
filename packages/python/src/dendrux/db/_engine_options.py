"""One engine option policy, without connections or session-level SQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import unquote

from sqlalchemy.engine import URL, make_url

POOL_RECYCLE_S = 1800
POOL_PRE_PING = True

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

_MYSQL_DRIVER = "mysql+asyncmy"
_MYSQL_UTC_INIT = "SET time_zone='+00:00'"
_SQLITE_MEMORY = ":memory:"
# Plain name=value URI options only; anything else (such as %00) is unclear.
_PLAIN_URI_QUERY = re.compile(r"\w+=\w+(?:&\w+=\w+)*", re.ASCII)
_T = TypeVar("_T")


def mysql_dependency_error(error: BaseException) -> ImportError | None:
    """Recognize missing optional drivers/auth dependencies without echoing their text."""
    if (
        isinstance(error, ImportError)
        and (getattr(error, "name", "") or "").split(".")[0] in {"asyncmy", "cryptography"}
    ) or (isinstance(error, RuntimeError) and "'cryptography' package is required" in str(error)):
        return ImportError(
            "MySQL persistence requires its optional drivers; install 'dendrux[mysql]'."
        )
    return None


def create_engine_with_hint(factory: Callable[..., _T], url: str | URL, **options: Any) -> _T:
    """Construct an engine with an actionable optional MySQL dependency error."""
    failure = None
    try:
        return factory(url, **options)
    except (ImportError, RuntimeError) as error:
        failure = mysql_dependency_error(error) if is_mysql_family_url(url) else None
        if failure is None:
            raise
    raise failure from None


def is_mysql_family_url(url: str | URL) -> bool:
    """True for MySQL and MariaDB URLs, whatever driver or dialect spelling they use."""
    return make_url(url).get_backend_name() in {"mysql", "mariadb"}


def mysql_engine_options(supplied: dict[str, Any]) -> dict[str, Any]:
    """Add the settings the MySQL store requires; refuse options that contradict them."""
    options = dict(supplied)
    if options.get("isolation_level", "READ COMMITTED") != "READ COMMITTED":
        raise ValueError('MySQLStateStore requires isolation_level="READ COMMITTED"')
    if (
        options.get("execution_options", {}).get("isolation_level", "READ COMMITTED")
        != "READ COMMITTED"
    ):
        raise ValueError(
            'MySQLStateStore requires isolation_level="READ COMMITTED" in execution_options'
        )
    options["isolation_level"] = "READ COMMITTED"
    connect = dict(options.get("connect_args", {}))
    if connect.get("init_command", _MYSQL_UTC_INIT) != _MYSQL_UTC_INIT:
        raise ValueError("MySQLStateStore requires the UTC connection init_command")
    connect["init_command"] = _MYSQL_UTC_INIT
    connect.setdefault("charset", "utf8mb4")
    options["connect_args"] = connect
    options["hide_parameters"] = True
    return options


def is_memory_sqlite(engine: AsyncEngine) -> bool:
    """True only when ``engine`` clearly opens an in-memory SQLite database.

    Reads the filename and ``uri`` flag the dialect passes to ``sqlite3`` and
    accepts an allow-list of clear forms, without copying SQLite's URI grammar:

    - the filename is exactly ``:memory:``, whatever the ``uri`` flag says;
    - with ``uri``: ``file:`` plus a path that percent-decodes to ``:memory:``
      (SQLite drops any ``#`` fragment first);
    - with ``uri``: ``file:<name>?<query>`` where the query is only plain
      ``name=value`` options of ASCII letters, digits and underscores, joined by
      ``&``, with at least one ``mode`` option and every ``mode`` equal to ``memory``.

    Anything unclear counts as a file, so the close guard stays on: percent-encoding
    or any other character in the query, a missing ``file:`` scheme, or no ``uri``
    flag. Options passed through ``connect_args`` or a custom ``creator`` are not
    visible here.
    """
    if engine.dialect.name != "sqlite":
        return False
    args, options = engine.dialect.create_connect_args(engine.url)
    if not args:
        return False
    if args[0] == _SQLITE_MEMORY:
        return True
    return bool(options.get("uri")) and _is_memory_uri(args[0])


def _is_memory_uri(filename: str) -> bool:
    if not filename.startswith("file:"):
        return False  # SQLite opens a URI without the file: scheme as a plain name
    path, _, query = filename.removeprefix("file:").partition("#")[0].partition("?")
    if unquote(path) == _SQLITE_MEMORY:
        return True
    if not _PLAIN_URI_QUERY.fullmatch(query):
        return False
    modes = [
        value
        for name, value in (option.split("=") for option in query.split("&"))
        if name == "mode"
    ]
    return bool(modes) and all(mode == "memory" for mode in modes)


def normalized_url(url: str | URL) -> URL:
    parsed = make_url(url)
    driver = {
        "postgres": "postgresql+asyncpg",
        "postgresql": "postgresql+asyncpg",
        "sqlite": "sqlite+aiosqlite",
    }.get(parsed.drivername, parsed.drivername)
    if is_mysql_family_url(parsed):
        # asyncmy is the one supported async driver; the server decides MySQL or MariaDB.
        driver = _MYSQL_DRIVER
    parsed = parsed.set(drivername=driver, query=dict(sorted(parsed.query.items())))
    if driver.startswith("sqlite"):
        database = parsed.database
        if not database or database == ":memory:":
            return parsed.set(database=":memory:")
        if not database.startswith("file:") and parsed.query.get("uri") != "true":
            parsed = parsed.set(database=str(Path(database).expanduser().resolve()))
    elif parsed.host:
        parsed = parsed.set(host=parsed.host.lower())
        if driver.startswith("postgresql") and parsed.port is None:
            parsed = parsed.set(port=5432)
    return parsed


def engine_options(url: str | URL, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = make_url(url)
    options: dict[str, Any] = {"echo": False}
    if parsed.drivername.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    else:
        options.update(pool_pre_ping=POOL_PRE_PING, pool_recycle=POOL_RECYCLE_S)
    incoming, _store = split_database_options(supplied)
    connect = dict(options.get("connect_args", {}))
    connect.update(incoming.pop("connect_args", {}))
    options.update(incoming)
    if connect:
        options["connect_args"] = connect
    options["hide_parameters"] = True
    if is_mysql_family_url(parsed):
        options = mysql_engine_options(options)
    return options


def frozen_options(value: Any) -> Any:
    """Preserve value types; opaque hashable driver objects compare by identity."""
    if isinstance(value, dict):
        return (dict, tuple(sorted((k, frozen_options(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(frozen_options(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(frozen_options(v) for v in value))
    try:
        hash(value)
    except TypeError:
        raise ValueError("database_options values must be hashable or containers") from None
    return (type(value), value)


def split_database_options(
    supplied: dict[str, Any] | None,
    *,
    replacement: str = "build a store with live_updates=... and pass it to Agent/sweep/RunStore",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep store-only configuration away from SQLAlchemy engine construction."""
    engine = dict(supplied or {})
    if {"live_updates", "notify_channel"}.intersection(engine):
        raise ValueError(f"Live updates require an explicit store; {replacement}")
    return engine, {}
