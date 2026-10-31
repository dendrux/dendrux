"""One engine option policy, without connections or session-level SQL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.engine import URL, make_url

_MYSQL_DRIVER = "mysql+asyncmy"
_MYSQL_UTC_INIT = "SET time_zone='+00:00'"


def is_mysql_family_url(url: str | URL) -> bool:
    """True for MySQL and MariaDB URLs, whatever driver or dialect spelling they use."""
    return make_url(url).get_backend_name() in {"mysql", "mariadb"}


def mysql_engine_options(supplied: dict[str, Any]) -> dict[str, Any]:
    """Add the settings the MySQL store requires; refuse options that contradict them."""
    options = dict(supplied)
    if options.get("isolation_level", "READ COMMITTED") != "READ COMMITTED":
        raise ValueError('MySQLStateStore requires isolation_level="READ COMMITTED"')
    options["isolation_level"] = "READ COMMITTED"
    connect = dict(options.get("connect_args", {}))
    if connect.get("init_command", _MYSQL_UTC_INIT) != _MYSQL_UTC_INIT:
        raise ValueError("MySQLStateStore requires the UTC connection init_command")
    connect["init_command"] = _MYSQL_UTC_INIT
    connect.setdefault("charset", "utf8mb4")
    options["connect_args"] = connect
    options["hide_parameters"] = True
    return options


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
        options.update(pool_pre_ping=True, pool_recycle=1800)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep store-only configuration away from SQLAlchemy engine construction."""
    engine = dict(supplied or {})
    store = {}
    if "notify_channel" in engine:
        store["notify_channel"] = engine.pop("notify_channel")
    return engine, store
