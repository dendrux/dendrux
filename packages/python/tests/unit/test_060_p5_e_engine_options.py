"""Library-owned MySQL/MariaDB engines get READ COMMITTED and UTC without extra options."""

from __future__ import annotations

import pytest

from dendrux.db._engine_options import engine_options, normalized_url

UTC_INIT = "SET time_zone='+00:00'"


@pytest.mark.parametrize(
    "url",
    [
        "mysql://u:p@db.example/app",
        "mysql+asyncmy://u:p@db.example/app",
        "mysql+aiomysql://u:p@db.example/app",
        "mariadb://u:p@db.example/app",
        "mariadb+asyncmy://u:p@db.example/app",
        "mariadb+aiomysql://u:p@db.example/app",
    ],
)
def test_every_mysql_family_url_uses_the_supported_async_driver(url: str) -> None:
    parsed = normalized_url(url)
    assert parsed.drivername == "mysql+asyncmy"
    assert parsed.host == "db.example"
    assert parsed.database == "app"


def test_normalising_keeps_credentials_and_query() -> None:
    parsed = normalized_url("mysql+aiomysql://u:p%40ss@DB.example:3307/app?charset=utf8mb4")
    assert (parsed.username, parsed.password, parsed.port) == ("u", "p@ss", 3307)
    assert parsed.host == "db.example"
    assert dict(parsed.query) == {"charset": "utf8mb4"}


def test_mysql_engines_are_read_committed_with_utc_at_connect() -> None:
    options = engine_options("mysql+asyncmy://u:p@h/app")
    assert options["isolation_level"] == "READ COMMITTED"
    assert options["connect_args"]["init_command"] == UTC_INIT
    assert options["connect_args"]["charset"] == "utf8mb4"
    assert options["hide_parameters"] is True
    assert options["pool_pre_ping"] is True


def test_caller_connect_args_are_kept_alongside_the_required_ones() -> None:
    options = engine_options(
        "mysql+asyncmy://u:p@h/app",
        {"connect_args": {"connect_timeout": 3}, "pool_size": 7},
    )
    assert options["connect_args"] == {
        "connect_timeout": 3,
        "init_command": UTC_INIT,
        "charset": "utf8mb4",
    }
    assert options["pool_size"] == 7


@pytest.mark.parametrize(
    "supplied",
    [
        {"isolation_level": "REPEATABLE READ"},
        {"connect_args": {"init_command": "SET time_zone='+05:00'"}},
    ],
)
def test_options_that_break_the_store_contract_are_refused(supplied: dict) -> None:
    with pytest.raises(ValueError, match="requires"):
        engine_options("mysql+asyncmy://u:p@h/app", supplied)


def test_the_required_values_may_be_restated() -> None:
    options = engine_options(
        "mysql+asyncmy://u:p@h/app",
        {"isolation_level": "READ COMMITTED", "connect_args": {"init_command": UTC_INIT}},
    )
    assert options["connect_args"]["init_command"] == UTC_INIT


@pytest.mark.parametrize("url", ["postgresql+asyncpg://u:p@h/app", "sqlite+aiosqlite:///:memory:"])
def test_other_databases_get_no_mysql_settings(url: str) -> None:
    options = engine_options(url)
    assert "isolation_level" not in options
    assert "init_command" not in options.get("connect_args", {})


class _StopError(Exception):
    pass


def _record_engines(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, dict]]:
    import sqlalchemy.ext.asyncio as asyncio_ext

    created: list[tuple[object, dict]] = []

    def fake(url: object, **kwargs: object) -> object:
        created.append((url, kwargs))
        raise _StopError

    monkeypatch.setattr(asyncio_ext, "create_async_engine", fake)
    return created


@pytest.mark.parametrize("command", ["status", "activate", "check"])
async def test_cli_database_commands_build_mysql_engines_the_same_way(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from dendrux.cli import db as cli_db

    created = _record_engines(monkeypatch)
    url = "mysql+aiomysql://u:p@h/app"
    with pytest.raises(_StopError):
        if command == "status":
            await cli_db._read_activation_status(url)
        elif command == "activate":
            await cli_db._activate_database(url, 100, False)
        else:
            await cli_db._read_schema_state(url)
    [(built, kwargs)] = created
    assert built.drivername == "mysql+asyncmy"  # type: ignore[attr-defined]
    assert kwargs["isolation_level"] == "READ COMMITTED"
    assert kwargs["connect_args"]["init_command"] == UTC_INIT


def test_cli_migrations_receive_the_supported_mysql_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dendrux.cli import db as cli_db

    monkeypatch.setenv("DENDRUX_DATABASE_URL", "mysql+aiomysql://u:p%40x@h/app")
    assert cli_db._database_url() == "mysql+asyncmy://u:p%40x@h/app"
    monkeypatch.setenv("DENDRUX_DATABASE_URL", "postgresql+asyncpg://u:p@h/app")
    assert cli_db._database_url() == "postgresql+asyncpg://u:p@h/app"
