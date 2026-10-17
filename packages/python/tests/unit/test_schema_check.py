"""Schema-currency check: comparison logic, error shape, skip rules (no database)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import dendrux
import dendrux.errors as errors
from dendrux.db._schema_check import SKIP_ENV, compare_schema, ensure_schema_current
from dendrux.db.models import Base
from dendrux.errors import SchemaNotInitializedError, SchemaOutOfDateError


def _full() -> dict[str, list[str]]:
    return {t.name: [c.name for c in t.columns] for t in Base.metadata.tables.values()}


def test_complete_schema_has_nothing_missing() -> None:
    assert compare_schema(_full()) == ([], [])


def test_missing_column_is_reported_as_table_dot_column() -> None:
    actual = _full()
    actual["agent_runs"].remove("pause_requested")
    assert compare_schema(actual) == ([], ["agent_runs.pause_requested"])


def test_missing_table_is_reported_without_its_columns() -> None:
    actual = _full()
    del actual["run_inbox"]
    assert compare_schema(actual) == (["run_inbox"], [])


def test_extra_tables_and_columns_are_ignored() -> None:
    actual = _full()
    actual["agent_runs"].append("app_owned_column")
    actual["app_owned_table"] = ["id"]
    assert compare_schema(actual) == ([], [])


def test_names_compare_case_insensitively() -> None:
    actual = {name.upper(): [c.upper() for c in cols] for name, cols in _full().items()}
    assert compare_schema(actual) == ([], [])


def test_error_message_lists_objects_and_the_remedy() -> None:
    err = SchemaOutOfDateError(["run_inbox"], ["agent_runs.pause_requested"])
    text = str(err)
    assert "agent_runs.pause_requested" in text
    assert "table run_inbox" in text
    assert "dendrux db migrate" in text
    assert "DENDRUX_DATABASE_URL" in text
    assert SKIP_ENV in text
    assert err.missing_tables == ("run_inbox",)
    assert err.missing_columns == ("agent_runs.pause_requested",)
    assert err.missing == ("agent_runs.pause_requested", "run_inbox")


def test_error_is_exported() -> None:
    assert dendrux.SchemaOutOfDateError is SchemaOutOfDateError
    assert "SchemaOutOfDateError" in dendrux.__all__
    assert "SchemaOutOfDateError" in errors.__all__
    assert issubclass(SchemaOutOfDateError, RuntimeError)


def _engine(dialect: str) -> MagicMock:
    engine = MagicMock()
    engine.dialect.name = dialect
    return engine


@pytest.mark.asyncio
async def test_sqlite_is_never_inspected() -> None:
    engine = _engine("sqlite")
    await ensure_schema_current(engine)
    engine.connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["1", "true", "YES"])
async def test_skip_env_var_never_inspects(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(SKIP_ENV, value)
    engine = _engine("postgresql")
    await ensure_schema_current(engine)
    engine.connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "0", "no"])
async def test_falsey_skip_env_var_still_checks(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SKIP_ENV, value)
    engine = _engine("postgresql")
    engine.connect.side_effect = RuntimeError("reached the database")
    with pytest.raises(RuntimeError, match="reached the database"):
        await ensure_schema_current(engine)


@pytest.mark.asyncio
async def test_empty_database_raises_not_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SKIP_ENV, raising=False)
    engine = _engine("postgresql")
    conn = MagicMock()

    async def run_sync(_fn: object) -> dict[str, list[str]]:
        return {}

    conn.run_sync = run_sync
    engine.connect.return_value.__aenter__ = _async_return(conn)
    engine.connect.return_value.__aexit__ = _async_return(None)
    with pytest.raises(SchemaNotInitializedError):
        await ensure_schema_current(engine)


def _async_return(value: object):
    async def _call(*_args: object) -> object:
        return value

    return _call
