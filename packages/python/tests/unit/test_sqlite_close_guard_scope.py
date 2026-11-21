"""The SQLite close guard instruments file databases only, never in-memory ones."""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db._pool_guard import (
    _finalize_before_close,
    _track_sqlite_statements,
    guard_sqlite_closes,
)
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

_LISTENERS = (
    ("connect", _track_sqlite_statements),
    ("checkout", _track_sqlite_statements),
    ("close", _finalize_before_close),
    ("close_detached", _finalize_before_close),
)

MEMORY_URLS = [
    "sqlite+aiosqlite://",
    "sqlite+aiosqlite:///:memory:",
    "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
    "sqlite+aiosqlite:///file:guard_scope?mode=memory&cache=shared&uri=true",
]


async def _driver_is_wrapped(engine) -> bool:
    async with engine.connect() as connection:
        raw = await connection.get_raw_connection()
        return "_execute" in vars(raw.driver_connection)


def _listening(engine) -> list[str]:
    return [name for name, fn in _LISTENERS if event.contains(engine.sync_engine, name, fn)]


@pytest.mark.parametrize("url", MEMORY_URLS)
async def test_memory_sqlite_gets_no_listener_or_wrapper(url) -> None:
    engine = create_async_engine(url)
    try:
        guard_sqlite_closes(engine)
        SQLiteStateStore(engine)
        assert _listening(engine) == []
        assert not await _driver_is_wrapped(engine)
    finally:
        await engine.dispose()


async def test_file_sqlite_is_still_guarded(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'guarded.db'}")
    try:
        SQLiteStateStore(engine)
        assert _listening(engine) == [name for name, _ in _LISTENERS]
        assert await _driver_is_wrapped(engine)
    finally:
        await engine.dispose()


# (URL, opens an in-memory database). The second value is checked against what SQLite
# actually opened before the guard and the store are judged.
URL_KINDS = [
    ("sqlite+aiosqlite://", True),
    ("sqlite+aiosqlite:///:memory:", True),
    ("sqlite+aiosqlite:///file::memory:?cache=shared&uri=true", True),
    ("sqlite+aiosqlite:///file:named?mode=memory&cache=shared&uri=true", True),
    ("sqlite+aiosqlite:///file:%3Amemory%3A?uri=true", True),  # percent-encoded name
    ("sqlite+aiosqlite:///file::memory:#ignored?uri=true", True),  # SQLite drops the fragment
    ("sqlite+aiosqlite:///ordinary.db?mode=memory", False),  # no uri=true: query not forwarded
    ("sqlite+aiosqlite:///disabled.db?mode=memory&uri=false", False),
    ("sqlite+aiosqlite:///no_scheme.db?mode=memory&uri=true", False),  # URI needs file:
    ("sqlite+aiosqlite:///file::memory:", False),  # URI off: a file with that literal name
    ("sqlite+aiosqlite:///file::memory:?uri=false", False),
    ("sqlite+aiosqlite:///file:other.db?mode=MEMORY&uri=true", False),  # unclear: stay guarded
    ("sqlite+aiosqlite:///normal.db", False),
]


@pytest.mark.parametrize(("url", "memory"), URL_KINDS)
async def test_guard_and_store_follow_the_database_sqlite_opens(
    url, memory, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = create_async_engine(url)
    try:
        if "MEMORY" not in url:  # SQLite refuses an unknown mode; nothing to open
            async with engine.connect() as connection:
                main = (await connection.exec_driver_sql("PRAGMA database_list")).one()
            assert (main[2] == "") is memory, "the case table names the wrong database kind"
        store = SQLiteStateStore(engine)
        assert store.capabilities.deployment == ("single_process" if memory else "single_host")
        assert _listening(engine) == ([] if memory else [name for name, _ in _LISTENERS])
    finally:
        await engine.dispose()
