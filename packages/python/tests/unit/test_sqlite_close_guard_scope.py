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
