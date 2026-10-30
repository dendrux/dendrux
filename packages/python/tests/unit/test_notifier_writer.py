"""The writer hook remains inside the event transaction and adds no SQLite work."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.runtime.state import SQLAlchemyStateStore


@pytest.mark.parametrize(
    "dialect,channel",
    [("postgresql", "dendrux_events"), ("postgresql", None), ("sqlite", "dendrux_events")],
)
async def test_writer_sql_only_for_enabled_postgres(dialect, channel):
    session = SimpleNamespace(
        bind=SimpleNamespace(dialect=SimpleNamespace(name=dialect)), info={}, execute=AsyncMock()
    )
    store = object.__new__(SQLAlchemyStateStore)
    store.notify_channel = channel
    await store._after_event_insert(session, "run")
    if dialect == "postgresql" and channel:
        session.execute.assert_awaited_once()
        sql, params = session.execute.call_args.args
        assert str(sql) == "SELECT pg_notify(:channel, :run_id)"
        assert params == {"channel": channel, "run_id": "run"}
    else:
        session.execute.assert_not_awaited()


async def test_writer_notify_error_propagates():
    session = SimpleNamespace(
        bind=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        info={},
        execute=AsyncMock(side_effect=RuntimeError("sentinel")),
    )
    store = object.__new__(SQLAlchemyStateStore)
    store.notify_channel = "dendrux_events"
    with pytest.raises(RuntimeError):
        await store._after_event_insert(session, "run")


@pytest.mark.parametrize("channel", ["", "x" * 64, "é" * 32, "a\x00b", 1, b"bytes", True])
async def test_p2_3_writer_rejects_invalid_channel_at_construction(channel):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        with pytest.raises(ValueError):
            SQLAlchemyStateStore(engine, notify_channel=channel)
    finally:
        await engine.dispose()
