"""Cursor pagination semantics on ``StateStore.get_run_events``.

Covers the ``after_sequence_index`` (exclusive ``>``) and ``limit``
parameters that back the read-router polling and SSE endpoints.
"""

from __future__ import annotations

import pytest


@pytest.fixture
async def store_with_events():
    """Fresh in-memory SQLite store seeded with 10 sequential events."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from dendrux.runtime.state import SQLAlchemyStateStore

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = SQLAlchemyStateStore(engine)

    run_id = "test-run-cursor"
    await store.create_run(
        run_id,
        "TestAgent",
        input_data={"input": "test"},
        model="mock",
        strategy="NativeToolCalling",
    )

    for i in range(10):
        await store.save_run_event(
            run_id,
            event_type=f"test.event.{i}",
            sequence_index=i,
            iteration_index=0,
            data={"index": i},
        )

    yield store, run_id
    await engine.dispose()


class TestGetRunEventsCursor:
    async def test_no_params_returns_all(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id)
        assert len(events) == 10
        assert events[0].sequence_index == 0
        assert events[9].sequence_index == 9

    async def test_after_sequence_index_exclusive(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=4)
        assert [e.sequence_index for e in events] == [5, 6, 7, 8, 9]

    async def test_after_sequence_index_zero(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=0)
        assert [e.sequence_index for e in events] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    async def test_after_last_sequence(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=9)
        assert events == []

    async def test_after_out_of_range(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=999)
        assert events == []

    async def test_limit(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=3)
        assert [e.sequence_index for e in events] == [0, 1, 2]

    async def test_after_and_limit(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, after_sequence_index=2, limit=3)
        assert [e.sequence_index for e in events] == [3, 4, 5]

    async def test_unknown_run_returns_empty(self, store_with_events) -> None:
        store, _ = store_with_events
        events = await store.get_run_events("no-such-run", after_sequence_index=0, limit=10)
        assert events == []

    async def test_limit_none_returns_all(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id, limit=None)
        assert len(events) == 10


class TestProtocolBackwardCompat:
    async def test_existing_callers_unaffected(self, store_with_events) -> None:
        store, run_id = store_with_events
        events = await store.get_run_events(run_id)
        assert len(events) == 10
        for i, event in enumerate(events):
            assert event.sequence_index == i
