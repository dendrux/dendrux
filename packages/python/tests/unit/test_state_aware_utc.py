"""Boundary contract: every datetime returned by ``StateStore`` is aware-UTC.

Covers the ``_to_aware_utc`` helper (idempotent on aware inputs, tags naive
inputs as UTC, passes through ``None``) and the read-side guarantee that no
naive datetime escapes the store. The latter matters because SQLite +
``DateTime(timezone=True)`` does not roundtrip tzinfo, so reads come back
naive even when the schema asks for TIMESTAMPTZ — without normalization the
invariant would silently break on SQLite while passing on Postgres.
"""

from __future__ import annotations

import datetime as dt

import pytest

from dendrux.runtime.state import _require_aware_utc, _to_aware_utc


class TestToAwareUtc:
    def test_none_passthrough(self) -> None:
        assert _to_aware_utc(None) is None

    def test_naive_tagged_as_utc(self) -> None:
        naive = dt.datetime(2026, 4, 25, 12, 0, 0)
        result = _to_aware_utc(naive)
        assert result.tzinfo is dt.UTC
        assert result.replace(tzinfo=None) == naive

    def test_aware_utc_passthrough(self) -> None:
        aware = dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.UTC)
        result = _to_aware_utc(aware)
        assert result == aware
        assert result.tzinfo is dt.UTC

    def test_aware_non_utc_converted_to_utc(self) -> None:
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        aware_ist = dt.datetime(2026, 4, 25, 17, 30, 0, tzinfo=ist)
        result = _to_aware_utc(aware_ist)
        assert result.tzinfo is dt.UTC
        assert result == dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.UTC)

    def test_idempotent_on_aware(self) -> None:
        aware = dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.UTC)
        assert _to_aware_utc(_to_aware_utc(aware)) == aware


@pytest.fixture
async def store():
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from dendrux.runtime.state import SQLAlchemyStateStore

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return SQLAlchemyStateStore(engine)


class TestStoreReadsAreAwareUtc:
    """Every datetime returned from store reads must be aware-UTC."""

    @pytest.mark.asyncio
    async def test_get_run_returns_aware_datetimes(self, store) -> None:
        await store.create_run("run-1", "agent-x")
        record = await store.get_run("run-1")
        assert record is not None
        assert record.created_at is not None
        assert record.created_at.tzinfo is dt.UTC
        assert record.updated_at is not None
        assert record.updated_at.tzinfo is dt.UTC
        assert record.last_progress_at is not None
        assert record.last_progress_at.tzinfo is dt.UTC

    @pytest.mark.asyncio
    async def test_get_traces_returns_aware_datetimes(self, store) -> None:
        await store.create_run("run-2", "agent-x")
        await store.save_trace(
            run_id="run-2",
            role="user",
            content="hello",
            order_index=0,
        )
        traces = await store.get_traces("run-2")
        assert len(traces) == 1
        assert traces[0].created_at is not None
        assert traces[0].created_at.tzinfo is dt.UTC

    @pytest.mark.asyncio
    async def test_get_run_events_returns_aware_datetimes(self, store) -> None:
        await store.create_run("run-3", "agent-x")
        await store.save_run_event(
            "run-3",
            event_type="run.started",
            sequence_index=0,
            iteration_index=0,
        )
        events = await store.get_run_events("run-3")
        assert len(events) == 1
        assert events[0].created_at is not None
        assert events[0].created_at.tzinfo is dt.UTC

    @pytest.mark.asyncio
    async def test_list_runs_returns_aware_datetimes(self, store) -> None:
        await store.create_run("run-4", "agent-x")
        runs = await store.list_runs(limit=10)
        assert len(runs) >= 1
        for run in runs:
            assert run.created_at is not None
            assert run.created_at.tzinfo is dt.UTC
            assert run.updated_at is not None
            assert run.updated_at.tzinfo is dt.UTC


class TestRequireAwareUtc:
    """Filter-boundary contract: naive datetimes are rejected with a clear error."""

    def test_none_passes(self) -> None:
        _require_aware_utc("started_after", None)  # no raise

    def test_aware_passes(self) -> None:
        _require_aware_utc("started_after", dt.datetime(2026, 4, 25, tzinfo=dt.UTC))

    def test_aware_non_utc_passes(self) -> None:
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        _require_aware_utc("started_after", dt.datetime(2026, 4, 25, tzinfo=ist))

    def test_naive_raises_with_helpful_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _require_aware_utc("started_after", dt.datetime(2026, 4, 25))
        msg = str(exc_info.value)
        assert "started_after" in msg
        assert "datetime.now(UTC)" in msg
        assert "TIMESTAMPTZ" in msg


class TestListRunsRejectsNaiveFilters:
    """Public API: list_runs / count_runs reject naive datetime filters."""

    @pytest.mark.asyncio
    async def test_list_runs_rejects_naive_started_after(self, store) -> None:
        with pytest.raises(ValueError, match="started_after"):
            await store.list_runs(started_after=dt.datetime(2026, 4, 25))

    @pytest.mark.asyncio
    async def test_list_runs_rejects_naive_started_before(self, store) -> None:
        with pytest.raises(ValueError, match="started_before"):
            await store.list_runs(started_before=dt.datetime(2026, 4, 25))

    @pytest.mark.asyncio
    async def test_count_runs_rejects_naive_started_after(self, store) -> None:
        with pytest.raises(ValueError, match="started_after"):
            await store.count_runs(started_after=dt.datetime(2026, 4, 25))

    @pytest.mark.asyncio
    async def test_count_runs_rejects_naive_started_before(self, store) -> None:
        with pytest.raises(ValueError, match="started_before"):
            await store.count_runs(started_before=dt.datetime(2026, 4, 25))

    @pytest.mark.asyncio
    async def test_list_runs_accepts_aware_filters(self, store) -> None:
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
        await store.list_runs(started_after=cutoff, started_before=dt.datetime.now(dt.UTC))
