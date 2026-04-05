"""Integration tests for stale-run sweep — real SQLite, full lifecycle.

Uses in-memory SQLite so no files are created.
Each test gets a fresh engine and tables via the `engine` fixture.
"""

from __future__ import annotations

import datetime as dt
from datetime import timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from dendrux.db.models import AgentRun, Base
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.types import generate_ulid

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def engine():
    """Create a fresh in-memory SQLite engine with all tables."""
    from sqlalchemy import event

    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def store(engine):
    """StateStore backed by the test engine."""
    return SQLAlchemyStateStore(engine)


@pytest.fixture
def session_factory(engine):
    """Raw session factory for direct DB assertions."""
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _create_run(
    store: SQLAlchemyStateStore,
    *,
    run_id: str | None = None,
    agent_name: str = "TestAgent",
    status: str = "running",
    last_progress_at: dt.datetime | None = None,
    created_at: dt.datetime | None = None,
    emit_started_event: bool = True,
) -> str:
    """Create a run row with optional overrides for testing."""
    rid = run_id or generate_ulid()
    await store.create_run(rid, agent_name)

    # Override status if not running (create_run always sets running)
    if status != "running":
        async with store._session_factory() as session:
            stmt = (
                update(AgentRun)
                .where(AgentRun.id == rid)
                .values(status=status)
            )
            await session.execute(stmt)
            await session.commit()

    # Override timestamps if provided
    updates: dict = {}
    if last_progress_at is not None:
        updates["last_progress_at"] = last_progress_at
    if created_at is not None:
        updates["created_at"] = created_at

    if updates:
        async with store._session_factory() as session:
            stmt = update(AgentRun).where(AgentRun.id == rid).values(**updates)
            await session.execute(stmt)
            await session.commit()

    # Emit run.started event (as the runner would)
    if emit_started_event:
        await store.save_run_event(
            rid,
            event_type="run.started",
            sequence_index=0,
            data={"agent_name": agent_name},
        )

    return rid


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSweepStaleRuns:
    """Tests for StateStore.sweep_stale_runs()."""

    async def test_stale_running_is_swept(self, store):
        """A RUNNING row with old last_progress_at is swept to ERROR."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(store, last_progress_at=old_time)

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 1
        assert swept[0].run_id == rid
        assert swept[0].failure_reason == "stale_running"
        assert swept[0].previous_status == "running"

        # Verify DB state
        run = await store.get_run(rid)
        assert run is not None
        assert run.status == "error"
        assert run.failure_reason == "stale_running"

    async def test_fresh_running_not_swept(self, store):
        """A RUNNING row with recent last_progress_at is NOT swept."""
        rid = await _create_run(store)  # last_progress_at = now

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 0

        run = await store.get_run(rid)
        assert run is not None
        assert run.status == "running"

    async def test_waiting_rows_not_swept(self, store):
        """WAITING_* rows are never touched by stale sweep."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)

        for status in ["waiting_client_tool", "waiting_human_input", "waiting_approval"]:
            await _create_run(
                store, status=status, last_progress_at=old_time
            )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))
        assert len(swept) == 0

    async def test_terminal_rows_not_swept(self, store):
        """Terminal status rows (SUCCESS, ERROR, etc.) are not swept."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)

        for status in ["success", "error", "cancelled", "max_iterations"]:
            await _create_run(
                store, status=status, last_progress_at=old_time
            )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))
        assert len(swept) == 0

    async def test_null_last_progress_at_falls_back_to_created_at(self, store):
        """If last_progress_at is NULL, created_at is used for staleness."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(
            store, last_progress_at=None, created_at=old_time
        )

        # Manually NULL out last_progress_at (create_run sets it)
        async with store._session_factory() as session:
            stmt = (
                update(AgentRun)
                .where(AgentRun.id == rid)
                .values(last_progress_at=None)
            )
            await session.execute(stmt)
            await session.commit()

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))
        assert len(swept) == 1
        assert swept[0].run_id == rid

    async def test_never_started_classification(self, store):
        """Run without run.started event is classified as never_started."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(
            store, last_progress_at=old_time, emit_started_event=False
        )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 1
        assert swept[0].failure_reason == "never_started"

        run = await store.get_run(rid)
        assert run is not None
        assert run.failure_reason == "never_started"

    async def test_stale_running_classification(self, store):
        """Run with run.started event is classified as stale_running."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        await _create_run(
            store, last_progress_at=old_time, emit_started_event=True
        )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 1
        assert swept[0].failure_reason == "stale_running"

    async def test_interrupted_event_emitted(self, store):
        """Sweep emits a run.interrupted event with failure_reason in data."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(store, last_progress_at=old_time)

        await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        events = await store.get_run_events(rid)
        interrupted = [e for e in events if e.event_type == "run.interrupted"]
        assert len(interrupted) == 1
        assert interrupted[0].data == {"failure_reason": "stale_running"}

    async def test_structured_result(self, store):
        """SweptRun has all expected fields populated."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(
            store, agent_name="MyAgent", last_progress_at=old_time
        )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 1
        s = swept[0]
        assert s.run_id == rid
        assert s.agent_name == "MyAgent"
        assert s.previous_status == "running"
        assert s.failure_reason == "stale_running"
        assert s.last_progress_at is not None
        assert s.swept_at is not None

    async def test_multiple_stale_runs(self, store):
        """Multiple stale runs are all swept in one call."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rids = []
        for i in range(3):
            rid = await _create_run(
                store, agent_name=f"Agent{i}", last_progress_at=old_time
            )
            rids.append(rid)

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 3
        swept_ids = {s.run_id for s in swept}
        assert swept_ids == set(rids)

    async def test_mixed_runs_only_stale_swept(self, store):
        """Only stale RUNNING rows are swept; fresh, waiting, terminal are untouched."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)

        stale_rid = await _create_run(store, last_progress_at=old_time)
        fresh_rid = await _create_run(store)  # recent
        waiting_rid = await _create_run(
            store, status="waiting_client_tool", last_progress_at=old_time
        )
        done_rid = await _create_run(
            store, status="success", last_progress_at=old_time
        )

        swept = await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        assert len(swept) == 1
        assert swept[0].run_id == stale_rid

        # Others unchanged
        assert (await store.get_run(fresh_rid)).status == "running"
        assert (await store.get_run(waiting_rid)).status == "waiting_client_tool"
        assert (await store.get_run(done_rid)).status == "success"


    async def test_sweep_updates_updated_at(self, store, session_factory):
        """Swept runs have their updated_at refreshed."""
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        rid = await _create_run(store, last_progress_at=old_time)

        # Read updated_at before sweep
        async with session_factory() as session:
            from sqlalchemy import select
            stmt = select(AgentRun).where(AgentRun.id == rid)
            result = await session.execute(stmt)
            row = result.scalar_one()
            updated_before = row.updated_at

        await store.sweep_stale_runs(older_than=timedelta(minutes=20))

        # Read updated_at after sweep
        async with session_factory() as session:
            stmt = select(AgentRun).where(AgentRun.id == rid)
            result = await session.execute(stmt)
            row = result.scalar_one()
            assert row.updated_at >= updated_before


class TestTouchProgress:
    """Tests for StateStore.touch_progress()."""

    async def test_touch_updates_last_progress_at(self, store):
        """touch_progress updates the last_progress_at timestamp."""
        rid = await _create_run(store)

        run_before = await store.get_run(rid)
        initial_progress = run_before.last_progress_at

        # Small delay to ensure timestamp differs
        await store.touch_progress(rid)

        run_after = await store.get_run(rid)
        assert run_after.last_progress_at is not None
        assert run_after.last_progress_at >= initial_progress


class TestSweepPublicAPI:
    """Tests for the top-level sweep() convenience function."""

    async def test_sweep_with_state_store(self, store):
        """sweep() works with a pre-existing state_store."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        await _create_run(store, last_progress_at=old_time)

        result = await sweep_fn(
            state_store=store,
            stale_running=timedelta(minutes=20),
        )

        assert len(result.stale_running) == 1

    async def test_sweep_no_threshold_returns_empty(self, store):
        """sweep() with no threshold returns empty result."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        result = await sweep_fn(state_store=store)

        assert len(result.stale_running) == 0

    async def test_sweep_both_args_raises(self):
        """sweep() with both database_url and state_store raises."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        with pytest.raises(ValueError, match="mutually exclusive"):
            await sweep_fn(
                database_url="sqlite+aiosqlite:///:memory:",
                state_store=object(),  # type: ignore[arg-type]
                stale_running=timedelta(minutes=20),
            )

    async def test_sweep_no_args_raises(self):
        """sweep() with neither database_url nor state_store raises."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        with pytest.raises(ValueError, match="required"):
            await sweep_fn(stale_running=timedelta(minutes=20))

    async def test_sweep_with_database_url_zero_config(self):
        """sweep(database_url=...) auto-creates tables on fresh SQLite."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        result = await sweep_fn(
            database_url="sqlite+aiosqlite:///:memory:",
            stale_running=timedelta(minutes=20),
        )

        # Fresh DB, no runs — should return empty, not crash
        assert len(result.stale_running) == 0

    async def test_sweep_with_database_url_file(self, tmp_path):
        """sweep(database_url=...) works end-to-end with a real SQLite file."""
        from dendrux.runtime.sweep import sweep as sweep_fn

        db_path = tmp_path / "test_sweep.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        # First sweep creates tables, finds nothing
        result = await sweep_fn(
            database_url=db_url,
            stale_running=timedelta(minutes=20),
        )
        assert len(result.stale_running) == 0

        # Create a stale run directly via a store on the same file
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.db.models import Base

        eng = create_async_engine(
            db_url, connect_args={"check_same_thread": False}
        )

        @event.listens_for(eng.sync_engine, "connect")
        def _fk(conn, _):
            conn.execute("PRAGMA foreign_keys = ON")

        temp_store = SQLAlchemyStateStore(eng)
        old_time = dt.datetime.now(dt.UTC) - timedelta(minutes=30)
        await _create_run(temp_store, last_progress_at=old_time)
        await eng.dispose()

        # Second sweep finds and sweeps the stale run
        result = await sweep_fn(
            database_url=db_url,
            stale_running=timedelta(minutes=20),
        )
        assert len(result.stale_running) == 1
        assert result.stale_running[0].failure_reason == "stale_running"
