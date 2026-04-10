"""Run sweep — public maintenance API.

Detects stale RUNNING and abandoned WAITING runs, marks them ERROR
with structured failure reasons. Developer calls this at app startup.

Usage::

    from dendrux import sweep

    results = await sweep(
        database_url="sqlite+aiosqlite:///runs.db",
        stale_running=timedelta(minutes=20),
        abandoned_waiting=timedelta(hours=2),
    )

    for run in results.stale_running:
        logger.warning("Swept stale: %s (%s)", run.run_id, run.failure_reason)

    for run in results.abandoned_waiting:
        logger.warning("Swept abandoned: %s (%s)", run.run_id, run.previous_status)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux.runtime.state import SweepResult

if TYPE_CHECKING:
    from datetime import timedelta

    from dendrux.runtime.state import StateStore


async def sweep(
    *,
    database_url: str | None = None,
    state_store: StateStore | None = None,
    stale_running: timedelta | None = None,
    abandoned_waiting: timedelta | None = None,
) -> SweepResult:
    """Sweep stale and abandoned runs. Developer calls this at app startup.

    Detects stale RUNNING rows and abandoned WAITING rows, marks them
    ERROR with structured failure reasons, emits lifecycle events, and
    returns a report of what changed.

    Args:
        database_url: Database URL. Creates a temporary engine, disposed
            after sweep. Mutually exclusive with ``state_store``.
        state_store: Pre-existing StateStore instance. Mutually exclusive
            with ``database_url``.
        stale_running: Threshold for stale RUNNING detection. Runs with
            no forward progress beyond this duration are swept. If None,
            no stale-running sweep is performed.
        abandoned_waiting: Threshold for abandoned WAITING detection.
            WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT, and
            WAITING_APPROVAL runs with no state change beyond this
            duration are swept. If None, no abandoned-waiting sweep
            is performed.

    Returns:
        SweepResult with lists of swept runs by category.

    Raises:
        ValueError: If both or neither of database_url/state_store provided.
    """
    if database_url is not None and state_store is not None:
        raise ValueError(
            "database_url and state_store are mutually exclusive. Pass one or the other."
        )
    if database_url is None and state_store is None:
        raise ValueError("Either database_url or state_store is required.")
    if stale_running is None and abandoned_waiting is None:
        return SweepResult(stale_running=[])

    store: StateStore
    engine = None

    if database_url is not None:
        store, engine = await _create_temp_store(database_url)
    else:
        assert state_store is not None
        store = state_store

    try:
        stale = await store.sweep_stale_runs(older_than=stale_running) if stale_running else []
        abandoned = (
            await store.sweep_abandoned_runs(older_than=abandoned_waiting)
            if abandoned_waiting
            else []
        )
        return SweepResult(stale_running=stale, abandoned_waiting=abandoned)
    finally:
        if engine is not None:
            await engine.dispose()


async def _create_temp_store(
    database_url: str,
) -> tuple[Any, Any]:
    """Create a temporary StateStore + engine from a database URL.

    For SQLite, auto-creates tables if they don't exist (matching the
    zero-config promise in Agent._create_private_engine). For Postgres
    or other backends, tables are assumed to exist via migrations.

    The caller is responsible for disposing the engine after use.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base
    from dendrux.runtime.state import SQLAlchemyStateStore

    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        database_url,
        echo=False,
        connect_args=connect_args,
    )

    # Auto-create tables for SQLite (zero-config promise)
    if database_url.startswith("sqlite"):
        from pathlib import Path

        db_path = database_url.split("///", 1)[-1] if "///" in database_url else None
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    store = SQLAlchemyStateStore(engine)
    return store, engine
