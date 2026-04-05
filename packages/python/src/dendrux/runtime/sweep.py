"""Stale-run sweep — public maintenance API.

Detects runs stuck in RUNNING beyond a threshold and marks them ERROR
with a structured failure reason. Developer calls this at app startup.

Usage::

    from dendrux import sweep

    results = await sweep(
        database_url="sqlite+aiosqlite:///runs.db",
        stale_running=timedelta(minutes=20),
    )

    for run in results.stale_running:
        logger.warning("Swept %s (%s), reason=%s",
                        run.run_id, run.agent_name, run.failure_reason)
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
) -> SweepResult:
    """Sweep stale runs. Developer calls this at app startup.

    Dendrux detects and marks stale RUNNING rows as ERROR, emits
    ``run.interrupted`` events, and returns a structured report of
    what it changed. The developer decides any follow-up action.

    Args:
        database_url: Database URL. Creates a temporary engine, disposed
            after sweep. Mutually exclusive with ``state_store``.
        state_store: Pre-existing StateStore instance. Mutually exclusive
            with ``database_url``.
        stale_running: Threshold for stale RUNNING detection. Runs with
            no forward progress beyond this duration are swept. If None,
            no stale-running sweep is performed.

    Returns:
        SweepResult with list of swept runs.

    Raises:
        ValueError: If both or neither of database_url/state_store provided,
            or if no sweep threshold is specified.
    """
    if database_url is not None and state_store is not None:
        raise ValueError(
            "database_url and state_store are mutually exclusive. Pass one or the other."
        )
    if database_url is None and state_store is None:
        raise ValueError(
            "Either database_url or state_store is required."
        )
    if stale_running is None:
        return SweepResult(stale_running=[])

    store: StateStore
    engine = None

    if database_url is not None:
        store, engine = await _create_temp_store(database_url)

    else:
        assert state_store is not None
        store = state_store

    try:
        swept = await store.sweep_stale_runs(older_than=stale_running)
        return SweepResult(stale_running=swept)
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
