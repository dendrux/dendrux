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
    batch_size: int = 100,
    max_runs: int | None = None,
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
            no stale-running sweep is performed. It must exceed the longest
            gap between progress updates (LLM calls, tool starts/ends,
            ``heartbeat()``), otherwise event sequence indexes of a run that
            is still executing may collide with the sweep's event.
        abandoned_waiting: Threshold for abandoned WAITING detection.
            WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT, WAITING_APPROVAL and
            PAUSED runs whose status has not changed beyond this duration are
            swept. If None, no abandoned-waiting sweep is performed.

        batch_size: Candidate rows per projected page (default 100, maximum 1000).
        max_runs: Optional candidate-cap override. Defaults independently to
            10,000 running rows (maximum 100,000) and 1,000 waiting rows
            (maximum 10,000). Fresh running rows count toward the cap; running
            population is bounded operationally by live concurrency plus
            orphan backlog. A structured ``sweep_cap_reached`` warning signals
            saturation, including when all inspected rows are fresh. Increase
            the cap to reach later running IDs; repeat to drain waiting work.
            No continuation cursor is returned.

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

    checks = [("batch_size", batch_size, 1000)]
    if max_runs is not None:
        checks.append(("max_runs", max_runs, 10000 if abandoned_waiting is not None else 100000))
    for name, value, ceiling in checks:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}")
    store: StateStore
    engine = None

    if database_url is not None:
        store, engine = await _create_temp_store(database_url)
    else:
        assert state_store is not None
        store = state_store

    try:
        # Preserve legacy custom-store signatures at the default settings.
        limits = {}
        if batch_size != 100:
            limits["batch_size"] = batch_size
        if max_runs is not None:
            limits["max_runs"] = max_runs
        stale = (
            await store.sweep_stale_runs(older_than=stale_running, **limits)
            if stale_running is not None
            else []
        )
        abandoned = (
            await store.sweep_abandoned_runs(older_than=abandoned_waiting, **limits)
            if abandoned_waiting is not None
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
    or other backends, tables must already exist via migrations; a database
    behind the models raises SchemaOutOfDateError.

    The caller is responsible for disposing the engine after use.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db._schema_check import ensure_schema_current
    from dendrux.db._sqlite_sync import create_sqlite_schema
    from dendrux.runtime.state import SQLAlchemyStateStore

    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        database_url,
        echo=False,
        connect_args=connect_args,
    )

    try:
        if database_url.startswith("sqlite"):
            # Auto-create tables for SQLite (zero-config promise)
            from pathlib import Path

            db_path = database_url.split("///", 1)[-1] if "///" in database_url else None
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            await create_sqlite_schema(engine)
        else:
            await ensure_schema_current(engine)
    except BaseException:
        await engine.dispose()
        raise

    store = SQLAlchemyStateStore(engine)
    return store, engine
