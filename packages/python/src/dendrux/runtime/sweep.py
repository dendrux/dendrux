"""Run sweep — public maintenance API.

Detects stale RUNNING and abandoned WAITING runs. Stale runs can be
marked for explicit recovery or finalised with structured failure reasons.

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

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from dendrux import _codes
from dendrux.runtime._bounds import (
    SWEEP_BATCH_DEFAULT,
    SWEEP_BATCH_MAX,
    SWEEP_MAX_RECLAIMS_DEFAULT,
    SWEEP_MAX_RECLAIMS_MAX,
    SWEEP_MAX_RETURNED_DEFAULT,
    SWEEP_MAX_RETURNED_MAX,
    SWEEP_MAX_RUNS_DEFAULT,
    SWEEP_MAX_RUNS_MAX,
)
from dendrux.runtime.durability import _live_updates_boundary
from dendrux.runtime.state import SweepResult, _validate_sweep_age

if TYPE_CHECKING:
    from datetime import timedelta

    from dendrux.runtime.state import StateStore, SweptRun


_CURSOR_PREFIX = "sweep:v1:"


@dataclass
class _SweepCursor:
    running: str | None = None
    waiting: tuple[str, datetime, str] | None = None
    running_done: bool = False
    waiting_done: bool = False

    @classmethod
    def decode(cls, cursor: str | None) -> _SweepCursor:
        if cursor is None or not cursor.startswith(_CURSOR_PREFIX):
            return cls(running=cursor)
        try:
            running, waiting, running_done, waiting_done = json.loads(cursor[len(_CURSOR_PREFIX) :])
            if running is not None and not isinstance(running, str):
                raise ValueError
            if type(running_done) is not bool or type(waiting_done) is not bool:
                raise ValueError
            return cls(running, _waiting_position(waiting), running_done, waiting_done)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Invalid sweep cursor") from exc

    def encode(self, *, with_waiting: bool) -> str | None:
        if self.running_done and self.waiting_done:
            return None
        if not with_waiting:
            return self.running
        waiting = None
        if self.waiting is not None:
            status, age, run_id = self.waiting
            waiting = [status, age.isoformat(), run_id]
        return _CURSOR_PREFIX + json.dumps(
            [self.running, waiting, self.running_done, self.waiting_done], separators=(",", ":")
        )


def _waiting_position(value: Any) -> tuple[str, datetime, str] | None:
    from dendrux._lifecycle import PAUSE_STATUS_VALUES

    if value is None:
        return None
    status, age, run_id = value
    if status not in PAUSE_STATUS_VALUES or not isinstance(run_id, str):
        raise ValueError("Invalid sweep cursor")
    # Keep the database's timestamp precision and timezone representation.
    return status, datetime.fromisoformat(age), run_id


async def _sweep_waiting(
    store: StateStore, threshold: timedelta | None, limits: dict[str, int], progress: _SweepCursor
) -> list[SweptRun]:
    if threshold is None:
        progress.waiting_done = True
        return []
    scan = getattr(store, "_scan_abandoned_runs", None)
    if getattr(store, "_uses_waiting_sweep_cursor", False) is True and callable(scan):
        if progress.waiting_done:
            return []
        result, progress.waiting = await scan(
            threshold,
            batch_size=limits.get("batch_size", SWEEP_BATCH_DEFAULT),
            max_runs=limits.get("max_runs", SWEEP_MAX_RUNS_DEFAULT),
            cursor=progress.waiting,
        )
        progress.waiting_done = progress.waiting is None
        return cast("list[SweptRun]", result)
    progress.waiting_done = True
    return await store.sweep_abandoned_runs(older_than=threshold, **limits)


async def sweep(
    *,
    database_url: str | None = None,
    state_store: StateStore | None = None,
    stale_running: timedelta | None = None,
    abandoned_waiting: timedelta | None = None,
    batch_size: int = SWEEP_BATCH_DEFAULT,
    max_runs: int = SWEEP_MAX_RUNS_DEFAULT,
    database_options: dict[str, Any] | None = None,
    on_stale: Literal["error", "reclaim"] | None = None,
    max_reclaims: int = SWEEP_MAX_RECLAIMS_DEFAULT,
    max_returned: int = SWEEP_MAX_RETURNED_DEFAULT,
    cursor: str | None = None,
) -> SweepResult:
    """Sweep stale and abandoned runs. Developer calls this at app startup.

    Detects stale RUNNING rows and abandoned WAITING rows, emits lifecycle
    events, and reports finalisations and runs marked for explicit recovery.

    Args:
        database_url: Database URL. Creates a temporary engine, disposed
            after sweep. Mutually exclusive with ``state_store``.
        state_store: Pre-existing StateStore instance. Mutually exclusive
            with ``database_url``.
        stale_running: Threshold for stale RUNNING detection. Runs with
            no forward progress beyond this duration are swept. If None,
            no stale-running sweep is performed. It must exceed the longest
            gap between progress updates (LLM calls, tool starts/ends,
            ``heartbeat()``). Use at least three times the lease interval;
            a shorter threshold can fence a worker that is still executing.
        abandoned_waiting: Threshold for abandoned WAITING detection.
            WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT, WAITING_APPROVAL and
            PAUSED runs are swept only when both status_changed_at and updated_at
            are older than this duration. The updated_at check protects pauses
            written by older workers; recent metadata edits delay abandonment.
            If None, no abandoned-waiting sweep is performed.

        batch_size: Candidate rows per projected page (default 100, maximum 1000).
        max_runs: Candidates inspected per category (default 10,000, maximum
            100,000), using the same integer bound for running and waiting.
            All waiting statuses share the waiting category's cap. Fresh running
            rows, exempt or corrupt managed waits, and losing CAS attempts count
            toward it. A structured
            ``sweep_cap_reached`` warning signals saturation. Increase the cap
            or continue with the returned cursor to reach later running IDs and
            waiting index windows, including past exempt or corrupt waits. Running
            population is bounded operationally by live concurrency plus orphan backlog.
        on_stale: ``"error"`` finalises stale runs; ``"reclaim"`` marks recoverable
            ReAct runs for explicit ``Agent.reclaim()``. Omission warns when stale
            rows are found and retains ``"error"`` behavior through 0.6.
        max_reclaims: Attempt limit stored by each new mark (default 3).
            Rediscovering an existing mark preserves its stored limit.
        max_returned: Maximum reclaimable runs returned (default 1,000).
        cursor: Continue after the previous page's opaque cursor. A call never
            wraps; None in the result means both scans reached their end. Continue
            with the same thresholds; omit the cursor to start a new scan. Native
            SQL stores continue waiting scans; custom stores retain their own
            sweep behavior.

    Returns:
        SweepResult with lists of swept runs by category.

    Raises:
        ValueError: If both or neither of database_url/state_store provided.
    """
    _validate_source(database_url, state_store, database_options)
    _validate_thresholds(stale_running, abandoned_waiting)
    if stale_running is None and abandoned_waiting is None and on_stale != "reclaim":
        return SweepResult(stale_running=[])

    if on_stale not in (None, "error", "reclaim"):
        raise ValueError("on_stale must be error or reclaim")
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("cursor must be a string or None")
    _validate_scan_limits(batch_size, max_runs, max_returned, max_reclaims)
    progress = _SweepCursor.decode(cursor)
    store: StateStore
    engine = None

    if database_url is not None:
        store, engine = await _create_temp_store(
            database_url,
            **({"database_options": database_options} if database_options is not None else {}),
        )
    else:
        assert state_store is not None
        store = state_store

    async with _live_updates_boundary(store, sweep=True):
        try:
            # Preserve legacy custom-store signatures at the default settings.
            limits = {}
            if batch_size != SWEEP_BATCH_DEFAULT:
                limits["batch_size"] = batch_size
            if max_runs != SWEEP_MAX_RUNS_DEFAULT:
                limits["max_runs"] = max_runs
            from dendrux.runtime._capabilities import capabilities, validate_capabilities

            if on_stale == "reclaim":
                validate_capabilities(store)
            if on_stale == "reclaim" and not capabilities(store).recovery:
                _codes.log_code(logging.getLogger(__name__), _codes.STORE_CAPABILITY, run_id=None)
                raise NotImplementedError(
                    f"StateStore {type(store).__name__} does not declare the 'recovery' capability"
                )
            reclaimable = []
            next_cursor = None
            stale = []
            if stale_running is not None and not progress.running_done:
                combined_scan = getattr(store, "_scan_expired_runs", None)
                if getattr(store, "_uses_combined_sweep", False) is True and callable(
                    combined_scan
                ):
                    page = await combined_scan(
                        cutoff=datetime.now(UTC) - stale_running,
                        batch_size=batch_size,
                        max_runs=max_runs,
                        max_returned=max_returned,
                        reclaim_limit=max_reclaims,
                        cursor=progress.running,
                        on_stale=on_stale or "error",
                    )
                    stale, reclaimable, next_cursor = page.finalised, page.reclaimable, page.cursor
                elif on_stale == "reclaim":
                    page = await store.mark_lease_expired(
                        cutoff=datetime.now(UTC) - stale_running,
                        batch_size=batch_size,
                        max_runs=max_runs,
                        max_returned=max_returned,
                        reclaim_limit=max_reclaims,
                        cursor=progress.running,
                        finalise_unrecoverable=True,
                    )
                    stale, reclaimable, next_cursor = page.finalised, page.reclaimable, page.cursor
                else:
                    stale = await store.sweep_stale_runs(older_than=stale_running, **limits)
                if on_stale is None and stale:
                    warnings.warn(
                        'sweep() will default to on_stale="reclaim" in 0.7; '
                        'pass on_stale="error" to keep finalising stale runs.',
                        FutureWarning,
                        stacklevel=2,
                    )
            progress.running = next_cursor
            progress.running_done = next_cursor is None
            abandoned = await _sweep_waiting(store, abandoned_waiting, limits, progress)
            return SweepResult(
                stale_running=stale,
                abandoned_waiting=abandoned,
                reclaimable=reclaimable,
                cursor=progress.encode(with_waiting=abandoned_waiting is not None),
            )
        finally:
            if engine is not None:
                await engine.dispose()


def _validate_scan_limits(
    batch_size: int, max_runs: int, max_returned: int, max_reclaims: int
) -> None:
    for name, value, ceiling in (
        ("batch_size", batch_size, SWEEP_BATCH_MAX),
        ("max_runs", max_runs, SWEEP_MAX_RUNS_MAX),
        ("max_returned", max_returned, SWEEP_MAX_RETURNED_MAX),
        ("max_reclaims", max_reclaims, SWEEP_MAX_RECLAIMS_MAX),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}")


def _validate_thresholds(*thresholds: timedelta | None) -> None:
    for threshold in thresholds:
        if threshold is not None:
            _validate_sweep_age(threshold)


def _validate_source(
    database_url: str | None,
    state_store: StateStore | None,
    database_options: dict[str, Any] | None,
) -> None:
    """Reject ambiguous persistence configuration before opening an engine."""
    from dendrux.db._engine_options import split_database_options

    split_database_options(database_options, replacement="use sweep(state_store=...)")
    if database_url is not None and state_store is not None:
        raise ValueError(
            "database_url and state_store are mutually exclusive. Pass one or the other."
        )
    if database_url is None and state_store is None:
        raise ValueError("Either database_url or state_store is required.")
    if database_options is not None and database_url is None:
        raise ValueError("database_options requires database_url")


async def _create_temp_store(
    database_url: str,
    *,
    database_options: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Create a temporary StateStore + engine from a database URL.

    For SQLite, auto-creates tables if they don't exist (matching the
    zero-config promise in Agent._create_private_engine). For Postgres
    or other backends, tables must already exist via migrations; a database
    behind the models raises SchemaOutOfDateError.

    The caller is responsible for disposing the engine after use.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db._engine_options import engine_options, normalized_url
    from dendrux.db._schema_check import ensure_schema_current
    from dendrux.db._sqlite_sync import create_sqlite_schema
    from dendrux.runtime.stores._factory import store_for_engine

    parsed = normalized_url(database_url)
    engine = create_async_engine(parsed, **engine_options(parsed, database_options))

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

    store = store_for_engine(engine)
    return store, engine
