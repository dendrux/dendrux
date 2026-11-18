"""Sweep stale running or abandoned waiting rows through bounded scan categories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUS_VALUES, FailureReason
from dendrux.runtime._bounds import (
    SWEEP_BATCH_DEFAULT,
    SWEEP_BATCH_MAX,
    SWEEP_MAX_RUNS_DEFAULT,
    SWEEP_MAX_RUNS_MAX,
)
from dendrux.runtime.durability import _safe_database_call
from dendrux.runtime.state import SweptRun, _validate_sweep_age
from dendrux.runtime.stores.sql import _sql_sweep_scan
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from datetime import datetime, timedelta


class _SqlSweep(_SqlWrites):
    def _waiting_sweep_page(
        self, statement: Any, after: tuple[datetime, str] | None, limit: int
    ) -> Any:
        from sqlalchemy import tuple_

        from dendrux.db.models import AgentRun

        if after is not None:
            statement = statement.where(tuple_(AgentRun.status_changed_at, AgentRun.id) > after)
        return statement

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = SWEEP_BATCH_DEFAULT,
        max_runs: int = SWEEP_MAX_RUNS_DEFAULT,
    ) -> list[SweptRun]:
        """Inspect running candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000. Fresh rows and losing CAS attempts count
        toward it; hitting the cap emits sweep_cap_reached.
        """
        return cast(
            "list[SweptRun]",
            (
                await self._sweep_runs(
                    older_than, waiting=False, batch_size=batch_size, max_runs=max_runs
                )
            )[0],
        )

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = SWEEP_BATCH_DEFAULT,
        max_runs: int = SWEEP_MAX_RUNS_DEFAULT,
    ) -> list[SweptRun]:
        """Inspect waiting candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000 across all waiting statuses. The final CAS
        requires both status_changed_at and updated_at to be stale. Losing CAS
        attempts count toward the cap; hitting it emits sweep_cap_reached.
        """
        return cast(
            "list[SweptRun]",
            (
                await self._sweep_runs(
                    older_than, waiting=True, batch_size=batch_size, max_runs=max_runs
                )
            )[0],
        )

    async def _scan_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int,
        max_runs: int,
        cursor: tuple[str, datetime, str] | None,
    ) -> tuple[list[SweptRun], tuple[str, datetime, str] | None]:
        return cast(
            "tuple[list[SweptRun], tuple[str, datetime, str] | None]",
            await self._sweep_runs(
                older_than, waiting=True, batch_size=batch_size, max_runs=max_runs, cursor=cursor
            ),
        )

    @_safe_database_call
    async def _sweep_runs(
        self,
        older_than: timedelta,
        *,
        waiting: bool,
        batch_size: int,
        max_runs: int,
        cursor: tuple[str, datetime, str] | None = None,
    ) -> tuple[list[SweptRun], tuple[str, datetime, str] | None]:
        # The database-call decorator retains the core's tuple result.
        return cast(
            "tuple[list[SweptRun], tuple[str, datetime, str] | None]",
            await _collect_runs(
                self,
                older_than,
                waiting=waiting,
                batch_size=batch_size,
                max_runs=max_runs,
                cursor=cursor,
            ),
        )


@_safe_database_call
async def _collect_runs(
    store: _SqlSweep,
    older_than: timedelta | None,
    *,
    waiting: bool,
    batch_size: int,
    max_runs: int,
    cursor: tuple[str, datetime, str] | None = None,
    adoptable_after: timedelta | None = None,
    adoptable: list[str] | None = None,
) -> tuple[list[SweptRun], tuple[str, datetime, str] | None]:
    for name, value, ceiling in (
        ("batch_size", batch_size, SWEEP_BATCH_MAX),
        ("max_runs", max_runs, SWEEP_MAX_RUNS_MAX),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}")
    ages = [age for age in (older_than, adoptable_after) if age is not None]
    for age in ages:
        _validate_sweep_age(age)
    if not ages:
        raise ValueError("sweep age must be a non-negative timedelta")
    scan = _sql_sweep_scan._SweepScan(store, waiting=waiting, older_than=min(ages))
    if adoptable_after is not None:
        scan.configure_adoption(
            abandoned_after=older_than, adoptable_after=adoptable_after, adoptable=adoptable
        )
    swept: list[SweptRun] = []
    visited = 0
    statuses = list(PAUSE_STATUS_VALUES) if waiting else [RunStatus.RUNNING.value]
    after = None
    if cursor is not None:
        statuses = statuses[statuses.index(cursor[0]) :]
        after = cursor[1:]
    for status_filter in statuses:
        done, count, end = await scan.collect(status_filter, after, batch_size, max_runs - visited)
        swept.extend(done)
        visited += count
        if visited == max_runs:
            _codes.log_code(
                logger,
                _codes.SWEEP_CAP_REACHED,
                category=FailureReason.ABANDONED_WAITING.value
                if waiting
                else FailureReason.STALE_RUNNING.value,
                cap=max_runs,
                inspected=visited,
            )
            return swept, (status_filter, *end) if waiting else None
        after = None
    return swept, None
