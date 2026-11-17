"""Maintain durable pause generations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUS_VALUES, GovernanceEventType, StoredEventType
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    PAUSE_REQUEST_META_KEY,
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    PauseSummary,
    TenantScope,
    _apply_tenant_scope,
    _approval_event_object,
    _run_in_scope,
    _usage_totals,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import PAUSE_INVALIDATION_EVENT_TYPES, logger
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from dendrux.types import UsageStats

from dendrux.runtime import state as _state


class _SqlPause(_SqlWrites):
    async def request_pause(self, run_id: str, *, reason: str | None = None) -> bool:
        """Set ``pause_requested`` on a RUNNING row; status guard in the UPDATE."""
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == RunStatus.RUNNING.value)
                    .values(pause_requested=True)
                )
                if not result.rowcount:
                    await session.rollback()
                    return False
                # The UPDATE holds the row lock; record the reason (or clear a
                # stale one) under it.
                meta = (
                    await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                ).scalar_one_or_none() or {}
                if reason is not None or PAUSE_REQUEST_META_KEY in meta:
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**meta, PAUSE_REQUEST_META_KEY: {"reason": reason}})
                    )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return True

        return await retry_transient_db(
            _attempt,
            label="request_pause",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def try_pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """CAS pause: RUNNING and no pending cancel, else False (caller honors winner)."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_sql_support._status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                self._remember_run_values(session, run_id, values)
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(
            _attempt,
            label="try_pause_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def pause_with_event(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        event_data: dict[str, Any] | None = None,
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """:meth:`try_pause_run` plus the ``run.paused`` event, in one transaction.

        A finalizer of paused runs (``cancel_run``, the sweeper) can only act
        once this commits, so it always sees ``run.paused`` and allocates a
        later sequence index. Returns the event's sequence index, or None
        when the CAS lost (a pending cancel or a non-RUNNING row).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_sql_support._status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                self._remember_run_values(session, run_id, values)
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                if not result.rowcount:
                    await session.rollback()
                    return None
                seq = await _state._insert_event_locked(
                    session,
                    run_id,
                    StoredEventType.RUN_PAUSED.value,
                    event_data,
                    iteration_index=iteration_count or 0,
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return seq

        return await retry_transient_db(
            _attempt,
            label="pause_with_event",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        """Persist pause state and set the paused status.

        ``total_usage`` writes the run's running totals (``total_*``).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_sql_support._status_values(status),
                    "pause_data": pause_data,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                if claim_epoch is not None:
                    stmt = stmt.where(
                        AgentRun.claim_epoch == claim_epoch,
                        AgentRun.status == RunStatus.RUNNING.value,
                    )
                self._remember_run_values(session, run_id, values)
                await session.execute(stmt)
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)

        await retry_transient_db(
            _attempt,
            label="pause_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve pause_data for a run. Returns None if no run or no pause data."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = select(AgentRun.pause_data).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            # row is None if run doesn't exist OR if pause_data column is NULL
            if row is None:
                return None
            return dict(row)  # type narrowing for mypy

    async def get_active_pause_sequence(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> int | None:
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun, RunEvent

        seq = (
            select(func.max(RunEvent.sequence_index))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type == StoredEventType.RUN_PAUSED.value,
            )
            .scalar_subquery()
        )
        aborted = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type == StoredEventType.RUN_RESUME_ABORTED.value,
            )
            .scalar_subquery()
        )
        invalidated = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(PAUSE_INVALIDATION_EVENT_TYPES),
                (RunEvent.event_type != StoredEventType.RUN_RESUMED.value)
                | (RunEvent.sequence_index > aborted),
            )
            .scalar_subquery()
        )
        async with self._session_factory() as session:
            value = (
                await session.execute(
                    _apply_tenant_scope(
                        select(seq).where(
                            AgentRun.id == run_id,
                            seq > invalidated,
                            AgentRun.status.in_(PAUSE_STATUS_VALUES),
                        ),
                        tenant_id,
                    )
                )
            ).scalar_one_or_none()
            return None if value is None else int(value)

    async def get_approval_params(
        self, run_id: str, pause_sequence: int, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, dict[str, Any]]:
        """Latest recorded model args per call in this pause's event interval.

        Read at most 1000 approval requests. Missing/overflowed calls fail closed
        in the display layer when PII mapping exists.
        """
        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        boundary = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(
                    [StoredEventType.RUN_PAUSED.value, StoredEventType.RUN_RESUMED.value]
                ),
                RunEvent.sequence_index < pause_sequence,
            )
            .scalar_subquery()
        )
        stmt = (
            select(RunEvent.data)
            .where(
                RunEvent.agent_run_id == run_id,
                _run_in_scope(run_id, tenant_id),
                RunEvent.event_type == GovernanceEventType.APPROVAL_REQUESTED.value,
                RunEvent.sequence_index > boundary,
                RunEvent.sequence_index < pause_sequence,
            )
            .order_by(RunEvent.sequence_index.desc())
            .limit(MAX_PAGE)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        result: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for data in rows:
            data = _approval_event_object(data)
            call_id = data.get("call_id")
            if isinstance(call_id, str) and call_id not in seen:
                seen.add(call_id)
                if isinstance(data.get("params"), dict):
                    result[call_id] = data["params"]
        return result

    async def get_pause_summary(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> PauseSummary | None:
        """Read bounded pause fields, including legacy plain-JSON documents."""
        return await self._read_pause_summary(run_id, tenant_id, pending=False)

    async def get_pending_pause_summary(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> PauseSummary | None:
        """Read pending fields using the adapter's bounded query plan."""
        return await self._read_pause_summary(run_id, tenant_id, pending=True)

    def pause_summary_query(self, run_id: str, tenant_scope: TenantScope, *, pending: bool) -> Any:
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun, RunEvent

        paused_seq = (
            select(func.max(RunEvent.sequence_index))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type == StoredEventType.RUN_PAUSED.value,
            )
            .scalar_subquery()
        )
        parts = self.pause_projection(AgentRun.pause_data)
        return _apply_tenant_scope(
            select(AgentRun.status, paused_seq, AgentRun.pii_mapping, *parts).where(
                AgentRun.id == run_id
            ),
            tenant_scope,
        )

    async def _read_pause_summary(
        self, run_id: str, tenant_id: TenantScope, *, pending: bool
    ) -> PauseSummary | None:
        from sqlalchemy.exc import DBAPIError

        stmt = self.pause_summary_query(run_id, tenant_id, pending=pending)
        try:
            async with self._session_factory() as session:
                row = (await session.execute(stmt)).first()
        except DBAPIError as exc:
            if not self.json_conversion_error(exc):
                raise
            from dendrux.runtime._pending import _whole_pause_summary

            _codes.log_code(logger, _codes.PAUSE_DATA_NOT_JSONB, run_id=run_id)
            return await _whole_pause_summary(self, run_id, tenant_id)
        if row is None:
            return None
        status, seq, mapping, *values = row
        mapping = dict(mapping) if isinstance(mapping, dict) else None
        decoded = list(values)
        pause = {
            k: v for k, v in zip(PAUSE_SUMMARY_KEYS, decoded[:-1], strict=True) if v is not None
        }
        last = decoded[-1]
        last_step = last if isinstance(last, dict) else None
        return PauseSummary(str(status), pause, last_step, seq, mapping)
