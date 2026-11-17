"""Maintain the fenced run lifecycle."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, select, true

import dendrux.runtime.state as _state
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux._meta_keys import ERROR_INFO_META_KEY
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    EventWrite,
    RunControl,
    TenantScope,
    TraceWrite,
    _finalized_run_read,
    _FinalizedRunRead,
    _to_aware_utc,
    _Unscoped,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._managed_reads import ManagedReads
from dendrux.runtime.stores.sql._sql_evidence import _SqlEvidence
from dendrux.runtime.stores.sql._types import SetFields
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.types import UsageStats


class _SqlLifecycle(_SqlEvidence, ManagedReads):
    def _frontier_predicate(self, column: Any, run_ids: Sequence[str]) -> Any:
        return self.id_in(column, run_ids)

    def _frontier_scope(self, statement: Any, tenant_id: TenantScope) -> Any:
        from dendrux.db.models import AgentRun
        from dendrux.runtime.state import _check_scope

        if isinstance(tenant_id, _Unscoped):
            return statement
        _check_scope(tenant_id)
        return statement.where(AgentRun.tenant_id.is_not_distinct_from(tenant_id))

    async def mark_cancel_requested(self, run_ids: Sequence[str], *, tenant_id: TenantScope) -> int:
        """Mark a frontier in one statement before scanning its children."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        if not run_ids:
            return 0

        async def _attempt() -> int:
            async with self._session_factory() as session:
                result = await session.execute(
                    self._frontier_scope(
                        update(AgentRun)
                        .where(
                            self._frontier_predicate(AgentRun.id, run_ids),
                            AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                        )
                        .values(cancel_requested=True)
                        .execution_options(synchronize_session=False),
                        tenant_id,
                    )
                )
                self._mark_announce(session, run_ids)
                await self._write_tx(session)
                return int(getattr(result, "rowcount", 0))

        return await retry_transient_db(
            _attempt,
            label="mark_cancel_requested",
            run_id=run_ids[0],
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        expected_current_status: str | None = None,
        pii_mapping: dict[str, str] | None = None,
        claim_epoch: int | None = None,
        error_info: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        """Finalize a run. Returns True if the update was applied.

        Args:
            expected_current_status: If provided, only updates the row if the
                current DB status matches. Returns False if status has changed
                (e.g. already cancelled). This prevents cancel/finalize races.
            pii_mapping: Final guardrail PII mapping. Persists past finalize
                (audit-first — framework never auto-clears).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_sql_support._status_values(status),
                    **_sql_support._final_progress(status, answer, claim_epoch),
                }
                values.update(
                    await _sql_support._final_output_values(
                        session,
                        run_id,
                        answer,
                        error,
                        iteration_count,
                        total_usage,
                        pii_mapping,
                        failure_reason,
                    )
                )
                # Clear pause_data on finalize (D1: execution state cleaned up)
                values["pause_data"] = None
                # Clear cooperative cancel flag — terminal state is final;
                # a stale True here would confuse later audits.
                values["cancel_requested"] = False
                values["pause_requested"] = False

                # Conditional finalize: only update if status is still 'running'
                # (or expected_status if provided). Prevents cancel/finalize races.
                if expected_current_status is not None:
                    stmt = (
                        update(AgentRun)
                        .where(AgentRun.id == run_id, AgentRun.status == expected_current_status)
                        .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                    )
                else:
                    stmt = (
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                    )
                if claim_epoch is not None:
                    stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
                self._remember_run_values(session, run_id, values)
                result = await session.execute(stmt)
                if result.rowcount and error_info is not None:
                    from sqlalchemy import select

                    old_meta = (
                        await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                    ).scalar_one_or_none()
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**(old_meta or {}), ERROR_INFO_META_KEY: error_info})
                    )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return (
                    bool(result.rowcount and result.rowcount > 0)
                    if expected_current_status or claim_epoch is not None
                    else True
                )

        return await retry_transient_db(
            _attempt,
            label="finalize_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def finalize_run_if_status_in(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Atomic CAS finalize over multiple acceptable current statuses.

        One SQL ``UPDATE ... WHERE status IN (...)`` — collapses what
        used to be N sequential CAS attempts into a single round-trip.
        Mirrors :meth:`finalize_run`'s side-effects: clears ``pause_data``
        and ``cancel_requested``.
        """

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return won

        return await retry_transient_db(
            _attempt,
            label="finalize_run_if_status_in",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def finalize_with_event(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        event_type: str,
        event_data: dict[str, Any] | None = None,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
        trace: TraceWrite | None = None,
        event: EventWrite | None = None,
    ) -> int | None:
        """:meth:`finalize_run_if_status_in` plus its lifecycle event, in one transaction.

        The event's sequence index is allocated after the CAS took the row
        lock, so nothing can observe the new status before the event is
        durable and no two transition events share an index. Returns the
        event's sequence index, or None when the CAS lost (nothing written).
        """

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                if not won:
                    await session.rollback()
                    return None
                await self._finalize_error_info(session, run_id, event_data)
                if trace is not None:
                    await self._insert_trace(session, run_id, trace)
                seq = (
                    await _state._write_event(session, run_id, event)
                    if event is not None
                    else await _state._insert_event_locked(session, run_id, event_type, event_data)
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return seq

        return await retry_transient_db(
            _attempt,
            label="finalize_with_event",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _finalize_error_info(
        self, session: AsyncSession, run_id: str, event_data: dict[str, Any] | None
    ) -> None:
        info = (event_data or {}).get("error_info")
        if info is None:
            return
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        old_meta = await session.scalar(select(AgentRun.meta).where(AgentRun.id == run_id))
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(meta={**(old_meta or {}), ERROR_INFO_META_KEY: info})
        )

    async def _cas_finalize(
        self,
        session: AsyncSession,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None,
        error: str | None,
        iteration_count: int | None,
        total_usage: UsageStats | None,
        pii_mapping: dict[str, str] | None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        from sqlalchemy import case, literal, update

        from dendrux.db.models import AgentRun

        # updated_at is filled by the model's onupdate hook (Python aware-UTC).
        values: dict[str, Any] = {
            **_sql_support._status_values(status),
            **_sql_support._final_progress(status, answer, claim_epoch),
            "pause_data": None,
            "cancel_requested": False,
            "pause_requested": False,
        }
        finalized_state = self.json_patch_expr(
            AgentRun.loop_state, SetFields({"phase": "finalizing"})
        )
        if finalized_state is None:
            if not self.supports_row_locks:
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .ordered_values(*self._reservation_values(AgentRun.__table__))
                )
            current_state = await session.scalar(
                self.lock_clause(select(AgentRun.loop_state).where(AgentRun.id == run_id), "update")
            )
            finalized_state = literal(
                {**(current_state or {}), "phase": "finalizing"}, type_=AgentRun.loop_state.type
            )
        values["loop_state"] = case(
            (AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED, finalized_state),
            else_=AgentRun.loop_state,
        )
        values.update(
            await _sql_support._final_output_values(
                session,
                run_id,
                answer,
                error,
                iteration_count,
                total_usage,
                pii_mapping,
                failure_reason,
            )
        )

        self._remember_run_values(session, run_id, values)
        stmt = update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status.in_(allowed_current_statuses)
        )
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        capture = _finalized_run_read.get()
        if capture is not None and capture[0] is self and capture[1] == run_id:
            row = (
                await self._update_one_rows(
                    session,
                    table=AgentRun.__table__,
                    pk=(run_id,),
                    predicate=and_(
                        AgentRun.status.in_(allowed_current_statuses),
                        AgentRun.claim_epoch == claim_epoch if claim_epoch is not None else true(),
                    ),
                    values=values,
                    returning=[
                        AgentRun.id,
                        AgentRun.tenant_id,
                        AgentRun.reclaim_attempts,
                        AgentRun.last_reclaimed_at,
                    ],
                )
            ).first()
            if row is not None:
                capture[2][:] = [
                    _FinalizedRunRead(
                        row.id,
                        row.tenant_id,
                        row.reclaim_attempts,
                        _to_aware_utc(row.last_reclaimed_at) if row.last_reclaimed_at else None,
                    )
                ]
            return row is not None
        result: Any = await session.execute(
            stmt.ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
        )
        return bool(result.rowcount and result.rowcount > 0)

    async def request_cancel(self, run_id: str) -> bool:
        """Set ``cancel_requested=True`` on a non-terminal run.

        The terminal-status guard is part of the SQL UPDATE — no race
        window between read and write.
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                    )
                    .values(cancel_requested=True)
                )
                result = await session.execute(stmt)
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(
            _attempt,
            label="request_cancel",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def update_run_meta(
        self, run_id: str, patch: dict[str, Any], *, claim_epoch: int | None = None
    ) -> None:
        """Shallow-merge ``patch`` into ``agent_runs.meta`` for ``run_id``.

        Read-modify-write in one transaction; only Dendrux-owned keys
        (``dendrux.*``) are written through here. No-op for missing runs.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                if claim_epoch is not None:
                    await self._fence(session, run_id, claim_epoch)
                elif not self.supports_row_locks:
                    await self.reserve_parent_writer(session, run_id)
                row = (
                    await session.execute(
                        self.lock_clause(select(AgentRun).where(AgentRun.id == run_id), "update")
                    )
                ).scalar_one_or_none()
                if row is None:
                    return
                row.meta = _sql_support._merge_public_meta(row.meta, patch)
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)

        await retry_transient_db(
            _attempt,
            label="update_run_meta",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def is_cancel_requested(self, run_id: str) -> bool:
        """Read ``cancel_requested`` flag. False for missing runs."""
        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                stmt = select(AgentRun.cancel_requested).where(AgentRun.id == run_id)
                return bool(await session.scalar(stmt))

        return await retry_transient_db(
            _attempt,
            label="is_cancel_requested",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def get_run_control(self, run_id: str) -> RunControl:
        """One read of cancel / pause flags plus an EXISTS over undelivered notes."""
        return await self._read_control(run_id, self._control_query(), RunControl())

    async def touch_progress(self, run_id: str, *, claim_epoch: int | None = None) -> bool:
        """Renew only a running run with the supplied epoch, in one statement."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                stmt = update(AgentRun).where(
                    AgentRun.id == run_id, AgentRun.status == RunStatus.RUNNING.value
                )
                if claim_epoch is not None:
                    stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
                stmt = self._dispatch_filter(stmt, run_id)
                result = await session.execute(
                    stmt.values(last_progress_at=_dt.datetime.now(_dt.UTC))
                )
                if not result.rowcount:
                    await self._check_dispatch_block(session, run_id, claim_epoch)
                await self._write_tx(session)
                return bool(result.rowcount)

        return await retry_transient_db(
            _attempt,
            label="touch_progress",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )
