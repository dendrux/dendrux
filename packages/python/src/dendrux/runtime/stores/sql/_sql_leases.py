"""Maintain epoch-fenced execution liveness."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import and_, true

from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    CancelReason,
    FailureReason,
    StoredEventType,
)
from dendrux._meta_keys import GOVERNANCE_KEY, STORE_SESSION_KEY
from dendrux.runtime import state as _state
from dendrux.runtime._bounds import (
    MAX_PAGE,
    RECOVERY_SCAN_MAX,
    SWEEP_BATCH_MAX,
    SWEEP_MAX_RECLAIMS_MAX,
)
from dendrux.runtime.durability import (
    _live_updates_page,
    _safe_database_call,
    retry_transient_db,
)
from dendrux.runtime.state import (
    ClaimResult,
    MarkResult,
    ReclaimableRun,
    SweptRun,
    TenantScope,
    _reclaimable,
    _require_aware_utc,
    _to_aware_utc,
    _Unscoped,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from datetime import datetime


class _SqlLeases(_SqlWrites):
    async def mark_lease_expired(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        max_runs: int,
        max_returned: int,
        reclaim_limit: int,
        cursor: str | None,
        finalise_unrecoverable: bool = False,
    ) -> MarkResult:
        """Scan running rows once by id, atomically fencing each expired lease."""
        _sql_support._validate_lease_batch_size(
            batch_size=batch_size,
            cap=SWEEP_BATCH_MAX,
        )
        return cast(
            "MarkResult",
            await self._scan_expired_runs(
                cutoff=cutoff,
                batch_size=batch_size,
                max_runs=max_runs,
                max_returned=max_returned,
                reclaim_limit=reclaim_limit,
                cursor=cursor,
                on_stale="reclaim" if finalise_unrecoverable else None,
            ),
        )

    @_safe_database_call
    async def _scan_expired_runs(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        max_runs: int,
        max_returned: int,
        reclaim_limit: int,
        cursor: str | None,
        on_stale: Literal["error", "reclaim"] | None = None,
    ) -> MarkResult:
        from sqlalchemy import case, func, literal_column, or_, select

        from dendrux.db.models import AgentRun, RunEvent

        _require_aware_utc("cutoff", cutoff)
        for bound in (batch_size, max_runs, max_returned, reclaim_limit):
            _sql_support._positive_page(bound)
        if max(batch_size, max_runs, max_returned) > RECOVERY_SCAN_MAX:
            _codes.log_code(logger, _codes.RECOVERY_SCAN_LIMIT, run_id=None)
            raise ValueError("Recovery scan bounds must not exceed 100000")
        if reclaim_limit > SWEEP_MAX_RECLAIMS_MAX:
            raise ValueError("reclaim_limit exceeds SMALLINT")
        result = MarkResult([], [], [], 0, None)
        inspected, after = 0, cursor
        started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column(f"'{StoredEventType.RUN_STARTED.value}'"),
            )
            .limit(1)
            .correlate(AgentRun)
            .scalar_subquery()
        )
        fields = (
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.tenant_id,
            AgentRun.status,
            AgentRun.last_progress_at,
            AgentRun.created_at,
            AgentRun.storage_format,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.claim_epoch,
            AgentRun.lease_expired_at,
            AgentRun.loop_state["loop"].as_string().label("loop"),
            AgentRun.cancel_requested,
            AgentRun.iteration_count,
            started.label("started"),
        )
        while inspected < max_runs and len(result.reclaimable) < max_returned:
            statement = select(*fields).where(
                AgentRun.status.between(RunStatus.RUNNING.value, RunStatus.RUNNING.value)
            )
            if after is not None:
                statement = statement.where(AgentRun.id > after)
            async with self._session_factory() as session:
                ids = (
                    await session.execute(
                        statement.order_by(AgentRun.status, AgentRun.id).limit(
                            min(batch_size, max_runs - inspected)
                        )
                    )
                ).all()
            if not ids:
                return MarkResult(
                    result.reclaimable, result.finalised, result.legacy, inspected, None
                )
            async with _live_updates_page(self):
                for candidate in ids:
                    run_id = candidate.id

                    async def attempt(
                        row: Any = candidate, run_id: str = run_id
                    ) -> ReclaimableRun | SweptRun | None:
                        async with self._session_factory() as session:
                            age = _to_aware_utc(row.last_progress_at or row.created_at)
                            if (
                                on_stale == "error" or row.lease_expired_at is None
                            ) and age >= cutoff:
                                return None
                            if (
                                row.storage_format == STORAGE_FORMAT_CHECKPOINTED
                                and row.loop
                                not in (
                                    "react",
                                    "single_call",
                                )
                            ):
                                if on_stale != "error":
                                    _codes.log_code(logger, _codes.LOOP_UNKNOWN, run_id=run_id)
                                if on_stale is None:
                                    return None
                            limit = (
                                row.reclaim_limit
                                if row.lease_expired_at is not None
                                and row.reclaim_limit is not None
                                else reclaim_limit
                            )
                            legacy_reason = (
                                FailureReason.STALE_RUNNING.value
                                if row.started or on_stale is None
                                else FailureReason.NEVER_STARTED.value
                            )
                            reason = (
                                legacy_reason
                                if on_stale == "error"
                                else FailureReason.STALE_RUNNING_LEGACY.value
                                if row.storage_format != STORAGE_FORMAT_CHECKPOINTED
                                else FailureReason.RECLAIM_LIMIT.value
                                if row.reclaim_attempts >= limit
                                else legacy_reason
                                if row.loop != "react"
                                else None
                            )
                            if reason is not None or row.cancel_requested:
                                flagged = AgentRun.cancel_requested.is_(True)
                                status_expr = case(
                                    (flagged, RunStatus.CANCELLED.value),
                                    else_=RunStatus.ERROR.value,
                                )
                                changed = (
                                    await self._update_one_rows(
                                        session,
                                        table=AgentRun.__table__,
                                        pk=(run_id,),
                                        predicate=and_(
                                            AgentRun.id == run_id,
                                            AgentRun.status == RunStatus.RUNNING.value,
                                            AgentRun.claim_epoch == row.claim_epoch,
                                            or_(
                                                AgentRun.lease_expired_at.is_not(None)
                                                if on_stale != "error"
                                                else False,
                                                func.coalesce(
                                                    AgentRun.last_progress_at, AgentRun.created_at
                                                )
                                                < cutoff,
                                            ),
                                        ),
                                        values=dict(
                                            **_sql_support._status_values(status_expr),
                                            claim_epoch=AgentRun.claim_epoch + 1,
                                            failure_reason=reason,
                                            error=case(
                                                (flagged, None), else_=f"Run swept as {reason}"
                                            ),
                                            pause_data=None,
                                            pause_requested=False,
                                            cancel_requested=False,
                                        ),
                                        returning=[AgentRun.status],
                                    )
                                ).scalar_one_or_none()
                                if changed is None:
                                    return None
                                status = changed
                                await _state._insert_event_locked(
                                    session,
                                    run_id,
                                    StoredEventType.RUN_CANCELLED.value
                                    if status == RunStatus.CANCELLED.value
                                    else StoredEventType.RUN_INTERRUPTED.value,
                                    {
                                        "failure_reason": reason,
                                        **(
                                            {"reason": CancelReason.CANCEL_REQUESTED.value}
                                            if status == RunStatus.CANCELLED.value
                                            else {}
                                        ),
                                    },
                                    iteration_index=row.iteration_count or 0,
                                )
                                self._mark_announce(session, (run_id,))
                                await self._write_tx(session)
                                _codes.log_code(
                                    logger,
                                    _codes.REGISTRY[
                                        "legacy_run_swept"
                                        if reason == FailureReason.STALE_RUNNING_LEGACY.value
                                        else reason or CancelReason.CANCEL_REQUESTED.value
                                    ],
                                    run_id=run_id,
                                )
                                return SweptRun(
                                    run_id,
                                    row.agent_name,
                                    RunStatus.RUNNING.value,
                                    reason or CancelReason.CANCEL_REQUESTED.value,
                                    _to_aware_utc(row.last_progress_at),
                                    _dt.datetime.now(_dt.UTC),
                                    status,
                                )
                            if row.lease_expired_at is not None:
                                return _reclaimable(row)
                            now = _dt.datetime.now(_dt.UTC)
                            # This session holds scalar projections, not ORM entities.
                            # Avoid identity synchronization of the CAS's RETURNING row.
                            marked = (
                                await self._update_one_rows(
                                    session,
                                    table=AgentRun.__table__,
                                    pk=(run_id,),
                                    predicate=and_(
                                        AgentRun.id == run_id,
                                        AgentRun.status == RunStatus.RUNNING.value,
                                        AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED,
                                        AgentRun.lease_expired_at.is_(None),
                                        AgentRun.claim_epoch == row.claim_epoch,
                                        func.coalesce(
                                            AgentRun.last_progress_at, AgentRun.created_at
                                        )
                                        < cutoff,
                                    ),
                                    values=dict(
                                        claim_epoch=AgentRun.claim_epoch + 1,
                                        lease_expired_at=now,
                                        reclaim_limit=limit,
                                    ),
                                    returning=[*fields],
                                )
                            ).first()
                            if marked is None:
                                current = (
                                    await session.execute(
                                        select(*fields).where(
                                            AgentRun.id == run_id,
                                            AgentRun.status == RunStatus.RUNNING.value,
                                            AgentRun.lease_expired_at.is_not(None),
                                        )
                                    )
                                ).first()
                                return _reclaimable(current) if current is not None else None
                            await _state._insert_event_locked(
                                session,
                                run_id,
                                StoredEventType.RUN_LEASE_EXPIRED.value,
                                {"gap_s": max(0.0, (now - age).total_seconds())},
                            )
                            item = _reclaimable(marked)
                            self._mark_announce(session, (run_id,))
                            await self._write_tx(session)
                            return item

                    item = await retry_transient_db(
                        attempt,
                        label="mark_lease_expired",
                        run_id=run_id,
                        classify=self.classify_db_error,
                        _raw_driver_errors=True,
                    )
                    inspected += 1
                    after = run_id
                    if isinstance(item, ReclaimableRun):
                        result.reclaimable.append(item)
                    elif isinstance(item, SweptRun):
                        result.finalised.append(item)
                        if item.failure_reason == FailureReason.STALE_RUNNING_LEGACY.value:
                            result.legacy.append(item)
                    if inspected >= max_runs or len(result.reclaimable) >= max_returned:
                        if inspected >= max_runs:
                            _codes.log_code(
                                logger,
                                _codes.SWEEP_CAP_REACHED,
                                category=FailureReason.STALE_RUNNING.value,
                                cap=max_runs,
                                inspected=inspected,
                            )
                        return MarkResult(
                            result.reclaimable, result.finalised, result.legacy, inspected, after
                        )
        return MarkResult(result.reclaimable, result.finalised, result.legacy, inspected, after)

    async def get_lease_expired(self, *, limit: int, cursor: str | None) -> list[ReclaimableRun]:
        """Rediscover marks without changing epochs or emitting duplicate events."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        _sql_support._positive_page(limit)
        stmt = select(
            AgentRun.id,
            AgentRun.tenant_id,
            AgentRun.agent_name,
            AgentRun.last_progress_at,
            AgentRun.lease_expired_at,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.claim_epoch,
        ).where(
            AgentRun.status == RunStatus.RUNNING.value,
            AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED,
            AgentRun.lease_expired_at.is_not(None),
        )
        if cursor is not None:
            stmt = stmt.where(AgentRun.id > cursor)
        async with self._session_factory() as session:
            return [
                _reclaimable(row)
                for row in (
                    await session.execute(stmt.order_by(AgentRun.id).limit(min(limit, MAX_PAGE)))
                ).all()
            ]

    async def reclaim_run(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        agent_name: str,
        expected_claim_epoch: int | None,
        governance: dict[str, Any],
        governance_policy: Literal["warn", "raise"],
        skill_pins: dict[str, str],
        loop_type: str,
    ) -> ClaimResult | Literal[False]:
        """Validate and reclaim under the same immutable-PK lock used by the update."""
        from types import SimpleNamespace

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunNotFoundError, RunNotReclaimableError
        from dendrux.runtime.state import _check_scope

        if governance_policy not in ("warn", "raise"):
            _codes.log_code(logger, _codes.GOVERNANCE_MISMATCH, run_id=run_id)
            raise ValueError("governance_policy must be warn or raise")
        _check_scope(tenant_id)
        table = AgentRun.__table__
        scope = (
            true()
            if isinstance(tenant_id, _Unscoped)
            else table.c.tenant_id.is_not_distinct_from(tenant_id)
        )
        projection = [
            table.c[name]
            for name in (
                "id",
                "tenant_id",
                "storage_format",
                "status",
                "lease_expired_at",
                "claim_epoch",
                "reclaim_attempts",
                FailureReason.RECLAIM_LIMIT.value,
                CancelReason.CANCEL_REQUESTED.value,
                "agent_name",
                "meta",
                "loop_state",
                "last_progress_at",
                "created_at",
                "last_reclaimed_at",
            )
        ]

        async def attempt() -> ClaimResult | Literal[False]:
            assignments: list[tuple[Any, Any]] = []
            previous_epoch = 0
            cancelled = False
            governance_changed = False
            gap = 0.0

            def validate(row_mapping: Any) -> None:
                nonlocal previous_epoch, cancelled, governance_changed, gap
                row = SimpleNamespace(**row_mapping)
                _sql_support._refuse_direct_managed_child(run_id, row.loop_state)
                checks = [
                    (row.storage_format == STORAGE_FORMAT_CHECKPOINTED, "legacy_run"),
                    (self.capabilities.recovery, "store_capability"),
                    (row.status == RunStatus.RUNNING.value, "not_running"),
                    (row.lease_expired_at is not None, "lease_fresh"),
                    (
                        expected_claim_epoch is None or row.claim_epoch == expected_claim_epoch,
                        "already_reclaimed",
                    ),
                    (
                        row.reclaim_limit is not None and row.reclaim_attempts < row.reclaim_limit,
                        FailureReason.RECLAIM_LIMIT.value,
                    ),
                ]
                for ok, reason in checks:
                    if not ok:
                        raise RunNotReclaimableError(run_id, reason)
                previous_epoch = row.claim_epoch
                cancelled = bool(row.cancel_requested)
                if cancelled:
                    assignments.extend(
                        self._ordered(
                            table,
                            {
                                **_sql_support._status_values(RunStatus.CANCELLED.value),
                                "claim_epoch": table.c.claim_epoch + 1,
                                "pause_data": None,
                                "cancel_requested": False,
                            },
                        )
                    )
                    return
                if row.agent_name != agent_name:
                    raise RunNotReclaimableError(run_id, "agent_mismatch")
                saved = (row.meta or {}).get(GOVERNANCE_KEY)
                governance_changed = saved is not None and saved.get("digest") != governance.get(
                    "digest"
                )
                if governance_changed:
                    _codes.log_code(logger, _codes.GOVERNANCE_MISMATCH, run_id=run_id)
                    if governance_policy == "raise":
                        raise RunNotReclaimableError(run_id, "governance_changed")
                if (((row.loop_state or {}).get("skills") or {}).get("pins", {})) != skill_pins:
                    raise RunNotReclaimableError(run_id, "skill_mismatch")
                saved_loop = (row.loop_state or {}).get("loop")
                if saved_loop not in ("react", "single_call"):
                    raise RunNotReclaimableError(run_id, "loop_unknown")
                if loop_type != "react" or saved_loop != "react":
                    raise RunNotReclaimableError(run_id, "loop_unsupported")
                now = _dt.datetime.now(_dt.UTC)
                gap = max(
                    0.0,
                    (now - _to_aware_utc(row.last_progress_at or row.created_at)).total_seconds(),
                )
                assignments.extend(
                    self._ordered(
                        table,
                        dict(
                            claim_epoch=table.c.claim_epoch + 1,
                            reclaim_attempts=table.c.reclaim_attempts + 1,
                            lease_expired_at=None,
                            last_progress_at=now,
                            last_reclaimed_at=now,
                        ),
                    )
                )

            async with self._session_factory() as session:
                session.info[STORE_SESSION_KEY] = self
                changed = await self.update_one(
                    session,
                    table=table,
                    pk=(run_id,),
                    predicate=scope,
                    values=assignments,
                    returning=[
                        table.c.claim_epoch,
                        table.c.storage_format,
                        table.c.loop_state,
                        table.c.reclaim_attempts,
                    ],
                    lock_projection=projection,
                    validate_locked=validate,
                )
                if changed is None:
                    raise RunNotFoundError(run_id)
                if cancelled:
                    await _state._insert_event_locked(
                        session,
                        run_id,
                        StoredEventType.RUN_CANCELLED.value,
                        {"reason": CancelReason.CANCEL_REQUESTED.value},
                    )
                    self._mark_announce(session, (run_id,))
                    await self._write_tx(session)
                    raise RunNotReclaimableError(run_id, "cancelled")
                await _state._insert_event_locked(
                    session,
                    run_id,
                    StoredEventType.RUN_RECLAIMED.value,
                    dict(
                        attempt=changed["reclaim_attempts"],
                        previous_epoch=previous_epoch,
                        gap_s=gap,
                        **({"governance_changed": True} if governance_changed else {}),
                    ),
                )
                token = ClaimResult(
                    changed["claim_epoch"],
                    changed["storage_format"],
                    changed["loop_state"],
                    (changed["loop_state"] or {}).get("submission"),
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            attempt,
            label="reclaim_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )
