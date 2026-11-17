"""Apply bounded sweep-category projection through the final row CAS."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, select, true

from dendrux import _codes
from dendrux._lifecycle import CancelReason, FailureReason, StoredEventType
from dendrux.runtime import state as _state
from dendrux.runtime.durability import _live_updates_page
from dendrux.runtime.state import MANAGED_BINDING_KEY, SweptRun, _to_aware_utc
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.types import MANAGED_WAIT_REASON, MANAGED_WAIT_SCHEMA_VERSION, RunStatus

if TYPE_CHECKING:
    from datetime import timedelta

    from dendrux.runtime.stores.sql._sql_host import _SweepStore


class _SweepScan:
    """One sweep category: projected candidate pages and each row's final CAS.

    Waiting pages also read a parent's managed-wait marks from ``pause_data``
    (schema version, pause reason, bound child id; no history, no JSON index). A
    v3 managed wait is never abandoned (0.7): it waits on a subagent, or holds a
    finished subagent's adoption. The indexed window is bounded before managed
    classification, so exempt and corrupt rows count toward the work cap. The
    final CAS rechecks the exemption. Partial marks are reported, never swept.
    """

    def __init__(self, store: _SweepStore, *, waiting: bool, older_than: timedelta) -> None:
        from sqlalchemy import func, literal_column

        from dendrux.db.models import AgentRun, RunEvent

        self.store, self.waiting = store, waiting
        self.now = _dt.datetime.now(_dt.UTC)
        self.cutoff = self.now - older_than
        self.age = (
            AgentRun.status_changed_at
            if waiting
            else func.coalesce(AgentRun.last_progress_at, AgentRun.created_at)
        )
        # A scalar LIMIT subquery prevents a hashed scan of every started event.
        self.started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column(f"'{StoredEventType.RUN_STARTED.value}'"),
            )
            .limit(1)
            .correlate(AgentRun)
            .scalar_subquery()
        )

    def _wait_marks(self) -> tuple[Any, Any, Any]:
        from dendrux.db.models import AgentRun

        col = AgentRun.pause_data
        return (
            self.store.json_get(col, ("schema_version",), "int"),
            self.store.json_get(col, ("pause_reason",), "str"),
            self.store.json_get(col, (MANAGED_BINDING_KEY, "child_run_id"), "str"),
        )

    def _not_managed_wait(self) -> Any:
        from sqlalchemy import or_

        version, reason, _child = self._wait_marks()
        return and_(
            or_(version.is_(None), version != MANAGED_WAIT_SCHEMA_VERSION),
            or_(reason.is_(None), reason != MANAGED_WAIT_REASON),
        )

    def page(self, status_filter: str, after: Any, limit: int) -> Any:
        from sqlalchemy import tuple_

        from dendrux.db.models import AgentRun

        marks = (
            [m.label(n) for m, n in zip(self._wait_marks(), _WAIT_MARK_LABELS, strict=True)]
            if self.waiting
            else []
        )
        statement = select(
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.last_progress_at,
            AgentRun.iteration_count,
            AgentRun.claim_epoch,
            self.started.label("started"),
            self.age.label("sweep_age"),
            *marks,
        ).limit(limit)
        if self.waiting:
            statement = statement.where(
                AgentRun.status == status_filter, self.age < self.cutoff
            ).order_by(self.age, AgentRun.id)
            return self.store._waiting_sweep_page(statement, after, limit)
        # Equal range bounds preserve the (status, id) ordering in the planner.
        # Equality alone lets it prefer the primary key and filter through
        # terminal history on every page.
        statement = statement.where(AgentRun.status.between(status_filter, status_filter)).order_by(
            AgentRun.status, AgentRun.id
        )
        if after is not None:
            statement = statement.where(
                tuple_(AgentRun.status, AgentRun.id) > (status_filter, after)
            )
        return statement

    def skipped(self, row: Any) -> bool:
        """A fresh running row (counted, not swept) or any managed wait."""
        if not self.waiting:
            # Running rows are bounded by live concurrency. Count every inspected
            # row against the work cap, including fresh ones.
            return bool(_to_aware_utc(row.sweep_age) >= self.cutoff)
        version = row.wait_version == MANAGED_WAIT_SCHEMA_VERSION
        reason = row.wait_reason == MANAGED_WAIT_REASON
        if not (version or reason):
            return False
        if not (version and reason and row.wait_child):
            _codes.log_code(
                logger, _codes.MANAGED_STATE_INVALID, run_id=row.id, managed_reason="wait_invalid"
            )
        return True

    async def collect(
        self, status: str, after: Any, batch_size: int, budget: int
    ) -> tuple[list[SweptRun], int, Any]:
        swept: list[SweptRun] = []
        visited = 0
        while visited < budget:
            limit = min(batch_size, budget - visited)
            statement = self.page(status, after, limit)
            async with self.store._session_factory() as session:
                rows = (await session.execute(statement)).all()
            if not rows:
                return swept, visited, None
            visited += len(rows)
            after = (rows[-1].sweep_age, rows[-1].id) if self.waiting else rows[-1].id
            async with _live_updates_page(self.store):
                for row in rows:
                    done = None if self.skipped(row) else await self.sweep_row(row)
                    if done is not None:
                        swept.append(done)
            if len(rows) < limit:
                return swept, visited, None
        return swept, visited, after

    def _reason(self, row: Any) -> str:
        if self.waiting:
            return FailureReason.ABANDONED_WAITING.value
        if row.started:
            return FailureReason.STALE_RUNNING.value
        return FailureReason.NEVER_STARTED.value

    async def sweep_row(self, row: Any) -> SweptRun | None:
        """The row's final CAS and its lifecycle event; None when another writer won."""
        from sqlalchemy import case

        from dendrux.db.models import AgentRun

        reason = self._reason(row)
        flagged = AgentRun.cancel_requested.is_(True)
        waiting = self.waiting
        new_status = (
            RunStatus.ERROR.value
            if waiting
            else case((flagged, RunStatus.CANCELLED.value), else_=RunStatus.ERROR.value)
        )
        async with self.store._session_factory() as session:
            result = await self.store._update_one_rows(
                session,
                table=AgentRun.__table__,
                pk=(row.id,),
                predicate=and_(
                    AgentRun.id == row.id,
                    AgentRun.status == row.status,
                    AgentRun.claim_epoch == row.claim_epoch,
                    self.age < self.cutoff,
                    and_(AgentRun.updated_at < self.cutoff, self._not_managed_wait())
                    if waiting
                    else true(),
                ),
                values=dict(
                    **_sql_support._status_values(new_status),
                    claim_epoch=AgentRun.claim_epoch + 1,
                    failure_reason=reason,
                    pause_data=None,
                    error=f"Run swept as {reason}"
                    if waiting
                    else case((flagged, None), else_=f"Run swept as {reason}"),
                    cancel_requested=False,
                    pause_requested=False,
                ),
                returning=[AgentRun.status],
            )
            status = result.scalar_one_or_none()
            if status is None:
                return None
            event_type = (
                StoredEventType.RUN_ABANDONED.value
                if waiting
                else StoredEventType.RUN_INTERRUPTED.value
            )
            data = {"failure_reason": reason}
            if status == RunStatus.CANCELLED.value:
                event_type = StoredEventType.RUN_CANCELLED.value
                data["reason"] = CancelReason.CANCEL_REQUESTED.value
            await _state._insert_event_locked(
                session, row.id, event_type, data, iteration_index=row.iteration_count or 0
            )
            self.store._mark_announce(session, (row.id,))
            await self.store._write_tx(session)
        return SweptRun(
            run_id=row.id,
            agent_name=row.agent_name,
            previous_status=row.status,
            failure_reason=reason,
            last_progress_at=_to_aware_utc(row.last_progress_at),
            swept_at=_to_aware_utc(self.now),
            status=status,
        )


_WAIT_MARK_LABELS = ("wait_version", "wait_reason", "wait_child")


_SweepScan.__module__ = "dendrux.runtime.stores.sql.base"
