"""Apply bounded sweep-category projection through the final row CAS."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import and_, select, true

from dendrux import _codes
from dendrux._lifecycle import TERMINAL_STATUSES, CancelReason, FailureReason, StoredEventType
from dendrux.db.models import AgentRun
from dendrux.errors import ManagedStateError
from dendrux.runtime import state as _state
from dendrux.runtime.durability import _live_updates_page
from dendrux.runtime.state import MANAGED_BINDING_KEY, ManagedHeader, SweptRun, _to_aware_utc
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.types import MANAGED_WAIT_REASON, PauseState, RunStatus

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import timedelta

    from dendrux.runtime._managed_fanout_store import ManagedChildRow, ManagedFanoutStore
    from dendrux.runtime.stores.sql._sql_host import _SweepStore
    from dendrux.types import ManagedWait


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

        from dendrux.db.models import RunEvent

        self.store, self.waiting = store, waiting
        self.now = _dt.datetime.now(_dt.UTC)
        self.cutoff = self.now - older_than
        self.abandoned_after: timedelta | None = older_than
        self.adoptable_cutoff: _dt.datetime | None = None
        self.adoptable: list[str] | None = None
        self._adoptable_roots: set[str] = set()
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

    def configure_adoption(
        self,
        *,
        abandoned_after: timedelta | None,
        adoptable_after: timedelta,
        adoptable: list[str] | None,
    ) -> None:
        self.abandoned_after = abandoned_after
        self.adoptable_cutoff = self.now - adoptable_after
        self.adoptable = adoptable
        self._adoptable_roots = set(adoptable or ())

    def _wait_marks(self) -> tuple[Any, Any, Any, Any]:
        col = AgentRun.pause_data
        return (
            self.store.json_get(col, ("schema_version",), "int"),
            self.store.json_get(col, ("pause_reason",), "str"),
            self.store.json_get(col, (MANAGED_BINDING_KEY, "child_run_id"), "str"),
            self.store.json_get(col, ("managed_fanout", "n"), "int"),
        )

    def _not_managed_wait(self) -> Any:
        from sqlalchemy import or_

        version, reason, _child, _count = self._wait_marks()
        return and_(
            or_(version.is_(None), version.not_in((3, 4))),
            or_(reason.is_(None), reason != MANAGED_WAIT_REASON),
        )

    def page(self, status_filter: str, after: Any, limit: int) -> Any:
        from sqlalchemy import tuple_

        marks = (
            [m.label(n) for m, n in zip(self._wait_marks(), _WAIT_MARK_LABELS, strict=True)]
            if self.waiting
            else []
        )
        statement = select(
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.tenant_id,
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
        version = row.wait_version in (3, 4)
        reason = row.wait_reason == MANAGED_WAIT_REASON
        if not (version or reason):
            return False
        if not (
            version
            and reason
            and (
                row.wait_child
                if row.wait_version == 3
                else type(row.wait_count) is int and 2 <= row.wait_count <= 32
            )
        ):
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
                    done = await self.inspect_row(row)
                    if done is not None:
                        swept.append(done)
            if len(rows) < limit:
                return swept, visited, None
        return swept, visited, after

    async def inspect_row(self, row: Any) -> SweptRun | None:
        if self.skipped(row):
            if self.waiting:
                await self.collect_adoptable(row)
            return None
        if self.waiting and (
            self.abandoned_after is None
            or _to_aware_utc(row.sweep_age) >= self.now - self.abandoned_after
        ):
            return None
        return await self.sweep_row(row)

    async def collect_adoptable(self, row: Any) -> None:
        if (
            self.adoptable is None
            or self.adoptable_cutoff is None
            or row.wait_reason != MANAGED_WAIT_REASON
        ):
            return
        if _to_aware_utc(row.sweep_age) >= self.adoptable_cutoff:
            return
        statement = select(
            AgentRun.pause_data,
            self.store.json_get(AgentRun.loop_state, ("managed",), "json"),
        ).where(
            AgentRun.id == row.id,
            AgentRun.tenant_id == row.tenant_id,
            AgentRun.status == RunStatus.PAUSED.value,
            AgentRun.claim_epoch == row.claim_epoch,
        )
        async with self.store._session_factory() as session:
            saved = (await session.execute(statement)).one_or_none()
        if saved is None:
            return
        try:
            wait = PauseState.from_managed_dict(saved[0], run_id=row.id)
            header = ManagedHeader.from_dict(saved[1], run_id=row.id)
        except ManagedStateError:
            _codes.log_code(
                logger, _codes.MANAGED_STATE_INVALID, run_id=row.id, managed_reason="wait_invalid"
            )
            return
        bindings = wait.managed_fanout.bindings if wait.managed_fanout else (wait.managed_wait,)
        ids = [binding.child_run_id for binding in bindings if binding is not None]
        # The native SQL scan is hosted by a store implementing the fan-out read protocol.
        store: ManagedFanoutStore = cast("Any", self.store)
        children = await store.get_managed_children(row.id, ids, tenant_id=row.tenant_id)
        if (
            _adoptable_children(children, bindings, row, header)
            and header.root_run_id not in self._adoptable_roots
        ):
            self._adoptable_roots.add(header.root_run_id)
            self.adoptable.append(header.root_run_id)

    def _reason(self, row: Any) -> str:
        if self.waiting:
            return FailureReason.ABANDONED_WAITING.value
        if row.started:
            return FailureReason.STALE_RUNNING.value
        return FailureReason.NEVER_STARTED.value

    async def sweep_row(self, row: Any) -> SweptRun | None:
        """The row's final CAS and its lifecycle event; None when another writer won."""
        from sqlalchemy import case

        reason = self._reason(row)
        cutoff = (
            self.now - self.abandoned_after if self.abandoned_after is not None else self.cutoff
        )
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
                    self.age < cutoff,
                    and_(AgentRun.updated_at < cutoff, self._not_managed_wait())
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


def _adoptable_children(
    children: Sequence[ManagedChildRow],
    bindings: Sequence[ManagedWait | None],
    row: Any,
    header: ManagedHeader,
) -> bool:
    expected = {
        binding.child_run_id: binding.tool_call_id for binding in bindings if binding is not None
    }
    if len(children) != len(expected) or {child.run_id for child in children} != set(expected):
        return False
    terminal = {status.value for status in TERMINAL_STATUSES}
    for child in children:
        try:
            saved = ManagedHeader.from_dict(child.header, run_id=child.run_id)
        except ManagedStateError:
            return False
        if (
            child.status not in terminal
            or saved.path[:-1] != header.path
            or (
                child.parent_run_id,
                child.tenant_id,
                saved.parent_run_id,
                saved.root_run_id,
                saved.tool_call_id,
            )
            != (row.id, row.tenant_id, row.id, header.root_run_id, expected[child.run_id])
        ):
            return False
    return True


_WAIT_MARK_LABELS = ("wait_version", "wait_reason", "wait_child", "wait_count")


_SweepScan.__module__ = "dendrux.runtime.stores.sql.base"
