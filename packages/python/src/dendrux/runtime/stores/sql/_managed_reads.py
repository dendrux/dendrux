"""Read bounded cancellation projections and combined managed control through shared SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import and_, literal, select

from dendrux.db.models import AgentRun, RunInbox
from dendrux.runtime._bounds import MAX_SMALL_PAGE
from dendrux.runtime.durability import _safe_database_call, retry_transient_db
from dendrux.runtime.state import _TERMINAL_STATUS_VALUES, MANAGED_HEADER_KEY, RunControl
from dendrux.runtime.stores.sql._types import RunParentCursor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.runtime.state import TenantScope


class ManagedReads:
    """Supply managed cancellation reads using each concrete store's SQL dialect hooks."""

    @_safe_database_call
    async def get_managed_cancel_children_page(
        self: Any,
        parent_ids: Sequence[str],
        *,
        after: str | None,
        limit: int,
        tenant_id: TenantScope,
    ) -> list[Any]:
        """Project (id, parent id, raw header), ordered by id, after the keyset cursor.

        Scope to tenant_id (None means the null tenant; UNSCOPED is explicit).
        A positive integer limit is capped at 500; empty parents return no rows.
        Do not load histories or validate/filter the raw header in this read.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not parent_ids:
            return []
        page = self.child_page_query(
            runs=AgentRun.__table__,
            parents=[RunParentCursor(parent, after) for parent in sorted(set(parent_ids))],
            tenant_scope=tenant_id,
            page_size=min(limit, MAX_SMALL_PAGE),
        ).subquery("cancel_children")
        statement = (
            select(
                AgentRun.id,
                AgentRun.parent_run_id,
                self.json_get(AgentRun.loop_state, (MANAGED_HEADER_KEY,), "json"),
            )
            .join(page, AgentRun.id == page.c.id)
            .order_by(AgentRun.id)
        )
        async with self._session_factory() as session:
            return list((await session.execute(statement)).all())

    def _managed_root_key(self: Any, column: Any) -> Any:
        return self.json_get(column, (MANAGED_HEADER_KEY, "root_run_id"), "str")

    def _managed_stop_predicate(self: Any) -> Any:
        root = AgentRun.__table__.alias("managed_root")
        root_id = self._managed_root_key(AgentRun.loop_state)
        live = (
            select(root.c.id)
            .where(
                root.c.id == root_id,
                root.c.tenant_id.is_not_distinct_from(AgentRun.tenant_id),
                root.c.cancel_requested.is_(False),
                root.c.status.notin_(_TERMINAL_STATUS_VALUES),
            )
            .exists()
        )
        return and_(root_id.is_not(None), ~live)

    @_safe_database_call
    async def is_managed_root_stopped(self: Any, run_id: str) -> bool:
        """Read eligibility of the referenced root within the own row's tenant.

        Missing own rows fail closed. Missing, foreign-tenant, cancelled or
        terminal roots are stopped; a legacy row without a root is not.
        """
        async with self._session_factory() as session:
            value = await session.scalar(
                select(self._managed_stop_predicate()).where(AgentRun.id == run_id)
            )
            return value is None or bool(value)

    def _control_query(self: Any) -> Any:
        pending = (
            select(literal(1))
            .where(RunInbox.run_id == AgentRun.id, RunInbox.delivered_at.is_(None))
            .exists()
        )
        return select(
            AgentRun.cancel_requested,
            AgentRun.pause_requested,
            pending,
            AgentRun.claim_epoch,
            AgentRun.status,
        )

    @_safe_database_call
    async def get_managed_run_control(self: Any, run_id: str) -> RunControl:
        """Read own control and same-tenant root stop in one statement.

        Cancel is own-cancel OR root-stopped; keep own pause, pending-note,
        epoch and status fields. A missing own row fails closed with cancel=True.
        """
        statement = self._control_query()
        statement = statement.with_only_columns(
            AgentRun.cancel_requested | self._managed_stop_predicate(),
            *tuple(statement.selected_columns)[1:],
        )
        return cast(
            "RunControl",
            await self._read_control(run_id, statement, RunControl(cancel_requested=True)),
        )

    async def _read_control(
        self: Any, run_id: str, statement: Any, missing: RunControl
    ) -> RunControl:
        async def _attempt() -> RunControl:
            async with self._session_factory() as session:
                row = (await session.execute(statement.where(AgentRun.id == run_id))).first()
                if row is None:
                    return missing
                return RunControl(
                    cancel_requested=bool(row[0]),
                    pause_requested=bool(row[1]),
                    has_pending_notes=bool(row[2]),
                    claim_epoch=row[3],
                    status=row[4],
                )

        return await retry_transient_db(
            _attempt,
            label="get_run_control",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )
