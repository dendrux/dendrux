"""Read bounded managed scope/control and enforce atomic run admission through shared SQL."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import and_, literal, select

from dendrux.db.models import AgentRun, RunInbox
from dendrux.errors import ManagedStateError
from dendrux.runtime._bounds import MAX_SMALL_PAGE
from dendrux.runtime._managed_limits import ManagedRunLimitError
from dendrux.runtime.durability import _safe_database_call, retry_transient_db
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    MANAGED_HEADER_KEY,
    ManagedHeader,
    RunControl,
)
from dendrux.runtime.stores.sql._types import RunParentCursor
from dendrux.types import Limits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.state import BindManagedChildRequest, TenantScope


def _counted_member(
    row: Any, root_run_id: str, visited: set[str]
) -> Literal["member", "other", "invalid"]:
    if not isinstance(row.header, dict) or row.header.get("root_run_id") != root_run_id:
        return "other"
    try:
        header = ManagedHeader.from_dict(row.header, run_id=row.id)
    except ManagedStateError:
        return "invalid"
    if row.id in visited or header.parent_run_id != row.parent_run_id:
        return "invalid"
    return "member"


def _managed_members(rows: Sequence[Any], root_run_id: str, visited: set[str]) -> list[str] | None:
    members: list[str] = []
    for row in rows:
        membership = _counted_member(row, root_run_id, visited)
        if membership == "invalid":
            return None
        if membership == "member":
            members.append(row.id)
    return members


class ManagedReads:
    """Supply managed scope/control reads and bind admission through SQL dialect hooks."""

    async def _lock_managed_root(
        self: Any, session: AsyncSession, request: BindManagedChildRequest
    ) -> None:
        """Serialize all binders root-first, including those with different parents."""
        await self.reserve_parent_writer(session, request.root_run_id)
        await session.execute(
            self.lock_clause(
                select(AgentRun.id).where(
                    AgentRun.id == request.root_run_id,
                    AgentRun.tenant_id.is_not_distinct_from(request.tenant_id),
                ),
                "update",
            )
        )

    async def _admit_managed_limit(
        self: Any, session: AsyncSession, request: BindManagedChildRequest, parent: Any
    ) -> ManagedHeader:
        """Keep replay free; certify admission after ordinary bind eligibility checks."""
        header = self._bind_admit(request, parent)
        root_state = (
            parent.loop_state
            if request.root_run_id == request.parent_run_id
            else await session.scalar(
                select(AgentRun.loop_state).where(AgentRun.id == request.root_run_id)
            )
        )
        config = (root_state or {}).get(
            "managed_limits", {"max_managed_runs": Limits.max_managed_runs}
        )
        cap = config.get("max_managed_runs") if isinstance(config, dict) else None
        if type(cap) is not int or cap < 1:
            raise ManagedRunLimitError(run_id=request.parent_run_id)
        count = await self._managed_run_count(session, request.root_run_id, cap, request.tenant_id)
        if count is None or count >= cap:
            raise ManagedRunLimitError(run_id=request.parent_run_id)
        # The dynamic mixin self resolves SQLStateStore._bind_admit's typed header.
        return cast("ManagedHeader", header)

    async def _managed_run_count(
        self: Any, session: AsyncSession, root_run_id: str, limit: int, tenant_id: str | None
    ) -> int | None:
        """Saturating explicit membership count, root included; None is inconclusive.

        Completed members count. Legacy parent links do not. Only ancestry reachable
        through matching managed headers is membership. No histories are loaded.
        Inspect at most limit+1 child candidates, including legacy rows; exhaustion
        before a conclusive count returns None rather than scanning unlimited siblings.
        Bind uses this same enumeration inside its root-locked transaction.
        """
        root = await session.scalar(
            select(self.json_get(AgentRun.loop_state, (MANAGED_HEADER_KEY,), "json")).where(
                AgentRun.id == root_run_id, AgentRun.tenant_id.is_not_distinct_from(tenant_id)
            )
        )
        try:
            header = ManagedHeader.from_dict(root, run_id=root_run_id)
        except ManagedStateError:
            return None
        if header.root_run_id != root_run_id or header.parent_run_id is not None:
            return None
        visited, frontier = {root_run_id}, deque([root_run_id])
        remaining = limit + 1
        while frontier and len(visited) < limit:
            parents = [frontier.popleft() for _ in range(min(len(frontier), MAX_SMALL_PAGE))]
            rows = (
                await session.execute(self._managed_count_query(parents, remaining, tenant_id))
            ).all()
            members = _managed_members(rows, root_run_id, visited)
            if members is None:
                return None
            visited.update(members)
            frontier.extend(members)
            if len(visited) >= limit:
                return limit
            remaining -= len(rows)
            if not remaining:
                return None
        return len(visited)

    def _managed_count_query(
        self: Any, parents: list[str], limit: int, tenant_id: str | None
    ) -> Any:
        return (
            select(
                AgentRun.id,
                AgentRun.parent_run_id,
                self.json_get(AgentRun.loop_state, (MANAGED_HEADER_KEY,), "json").label("header"),
            )
            .where(
                AgentRun.parent_run_id.in_(parents),
                AgentRun.tenant_id.is_not_distinct_from(tenant_id),
            )
            .limit(limit)
        )

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
