"""Read bounded ancestry, descendants and aggregate subtree usage."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from sqlalchemy import select

from dendrux import _codes
from dendrux.runtime._bounds import (
    DEFAULT_READ_LIMIT,
    DEFAULT_SUBTREE_DEPTH,
    DEFAULT_SUBTREE_ROWS,
    MAX_SMALL_PAGE,
    MAX_SUBTREE_DEPTH,
    MAX_SUBTREE_ROWS,
)
from dendrux.runtime._bounds import IN_CHUNK as _IN_CHUNK
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    _COST_DECIMALS,
    UNSCOPED,
    DelegationInfo,
    ParentRef,
    RunBrief,
    SubtreeNodeRecord,
    SubtreeRecord,
    SubtreeSummary,
    TenantScope,
    UsageRollup,
    _apply_tenant_scope,
    _extract_status,
    _run_to_record,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_host import _StoreHost
from dendrux.runtime.stores.sql._sql_reads import _waiting_for_expression
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._types import RunParentCursor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.db.models import AgentRun


class _SqlTree(_StoreHost):
    async def get_children_page(
        self, parent_ids: Sequence[str], *, after: str | None, limit: int, tenant_id: TenantScope
    ) -> list[str]:
        """Read a tenant-scoped keyset page, capped at 500 children of a marked frontier."""

        from dendrux.db.models import AgentRun

        _sql_support._positive_page(limit)
        if not parent_ids:
            return []
        size = min(limit, MAX_SMALL_PAGE)
        statement = self.child_page_query(
            runs=AgentRun.__table__,
            parents=[RunParentCursor(parent, after) for parent in sorted(set(parent_ids))],
            tenant_scope=tenant_id,
            page_size=size,
        )

        async def _attempt() -> list[str]:
            async with self._session_factory() as session:
                return list((await session.scalars(statement)).all())

        return await retry_transient_db(
            _attempt,
            label="get_children_page",
            run_id=parent_ids[0],
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def get_delegation_info(
        self,
        run_id: str,
        *,
        max_rows: int = DEFAULT_SUBTREE_ROWS,
        max_depth: int = DEFAULT_SUBTREE_DEPTH,
    ) -> DelegationInfo | None:
        """Bounded delegation context; summaries cover only the visited subtree.

        max_rows includes the root, ancestors and descendants (1..10000).
        max_depth bounds each direction (0..100). One lookahead row detects
        truncation; large JSON execution/payload columns are never selected.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from dendrux.db.models import AgentRun

        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or not 1 <= max_rows <= MAX_SUBTREE_ROWS
        ):
            raise ValueError("max_rows must be an int between 1 and 10000.")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 0 <= max_depth <= MAX_SUBTREE_DEPTH
        ):
            raise ValueError("max_depth must be an int between 0 and 100.")
        projection = load_only(
            AgentRun.id,
            AgentRun.parent_run_id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.delegation_level,
            AgentRun.total_input_tokens,
            AgentRun.total_output_tokens,
            AgentRun.total_cost_usd,
        )

        def brief(row: AgentRun) -> RunBrief:
            return RunBrief(
                run_id=row.id,
                agent_name=row.agent_name,
                status=_extract_status(row),
                delegation_level=row.delegation_level,
            )

        async with self._session_factory() as session:
            run_row = (
                await session.execute(
                    select(AgentRun).options(projection).where(AgentRun.id == run_id)
                )
            ).scalar_one_or_none()
            if run_row is None:
                return None
            remaining = max_rows - 1
            seen = {run_id}
            ancestry: list[RunBrief] = []
            parent = (
                ParentRef(run_id=run_row.parent_run_id, resolved=False)
                if run_row.parent_run_id
                else None
            )
            current = run_row.parent_run_id
            truncated = False
            ancestry_complete = True
            while current:
                if current in seen:
                    ancestry_complete = False
                    break
                if remaining == 0 or len(ancestry) >= max_depth:
                    truncated = True
                    ancestry_complete = False
                    break
                row = (
                    await session.execute(
                        select(AgentRun).options(projection).where(AgentRun.id == current)
                    )
                ).scalar_one_or_none()
                if row is None:
                    ancestry_complete = False
                    break
                seen.add(row.id)
                remaining -= 1
                ancestry.append(brief(row))
                if len(ancestry) == 1:
                    parent = ParentRef(
                        run_id=row.id,
                        resolved=True,
                        agent_name=row.agent_name,
                        status=_extract_status(row),
                        delegation_level=row.delegation_level,
                    )
                current = row.parent_run_id
            ancestry.reverse()
            descendants: list[tuple[AgentRun, int]] = []
            frontier = [run_id]
            depth = 0
            seen = {run_id}
            while frontier:
                depth += 1
                next_level = []
                stop = False
                for i in range(0, len(frontier), _IN_CHUNK):
                    allowance = remaining if depth <= max_depth else 0
                    stmt = (
                        select(AgentRun)
                        .options(projection)
                        .where(
                            AgentRun.parent_run_id.in_(frontier[i : i + _IN_CHUNK]),
                            AgentRun.id.not_in(seen),
                        )
                        .order_by(AgentRun.created_at, AgentRun.id)
                        .limit(allowance + 1)
                    )
                    rows = list((await session.execute(self._parent_run_query(stmt))).scalars())
                    if len(rows) > allowance:
                        truncated = True
                        stop = True
                    for row in rows[:allowance]:
                        seen.add(row.id)
                        remaining -= 1
                        descendants.append((row, depth))
                        next_level.append(row.id)
                    if stop:
                        break
                if stop:
                    break
                frontier = next_level
            children, summary = await self._traverse_subtree_bfs(
                session, run_row, descendants=descendants
            )
            return DelegationInfo(
                parent=parent,
                children=children,
                ancestry=ancestry,
                subtree_summary=summary,
                ancestry_complete=ancestry_complete,
                truncated=truncated,
            )

    async def _walk_ancestry(
        self,
        session: AsyncSession,
        run_row: AgentRun,
    ) -> tuple[list[RunBrief], bool]:
        """Walk parent chain upward. Returns (ancestry, complete) root-first."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        ancestry: list[RunBrief] = []
        complete = True
        current_parent_id = run_row.parent_run_id
        visited: set[str] = {run_row.id}

        while current_parent_id and current_parent_id not in visited:
            visited.add(current_parent_id)
            stmt = select(AgentRun).where(AgentRun.id == current_parent_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                complete = False
                break
            ancestry.append(
                RunBrief(
                    run_id=row.id,
                    agent_name=row.agent_name,
                    status=_extract_status(row),
                    delegation_level=row.delegation_level,
                )
            )
            current_parent_id = row.parent_run_id

        # Cycle detected — parent_run_id pointed back into visited set
        if current_parent_id and current_parent_id in visited:
            complete = False

        ancestry.reverse()
        return ancestry, complete

    async def _descendant_batch(
        self,
        session: AsyncSession,
        root_id: str,
        parent_ids: list[str],
        *,
        tenant_id: TenantScope,
        light: bool,
        max_nodes: int | None,
        loaded: int,
        waiting_labels: dict[str, str | None] | None = None,
    ) -> list[AgentRun]:
        from sqlalchemy.orm import load_only

        from dendrux.db.models import AgentRun

        stmt = _apply_tenant_scope(
            select(AgentRun).where(AgentRun.parent_run_id.in_(parent_ids)), tenant_id
        )
        if waiting_labels is not None:
            stmt = stmt.add_columns(_waiting_for_expression(self))
        if light:
            stmt = stmt.options(load_only(AgentRun.id, AgentRun.parent_run_id, AgentRun.tenant_id))
        if max_nodes is not None:
            stmt = stmt.where(AgentRun.id != root_id).limit(max_nodes - loaded)
        result = await session.execute(
            self._parent_run_query(stmt.order_by(AgentRun.created_at, AgentRun.id))
        )
        if waiting_labels is not None:
            rows = result.all()
            waiting_labels.update((row.id, label) for row, label in rows)
            return [row for row, _ in rows]
        return list(result.scalars())

    async def _bfs_descendants(
        self,
        session: AsyncSession,
        root_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        light: bool = False,
        max_nodes: int | None = None,
        waiting_labels: dict[str, str | None] | None = None,
    ) -> list[tuple[AgentRun, int]]:
        """Descendants of ``root_id`` as ``(row, depth)`` in BFS order, root excluded.

        ``depth`` is 1 for direct children. Cycle-safe via a seen set. With
        ``tenant_id`` set, other tenants' runs are skipped and so is
        everything below them. ``light`` loads only id / parent / tenant so
        usage-only callers don't pull large JSON columns.
        """
        batch = self._descendant_batch
        if waiting_labels is not None:
            batch = partial(batch, waiting_labels=waiting_labels)
        seen: set[str] = {root_id}
        current_level = [root_id]
        found: list[tuple[AgentRun, int]] = []
        depth = 0

        while current_level:
            depth += 1
            rows: list[AgentRun] = []
            for i in range(0, len(current_level), _IN_CHUNK):
                rows.extend(
                    await batch(
                        session,
                        root_id,
                        current_level[i : i + _IN_CHUNK],
                        tenant_id=tenant_id,
                        light=light,
                        max_nodes=max_nodes,
                        loaded=len(found) + len(rows),
                    )
                )
                if max_nodes is not None and len(found) + len(rows) >= max_nodes:
                    from dendrux.errors import BudgetConfigurationError

                    _codes.log_code(
                        logger,
                        _codes.SUBTREE_LIMIT,
                        run_id=root_id,
                        limit=max_nodes,
                    )
                    raise BudgetConfigurationError(reason="subtree_limit")

            next_level: list[str] = []
            for cr in rows:
                if cr.id in seen:
                    continue
                seen.add(cr.id)
                next_level.append(cr.id)
                found.append((cr, depth))
            current_level = next_level

        return found

    async def _traverse_subtree_bfs(
        self,
        session: AsyncSession,
        run_row: AgentRun,
        *,
        descendants: list[tuple[AgentRun, int]] | None = None,
    ) -> tuple[list[RunBrief], SubtreeSummary]:
        """BFS downward from run_row. Returns (direct_children, summary).

        Summary includes the root run itself in token/cost/status counts.
        """
        if descendants is None:
            descendants = await self._bfs_descendants(session, run_row.id)

        sum_input = run_row.total_input_tokens or 0
        sum_output = run_row.total_output_tokens or 0
        root_cost = run_row.total_cost_usd
        has_cost = root_cost is not None
        sum_cost = float(root_cost) if root_cost is not None else 0.0
        unknown_cost_count = 0 if has_cost else 1
        status_counts: dict[str, int] = {_extract_status(run_row): 1}
        children: list[RunBrief] = []

        for cr, depth in descendants:
            sum_input += cr.total_input_tokens or 0
            sum_output += cr.total_output_tokens or 0
            if cr.total_cost_usd is not None:
                sum_cost += float(cr.total_cost_usd)
                has_cost = True
            else:
                unknown_cost_count += 1
            s = _extract_status(cr)
            status_counts[s] = status_counts.get(s, 0) + 1
            if depth == 1:
                children.append(
                    RunBrief(
                        run_id=cr.id,
                        agent_name=cr.agent_name,
                        status=s,
                        delegation_level=cr.delegation_level,
                    )
                )

        summary = SubtreeSummary(
            direct_child_count=len(children),
            descendant_count=len(descendants),
            max_depth=max((d for _, d in descendants), default=0),
            subtree_input_tokens=sum_input,
            subtree_output_tokens=sum_output,
            subtree_cost_usd=sum_cost if has_cost else None,
            unknown_cost_count=unknown_cost_count,
            status_counts=status_counts,
        )
        return children, summary

    async def _usage_by_run(
        self, session: AsyncSession, run_ids: list[str]
    ) -> dict[str, UsageRollup]:
        """Per-run live usage, one grouped query per chunk of ids.

        Runs without any ``llm_interactions`` row are absent from the result.
        """
        from sqlalchemy import case, func, select

        from dendrux.db.models import LLMInteraction as L

        out: dict[str, UsageRollup] = {}
        for i in range(0, len(run_ids), _IN_CHUNK):
            stmt = (
                select(
                    L.agent_run_id,
                    func.coalesce(func.sum(L.input_tokens), 0),
                    func.coalesce(func.sum(L.output_tokens), 0),
                    func.coalesce(func.sum(L.cache_read_input_tokens), 0),
                    func.coalesce(func.sum(L.cache_creation_input_tokens), 0),
                    func.coalesce(func.sum(L.reasoning_tokens), 0),
                    func.coalesce(func.sum(L.cost_usd), 0),
                    func.coalesce(func.sum(case((L.cost_usd.is_(None), 1), else_=0)), 0),
                    func.count(),
                )
                .where(L.agent_run_id.in_(run_ids[i : i + _IN_CHUNK]))
                .group_by(L.agent_run_id)
            )
            for row in (await session.execute(stmt)).all():
                out[row[0]] = UsageRollup(
                    input_tokens=int(row[1]),
                    output_tokens=int(row[2]),
                    cache_read_tokens=int(row[3]),
                    cache_write_tokens=int(row[4]),
                    reasoning_tokens=int(row[5]),
                    known_cost_usd=round(float(row[6]), _COST_DECIMALS),
                    unpriced_calls=int(row[7]),
                    llm_calls=int(row[8]),
                )
        return out

    async def get_subtree(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED, max_nodes: int | None = None
    ) -> SubtreeRecord | None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(
                select(AgentRun, _waiting_for_expression(self)).where(AgentRun.id == run_id),
                tenant_id,
            )
            row = (await session.execute(stmt)).one_or_none()
            if row is None:
                return None
            root, label = row
            waiting_labels = {root.id: label}

            rows = [
                (root, 0),
                *await self._bfs_descendants(
                    session,
                    root.id,
                    tenant_id=tenant_id,
                    max_nodes=max_nodes,
                    waiting_labels=waiting_labels,
                ),
            ]
            usage = await self._usage_by_run(session, [r.id for r, _ in rows])
            nodes = [
                SubtreeNodeRecord(
                    run=_run_to_record(r, waiting_for=waiting_labels[r.id]),
                    depth=d,
                    usage=usage.get(r.id, UsageRollup()),
                )
                for r, d in rows
            ]
            return SubtreeRecord(nodes=nodes, usage=UsageRollup.combine(n.usage for n in nodes))

    async def get_descendant_page(
        self,
        run_id: str,
        *,
        after: str | None = None,
        limit: int = DEFAULT_READ_LIMIT,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[str]:
        """Read one page of descendant ids, without run payloads.

        A recursive UNION deduplicates ids, including malformed cycles. Each
        page uses a fresh statement so cancellation writes never hold its cursor.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        tree = _apply_tenant_scope(select(AgentRun.id).where(AgentRun.id == run_id), tenant_id).cte(
            "descendants", recursive=True
        )
        children = _apply_tenant_scope(
            select(AgentRun.id).join(tree, AgentRun.parent_run_id == tree.c.id), tenant_id
        )
        tree = tree.union(children)
        stmt = select(tree.c.id).where(tree.c.id != run_id)
        if after is not None:
            stmt = stmt.where(tree.c.id > after)
        stmt = stmt.order_by(tree.c.id).limit(max(1, min(limit, MAX_SMALL_PAGE)))
        async with self._session_factory() as session:
            return list((await session.scalars(stmt)).all())

    async def subtree_usage(self, run_id: str, *, max_nodes: int = 10000) -> UsageRollup:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        async with self._session_factory() as session:
            descendants = await self._bfs_descendants(
                session, run_id, light=True, max_nodes=max_nodes
            )
            usage = await self._usage_by_run(session, [run_id, *(r.id for r, _ in descendants)])
            return UsageRollup.combine(usage.values())

    async def run_usage(self, run_id: str) -> UsageRollup:
        async with self._session_factory() as session:
            return (await self._usage_by_run(session, [run_id])).get(run_id, UsageRollup())
