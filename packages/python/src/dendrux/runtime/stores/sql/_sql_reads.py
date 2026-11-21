"""Serve bounded run, trace, tool and listing reads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import ColumnElement, and_, case, select

from dendrux._lifecycle import StoredEventType
from dendrux._meta_keys import (
    ERROR_INFO_META_KEY,
    LATEST_REWRITE_META_KEY,
    OBSERVATION_META_KEY,
    STOP_REASON_META_KEY,
)
from dendrux.db.models import AgentRun
from dendrux.runtime._bounds import DEFAULT_LIST_LIMIT, DEFAULT_READ_LIMIT, MAX_PAGE
from dendrux.runtime._bounds import IN_CHUNK as _IN_CHUNK
from dendrux.runtime.durability import _safe_database_call
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    UNSCOPED,
    ConversationEntry,
    ConversationPage,
    LLMInteractionRecord,
    RunRecord,
    TenantScope,
    ToolCallReadRecord,
    TraceRecord,
    _apply_tenant_scope,
    _check_offset,
    _require_aware_utc,
    _run_in_scope,
    _run_to_record,
    _to_aware_utc,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_host import _StoreHost

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

_TRACE_POSITION_STRIDE = 1000


def _waiting_for_expression(store: _StoreHost) -> ColumnElement[str | None]:
    return case(
        (
            and_(
                AgentRun.status == "paused",
                store.json_get(AgentRun.pause_data, ["pause_reason"], "str") == "managed_wait",
            ),
            "subagents",
        ),
        else_=None,
    )


class _SqlReads(_StoreHost):
    async def get_loop_state(self, run_id: str) -> dict[str, Any] | None:
        """Read the authoritative execution projection."""
        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            value = await session.scalar(select(AgentRun.loop_state).where(AgentRun.id == run_id))
            return cast("dict[str, Any] | None", value)

    async def get_trace(
        self, run_id: str, trace_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> TraceRecord | None:
        """Read a referenced state trace by id, including superseded prefixes."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            row = await session.scalar(
                select(ReactTrace).where(
                    ReactTrace.agent_run_id == run_id,
                    ReactTrace.id == trace_id,
                    _run_in_scope(run_id, tenant_id),
                )
            )
            return _sql_support._trace_record(row) if row is not None else None

    async def get_traces_after(
        self,
        run_id: str,
        *,
        after_order_index: int | None,
        limit: int = DEFAULT_READ_LIMIT,
        kinds: tuple[str, ...] = ("seed", "turn", "rewrite", "state"),
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceRecord]:
        """Read a bounded keyset page of physical trace rows."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        _sql_support._positive_page(limit)
        async with self._session_factory() as session:
            stmt = select(ReactTrace).where(
                ReactTrace.agent_run_id == run_id,
                ReactTrace.trace_kind.in_(kinds),
                _run_in_scope(run_id, tenant_id),
            )
            if after_order_index is not None:
                stmt = stmt.where(ReactTrace.order_index > after_order_index)
            rows = (
                await session.scalars(
                    stmt.order_by(ReactTrace.order_index).limit(min(limit, MAX_PAGE))
                )
            ).all()
            return [_sql_support._trace_record(row) for row in rows]

    async def conversation_traces(
        self,
        run_id: str,
        *,
        after_position: int | None,
        limit: int = DEFAULT_READ_LIMIT,
        include_superseded: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> ConversationPage:
        """Project the latest replacement prefix followed by its conversation suffix."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun, ReactTrace

        _sql_support._positive_page(limit)
        limit = min(limit, MAX_PAGE)
        after = -1 if after_position is None else after_position
        entries: list[ConversationEntry] = []
        async with self._session_factory() as session:
            base = select(ReactTrace).where(
                ReactTrace.agent_run_id == run_id, _run_in_scope(run_id, tenant_id)
            )
            rewrite = None
            rewrite_order = None
            if not include_superseded:
                pointer = await session.scalar(
                    select(AgentRun.meta[LATEST_REWRITE_META_KEY]).where(
                        AgentRun.id == run_id,
                        _run_in_scope(run_id, tenant_id),
                    )
                )
                if isinstance(pointer, dict):
                    rewrite_order = pointer["order_index"]
                    if after < rewrite_order * _TRACE_POSITION_STRIDE + (
                        _TRACE_POSITION_STRIDE - 1
                    ):
                        rewrite = await session.scalar(
                            base.where(ReactTrace.id == pointer["trace_id"])
                        )
            if rewrite is not None:
                for index, message in enumerate(json.loads(rewrite.content), 1):
                    position = rewrite.order_index * _TRACE_POSITION_STRIDE + index
                    if position > after:
                        entries.append(
                            ConversationEntry(
                                position,
                                message["role"],
                                message["content"],
                                {
                                    **(rewrite.meta or {}),
                                    **(message.get("meta") or {}),
                                    "expanded_index": index,
                                },
                                "rewrite",
                                rewrite.id,
                            )
                        )
                        if len(entries) > limit:
                            break
            if len(entries) <= limit:
                kinds = ("seed", "turn", "rewrite") if include_superseded else ("seed", "turn")
                stmt = base.where(
                    ReactTrace.trace_kind.in_(kinds),
                    ReactTrace.order_index > after // _TRACE_POSITION_STRIDE,
                )
                if rewrite_order is not None:
                    stmt = stmt.where(ReactTrace.order_index > rewrite_order)
                rows = (
                    await session.scalars(
                        stmt.order_by(ReactTrace.order_index).limit(limit + 1 - len(entries))
                    )
                ).all()
                entries.extend(
                    ConversationEntry(
                        row.order_index * _TRACE_POSITION_STRIDE,
                        row.role,
                        row.content,
                        row.meta or {},
                        row.trace_kind,
                        row.id,
                    )
                    for row in rows
                )
        more = len(entries) > limit
        entries = entries[:limit]
        return ConversationPage(entries, entries[-1].position if more else None)

    async def get_llm_interactions(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[LLMInteractionRecord]:
        """LLM interactions of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        _require_aware_utc("before", before)

        from dendrux.db.models import LLMInteraction

        async with self._session_factory() as session:
            stmt = (
                select(LLMInteraction)
                .where(LLMInteraction.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(LLMInteraction.iteration_index, LLMInteraction.id)
            )
            if iteration_index is not None:
                stmt = stmt.where(LLMInteraction.iteration_index == iteration_index)
            if before is not None:
                stmt = stmt.where(LLMInteraction.created_at < _to_aware_utc(before))
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, MAX_PAGE)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                LLMInteractionRecord(
                    id=r.id,
                    iteration_index=r.iteration_index,
                    model=r.model,
                    provider=r.provider,
                    semantic_request=r.semantic_request,
                    semantic_response=r.semantic_response,
                    provider_request=r.provider_request,
                    provider_response=r.provider_response,
                    input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens,
                    cost_usd=float(r.cost_usd) if r.cost_usd is not None else None,
                    duration_ms=r.duration_ms,
                    cache_read_input_tokens=r.cache_read_input_tokens,
                    cache_creation_input_tokens=r.cache_creation_input_tokens,
                    reasoning_tokens=r.reasoning_tokens,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def get_run_status(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> str | None:
        """Status-only projection for polling; never loads execution history."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            value = (
                await session.execute(
                    _apply_tenant_scope(
                        select(AgentRun.status).where(AgentRun.id == run_id), tenant_id
                    )
                )
            ).scalar_one_or_none()
            return None if value is None else str(value)

    async def get_pii_mapping(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, str] | None:
        """Retrieve pii_mapping for a run. None if not set or outside ``tenant_id``."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(
                select(AgentRun.pii_mapping).where(AgentRun.id == run_id), tenant_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return dict(row)

    @_safe_database_call
    async def _get_result_record(self, run_id: str, *, tenant_id: TenantScope) -> RunRecord | None:
        """Project one browser-result record without prompts or pause history.

        Private optimization; custom stores keep the existing scoped get_run
        contract. All reads use the same tenant predicate as get_run.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        fields = (
            AgentRun.strategy,
            AgentRun.storage_format,
            AgentRun.claim_epoch,
            AgentRun.status_changed_at,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.lease_expired_at,
            AgentRun.last_reclaimed_at,
            AgentRun.trace_link,
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.tenant_id,
            AgentRun.iteration_count,
            AgentRun.error,
            AgentRun.cancel_requested,
            AgentRun.total_input_tokens,
            AgentRun.total_output_tokens,
            AgentRun.total_cache_read_tokens,
            AgentRun.total_cache_creation_tokens,
            AgentRun.total_reasoning_tokens,
            AgentRun.total_cost_usd,
            AgentRun.output_data["answer"].as_string().label("answer"),
            AgentRun.meta[ERROR_INFO_META_KEY].label("error_info"),
            AgentRun.meta[STOP_REASON_META_KEY].as_string().label("stop_reason"),
        )
        stmt = _apply_tenant_scope(select(*fields).where(AgentRun.id == run_id), tenant_id)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).mappings().first()
        if row is None:
            return None
        values = dict(row)
        for name in ("status_changed_at", "lease_expired_at", "last_reclaimed_at"):
            values[name] = _to_aware_utc(values[name])
        info, stop_reason = values.pop("error_info"), values.pop("stop_reason")
        values["total_cost_usd"] = (
            float(values["total_cost_usd"]) if values["total_cost_usd"] is not None else None
        )
        return RunRecord(
            **values,
            meta={ERROR_INFO_META_KEY: info, STOP_REASON_META_KEY: stop_reason},
        )

    async def get_run(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> RunRecord | None:
        """The run, or None when it does not exist or is outside ``tenant_id``.

        The tenant predicate is part of the query, so a foreign run reads
        exactly like a missing one.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(
                select(AgentRun, _waiting_for_expression(self)).where(AgentRun.id == run_id),
                tenant_id,
            )
            result = await session.execute(stmt)
            row = result.one_or_none()
            if row is None:
                return None
            return _run_to_record(row[0], waiting_for=row[1])

    async def get_next_trace_index(self, run_id: str) -> int:
        """Read the next trace order from the run/order index, without payloads."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            value = await session.scalar(
                select(ReactTrace.order_index)
                .where(ReactTrace.agent_run_id == run_id)
                .order_by(ReactTrace.order_index.desc())
                .limit(1)
            )
            return int(value) + 1 if value is not None else 0

    async def get_public_traces(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        limit: int | None = None,
        offset: int = 0,
        retry: bool = False,
    ) -> list[TraceRecord]:
        """Read one public page with its reservation cutoff from the same snapshot."""
        from sqlalchemy import and_, case, false, func, or_, select

        from dendrux._meta_keys import SEED_PREFIX_META_KEY
        from dendrux.db.models import AgentRun, ReactTrace

        visible_seed = (
            and_(
                ReactTrace.trace_kind == "seed",
                func.coalesce(
                    self.json_get(ReactTrace.meta, (SEED_PREFIX_META_KEY,), "bool"), false()
                )
                == false(),
            )
            if not retry
            else false()
        )
        pending_projection = (
            select(
                case(
                    (AgentRun.status.in_(_TERMINAL_STATUS_VALUES), None),
                    else_=self.json_get(AgentRun.loop_state, ("pending",), "json"),
                )
            )
            .where(AgentRun.id == run_id, _run_in_scope(run_id, tenant_id))
            .correlate(None)
            .scalar_subquery()
        )
        stmt = (
            select(ReactTrace, pending_projection)
            .where(
                ReactTrace.agent_run_id == run_id,
                _run_in_scope(run_id, tenant_id),
                ReactTrace.role.in_(("user", "assistant", "tool")),
                or_(ReactTrace.trace_kind == "turn", visible_seed),
            )
            .order_by(ReactTrace.order_index)
            .offset(offset)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._session_factory() as session:
            result = (await session.execute(stmt)).all()
            # A separately fetched checkpoint could expose a late row after a reservation moves.
            pending = (result[0][1] or []) if result else []
            boundary = min((p["order_index"] for p in pending), default=None)
            rows = [r for r, _ in result if boundary is None or r.order_index < boundary]
            return [_sql_support._trace_record(row) for row in rows]

    async def get_traces(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceRecord]:
        """Trace rows of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            stmt = (
                select(ReactTrace)
                .where(ReactTrace.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(ReactTrace.order_index)
            )
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, MAX_PAGE)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                TraceRecord(
                    id=r.id,
                    trace_kind=r.trace_kind,
                    role=r.role,
                    content=r.content,
                    order_index=r.order_index,
                    meta=r.meta,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def get_tool_calls(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[ToolCallReadRecord]:
        """Tool calls of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        _require_aware_utc("before", before)

        from dendrux.db.models import ToolCallRecord

        async with self._session_factory() as session:
            stmt = (
                select(ToolCallRecord)
                .where(ToolCallRecord.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(ToolCallRecord.created_at, ToolCallRecord.id)
            )
            if iteration_index is not None:
                stmt = stmt.where(ToolCallRecord.iteration_index == iteration_index)
            if before is not None:
                stmt = stmt.where(ToolCallRecord.created_at < _to_aware_utc(before))
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, MAX_PAGE)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                ToolCallReadRecord(
                    id=r.id,
                    outcome=r.outcome,
                    observation=(r.meta or {}).get(OBSERVATION_META_KEY),
                    content_digest=r.content_digest,
                    managed_child_run_id=_bound_child(r.meta),
                    idempotent=r.idempotent,
                    replayed_at=_to_aware_utc(r.replayed_at),
                    tool_call_id=r.tool_call_id,
                    provider_tool_call_id=r.provider_tool_call_id,
                    tool_name=r.tool_name,
                    target=r.target,
                    params=r.params,
                    result=r.result,
                    success=r.success,
                    duration_ms=r.duration_ms,
                    iteration_index=r.iteration_index,
                    error_message=r.error_message,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def get_run_statuses(self, run_ids: Sequence[str]) -> dict[str, str]:
        """Read status projections for already-authorized run ids."""
        from dendrux.db.models import AgentRun
        from dendrux.runtime.stores.sql._batched_reads import _validate_run_count

        _validate_run_count(run_ids)
        if not run_ids:
            return {}
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AgentRun.id, AgentRun.status).where(self.id_in(AgentRun.id, run_ids))
                )
            ).all()
        return dict(rows)

    async def list_runs(
        self,
        *,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_waiting: bool = True,
    ) -> list[RunRecord]:
        """Runs newest first, filtered in SQL. Never loads ``pause_data`` or ``pii_mapping``.

        ``limit`` is clamped to 1..1000; ``offset`` above 10,000 raises
        ``ValueError`` (deep OFFSET pages scan every skipped row).
        """
        from sqlalchemy import select
        from sqlalchemy.orm import defer

        from dendrux.db.models import AgentRun

        capped_limit = min(max(1, limit), self._MAX_LIST_LIMIT)
        clamped_offset = _check_offset(offset)

        async with self._session_factory() as session:
            # Secondary sort by id keeps pagination stable when created_at
            # ties — SQLite's DateTime is second-resolution, so concurrent
            # runs share timestamps and offset paging would otherwise skip
            # or duplicate rows across pages.
            stmt = (
                select(AgentRun, _waiting_for_expression(self))
                .options(
                    defer(AgentRun.pause_data, raiseload=True),
                    defer(AgentRun.pii_mapping, raiseload=True),
                )
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
            )
            stmt = self._apply_run_filters(
                stmt,
                tenant_id=tenant_id,
                status=status,
                agent_name=agent_name,
                parent_run_id=parent_run_id,
                started_after=started_after,
                started_before=started_before,
                metadata_filter=metadata_filter,
                include_waiting=include_waiting,
            )
            stmt = stmt.limit(capped_limit).offset(clamped_offset)

            result = await session.execute(stmt)
            return [_run_to_record(row, waiting_for=label) for row, label in result.all()]

    async def count_pauses_per_run(
        self,
        run_ids: list[str],
    ) -> dict[str, int]:
        if not run_ids:
            return {}

        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        async with self._session_factory() as session:
            counts: dict[str, int] = {}
            ids = sorted(set(run_ids))
            for start in range(0, len(ids), _IN_CHUNK):
                stmt = (
                    select(RunEvent.agent_run_id, func.count())
                    .where(RunEvent.agent_run_id.in_(ids[start : start + _IN_CHUNK]))
                    .where(RunEvent.event_type == StoredEventType.RUN_PAUSED.value)
                    .group_by(RunEvent.agent_run_id)
                )
                result = await session.execute(stmt)
                counts.update((row[0], int(row[1])) for row in result.all())
            return counts

    async def count_runs(
        self,
        *,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_waiting: bool = True,
    ) -> int:
        """Count runs matching the ``list_runs`` filters (same tenant semantics)."""
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = select(func.count()).select_from(AgentRun)
            stmt = self._apply_run_filters(
                stmt,
                tenant_id=tenant_id,
                status=status,
                agent_name=agent_name,
                parent_run_id=parent_run_id,
                started_after=started_after,
                started_before=started_before,
                metadata_filter=metadata_filter,
                include_waiting=include_waiting,
            )
            result = await session.execute(stmt)
            total = result.scalar_one()
            return int(total)

    def _apply_run_filters(
        self,
        stmt: Any,
        *,
        tenant_id: TenantScope,
        status: str | list[str] | None,
        agent_name: str | None,
        parent_run_id: str | None,
        started_after: datetime | None,
        started_before: datetime | None,
        metadata_filter: dict[str, Any] | None = None,
        include_waiting: bool = True,
    ) -> Any:
        from sqlalchemy import false

        from dendrux.db.models import AgentRun

        # Naive datetimes here would silently break PG comparisons against
        # TIMESTAMPTZ. Reject at the boundary so future filter additions
        # inherit the guard automatically by routing through this helper.
        _require_aware_utc("started_after", started_after)
        _require_aware_utc("started_before", started_before)

        stmt = _apply_tenant_scope(stmt, tenant_id)
        if not include_waiting:
            stmt = stmt.where(_waiting_for_expression(self).is_(None))
        if status is not None:
            if isinstance(status, list):
                if not status:
                    # Empty list = "match no status" (vs. "no filter at all").
                    stmt = stmt.where(false())
                else:
                    stmt = stmt.where(AgentRun.status.in_(status))
            else:
                stmt = stmt.where(AgentRun.status == status)
        if agent_name is not None:
            stmt = stmt.where(AgentRun.agent_name == agent_name)
        if parent_run_id is not None:
            stmt = self._parent_run_query(stmt.where(AgentRun.parent_run_id == parent_run_id))
        # Normalized to UTC: SQLite stores UTC wall-clock text, so a non-UTC
        # offset would otherwise compare against the wrong instant.
        if started_after is not None:
            stmt = stmt.where(AgentRun.created_at >= _to_aware_utc(started_after))
        if started_before is not None:
            stmt = stmt.where(AgentRun.created_at < _to_aware_utc(started_before))
        if metadata_filter:
            stmt = self._apply_metadata_filter(stmt, metadata_filter)
        return stmt

    def _apply_metadata_filter(self, stmt: Any, metadata_filter: dict[str, Any]) -> Any:
        """Filter by exact-match equality on top-level ``meta`` keys.

        AND across all keys. ``thread_id="abc"`` is the canonical chatbot
        case; arbitrary scalar values (str/int/bool) work too. Nested
        paths (``user.id``) and list-membership checks are intentionally
        unsupported in v1 — add them when a real use case arrives.
        """

        from dendrux.db.models import AgentRun
        from dendrux.types import _validate_persistable

        _validate_persistable(metadata_filter, "metadata_filter")
        for key in metadata_filter:
            if not isinstance(key, str) or not self._METADATA_KEY_RE.match(key):
                raise ValueError(
                    f"metadata_filter keys must match {self._METADATA_KEY_RE.pattern!r}; "
                    f"got {key!r}. Nested paths are not supported in v1."
                )

        return stmt.where(self.metadata_filter_predicate(AgentRun.meta, metadata_filter))


def _bound_child(meta: dict[str, Any] | None) -> str | None:
    """The child a managed intent row was bound to (its ``meta.managed.child_run_id``)."""
    managed = (meta or {}).get("managed")
    child = managed.get("child_run_id") if isinstance(managed, dict) else None
    return child if isinstance(child, str) else None
