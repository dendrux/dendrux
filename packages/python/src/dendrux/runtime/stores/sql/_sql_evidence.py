"""Maintain durable execution evidence."""

from __future__ import annotations

import datetime as _dt
import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, select

import dendrux.runtime._blob_catalog as _blobs
import dendrux.runtime.state as _state
from dendrux import _codes
from dendrux._meta_keys import LATEST_REWRITE_META_KEY
from dendrux.runtime._bounds import DEFAULT_EVENT_LIMIT, MAX_PAGE
from dendrux.runtime.state import (
    UNSCOPED,
    EventWrite,
    LoopStatePatch,
    TenantScope,
    TraceWrite,
    _apply_loop_patch,
    _finite_json_float,
    _run_in_scope,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.runtime.stores.sql._types import PatchLoopState, SetFields
from dendrux.types import UsageStats, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.media import BlobEdge


class _SqlEvidence(_SqlWrites):
    async def save_trace(
        self,
        run_id: str,
        role: str,
        content: str,
        *,
        order_index: int,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        claim_epoch: int | None = None,
        trace_id: str | None = None,
        trace_kind: str = "turn",
        content_digest: str | None = None,
    ) -> str:
        trace = TraceWrite(
            trace_id or generate_ulid(),
            role,
            content,
            order_index,
            trace_kind,
            meta,
            blob_refs,
            blob_refs_scope,
            content_digest,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            row_id = await self._insert_trace(session, run_id, trace)
            await self._write_tx(session)
            return row_id

    async def save_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        provider_tool_call_id: str | None,
        tool_name: str,
        target: str,
        params: dict[str, Any] | None,
        result_payload: str,
        success: bool,
        duration_ms: int,
        iteration_index: int,
        error_message: str | None = None,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        claim_epoch: int | None = None,
        row_id: str | None = None,
        content_digest: str | None = None,
        outcome: str = "completed",
    ) -> str:
        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest as digest_of

        digest = digest_of(
            dict(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                target=target,
                params=params,
                result_payload=result_payload,
                success=success,
                outcome=outcome,
            )
        )
        if content_digest is not None and content_digest != digest:
            self._write_conflict(run_id)
        try:
            result_dict = json.loads(
                result_payload, parse_constant=_finite_json_float, parse_float=_finite_json_float
            )
        except (ValueError, TypeError):
            logger.warning("Tool call result JSON decode failed", extra={"run_id": run_id})
            result_dict = {"raw": result_payload}
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            saved = await self._insert_retry(
                session,
                ToolCallRecord,
                dict(
                    id=row_id or generate_ulid(),
                    agent_run_id=run_id,
                    tool_call_id=tool_call_id,
                    provider_tool_call_id=provider_tool_call_id,
                    tool_name=tool_name,
                    target=target,
                    params=params,
                    result=result_dict,
                    success=success,
                    outcome=outcome,
                    duration_ms=duration_ms,
                    iteration_index=iteration_index,
                    error_message=error_message,
                    meta=meta,
                    content_digest=digest,
                ),
                digest=digest,
            )
            if blob_refs:
                await _blobs.write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await self._write_tx(session)
            return saved

    async def save_usage(
        self,
        run_id: str,
        *,
        iteration_index: int,
        usage: UsageStats,
        model: str | None = None,
        provider: str | None = None,
        duration_ms: int | None = None,
        meta: dict[str, Any] | None = None,
        usage_id: str | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        from decimal import ROUND_HALF_UP, Decimal

        from dendrux.db.models import TokenUsage

        values = dict(
            id=usage_id or generate_ulid(),
            agent_run_id=run_id,
            iteration_index=iteration_index,
            model=model,
            provider=provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=(
                Decimal(str(usage.cost_usd)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                if usage.cost_usd is not None
                else None
            ),
            duration_ms=duration_ms,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            meta=meta,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            # §12.18: equality covers persisted columns, excluding non-persisted UsageStats fields.
            await self._insert_retry(
                session,
                TokenUsage,
                values,
                equality=(
                    "iteration_index",
                    "model",
                    "provider",
                    "duration_ms",
                    "input_tokens",
                    "output_tokens",
                    "cost_usd",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "reasoning_tokens",
                ),
            )
            await self._write_tx(session)

    async def save_llm_interaction(
        self,
        run_id: str,
        *,
        iteration_index: int,
        usage: UsageStats,
        model: str | None = None,
        provider: str | None = None,
        duration_ms: int | None = None,
        semantic_request: dict[str, Any] | None = None,
        semantic_response: dict[str, Any] | None = None,
        provider_request: dict[str, Any] | None = None,
        provider_response: dict[str, Any] | None = None,
        guardrail_findings: dict[str, Any] | None = None,
        interaction_id: str | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        from dendrux.db.models import LLMInteraction

        values = dict(
            id=interaction_id or generate_ulid(),
            agent_run_id=run_id,
            iteration_index=iteration_index,
            model=model,
            provider=provider,
            semantic_request=semantic_request,
            semantic_response=semantic_response,
            provider_request=provider_request,
            provider_response=provider_response,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            duration_ms=duration_ms,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            guardrail_findings=guardrail_findings,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            await self._insert_retry(
                session,
                LLMInteraction,
                values,
                equality=(
                    "iteration_index",
                    "model",
                    "provider",
                    "semantic_request",
                    "semantic_response",
                    "provider_request",
                    "provider_response",
                ),
            )
            await self._write_tx(session)

    async def _insert_trace(self, session: AsyncSession, run_id: str, trace: TraceWrite) -> str:
        from dendrux.db.models import ReactTrace
        from dendrux.runtime._digest import trace_digest

        _sql_support._validate_trace_write(run_id, trace)
        digest = trace_digest(
            trace.role, trace.content, trace.order_index, trace.trace_kind, trace.meta
        )
        if trace.content_digest is not None and trace.content_digest != digest:
            self._write_conflict(run_id)
        row_id = await self._insert_retry(
            session,
            ReactTrace,
            dict(
                id=trace.trace_id,
                agent_run_id=run_id,
                role=trace.role,
                content=trace.content,
                order_index=trace.order_index,
                trace_kind=trace.trace_kind,
                meta=trace.meta,
                content_digest=digest,
            ),
            digest=digest,
        )
        if trace.trace_kind == "rewrite":
            # A run-owned pointer avoids scanning every conversation row to
            # discover whether a rewrite exists. It commits with the trace.
            from sqlalchemy import update

            from dendrux.db.models import AgentRun

            pointer = {"trace_id": trace.trace_id, "order_index": trace.order_index}
            current_order = self.json_get(
                AgentRun.meta, (LATEST_REWRITE_META_KEY, "order_index"), "int"
            )
            metadata = self.json_patch_expr(
                AgentRun.meta, SetFields({LATEST_REWRITE_META_KEY: pointer})
            )
            if metadata is None:
                current_meta = await session.scalar(
                    self.lock_clause(select(AgentRun.meta).where(AgentRun.id == run_id), "update")
                )
                metadata = {**(current_meta or {}), LATEST_REWRITE_META_KEY: pointer}
            await session.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    (current_order.is_(None)) | (current_order <= trace.order_index),
                )
                .values(meta=metadata)
            )
        if trace.blob_refs:
            await _blobs.write_blob_refs(
                session,
                run_id=run_id,
                blob_refs=trace.blob_refs,
                blob_refs_scope=trace.blob_refs_scope,
            )
        return row_id

    async def checkpoint_step(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        loop_state: dict[str, Any],
        pii_mapping: dict[str, str] | None,
        trace: TraceWrite | None,
    ) -> None:
        """Commit a bounded execution projection and optional trace under one fence."""
        _sql_support._validate_checkpoint(run_id, loop_state)
        values: dict[str, Any] = {
            "loop_state": loop_state,
            **_sql_support._checkpoint_progress(trace),
        }
        if pii_mapping is not None:
            values["pii_mapping"] = pii_mapping
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch, **values)
            if trace is not None:
                await self._insert_trace(session, run_id, trace)
            await self._write_tx(session)

    async def _patch_fence(
        self,
        session: AsyncSession,
        run_id: str,
        claim_epoch: int,
        patch: LoopStatePatch,
        *,
        count_tool: str | None = None,
        pii_mapping: dict[str, str] | None = None,
        progress: bool = False,
    ) -> None:
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunFencedError

        if "pending" in patch.set_fields:
            raise ValueError("A patch cannot replace pending; remove calls by identity")
        _sql_support._validate_checkpoint(run_id, dict(patch.set_fields))
        progress_values = _sql_support._progress_values(progress)
        changed = self.json_patch_expr(AgentRun.loop_state, PatchLoopState(patch, count_tool))
        if changed is not None:
            result = await self._update_one_rows(
                session,
                table=AgentRun.__table__,
                pk=(run_id,),
                predicate=and_(AgentRun.id == run_id, AgentRun.claim_epoch == claim_epoch),
                values=dict(
                    updated_at=_dt.datetime.now(_dt.UTC),
                    last_progress_at=_dt.datetime.now(_dt.UTC),
                    loop_state=changed,
                    **progress_values,
                    **{"pii_mapping": pii_mapping} if pii_mapping is not None else {},
                ),
                returning=[AgentRun.loop_state],
            )
            merged = result.scalar_one_or_none()
            if merged is not None:
                # Same compact UTF-8 measure as checkpoint_step and SQLite.
                # A failed bound rolls back the patch and the whole completion.
                _sql_support._validate_checkpoint(run_id, merged)
                self._remember_run_values(
                    session, run_id, {"loop_state": merged, **progress_values}
                )
            else:
                epoch = await session.scalar(
                    select(AgentRun.claim_epoch).where(AgentRun.id == run_id)
                )
                if epoch == claim_epoch:
                    _sql_support._checkpoint_error(run_id)
                _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
                raise RunFencedError(run_id, claim_epoch=claim_epoch)
        else:
            await self._fence(
                session,
                run_id,
                claim_epoch,
                **progress_values,
                **({"pii_mapping": pii_mapping} if pii_mapping is not None else {}),
            )
            current = await session.scalar(select(AgentRun.loop_state).where(AgentRun.id == run_id))
            changed = _apply_loop_patch(current or {}, patch)
            if count_tool is not None:
                counts = dict(changed.get("tool_counts", {}))
                counts[count_tool] = counts.get(count_tool, 0) + 1
                changed["tool_counts"] = counts
            _sql_support._validate_checkpoint(run_id, changed)
            self._remember_run_values(session, run_id, {"loop_state": changed})
            await session.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(loop_state=changed)
            )

    async def start_tool_call(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        row_id: str,
        tool_call_id: str,
        provider_tool_call_id: str | None,
        tool_name: str,
        target: str,
        params: dict[str, Any] | None,
        iteration_index: int,
        idempotent: bool,
        event: EventWrite,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> None:
        """Commit server-side intent and its event before invoking a tool."""
        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest

        if target == "client":
            raise ValueError("Client calls cannot have execution intents")
        digest = content_digest(
            dict(tool_call_id=tool_call_id, tool_name=tool_name, target=target, params=params)
        )
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch)
            await self._insert_retry(
                session,
                ToolCallRecord,
                dict(
                    id=row_id,
                    agent_run_id=run_id,
                    tool_call_id=tool_call_id,
                    provider_tool_call_id=provider_tool_call_id,
                    tool_name=tool_name,
                    target=target,
                    params=params,
                    iteration_index=iteration_index,
                    outcome="started",
                    result="",
                    success=False,
                    duration_ms=0,
                    idempotent=idempotent,
                    meta=meta,
                    content_digest=digest,
                ),
                digest=digest,
            )
            if blob_refs:
                await _blobs.write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await _state._write_event(session, run_id, event)
            self._mark_announce(session, (run_id,))
            await self._write_tx(session)

    async def complete_tool_call(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        tool_call_id: str,
        outcome: str,
        result_payload: str,
        success: bool,
        duration_ms: int,
        error_message: str | None,
        replayed: bool,
        observation: TraceWrite,
        event: EventWrite,
        loop_state_patch: LoopStatePatch,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        pii_mapping: dict[str, str] | None = None,
    ) -> None:
        """Atomically resolve one intent, its reserved observation and pending entry."""
        from sqlalchemy import select

        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest

        if outcome not in ("completed", "failed", "unknown"):
            raise ValueError("Unsupported tool outcome")
        digest = content_digest(
            dict(
                tool_call_id=tool_call_id,
                outcome=outcome,
                result_payload=result_payload,
                success=success,
                error_message=error_message,
            )
        )
        try:
            payload = json.loads(
                result_payload, parse_constant=_finite_json_float, parse_float=_finite_json_float
            )
        except (ValueError, TypeError):
            payload = {"raw": result_payload}
        values: dict[str, Any] = dict(
            outcome=outcome,
            result=payload,
            success=success,
            duration_ms=duration_ms,
            error_message=error_message,
            content_digest=digest,
        )
        if replayed:
            values["replayed_at"] = _dt.datetime.now(_dt.UTC)
        if meta is not None:
            values["meta"] = meta
        async with self._session_factory() as session:
            await self._patch_fence(
                session,
                run_id,
                claim_epoch,
                loop_state_patch,
                count_tool=(observation.meta or {}).get("tool_name"),
                pii_mapping=pii_mapping,
                progress=_sql_support._advancing_turn(observation),
            )
            row_id = (
                await self._update_many_rows(
                    session,
                    table=ToolCallRecord.__table__,
                    page_size=1,
                    predicate=and_(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.tool_call_id == tool_call_id,
                        ToolCallRecord.outcome == "started",
                    ),
                    values=dict(**values),
                    returning=[ToolCallRecord.id],
                )
            ).scalar_one_or_none()
            if row_id is None:
                stored = await session.scalar(
                    select(ToolCallRecord.content_digest).where(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.tool_call_id == tool_call_id,
                    )
                )
                await session.rollback()
                if stored != digest:
                    self._write_conflict(run_id)
                return
            await self._insert_trace(session, run_id, observation)
            if blob_refs:
                await _blobs.write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await _state._write_event(session, run_id, event)
            self._mark_announce(session, (run_id,))
            await self._write_tx(session)

    async def set_trace_link(self, run_id: str, *, claim_epoch: int, trace_link: str) -> None:
        """Persist a validated traceparent under the execution fence."""
        from dendrux.runtime._tracelink import validate_trace_link

        if validate_trace_link(trace_link) is None:
            _codes.log_code(logger, _codes.TRACE_LINK_INVALID, run_id=run_id)
            raise ValueError("Invalid trace link")
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch, trace_link=trace_link)
            await self._write_tx(session)

    async def get_replayed_tool_calls(
        self, run_id: str, *, limit: int = DEFAULT_EVENT_LIMIT, tenant_id: TenantScope = UNSCOPED
    ) -> tuple[list[str], bool]:
        """Read replayed call ids with one bounded lookahead for truncation."""
        from sqlalchemy import select

        from dendrux.db.models import ToolCallRecord

        _sql_support._positive_page(limit)
        limit = min(limit, MAX_PAGE)
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(ToolCallRecord.tool_call_id)
                    .where(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.replayed_at.is_not(None),
                        _run_in_scope(run_id, tenant_id),
                    )
                    .order_by(ToolCallRecord.replayed_at, ToolCallRecord.id)
                    .limit(limit + 1)
                )
            ).all()
            return list(rows[:limit]), len(rows) > limit
