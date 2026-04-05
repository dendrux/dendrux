"""StateStore — persistence interface for agent runs.

Protocol-based so developers can swap in Redis, DynamoDB, or any other
backend. The default implementation uses SQLAlchemy async (SQLite/Postgres).

Usage:
    store = SQLAlchemyStateStore(engine)
    run_id = await store.create_run(...)
    # ... loop runs with recorder writing traces/tool_calls/usage ...
    await store.finalize_run(run_id, status=..., answer=..., ...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from dendrux.types import (
    CreateRunResult,
    IdempotencyConflictError,
    RunStatus,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.db.models import AgentRun

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Lightweight read model for a persisted run. Used by CLI and queries."""

    id: str
    agent_name: str
    status: str
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    iteration_count: int = 0
    model: str | None = None
    strategy: str | None = None
    parent_run_id: str | None = None
    delegation_level: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    meta: dict[str, Any] | None = None
    last_progress_at: datetime | None = None
    failure_reason: str | None = None
    retry_of_run_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Delegation read models
# ---------------------------------------------------------------------------


@dataclass
class RunBrief:
    """Minimal run reference for delegation payloads."""

    run_id: str
    agent_name: str
    status: str
    delegation_level: int


@dataclass
class ParentRef:
    """Reference to a parent run — may or may not be resolvable.

    ``resolved=False`` means parent_run_id exists on the run row but the
    parent row itself is missing (different store, deleted, etc.).
    ``parent is None`` on DelegationInfo means the run has no parent_run_id.
    """

    run_id: str
    resolved: bool
    agent_name: str | None = None
    status: str | None = None
    delegation_level: int | None = None


@dataclass
class SubtreeSummary:
    """Aggregated metrics for a run and all its descendants.

    All ``subtree_*`` fields and ``status_counts`` include the current run.
    ``descendant_count`` does not — it counts only runs below this one.
    """

    direct_child_count: int
    descendant_count: int
    max_depth: int
    subtree_input_tokens: int
    subtree_output_tokens: int
    subtree_cost_usd: float | None
    unknown_cost_count: int
    status_counts: dict[str, int]


@dataclass
class DelegationInfo:
    """Complete delegation context for a single run.

    Returned by ``StateStore.get_delegation_info()``.
    """

    parent: ParentRef | None
    children: list[RunBrief]
    ancestry: list[RunBrief]
    subtree_summary: SubtreeSummary
    ancestry_complete: bool


@dataclass
class TraceRecord:
    """Lightweight read model for a persisted trace entry."""

    id: str
    role: str
    content: str
    order_index: int
    meta: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass
class ToolCallReadRecord:
    """Lightweight read model for a persisted tool call."""

    id: str
    tool_call_id: str
    provider_tool_call_id: str | None
    tool_name: str
    target: str
    params: dict[str, Any] | None
    result: dict[str, Any] | None
    success: bool
    duration_ms: int | None
    iteration_index: int | None
    error_message: str | None
    created_at: datetime | None = None


@dataclass
class RunEventRecord:
    """Lightweight read model for a persisted run event."""

    id: str
    event_type: str
    sequence_index: int
    iteration_index: int
    correlation_id: str | None = None
    data: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class SweptRun:
    """A run that was swept by the stale-run maintenance API."""

    run_id: str
    agent_name: str
    previous_status: str
    failure_reason: str
    last_progress_at: datetime | None
    swept_at: datetime


@dataclass(frozen=True)
class SweepResult:
    """Result of a sweep operation."""

    stale_running: list[SweptRun]
    abandoned_waiting: list[SweptRun] = field(default_factory=list)


@dataclass
class LLMInteractionRecord:
    """Lightweight read model for a persisted LLM interaction."""

    id: str
    iteration_index: int
    model: str | None = None
    provider: str | None = None
    semantic_request: dict[str, Any] | None = None
    semantic_response: dict[str, Any] | None = None
    provider_request: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int | None = None
    created_at: datetime | None = None


class StateStore(Protocol):
    """Persistence interface for agent runs.

    Implement this protocol to use a custom storage backend.
    The SQLAlchemy implementation is the default.
    """

    async def create_run(
        self,
        run_id: str,
        agent_name: str,
        *,
        input_data: dict[str, Any] | None = None,
        model: str | None = None,
        strategy: str | None = None,
        parent_run_id: str | None = None,
        delegation_level: int = 0,
        tenant_id: str | None = None,
        meta: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
        retry_of_run_id: str | None = None,
    ) -> CreateRunResult: ...

    async def save_trace(
        self,
        run_id: str,
        role: str,
        content: str,
        *,
        order_index: int,
        meta: dict[str, Any] | None = None,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    async def get_llm_interactions(self, run_id: str) -> list[LLMInteractionRecord]: ...

    async def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int = 0,
        total_usage: UsageStats | None = None,
        expected_current_status: str | None = None,
    ) -> bool: ...

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
    ) -> None: ...

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None: ...

    async def claim_paused_run(self, run_id: str, *, expected_status: str) -> bool: ...

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
    ) -> bool:
        """Atomically save submitted data to pause_data and claim the run.

        Merges *submitted_data* into the existing pause_data blob and
        transitions the run from *expected_status* to RUNNING in a single
        conditional UPDATE.  First-writer-wins: returns True if this call
        won the race, False if the status had already changed.

        Used by the bridge for persist-first handoff: the bridge spawns a
        resume task, the task calls this method, and signals back to the
        HTTP handler whether it won.

        Args:
            run_id: The paused run's ID.
            expected_status: Guard value (e.g. "waiting_client_tool").
            submitted_data: Keys to merge into pause_data
                (e.g. ``{"submitted_tool_results": [...]}``)

        Returns:
            True if saved + claimed, False if another caller already won.
        """
        ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def get_traces(self, run_id: str) -> list[TraceRecord]: ...

    async def get_tool_calls(self, run_id: str) -> list[ToolCallReadRecord]: ...

    async def save_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        sequence_index: int = 0,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None: ...

    async def get_run_events(self, run_id: str) -> list[RunEventRecord]: ...

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]: ...

    async def touch_progress(self, run_id: str) -> None:
        """Update last_progress_at to now. Called on forward progress."""
        ...

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
    ) -> list[SweptRun]:
        """Find RUNNING rows older than threshold and mark them ERROR.

        Returns list of swept runs with classification.
        """
        ...

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
    ) -> list[SweptRun]:
        """Find WAITING_CLIENT_TOOL/WAITING_HUMAN_INPUT rows older than
        threshold and mark them ERROR with failure_reason=abandoned_waiting.

        Detection uses updated_at (last state change). Waiting runs have
        no forward progress by design, so last_progress_at is wrong here.

        Returns list of swept runs with previous_status reflecting the
        original waiting status.
        """
        ...

    async def get_delegation_info(self, run_id: str) -> DelegationInfo | None:
        """Get full delegation context for a run.

        Returns parent ref, direct children, ancestry chain, and subtree
        summary with rolled-up metrics. Returns None if the run doesn't exist.
        """
        ...


class SQLAlchemyStateStore:
    """Default StateStore implementation using SQLAlchemy async.

    Works with SQLite (zero-config) and Postgres (via DENDRUX_DATABASE_URL).
    """

    def __init__(self, engine: AsyncEngine) -> None:
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        self._engine = engine
        self._session_factory = sessionmaker(  # type: ignore[call-overload]
            engine, class_=AsyncSession, expire_on_commit=False
        )

    @property
    def store_identity(self) -> str:
        """Comparable identity for this store — the database URL.

        Two stores pointing at the same database have the same identity,
        even if they are different Python objects wrapping different engine
        instances. Used by delegation context to determine safe parent-child
        linking.
        """
        return str(self._engine.url)

    async def create_run(
        self,
        run_id: str,
        agent_name: str,
        *,
        input_data: dict[str, Any] | None = None,
        model: str | None = None,
        strategy: str | None = None,
        parent_run_id: str | None = None,
        delegation_level: int = 0,
        tenant_id: str | None = None,
        meta: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
        retry_of_run_id: str | None = None,
    ) -> CreateRunResult:
        import datetime as _dt

        from sqlalchemy.exc import IntegrityError

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        # Fast path: if an idempotency key is provided, check for existing run
        if idempotency_key is not None:
            existing = await self._check_idempotency(idempotency_key, idempotency_fingerprint or "")
            if existing is not None:
                return existing

        # Insert fresh row
        async with self._session_factory() as session:
            run = AgentRun(
                id=run_id,
                agent_name=agent_name,
                status=AgentRunStatus.RUNNING,
                input_data=input_data,
                model=model,
                strategy=strategy,
                parent_run_id=parent_run_id,
                delegation_level=delegation_level,
                tenant_id=tenant_id,
                meta=meta,
                last_progress_at=_dt.datetime.now(_dt.UTC),
                idempotency_key=idempotency_key,
                idempotency_fingerprint=idempotency_fingerprint,
                retry_of_run_id=retry_of_run_id,
            )
            session.add(run)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                # Only resolve via idempotency if this is the idempotency
                # unique constraint, not an arbitrary integrity error
                if idempotency_key is not None and "idempotency_key" in str(exc):
                    existing = await self._check_idempotency(
                        idempotency_key, idempotency_fingerprint or ""
                    )
                    if existing is not None:
                        return existing
                raise

        return CreateRunResult(
            run_id=run_id,
            outcome="created",
            status=RunStatus.RUNNING,
        )

    async def _check_idempotency(
        self,
        idempotency_key: str,
        fingerprint: str,
    ) -> CreateRunResult | None:
        """Check for existing run with this idempotency key.

        Returns None if no match (caller should create fresh).
        Returns CreateRunResult if match found.
        Raises IdempotencyConflictError if key reused with different input.
        """
        from sqlalchemy import select

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        terminal = {
            AgentRunStatus.SUCCESS,
            AgentRunStatus.ERROR,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.MAX_ITERATIONS,
        }

        async with self._session_factory() as session:
            stmt = select(AgentRun).where(AgentRun.idempotency_key == idempotency_key)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                return None

            # Conflict check: same key, different request
            if row.idempotency_fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    run_id=row.id,
                    idempotency_key=idempotency_key,
                )

            # Classify by status
            status = RunStatus(row.status)
            if row.status in terminal:
                return CreateRunResult(
                    run_id=row.id,
                    outcome="existing_terminal",
                    status=status,
                )
            return CreateRunResult(
                run_id=row.id,
                outcome="existing_active",
                status=status,
            )

    async def save_trace(
        self,
        run_id: str,
        role: str,
        content: str,
        *,
        order_index: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            trace = ReactTrace(
                id=generate_ulid(),
                agent_run_id=run_id,
                role=role,
                content=content,
                order_index=order_index,
                meta=meta,
            )
            session.add(trace)
            await session.commit()

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
    ) -> None:
        from dendrux.db.models import ToolCallRecord

        # Parse result_payload back to dict for JSON column
        try:
            result_dict = json.loads(result_payload)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Tool call result JSON decode failed — run=%s tool=%s payload_len=%d",
                run_id,
                tool_name,
                len(result_payload) if result_payload else 0,
            )
            result_dict = {"raw": result_payload}

        async with self._session_factory() as session:
            record = ToolCallRecord(
                id=generate_ulid(),
                agent_run_id=run_id,
                tool_call_id=tool_call_id,
                provider_tool_call_id=provider_tool_call_id,
                tool_name=tool_name,
                target=target,
                params=params,
                result=result_dict,
                success=success,
                duration_ms=duration_ms,
                iteration_index=iteration_index,
                error_message=error_message,
                meta=meta,
            )
            session.add(record)
            await session.commit()

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
    ) -> None:
        from dendrux.db.models import TokenUsage

        async with self._session_factory() as session:
            record = TokenUsage(
                id=generate_ulid(),
                agent_run_id=run_id,
                iteration_index=iteration_index,
                model=model,
                provider=provider,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                duration_ms=duration_ms,
                meta=meta,
            )
            session.add(record)
            await session.commit()

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
    ) -> None:
        from dendrux.db.models import LLMInteraction

        async with self._session_factory() as session:
            record = LLMInteraction(
                id=generate_ulid(),
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
            )
            session.add(record)
            await session.commit()

    async def get_llm_interactions(self, run_id: str) -> list[LLMInteractionRecord]:
        from sqlalchemy import select

        from dendrux.db.models import LLMInteraction

        async with self._session_factory() as session:
            stmt = (
                select(LLMInteraction)
                .where(LLMInteraction.agent_run_id == run_id)
                .order_by(LLMInteraction.iteration_index)
            )
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
                    created_at=r.created_at,
                )
                for r in rows
            ]

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
    ) -> bool:
        """Finalize a run. Returns True if the update was applied.

        Args:
            expected_current_status: If provided, only updates the row if the
                current DB status matches. Returns False if status has changed
                (e.g. already cancelled). This prevents cancel/finalize races.
        """
        from sqlalchemy import func, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            values: dict[str, Any] = {
                "status": status,
                "updated_at": func.now(),
            }
            if iteration_count is not None:
                values["iteration_count"] = iteration_count
            if answer is not None:
                values["output_data"] = {"answer": answer}
            if error is not None:
                values["error"] = error
            if total_usage is not None:
                values["total_input_tokens"] = total_usage.input_tokens
                values["total_output_tokens"] = total_usage.output_tokens
                values["total_cost_usd"] = total_usage.cost_usd

            # Clear pause_data on finalize (D1: execution state cleaned up)
            values["pause_data"] = None

            # Conditional finalize: only update if status is still 'running'
            # (or expected_status if provided). Prevents cancel/finalize races.
            if expected_current_status is not None:
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_current_status)
                    .values(**values)
                )
            else:
                stmt = update(AgentRun).where(AgentRun.id == run_id).values(**values)
            result = await session.execute(stmt)
            await session.commit()
            return (
                bool(result.rowcount and result.rowcount > 0) if expected_current_status else True
            )

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
    ) -> None:
        """Persist pause state and set WAITING status."""
        from sqlalchemy import func, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            values: dict[str, Any] = {
                "status": status,
                "pause_data": pause_data,
                "updated_at": func.now(),
            }
            if iteration_count is not None:
                values["iteration_count"] = iteration_count
            stmt = update(AgentRun).where(AgentRun.id == run_id).values(**values)
            await session.execute(stmt)
            await session.commit()

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

    async def claim_paused_run(self, run_id: str, *, expected_status: str) -> bool:
        """Atomically transition a paused run to RUNNING.

        Returns True if the claim succeeded (run was in expected_status).
        Returns False if someone else already claimed it or status didn't match.
        Uses UPDATE ... WHERE status=? for atomicity — no race window.
        """
        from sqlalchemy import func, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = (
                update(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                .values(status="running", updated_at=func.now())
            )
            result = await session.execute(stmt)
            await session.commit()
            return bool(result.rowcount and result.rowcount > 0)

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
    ) -> bool:
        """Atomically save submitted data and claim the run.

        Implementation:
          1. Read pause_data + status in a session.
          2. Merge submitted_data into pause_data (Python-side).
          3. Conditional UPDATE: set merged blob + status='running'
             WHERE status still equals *expected_status*.

        The CAS on the status column is the concurrency boundary.
        On SQLite, writes are serialized (single-writer lock) so the
        second writer's UPDATE sees the first writer's committed status
        change and matches 0 rows.  On Postgres, the UPDATE takes a
        row-level lock achieving the same effect.
        """
        from sqlalchemy import func, select, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            # 1. Read current pause_data
            stmt = select(AgentRun.pause_data, AgentRun.status).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            row = result.one_or_none()
            if row is None:
                return False

            current_pause_data, current_status = row
            if current_status != expected_status:
                return False
            if current_pause_data is None:
                return False

            # 2. Merge submitted_data into pause_data
            merged = dict(current_pause_data)
            merged.update(submitted_data)

            # 3. Conditional UPDATE — CAS on status column
            update_stmt = (
                update(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                .values(
                    pause_data=merged,
                    status="running",
                    updated_at=func.now(),
                )
            )
            update_result = await session.execute(update_stmt)
            await session.commit()
            return bool(update_result.rowcount and update_result.rowcount > 0)

    async def get_run(self, run_id: str) -> RunRecord | None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = select(AgentRun).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _run_to_record(row)

    async def get_traces(self, run_id: str) -> list[TraceRecord]:
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            stmt = (
                select(ReactTrace)
                .where(ReactTrace.agent_run_id == run_id)
                .order_by(ReactTrace.order_index)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                TraceRecord(
                    id=r.id,
                    role=r.role,
                    content=r.content,
                    order_index=r.order_index,
                    meta=r.meta,
                    created_at=r.created_at,
                )
                for r in rows
            ]

    async def get_tool_calls(self, run_id: str) -> list[ToolCallReadRecord]:
        from sqlalchemy import select

        from dendrux.db.models import ToolCallRecord

        async with self._session_factory() as session:
            stmt = (
                select(ToolCallRecord)
                .where(ToolCallRecord.agent_run_id == run_id)
                .order_by(ToolCallRecord.created_at)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                ToolCallReadRecord(
                    id=r.id,
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
                    created_at=r.created_at,
                )
                for r in rows
            ]

    async def save_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        sequence_index: int = 0,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        from dendrux.db.models import RunEvent

        async with self._session_factory() as session:
            record = RunEvent(
                id=generate_ulid(),
                agent_run_id=run_id,
                event_type=event_type,
                sequence_index=sequence_index,
                iteration_index=iteration_index,
                correlation_id=correlation_id,
                data=data,
            )
            session.add(record)
            await session.commit()

    async def get_run_events(self, run_id: str) -> list[RunEventRecord]:
        from sqlalchemy import select

        from dendrux.db.models import RunEvent

        async with self._session_factory() as session:
            stmt = (
                select(RunEvent)
                .where(RunEvent.agent_run_id == run_id)
                .order_by(RunEvent.sequence_index)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                RunEventRecord(
                    id=r.id,
                    event_type=r.event_type,
                    sequence_index=r.sequence_index,
                    iteration_index=r.iteration_index,
                    correlation_id=r.correlation_id,
                    data=r.data,
                    created_at=r.created_at,
                )
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Progress tracking & stale-run sweep
    # ------------------------------------------------------------------

    async def touch_progress(self, run_id: str) -> None:
        """Update last_progress_at to now. Lightweight single-column UPDATE."""
        import datetime as _dt

        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = (
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(last_progress_at=_dt.datetime.now(_dt.UTC))
            )
            await session.execute(stmt)
            await session.commit()

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
    ) -> list[SweptRun]:
        """Find RUNNING rows older than threshold and mark them ERROR.

        Detection uses last_progress_at (or created_at if last_progress_at
        is NULL — run never got far enough to record progress).

        Classification is best-effort: absence of a run.started event
        does NOT strictly prove 'never started' because that event is
        emitted through a helper that swallows failures. Useful heuristic
        for operator triage, not authoritative.
        """
        import datetime as _dt

        from sqlalchemy import func, select, update

        from dendrux.db.models import AgentRun, RunEvent

        now = _dt.datetime.now(_dt.UTC)
        cutoff = now - older_than
        swept: list[SweptRun] = []

        async with self._session_factory() as session:
            # Find stale RUNNING rows
            stmt = select(AgentRun).where(
                AgentRun.status == "running",
                # Stale if last_progress_at < cutoff, or if NULL then fall
                # back to created_at < cutoff
                func.coalesce(AgentRun.last_progress_at, AgentRun.created_at) < cutoff,
            )
            result = await session.execute(stmt)
            stale_rows = result.scalars().all()

            if not stale_rows:
                return []

            # Collect run IDs for batch event lookup
            stale_ids = [r.id for r in stale_rows]

            # Check which runs have a run.started event (for classification)
            started_stmt = select(RunEvent.agent_run_id).where(
                RunEvent.agent_run_id.in_(stale_ids),
                RunEvent.event_type == "run.started",
            )
            started_result = await session.execute(started_stmt)
            started_run_ids = {row[0] for row in started_result.all()}

            # Get max sequence_index per run for event ordering
            seq_stmt = (
                select(
                    RunEvent.agent_run_id,
                    func.max(RunEvent.sequence_index).label("max_seq"),
                )
                .where(RunEvent.agent_run_id.in_(stale_ids))
                .group_by(RunEvent.agent_run_id)
            )
            seq_result = await session.execute(seq_stmt)
            max_seqs: dict[str, int] = {row[0]: row[1] for row in seq_result.all()}

            # Process each stale run
            for row in stale_rows:
                classification = "stale_running" if row.id in started_run_ids else "never_started"

                # Mark as ERROR with failure_reason
                update_stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == row.id, AgentRun.status == "running")
                    .values(
                        status="error",
                        failure_reason=classification,
                        error=f"Run swept as {classification}",
                        updated_at=func.now(),
                    )
                )
                update_result = await session.execute(update_stmt)

                # CAS guard: skip if status already changed (race)
                if not (update_result.rowcount and update_result.rowcount > 0):
                    continue

                # Emit run.interrupted event
                next_seq = max_seqs.get(row.id, -1) + 1
                event = RunEvent(
                    id=generate_ulid(),
                    agent_run_id=row.id,
                    event_type="run.interrupted",
                    sequence_index=next_seq,
                    iteration_index=row.iteration_count or 0,
                    data={"failure_reason": classification},
                )
                session.add(event)

                # Commit per-run so partial progress is durable.
                # If run N+1 fails, runs 0..N are already committed.
                await session.commit()

                swept.append(
                    SweptRun(
                        run_id=row.id,
                        agent_name=row.agent_name,
                        previous_status="running",
                        failure_reason=classification,
                        last_progress_at=row.last_progress_at,
                        swept_at=now,
                    )
                )

        return swept

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
    ) -> list[SweptRun]:
        """Find WAITING_CLIENT_TOOL/WAITING_HUMAN_INPUT rows older than
        threshold and mark them ERROR with failure_reason=abandoned_waiting.
        """
        import datetime as _dt

        from sqlalchemy import select, update

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun, RunEvent

        now = _dt.datetime.now(_dt.UTC)
        cutoff = now - older_than

        waiting_statuses = [
            AgentRunStatus.WAITING_CLIENT_TOOL,
            AgentRunStatus.WAITING_HUMAN_INPUT,
        ]

        swept: list[SweptRun] = []

        async with self._session_factory() as session:
            # Find abandoned waiting rows
            stmt = select(AgentRun).where(
                AgentRun.status.in_(waiting_statuses),
                AgentRun.updated_at < cutoff,
            )
            result = await session.execute(stmt)
            abandoned_rows = result.scalars().all()

            if not abandoned_rows:
                return swept

            # Get max sequence_index per run for event ordering
            from sqlalchemy import func as sa_func

            run_ids = [r.id for r in abandoned_rows]
            seq_stmt = (
                select(
                    RunEvent.agent_run_id,
                    sa_func.max(RunEvent.sequence_index),
                )
                .where(RunEvent.agent_run_id.in_(run_ids))
                .group_by(RunEvent.agent_run_id)
            )
            seq_result = await session.execute(seq_stmt)
            max_seqs: dict[str, int] = {row[0]: row[1] for row in seq_result.all()}

            # Process each abandoned run
            for row in abandoned_rows:
                previous_status = row.status

                # CAS-guarded update: only if still in the expected waiting status
                update_stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == row.id, AgentRun.status == previous_status)
                    .values(
                        status=AgentRunStatus.ERROR,
                        failure_reason="abandoned_waiting",
                        updated_at=sa_func.now(),
                    )
                )
                update_result = await session.execute(update_stmt)

                if update_result.rowcount == 0:
                    continue  # Status changed between SELECT and UPDATE

                # Emit run.abandoned event
                next_seq = (max_seqs.get(row.id, -1)) + 1
                max_seqs[row.id] = next_seq

                event = RunEvent(
                    id=generate_ulid(),
                    agent_run_id=row.id,
                    event_type="run.abandoned",
                    sequence_index=next_seq,
                    data={"failure_reason": "abandoned_waiting"},
                )
                session.add(event)

                # Per-run commit for isolation
                await session.commit()

                swept.append(
                    SweptRun(
                        run_id=row.id,
                        agent_name=row.agent_name,
                        previous_status=previous_status,
                        failure_reason="abandoned_waiting",
                        last_progress_at=row.last_progress_at,
                        swept_at=now,
                    )
                )

        return swept

    # ------------------------------------------------------------------
    # Delegation queries
    # ------------------------------------------------------------------

    async def get_delegation_info(self, run_id: str) -> DelegationInfo | None:
        """Get full delegation context for a run.

        Steps: load run → resolve parent → walk ancestry → BFS subtree → return.
        Cycle-safe via visited sets. Cross-DB portable (no recursive CTE).
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            # 1. Load the run
            stmt = select(AgentRun).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            run_row = result.scalar_one_or_none()
            if run_row is None:
                return None

            # 2. Resolve parent ref (or broken-chain marker)
            parent: ParentRef | None = None
            if run_row.parent_run_id:
                p_stmt = select(AgentRun).where(AgentRun.id == run_row.parent_run_id)
                p_result = await session.execute(p_stmt)
                p_row = p_result.scalar_one_or_none()
                if p_row is not None:
                    parent = ParentRef(
                        run_id=p_row.id,
                        resolved=True,
                        agent_name=p_row.agent_name,
                        status=_extract_status(p_row),
                        delegation_level=p_row.delegation_level,
                    )
                else:
                    parent = ParentRef(run_id=run_row.parent_run_id, resolved=False)

            # 3. Walk ancestry upward (root-first)
            ancestry, ancestry_complete = await self._walk_ancestry(session, run_row)

            # 4. BFS subtree downward (collects direct children + rolled-up metrics)
            children, subtree_summary = await self._traverse_subtree_bfs(session, run_row)

            return DelegationInfo(
                parent=parent,
                children=children,
                ancestry=ancestry,
                subtree_summary=subtree_summary,
                ancestry_complete=ancestry_complete,
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

    async def _traverse_subtree_bfs(
        self,
        session: AsyncSession,
        run_row: AgentRun,
    ) -> tuple[list[RunBrief], SubtreeSummary]:
        """BFS downward from run_row. Returns (direct_children, summary).

        Summary includes the root run itself in token/cost/status counts.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        run_id = run_row.id
        seen: set[str] = {run_id}
        current_level = [run_id]
        children: list[RunBrief] = []
        max_depth = 0
        depth = 0

        # Seed metrics with the root run
        sum_input = run_row.total_input_tokens or 0
        sum_output = run_row.total_output_tokens or 0
        sum_cost = 0.0
        has_cost = run_row.total_cost_usd is not None
        unknown_cost_count = 0 if has_cost else 1
        if run_row.total_cost_usd is not None:
            sum_cost = float(run_row.total_cost_usd)
        descendant_count = 0
        status_counts: dict[str, int] = {}
        root_status = _extract_status(run_row)
        status_counts[root_status] = 1
        is_first_level = True

        while current_level:
            stmt = (
                select(AgentRun)
                .where(AgentRun.parent_run_id.in_(current_level))
                .order_by(AgentRun.created_at)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            next_level: list[str] = []
            for cr in rows:
                if cr.id in seen:
                    continue
                seen.add(cr.id)
                next_level.append(cr.id)
                descendant_count += 1
                sum_input += cr.total_input_tokens or 0
                sum_output += cr.total_output_tokens or 0
                if cr.total_cost_usd is not None:
                    sum_cost += float(cr.total_cost_usd)
                    has_cost = True
                else:
                    unknown_cost_count += 1
                s = _extract_status(cr)
                status_counts[s] = status_counts.get(s, 0) + 1
                if is_first_level:
                    children.append(
                        RunBrief(
                            run_id=cr.id,
                            agent_name=cr.agent_name,
                            status=s,
                            delegation_level=cr.delegation_level,
                        )
                    )

            is_first_level = False
            if next_level:
                depth += 1
                max_depth = depth
            current_level = next_level

        summary = SubtreeSummary(
            direct_child_count=len(children),
            descendant_count=descendant_count,
            max_depth=max_depth,
            subtree_input_tokens=sum_input,
            subtree_output_tokens=sum_output,
            subtree_cost_usd=sum_cost if has_cost else None,
            unknown_cost_count=unknown_cost_count,
            status_counts=status_counts,
        )
        return children, summary

    _MAX_LIST_LIMIT = 1000

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        capped_limit = min(max(1, limit), self._MAX_LIST_LIMIT)
        clamped_offset = max(0, offset)

        async with self._session_factory() as session:
            stmt = select(AgentRun).order_by(AgentRun.created_at.desc())
            if tenant_id is not None:
                stmt = stmt.where(AgentRun.tenant_id == tenant_id)
            if status is not None:
                stmt = stmt.where(AgentRun.status == status)
            stmt = stmt.limit(capped_limit).offset(clamped_offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_run_to_record(r) for r in rows]


def _extract_status(row: AgentRun) -> str:
    """Extract status string from an AgentRun, handling enum or raw string."""
    s = row.status
    return s.value if hasattr(s, "value") else str(s)


def _run_to_record(row: AgentRun) -> RunRecord:
    """Convert an AgentRun ORM object to a RunRecord dataclass."""
    answer = None
    if row.output_data and isinstance(row.output_data, dict):
        answer = row.output_data.get("answer")
    return RunRecord(
        id=row.id,
        agent_name=row.agent_name,
        status=_extract_status(row),
        input_data=row.input_data,
        output_data=row.output_data,
        answer=answer,
        error=row.error,
        iteration_count=row.iteration_count,
        model=row.model,
        strategy=row.strategy,
        parent_run_id=row.parent_run_id,
        delegation_level=row.delegation_level,
        total_input_tokens=row.total_input_tokens,
        total_output_tokens=row.total_output_tokens,
        total_cost_usd=float(row.total_cost_usd) if row.total_cost_usd is not None else None,
        meta=row.meta,
        last_progress_at=row.last_progress_at,
        failure_reason=row.failure_reason,
        retry_of_run_id=row.retry_of_run_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
