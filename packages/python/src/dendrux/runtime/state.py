"""StateStore — persistence interface for agent runs.

Protocol-based so developers can swap in Redis, DynamoDB, or any other
backend. The default implementation uses SQLAlchemy async (SQLite/Postgres).

Usage:
    store = SQLAlchemyStateStore(engine)
    run_id = await store.create_run(...)
    # ... loop runs with observer writing traces/tool_calls/usage ...
    await store.finalize_run(run_id, status=..., answer=..., ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from dendrux.types import UsageStats, generate_ulid

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.db.models import AgentRun


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
    created_at: datetime | None = None
    updated_at: datetime | None = None


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
    ) -> None: ...

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
    ) -> None:
        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

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
            )
            session.add(run)
            await session.commit()

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


def _run_to_record(row: AgentRun) -> RunRecord:
    """Convert an AgentRun ORM object to a RunRecord dataclass."""
    answer = None
    if row.output_data and isinstance(row.output_data, dict):
        answer = row.output_data.get("answer")
    return RunRecord(
        id=row.id,
        agent_name=row.agent_name,
        status=row.status.value if hasattr(row.status, "value") else str(row.status),
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
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
