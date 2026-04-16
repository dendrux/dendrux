"""Public read facade over persisted agent-run state.

``RunStore`` is the stable contract developers use to build dashboards,
monitoring, billing, and any other observation layer on top of Dendrux.

It is read-only. Write paths remain on ``runtime.state.SQLAlchemyStateStore``.
Response types are frozen dataclasses defined in this module; runtime read
records stay internal and are converted at the facade boundary so runtime
internals can evolve without breaking the public contract.

Example
-------
::

    from dendrux.store import RunStore

    store = RunStore.from_database_url("sqlite+aiosqlite:///dendrux.db")

    async for event in store.stream_events(run_id):
        print(event.event_type, event.sequence_index)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux.runtime.state import SQLAlchemyStateStore, StateStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine


class RunNotFoundError(Exception):
    """Raised when streaming is requested for a non-existent run."""


# ---------------------------------------------------------------------------
# Public response dataclasses — frozen, slotted, stable contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """A run row suitable for list views."""

    run_id: str
    agent_name: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    iteration_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    total_cost_usd: float | None
    model: str | None
    parent_run_id: str | None
    delegation_level: int


@dataclass(frozen=True, slots=True)
class RunDetail:
    """A single run's full detail — summary fields plus answer and error."""

    run_id: str
    agent_name: str
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    iteration_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_creation_tokens: int
    total_cost_usd: float | None
    model: str | None
    strategy: str | None
    parent_run_id: str | None
    delegation_level: int
    input_data: dict[str, Any] | None
    answer: str | None
    error: str | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One row from the durable run_events log."""

    sequence_index: int
    iteration_index: int
    event_type: str
    correlation_id: str | None
    data: dict[str, Any] | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One LLM round-trip recorded during a run."""

    iteration: int
    provider: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    cost_usd: float | None
    duration_ms: int | None
    provider_request: dict[str, Any] | None
    provider_response: dict[str, Any] | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One tool invocation recorded during a run."""

    iteration: int
    tool_name: str
    tool_call_id: str
    provider_tool_call_id: str | None
    target: str
    params: dict[str, Any] | None
    result: dict[str, Any] | None
    success: bool
    error: str | None
    duration_ms: int | None
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class PausePair:
    """A pause/resume cycle derived from ``run_events``.

    Never reads ``AgentRun.pause_data`` — that column is execution state,
    potentially unredacted, and cleared on finalize. Public pause history
    is built from the ``run.paused`` and ``run.resumed`` events only.
    """

    pause_sequence_index: int
    pause_at: datetime | None
    resume_sequence_index: int | None
    resume_at: datetime | None
    reason: str
    pending_tool_calls: list[dict[str, Any]]
    submitted_results: list[dict[str, Any]]
    user_input: str | None


# ---------------------------------------------------------------------------
# RunStore — the facade
# ---------------------------------------------------------------------------


class RunStore:
    """Read-only public facade over persisted run state.

    Instantiate via ``RunStore.from_database_url(...)`` for the common case,
    or ``RunStore.from_engine(engine)`` when you already own a SQLAlchemy
    async engine. The primary constructor accepts any ``StateStore`` so
    alternative backends can be injected in tests.
    """

    def __init__(self, state_store: StateStore) -> None:
        self._state = state_store

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> RunStore:
        """Build a RunStore from an existing SQLAlchemy async engine."""
        return cls(SQLAlchemyStateStore(engine))

    @classmethod
    def from_database_url(cls, database_url: str) -> RunStore:
        """Build a RunStore from a SQLAlchemy async database URL."""
        from sqlalchemy.ext.asyncio import create_async_engine

        return cls.from_engine(create_async_engine(database_url))

    async def list_runs(
        self,
        *,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List runs with SQL-side filtering and pagination."""
        records = await self._state.list_runs(
            limit=limit,
            offset=offset,
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=started_after,
            started_before=started_before,
        )
        return [_run_to_summary(r) for r in records]

    async def count_runs(
        self,
        *,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        tenant_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
    ) -> int:
        """Count runs matching the same filters as ``list_runs``."""
        return await self._state.count_runs(
            tenant_id=tenant_id,
            status=status,
            agent_name=agent_name,
            parent_run_id=parent_run_id,
            started_after=started_after,
            started_before=started_before,
        )

    async def get_run(self, run_id: str) -> RunDetail | None:
        """Fetch a single run's detail, or ``None`` if it does not exist."""
        record = await self._state.get_run(run_id)
        if record is None:
            return None
        return _run_to_detail(record)

    async def get_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
    ) -> list[StoredEvent]:
        """Read events for a run in ``sequence_index`` order.

        ``after_sequence_index`` is an exclusive cursor. Pass the previous
        batch's last ``sequence_index`` to continue.
        """
        rows = await self._state.get_run_events(
            run_id,
            after_sequence_index=after_sequence_index,
            limit=limit,
        )
        return [_event_to_stored(r) for r in rows]

    async def stream_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        poll_interval_s: float = 0.5,
    ) -> AsyncIterator[StoredEvent]:
        """Stream events as they land in the durable log.

        Raises :class:`RunNotFound` before the first yield if ``run_id``
        does not exist. The caller decides when to stop — typically by
        breaking out of ``async for`` on a terminal ``event_type`` or by
        calling ``aclose()`` on the returned generator.
        """
        run = await self._state.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"Run '{run_id}' does not exist.")

        cursor = after_sequence_index
        while True:
            rows = await self._state.get_run_events(
                run_id,
                after_sequence_index=cursor,
                limit=100,
            )
            if rows:
                for row in rows:
                    event = _event_to_stored(row)
                    yield event
                    cursor = event.sequence_index
                # If the batch was full, there may be more buffered — keep
                # draining before we sleep.
                if len(rows) == 100:
                    continue
            await asyncio.sleep(poll_interval_s)

    async def get_llm_calls(
        self,
        run_id: str,
        *,
        iteration: int | None = None,
    ) -> list[LLMCall]:
        """Return LLM interactions for a run, optionally filtered by iteration."""
        rows = await self._state.get_llm_interactions(run_id)
        out = [_llm_to_public(r) for r in rows]
        if iteration is not None:
            out = [r for r in out if r.iteration == iteration]
        return out

    async def get_tool_invocations(
        self,
        run_id: str,
        *,
        iteration: int | None = None,
    ) -> list[ToolInvocation]:
        """Return tool invocations for a run, optionally filtered by iteration."""
        rows = await self._state.get_tool_calls(run_id)
        out = [_tool_to_public(r) for r in rows]
        if iteration is not None:
            out = [r for r in out if r.iteration == iteration]
        return out

    async def get_pauses(self, run_id: str) -> list[PausePair]:
        """Derive pause/resume cycles from ``run_events``.

        Pairing is nearest-neighbor: each ``run.paused`` pairs with the
        next ``run.resumed`` in ``sequence_index`` order. An unpaired
        ``run.paused`` at the end of the log represents an active pause
        (``resume_sequence_index`` is ``None``).

        This method never reads ``AgentRun.pause_data``.
        """
        rows = await self._state.get_run_events(run_id)
        return _derive_pauses(rows)


# ---------------------------------------------------------------------------
# Runtime -> public type converters
# ---------------------------------------------------------------------------


def _run_to_summary(record: Any) -> RunSummary:
    return RunSummary(
        run_id=record.id,
        agent_name=record.agent_name,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration_count=record.iteration_count,
        total_input_tokens=record.total_input_tokens,
        total_output_tokens=record.total_output_tokens,
        total_cache_read_tokens=record.total_cache_read_tokens,
        total_cache_creation_tokens=record.total_cache_creation_tokens,
        total_cost_usd=record.total_cost_usd,
        model=record.model,
        parent_run_id=record.parent_run_id,
        delegation_level=record.delegation_level,
    )


def _run_to_detail(record: Any) -> RunDetail:
    return RunDetail(
        run_id=record.id,
        agent_name=record.agent_name,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        iteration_count=record.iteration_count,
        total_input_tokens=record.total_input_tokens,
        total_output_tokens=record.total_output_tokens,
        total_cache_read_tokens=record.total_cache_read_tokens,
        total_cache_creation_tokens=record.total_cache_creation_tokens,
        total_cost_usd=record.total_cost_usd,
        model=record.model,
        strategy=record.strategy,
        parent_run_id=record.parent_run_id,
        delegation_level=record.delegation_level,
        input_data=record.input_data,
        answer=record.answer,
        error=record.error,
        failure_reason=record.failure_reason,
    )


def _event_to_stored(record: Any) -> StoredEvent:
    return StoredEvent(
        sequence_index=record.sequence_index,
        iteration_index=record.iteration_index,
        event_type=record.event_type,
        correlation_id=record.correlation_id,
        data=record.data,
        created_at=record.created_at,
    )


def _llm_to_public(record: Any) -> LLMCall:
    return LLMCall(
        iteration=record.iteration_index,
        provider=record.provider,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.input_tokens + record.output_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        cost_usd=record.cost_usd,
        duration_ms=record.duration_ms,
        provider_request=record.provider_request,
        provider_response=record.provider_response,
        created_at=record.created_at,
    )


def _tool_to_public(record: Any) -> ToolInvocation:
    iteration = record.iteration_index if record.iteration_index is not None else 0
    return ToolInvocation(
        iteration=iteration,
        tool_name=record.tool_name,
        tool_call_id=record.tool_call_id,
        provider_tool_call_id=record.provider_tool_call_id,
        target=record.target,
        params=record.params,
        result=record.result,
        success=record.success,
        error=record.error_message,
        duration_ms=record.duration_ms,
        created_at=record.created_at,
    )


def _derive_pauses(events: list[Any]) -> list[PausePair]:
    """Pair ``run.paused`` with the nearest following ``run.resumed``.

    Known limitation (hardening H-007): pairing is order-based, not
    correlation-id-based. An id-based fix will upgrade this function once
    correlation IDs are added to the pause/resume pair.
    """
    pauses: list[PausePair] = []
    pending: Any | None = None
    for ev in events:
        if ev.event_type == "run.paused":
            if pending is not None:
                pauses.append(_build_pause_pair(pending, None))
            pending = ev
        elif ev.event_type == "run.resumed":
            if pending is not None:
                pauses.append(_build_pause_pair(pending, ev))
                pending = None
    if pending is not None:
        pauses.append(_build_pause_pair(pending, None))
    return pauses


def _build_pause_pair(paused: Any, resumed: Any | None) -> PausePair:
    pdata = paused.data or {}
    rdata = (resumed.data or {}) if resumed is not None else {}
    return PausePair(
        pause_sequence_index=paused.sequence_index,
        pause_at=paused.created_at,
        resume_sequence_index=resumed.sequence_index if resumed is not None else None,
        resume_at=resumed.created_at if resumed is not None else None,
        reason=str(pdata.get("status", "")),
        pending_tool_calls=list(pdata.get("pending_tool_calls") or []),
        submitted_results=list(rdata.get("submitted_results") or []),
        user_input=rdata.get("user_input"),
    )


__all__ = [
    "LLMCall",
    "PausePair",
    "RunDetail",
    "RunNotFoundError",
    "RunStore",
    "RunSummary",
    "StoredEvent",
    "ToolInvocation",
]
