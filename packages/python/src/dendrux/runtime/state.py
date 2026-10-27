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

import datetime as _dt
import json
import logging
import math
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

from dendrux.runtime._blob_catalog import SQLBlobCatalog, write_blob_refs
from dendrux.runtime.durability import (
    _safe_database_call,
    retry_transient_db,
    safe_database_operations,
)
from dendrux.types import (
    CreateRunResult,
    IdempotencyConflictError,
    RunStatus,
    UsageStats,
    _usage_to_dict,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.db.models import AgentRun
    from dendrux.media import BlobEdge

logger = logging.getLogger(__name__)


class _Unscoped:
    """Type of the ``UNSCOPED`` sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSCOPED"


UNSCOPED = _Unscoped()

# First key of the Postgres advisory lock that serializes note inserts of one
# run under ``Limits.max_notes_per_run`` (second key: ``hashtext(run_id)``).
_NOTE_LOCK_NAMESPACE = 0x44584E54
"""Omitted tenant scope: trusted, no tenant predicate. Distinct from ``None``,
which scopes to untenanted runs only."""

TenantScope = str | None | _Unscoped

PAUSE_REQUEST_META_KEY = "dendrux.pause_request"
"""``agent_runs.meta`` key holding the latest ``request_pause`` reason."""

_TERMINAL_STATUS_VALUES = (
    RunStatus.SUCCESS.value,
    RunStatus.ERROR.value,
    RunStatus.CANCELLED.value,
    RunStatus.MAX_ITERATIONS.value,
)

_COST_DECIMALS = 6  # llm_interactions.cost_usd is Numeric(10, 6)
_IN_CHUNK = 500  # keep IN (...) lists well under SQLite's variable limit


@overload
def _to_aware_utc(dt: None) -> None: ...
@overload
def _to_aware_utc(dt: _dt.datetime) -> _dt.datetime: ...
def _to_aware_utc(dt: _dt.datetime | None) -> _dt.datetime | None:
    """Normalize a datetime to aware-UTC at the store/state boundary.

    - ``None`` → ``None`` (passthrough for nullable columns).
    - Aware datetime → converted to UTC (idempotent if already UTC).
    - Naive datetime → tagged as UTC (assumes stored values are UTC).

    The third case is load-bearing: SQLite + ``DateTime(timezone=True)`` does
    not roundtrip tzinfo through SQLAlchemy, so reads come back naive even
    when the schema asks for TIMESTAMPTZ. Tagging on read keeps the
    invariant — every datetime returned from the store is aware-UTC —
    independent of the backend.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.UTC)
    return dt.astimezone(_dt.UTC)


def _usage_totals(total_usage: UsageStats) -> dict[str, Any]:
    """``agent_runs.total_*`` columns for a run-level usage total."""
    return {
        "total_input_tokens": total_usage.input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "total_cost_usd": total_usage.cost_usd,
        # Cache/reasoning rollup: None treated as 0 in aggregation
        "total_cache_read_tokens": total_usage.cache_read_input_tokens or 0,
        "total_cache_creation_tokens": total_usage.cache_creation_input_tokens or 0,
        "total_reasoning_tokens": total_usage.reasoning_tokens or 0,
    }


def _require_aware_utc(name: str, value: _dt.datetime | None) -> None:
    """Reject naive datetimes at the public-API boundary.

    Counterpart to ``_to_aware_utc``. Reads tolerate naive inputs (legacy
    SQLite storage); writes from callers do not. Naive ``started_after`` /
    ``started_before`` filters compared against ``TIMESTAMPTZ`` columns are
    silently broken on Postgres and silently "work" on SQLite — exactly the
    asymmetry this sprint set out to eliminate. Fail loudly instead.
    """
    if value is None:
        return
    if not _is_aware_datetime(value):
        raise ValueError(
            f"{name} must be a timezone-aware datetime (e.g. datetime.now(UTC)); "
            "Dendrux compares TIMESTAMPTZ columns."
        )


def _is_aware_datetime(value: Any) -> bool:
    return isinstance(value, _dt.datetime) and value.tzinfo is not None


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
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    meta: dict[str, Any] | None = None
    last_progress_at: datetime | None = None
    failure_reason: str | None = None
    retry_of_run_id: str | None = None
    cancel_requested: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tenant_id: str | None = None
    pause_requested: bool = False


@dataclass(frozen=True)
class RunControl:
    """Cooperative control flags a runner reads at each checkpoint.

    ``has_pending_notes`` is True when the run's inbox holds undelivered notes.
    A missing run reads as all False.
    """

    cancel_requested: bool = False
    pause_requested: bool = False
    has_pending_notes: bool = False
    claim_epoch: int | None = None
    status: str | None = None


# Pause-state keys a pending read needs: everything but history, steps and usage.
def _validate_sweep_age(value: timedelta) -> None:
    if not isinstance(value, _dt.timedelta) or value < _dt.timedelta(0):
        raise ValueError("sweep age must be a non-negative timedelta")


PAUSE_SUMMARY_KEYS = (
    "agent_name",
    "pending_tool_calls",
    "pending_targets",
    "approval_gates",
    "pause_reason",
    "pause_message",
    "pause_id",
    "budget_exceeded",
)


_PAUSE_SCALAR_KEYS = frozenset({"agent_name", "pause_reason", "pause_message", "pause_id"})


@dataclass(frozen=True)
class PauseSummary:
    """The parts of a run's pause state that say what it waits for (no history).

    ``pause`` holds :data:`PAUSE_SUMMARY_KEYS` from ``pause_data`` (missing keys
    absent), ``last_step`` the last recorded step (its ``action`` carries a
    clarification question), ``paused_event_sequence`` the index of the latest
    ``run.paused`` event.
    """

    status: str
    pause: dict[str, Any]
    last_step: dict[str, Any] | None = None
    paused_event_sequence: int | None = None
    pii_mapping: dict[str, str] | None = None


@dataclass(frozen=True)
class NoteRecord:
    """Lightweight read model for a ``run_inbox`` row."""

    id: str
    run_id: str
    text: str
    metadata: Any = None
    created_at: datetime | None = None
    delivered_at: datetime | None = None
    delivered_iteration: int | None = None


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
    """Bounded delegation context for a single run.

    Returned by ``StateStore.get_delegation_info()``. When truncated, subtree
    totals cover only visited rows and are lower bounds on the full tree.
    """

    parent: ParentRef | None
    children: list[RunBrief]
    ancestry: list[RunBrief]
    subtree_summary: SubtreeSummary
    ancestry_complete: bool
    truncated: bool = False


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
class UsageRollup:
    """Token and cost totals summed live from ``llm_interactions``.

    Root inclusive when produced by ``subtree_usage``. ``known_cost_usd`` is
    the sum of non-null call costs. ``cost_usd`` equals it only when
    ``unpriced_calls == 0``, else ``None``, so a partial sum never reads as
    the whole spend. Cache/reasoning columns that
    are NULL (provider did not report) count as zero.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    known_cost_usd: float = 0.0
    unpriced_calls: int = 0
    llm_calls: int = 0

    @property
    def cost_usd(self) -> float | None:
        return None if self.unpriced_calls else self.known_cost_usd

    @classmethod
    def combine(cls, rollups: Iterable[UsageRollup]) -> UsageRollup:
        items = list(rollups)
        return cls(
            input_tokens=sum(r.input_tokens for r in items),
            output_tokens=sum(r.output_tokens for r in items),
            cache_read_tokens=sum(r.cache_read_tokens for r in items),
            cache_write_tokens=sum(r.cache_write_tokens for r in items),
            reasoning_tokens=sum(r.reasoning_tokens for r in items),
            known_cost_usd=round(sum(r.known_cost_usd for r in items), _COST_DECIMALS),
            unpriced_calls=sum(r.unpriced_calls for r in items),
            llm_calls=sum(r.llm_calls for r in items),
        )


@dataclass(frozen=True)
class SubtreeNodeRecord:
    """One run in a subtree with its depth below the root and its own live usage."""

    run: RunRecord
    depth: int
    usage: UsageRollup


@dataclass(frozen=True)
class SubtreeRecord:
    """A root run, its descendants (BFS order, root first) and rolled-up usage."""

    nodes: list[SubtreeNodeRecord]
    usage: UsageRollup


@dataclass(frozen=True)
class SweptRun:
    """A run that was swept by the stale-run maintenance API."""

    run_id: str
    agent_name: str
    previous_status: str
    failure_reason: str
    last_progress_at: datetime | None
    swept_at: datetime
    # Status the sweep wrote: "error", or "cancelled" for a stale RUNNING row
    # whose cancellation had been requested.
    status: str = "error"


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
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    created_at: datetime | None = None


class StateStore(Protocol):
    """Persistence interface for agent runs.

    Implement this protocol to use a custom storage backend.
    The SQLAlchemy implementation is the default.

    ``blob_refs`` / ``blob_refs_scope`` (0.4.0) are passed only when a record
    references images; a custom store that never sees images can ignore them.
    Images need the blob catalog, which only the built-in store implements.
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
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> CreateRunResult: ...

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
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
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
        usage_id: str | None = None,
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
        guardrail_findings: dict[str, Any] | None = None,
        interaction_id: str | None = None,
    ) -> None: ...

    async def get_llm_interactions(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
    ) -> list[LLMInteractionRecord]: ...

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
        pii_mapping: dict[str, str] | None = None,
        claim_epoch: int | None = None,
        error_info: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> bool: ...

    async def finalize_run_if_status_in(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Atomic CAS over a set of allowed current statuses.

        Single SQL ``UPDATE ... WHERE status IN (...)`` — no race window.
        Match :meth:`finalize_run` semantics: clears ``pause_data`` to
        ``None`` and ``cancel_requested`` to ``False`` on success.

        Returns:
            True if the row was updated; False if the run's status was
            not in ``allowed_current_statuses`` (someone else won).
        """
        ...

    async def request_cancel(self, run_id: str) -> bool:
        """Set the cooperative cancellation flag on a non-terminal run.

        Returns True if the flag was set (or already True on a
        non-terminal row), False if the row was missing OR already in a
        terminal state. The "non-terminal" check is part of the same
        atomic UPDATE so a finalize racing between the caller's
        preflight and this call cannot leave a stale True on a
        completed/errored/cancelled row.

        Idempotent on non-terminal rows.
        """
        ...

    async def is_cancel_requested(self, run_id: str) -> bool:
        """Read the cooperative cancellation flag.

        Returns False for missing runs.
        """
        ...

    async def get_run_control(self, run_id: str) -> RunControl:
        """Read cancel / pause flags and whether undelivered notes exist, in one read."""
        ...

    async def request_pause(self, run_id: str, *, reason: str | None = None) -> bool:
        """Set ``pause_requested`` on a RUNNING run (atomic, status-guarded).

        Returns False for any other status (WAITING_* / PAUSED / terminal /
        missing): a pause request never latches on a run that is not running.
        ``reason`` is kept under ``meta[PAUSE_REQUEST_META_KEY]`` for the
        runner to copy into the pause.
        """
        ...

    async def try_pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Persist a pause with one CAS: only a RUNNING row without a pending cancel.

        ``UPDATE ... WHERE status='running' AND cancel_requested=false``; also
        clears ``pause_requested`` (any durable pause satisfies a request).
        Returns False when the CAS lost — the caller reloads and honors the
        winning state (cancellation wins over pause).
        """
        ...

    async def pause_with_event(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        event_data: dict[str, Any] | None = None,
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """``try_pause_run`` and the ``run.paused`` event in one transaction.

        ``total_usage`` writes the run's running totals (``total_*``) so a
        paused run does not read as zero spend.

        Returns the event's sequence index, or None when the CAS lost.
        """
        ...

    async def finalize_with_event(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        event_type: str,
        event_data: dict[str, Any] | None = None,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """``finalize_run_if_status_in`` and its event in one transaction.

        Returns the event's sequence index, or None when the CAS lost.
        """
        ...

    async def add_note(
        self,
        run_id: str,
        *,
        note_id: str,
        text: str,
        metadata: Any = None,
        tenant_id: TenantScope = UNSCOPED,
        max_notes: int | None = None,
    ) -> bool:
        """Insert an inbox note only if the run exists, is non-terminal and in scope.

        One conditional ``INSERT ... SELECT`` that share-locks the run row, so
        a concurrent finalize either sees the note or the note sees the
        terminal status. Returns False when nothing was inserted.

        ``max_notes`` (``Limits.max_notes_per_run``): when set, the insert is
        also conditional on the run having fewer notes, and
        :class:`~dendrux.errors.NoteLimitExceededError` is raised when an
        in-scope, non-terminal run has that many already. The count and the
        insert must be atomic: concurrent submitters must never exceed the cap.
        """
        ...

    async def get_notes(
        self,
        run_id: str,
        *,
        undelivered_only: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[NoteRecord]:
        """Inbox notes of a run ordered by ``(created_at, id)``.

        A scoped ``tenant_id`` (incl. ``None``) returns nothing unless the run
        belongs to exactly that tenant.
        """
        ...

    async def mark_notes_delivered(
        self, run_id: str, note_ids: list[str], *, iteration: int
    ) -> int:
        """Stamp ``delivered_at`` / ``delivered_iteration`` on undelivered notes."""
        ...

    async def get_next_event_sequence(self, run_id: str) -> int:
        """Return ``max(sequence_index) + 1`` for the run's events.

        Used by ad-hoc emitters (e.g. ``cancel_run``) that aren't part
        of an ongoing runner sequence but still need to append events
        in monotonic order so SSE clients reading via
        ``after_sequence_index`` don't miss them. Returns 0 for runs
        with no prior events.
        """
        ...

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> None: ...

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None: ...

    async def get_pii_mapping(self, run_id: str) -> dict[str, str] | None: ...

    async def claim_paused_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        expected_pause_id: str | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> int: ...

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
        validate: Callable[[dict[str, Any]], None] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> int:
        """Atomically save submitted data to pause_data and claim the run.

        Merges *submitted_data* into the existing pause_data blob and
        transitions the run from *expected_status* to RUNNING in a single
        conditional UPDATE.  First-writer-wins: returns True if this call
        won the race, False if the status had already changed.

        Used by the agent submit methods for persist-first handoff:
        callers merge ``submitted_data`` here, then invoke ``resume_claimed``
        to re-enter the loop.

        Args:
            run_id: The paused run's ID.
            expected_status: Guard value (e.g. "waiting_client_tool").
            submitted_data: Keys to merge into pause_data
                (e.g. ``{"submitted_tool_results": [...]}``)
            validate: Optional check run on the pause_data being claimed,
                inside the claiming transaction. If it raises, nothing is
                written (the run stays in *expected_status*) and the
                exception propagates. Binds a submission to the exact pause
                generation it was validated against.

        Returns:
            True if saved + claimed, False if another caller already won.
        """
        ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def get_traces(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TraceRecord]: ...

    async def get_tool_calls(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
    ) -> list[ToolCallReadRecord]: ...

    async def save_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> int:
        """Append an event and return the ``sequence_index`` it was given.

        The store allocates the index (``max + 1`` for the run, starting at 0)
        in the same atomic step as the insert, so concurrent writers of one
        run get distinct indexes in commit order. Changed in 0.5.0: callers
        no longer pass ``sequence_index``.
        """
        ...

    async def get_run_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
        event_types: list[str] | None = None,
        before: datetime | None = None,
    ) -> list[RunEventRecord]: ...

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RunRecord]:
        """Runs newest first. ``tenant_id``: ``UNSCOPED`` (default) for every
        tenant, a string for that tenant, ``None`` for untenanted runs only
        (changed in 0.5.0: ``None`` used to mean "no filter")."""
        ...

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
    ) -> int: ...

    async def touch_progress(self, run_id: str) -> None:
        """Update last_progress_at to now. Called on forward progress."""
        ...

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Find RUNNING rows older than threshold and mark them ERROR.

        Inspect at most max_runs candidates (default 10,000, maximum 100,000),
        counting fresh rows and losing CAS attempts. Saturation emits a
        structured sweep_cap_reached warning. Return swept runs with classification.
        """
        ...

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Find WAITING_* and PAUSED rows older than threshold and mark them
        ERROR with failure_reason=abandoned_waiting.

        Candidates use status_changed_at (last status transition). The final
        CAS requires both status_changed_at and updated_at to be stale, protecting
        pauses written by older workers. A recent metadata edit delays abandonment.

        Inspect at most max_runs candidates across waiting statuses (default
        10,000, maximum 100,000), including losing CAS attempts. Saturation emits
        a structured sweep_cap_reached warning. Return swept runs with
        previous_status reflecting the original waiting status.
        """
        ...

    async def get_delegation_info(
        self, run_id: str, *, max_rows: int = 1000, max_depth: int = 32
    ) -> DelegationInfo | None:
        """Get full delegation context for a run.

        Returns parent ref, direct children, ancestry chain, and subtree
        summary with rolled-up metrics. Returns None if the run doesn't exist.
        """
        ...

    async def get_subtree(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> SubtreeRecord | None:
        """Root run plus all descendants with per-node and rolled-up live usage.

        Usage is summed from ``llm_interactions`` so running and paused runs
        count. ``tenant_id`` omitted (``UNSCOPED``) applies no tenant
        predicate. A string keeps only that tenant's runs; explicit ``None``
        keeps only untenanted runs. Other-scope nodes (and everything below
        them) are excluded and a root outside the scope reads as missing.
        Returns None if the run doesn't exist.
        """
        ...

    async def subtree_usage(self, run_id: str) -> UsageRollup:
        """Live usage rolled up over a run and all its descendants."""
        ...

    async def run_usage(self, run_id: str) -> UsageRollup:
        """Live usage of one run only (no descendants)."""
        ...


class _SerializedSessionFactory:
    """Session factory that holds the pool's session lock for each session's life.

    The session is closed (ROLLBACK and pool reset included) before the lock
    is released, even when the caller is cancelled during the close.
    """

    def __init__(self, engine: AsyncEngine, factory: Callable[[], AsyncSession]) -> None:
        self._engine = engine
        engine.sync_engine.hide_parameters = True
        self._factory = factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        from dendrux.db._pool_guard import close_fully, serialized

        async with serialized(self._engine):
            session = self._factory()
            try:
                yield session
            finally:
                await close_fully(session.close)


def _status_values(status: Any, *, initial: bool = False) -> dict[str, Any]:
    from sqlalchemy import case

    from dendrux.db.models import AgentRun

    now = _dt.datetime.now(_dt.UTC)
    return {
        "status": status,
        "status_changed_at": now
        if initial
        else case((AgentRun.status != status, now), else_=AgentRun.status_changed_at),
    }


def _check_create_identifiers(
    run_id: str,
    agent_name: str,
    tenant_id: str | None,
    idempotency_key: str | None,
    model: str | None,
) -> None:
    from dendrux.types import _validate_identifier

    _validate_identifier(run_id, "run_id", maximum=26)
    _validate_identifier(agent_name, "agent_name")
    for name, optional in (
        ("tenant_id", tenant_id),
        ("idempotency_key", idempotency_key),
        ("model", model),
    ):
        if optional is not None:
            _validate_identifier(optional, name)


_PERSISTED_VALUE_FIELDS = frozenset(
    {
        "content",
        "params",
        "result_payload",
        "error_message",
        "meta",
        "patch",
        "pause_data",
        "submitted_data",
        "event_data",
        "data",
        "text",
        "metadata",
        "reason",
        "answer",
        "error",
        "pii_mapping",
        "error_info",
        "failure_reason",
        "semantic_request",
        "semantic_response",
        "provider_request",
        "provider_response",
        "guardrail_findings",
    }
)


def _validated_store_write(method: Any) -> Any:
    from functools import wraps
    from inspect import signature

    sig = signature(method)

    @wraps(method)
    async def write(*args: Any, **kwargs: Any) -> Any:
        from dendrux.types import _validate_persistable

        values = sig.bind(*args, **kwargs).arguments
        for name in _PERSISTED_VALUE_FIELDS.intersection(values):
            _validate_persistable(values[name], name)
        return await method(*args, **kwargs)

    return write


def _validated_store_writes(cls: Any) -> Any:
    for name, method in list(vars(cls).items()):
        if name.startswith(
            (
                "save_",
                "create_",
                "finalize_",
                "pause_",
                "try_pause_",
                "submit_",
                "add_note",
                "update_run_meta",
                "request_pause",
            )
        ):
            setattr(cls, name, _validated_store_write(method))
    return cls


def _readable_json_document(column: Any) -> Any:
    from sqlalchemy import JSON, Text, cast, func

    # Older plain-JSON columns can contain escaped NUL. Preserve escaped
    # backslashes while replacing only an active NUL escape before extraction.
    cleaned = func.regexp_replace(
        cast(column, Text), r"((?:^|[^\\])(?:\\\\)*)\\u0000", r"\1\\ufffd", "g"
    )
    return cast(cleaned, JSON)


@safe_database_operations
@_validated_store_writes
class SQLAlchemyStateStore(SQLBlobCatalog):
    """Default StateStore implementation using SQLAlchemy async.

    Works with SQLite (zero-config) and Postgres (via DENDRUX_DATABASE_URL).
    Also the blob catalog (``dendrux.media.BlobCatalog``) for image blobs.

    ``create_run``, ``save_trace``, ``save_tool_call`` and ``submit_and_claim``
    take optional ``blob_refs`` + ``blob_refs_scope``: the record's blob edges,
    written in the record's transaction after asserting the run's tenant and
    claiming each blob row (``write_blob_refs``). Passed only when non-empty.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        from dendrux.db._pool_guard import shares_one_connection, warn_if_shared_connection

        self._engine = engine
        self._session_factory = sessionmaker(  # type: ignore[call-overload]
            engine, class_=AsyncSession, expire_on_commit=False
        )
        if shares_one_connection(engine):
            # One shared connection = one shared transaction: overlapping sessions
            # would roll back each other's pending writes (see db/_pool_guard.py).
            warn_if_shared_connection(engine)
            self._session_factory = _SerializedSessionFactory(engine, self._session_factory)

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
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> CreateRunResult:
        from dendrux.types import _validate_persistable

        _check_create_identifiers(run_id, agent_name, tenant_id, idempotency_key, model)
        _validate_persistable(meta, "metadata")
        _validate_persistable(input_data, "input")
        import datetime as _dt

        from sqlalchemy.exc import IntegrityError

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        if tenant_id == "":
            raise ValueError("tenant_id must be a non-empty string or None, not ''.")
        # Fast path: if an idempotency key is provided, check for an existing
        # run of this tenant (keys are unique per tenant, not globally).
        if idempotency_key is not None:
            existing = await self._check_idempotency(
                idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
            )
            if existing is not None:
                return existing

        async def _attempt() -> CreateRunResult:
            async with self._session_factory() as session:
                run = AgentRun(
                    id=run_id,
                    agent_name=agent_name,
                    **_status_values(AgentRunStatus.RUNNING, initial=True),
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
                    if blob_refs:
                        await session.flush()
                        await write_blob_refs(
                            session,
                            run_id=run_id,
                            blob_refs=blob_refs,
                            blob_refs_scope=blob_refs_scope,
                            run_tenant_id=tenant_id,
                        )
                    await session.commit()
                except IntegrityError as exc:
                    await session.rollback()
                    # Only resolve via idempotency when the violated constraint is
                    # an idempotency-key unique index, not an arbitrary integrity
                    # error. IntegrityError is NOT transient — don't let
                    # retry_transient_db see it.
                    if idempotency_key is None or not _is_idempotency_violation(exc):
                        raise
                    existing = await self._check_idempotency(
                        idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
                    )
                    if existing is not None:
                        return existing
                    # The key is held by another tenant's run under a pre-0.5
                    # global unique index (legacy SQLite file). Fail closed
                    # without disclosing that run.
                    logger.warning(
                        "idempotency key held by another tenant under a pre-0.5 global "
                        "unique index: run_id=%s code=idempotency_conflict_legacy",
                        run_id,
                        extra={"run_id": run_id, "code": "idempotency_conflict_legacy"},
                    )
                    raise IdempotencyConflictError(
                        run_id=None, idempotency_key=idempotency_key
                    ) from None

            return CreateRunResult(
                run_id=run_id,
                outcome="created",
                status=RunStatus.RUNNING,
            )

        return await retry_transient_db(_attempt, label="create_run", run_id=run_id)

    async def _check_idempotency(
        self,
        idempotency_key: str,
        fingerprint: str,
        *,
        tenant_id: str | None = None,
    ) -> CreateRunResult | None:
        """Check for an existing run of ``tenant_id`` with this idempotency key.

        Keys are scoped per tenant (``None`` is the untenanted scope, stored
        as ``''`` in the index), so another tenant's run is never returned.
        The predicate spells the index expression so it uses
        ``uq_agent_runs_tenant_idempotency_key``.

        Returns None if no match (caller should create fresh).
        Returns CreateRunResult if match found.
        Raises IdempotencyConflictError if key reused with different input.
        """
        from sqlalchemy import func, literal_column, select

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        terminal = {
            AgentRunStatus.SUCCESS,
            AgentRunStatus.ERROR,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.MAX_ITERATIONS,
        }

        async with self._session_factory() as session:
            stmt = select(AgentRun).where(
                func.coalesce(AgentRun.tenant_id, literal_column("''")) == (tenant_id or ""),
                AgentRun.idempotency_key == idempotency_key,
            )
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
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
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
            if blob_refs:
                await session.flush()
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
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
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> None:
        from dendrux.db.models import ToolCallRecord

        # Non-JSON text is stored as {"raw": original_payload}, as in 0.4.1.
        # Any NaN/Infinity or overflowing float anywhere in a result takes
        # that fallback for the entire payload; quoted strings remain JSON.
        try:
            result_dict = json.loads(
                result_payload, parse_constant=_finite_json_float, parse_float=_finite_json_float
            )
        except (ValueError, TypeError):
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
            if blob_refs:
                await session.flush()
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
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
        usage_id: str | None = None,
    ) -> None:
        from dendrux.db.models import TokenUsage

        async with self._session_factory() as session:
            values = dict(
                id=usage_id or generate_ulid(),
                agent_run_id=run_id,
                iteration_index=iteration_index,
                model=model,
                provider=provider,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
                duration_ms=duration_ms,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                meta=meta,
            )
            if usage_id is not None and self._engine.dialect.name in {"sqlite", "postgresql"}:
                stmt: Any
                if self._engine.dialect.name == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert as pg_insert

                    stmt = (
                        pg_insert(TokenUsage)
                        .values(**values)
                        .on_conflict_do_nothing(index_elements=["id"])
                    )
                else:
                    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                    stmt = (
                        sqlite_insert(TokenUsage)
                        .values(**values)
                        .on_conflict_do_nothing(index_elements=["id"])
                    )
                await session.execute(stmt)
            else:
                session.add(TokenUsage(**values))
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
        guardrail_findings: dict[str, Any] | None = None,
        interaction_id: str | None = None,
    ) -> None:
        from dendrux.db.models import LLMInteraction

        async with self._session_factory() as session:
            record = LLMInteraction(
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
            session.add(record)
            await session.commit()

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
                stmt = stmt.limit(max(1, min(limit, 1000)))
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
        pii_mapping: dict[str, str] | None = None,
        claim_epoch: int | None = None,
        error_info: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        """Finalize a run. Returns True if the update was applied.

        Args:
            expected_current_status: If provided, only updates the row if the
                current DB status matches. Returns False if status has changed
                (e.g. already cancelled). This prevents cancel/finalize races.
            pii_mapping: Final guardrail PII mapping. Persists past finalize
                (audit-first — framework never auto-clears).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_status_values(status),
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if answer is not None or total_usage is not None:
                    values["output_data"] = await _finalized_output_data(
                        session, run_id, answer, total_usage
                    )
                if error is not None:
                    values["error"] = error
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping

                if failure_reason is not None:
                    values["failure_reason"] = failure_reason
                # Clear pause_data on finalize (D1: execution state cleaned up)
                values["pause_data"] = None
                # Clear cooperative cancel flag — terminal state is final;
                # a stale True here would confuse later audits.
                values["cancel_requested"] = False
                values["pause_requested"] = False

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
                if claim_epoch is not None:
                    stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
                result = await session.execute(stmt)
                if result.rowcount and error_info is not None:
                    from sqlalchemy import select

                    old_meta = (
                        await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                    ).scalar_one_or_none()
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**(old_meta or {}), "dendrux.error_info": error_info})
                    )
                await session.commit()
                return (
                    bool(result.rowcount and result.rowcount > 0)
                    if expected_current_status or claim_epoch is not None
                    else True
                )

        return await retry_transient_db(_attempt, label="finalize_run", run_id=run_id)

    async def finalize_run_if_status_in(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Atomic CAS finalize over multiple acceptable current statuses.

        One SQL ``UPDATE ... WHERE status IN (...)`` — collapses what
        used to be N sequential CAS attempts into a single round-trip.
        Mirrors :meth:`finalize_run`'s side-effects: clears ``pause_data``
        and ``cancel_requested``.
        """

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                await session.commit()
                return won

        return await retry_transient_db(_attempt, label="finalize_run_if_status_in", run_id=run_id)

    async def finalize_with_event(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        event_type: str,
        event_data: dict[str, Any] | None = None,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """:meth:`finalize_run_if_status_in` plus its lifecycle event, in one transaction.

        The event's sequence index is allocated after the CAS took the row
        lock, so nothing can observe the new status before the event is
        durable and no two transition events share an index. Returns the
        event's sequence index, or None when the CAS lost (nothing written).
        """

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                if not won:
                    await session.rollback()
                    return None
                seq = await _insert_event_locked(session, run_id, event_type, event_data)
                await session.commit()
                return seq

        return await retry_transient_db(_attempt, label="finalize_with_event", run_id=run_id)

    async def _cas_finalize(
        self,
        session: AsyncSession,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None,
        error: str | None,
        iteration_count: int | None,
        total_usage: UsageStats | None,
        pii_mapping: dict[str, str] | None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        # updated_at is filled by the model's onupdate hook (Python aware-UTC).
        values: dict[str, Any] = {
            **_status_values(status),
            "pause_data": None,
            "cancel_requested": False,
            "pause_requested": False,
        }
        if iteration_count is not None:
            values["iteration_count"] = iteration_count
        if answer is not None or total_usage is not None:
            values["output_data"] = await _finalized_output_data(
                session, run_id, answer, total_usage
            )
        if error is not None:
            values["error"] = error
        if total_usage is not None:
            values.update(_usage_totals(total_usage))
        if pii_mapping is not None:
            values["pii_mapping"] = pii_mapping
        if failure_reason is not None:
            values["failure_reason"] = failure_reason

        stmt = update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status.in_(allowed_current_statuses)
        )
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        result: Any = await session.execute(stmt.values(**values))
        return bool(result.rowcount and result.rowcount > 0)

    async def request_cancel(self, run_id: str) -> bool:
        """Set ``cancel_requested=True`` on a non-terminal run.

        The terminal-status guard is part of the SQL UPDATE — no race
        window between read and write.
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                    )
                    .values(cancel_requested=True)
                )
                result = await session.execute(stmt)
                await session.commit()
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(_attempt, label="request_cancel", run_id=run_id)

    async def update_run_meta(self, run_id: str, patch: dict[str, Any]) -> None:
        """Shallow-merge ``patch`` into ``agent_runs.meta`` for ``run_id``.

        Read-modify-write in one transaction; only Dendrux-owned keys
        (``dendrux.*``) are written through here. No-op for missing runs.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        select(AgentRun).where(AgentRun.id == run_id).with_for_update()
                    )
                ).scalar_one_or_none()
                if row is None:
                    return
                row.meta = {**(row.meta or {}), **patch}
                await session.commit()

        await retry_transient_db(_attempt, label="update_run_meta", run_id=run_id)

    async def is_cancel_requested(self, run_id: str) -> bool:
        """Read ``cancel_requested`` flag. False for missing runs."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                stmt = select(AgentRun.cancel_requested).where(AgentRun.id == run_id)
                result = await session.execute(stmt)
                row = result.first()
                return bool(row[0]) if row is not None else False

        return await retry_transient_db(_attempt, label="is_cancel_requested", run_id=run_id)

    async def get_run_control(self, run_id: str) -> RunControl:
        """One read of cancel / pause flags plus an EXISTS over undelivered notes."""
        from sqlalchemy import literal, select

        from dendrux.db.models import AgentRun, RunInbox

        pending = (
            select(literal(1))
            .where(RunInbox.run_id == AgentRun.id, RunInbox.delivered_at.is_(None))
            .exists()
        )

        async def _attempt() -> RunControl:
            async with self._session_factory() as session:
                stmt = select(
                    AgentRun.cancel_requested,
                    AgentRun.pause_requested,
                    pending,
                    AgentRun.claim_epoch,
                    AgentRun.status,
                ).where(AgentRun.id == run_id)
                row = (await session.execute(stmt)).first()
                if row is None:
                    return RunControl()
                return RunControl(
                    cancel_requested=bool(row[0]),
                    pause_requested=bool(row[1]),
                    has_pending_notes=bool(row[2]),
                    claim_epoch=row[3],
                    status=row[4],
                )

        return await retry_transient_db(_attempt, label="get_run_control", run_id=run_id)

    async def request_pause(self, run_id: str, *, reason: str | None = None) -> bool:
        """Set ``pause_requested`` on a RUNNING row; status guard in the UPDATE."""
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == RunStatus.RUNNING.value)
                    .values(pause_requested=True)
                )
                if not result.rowcount:
                    await session.rollback()
                    return False
                # The UPDATE holds the row lock; record the reason (or clear a
                # stale one) under it.
                meta = (
                    await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                ).scalar_one_or_none() or {}
                if reason is not None or PAUSE_REQUEST_META_KEY in meta:
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**meta, PAUSE_REQUEST_META_KEY: {"reason": reason}})
                    )
                await session.commit()
                return True

        return await retry_transient_db(_attempt, label="request_pause", run_id=run_id)

    async def try_pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """CAS pause: RUNNING and no pending cancel, else False (caller honors winner)."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .values(**values)
                )
                await session.commit()
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(_attempt, label="try_pause_run", run_id=run_id)

    async def pause_with_event(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        event_data: dict[str, Any] | None = None,
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """:meth:`try_pause_run` plus the ``run.paused`` event, in one transaction.

        A finalizer of paused runs (``cancel_run``, the sweeper) can only act
        once this commits, so it always sees ``run.paused`` and allocates a
        later sequence index. Returns the event's sequence index, or None
        when the CAS lost (a pending cancel or a non-RUNNING row).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .values(**values)
                )
                if not result.rowcount:
                    await session.rollback()
                    return None
                seq = await _insert_event_locked(
                    session,
                    run_id,
                    "run.paused",
                    event_data,
                    iteration_index=iteration_count or 0,
                )
                await session.commit()
                return seq

        return await retry_transient_db(_attempt, label="pause_with_event", run_id=run_id)

    async def add_note(
        self,
        run_id: str,
        *,
        note_id: str,
        text: str,
        metadata: Any = None,
        tenant_id: TenantScope = UNSCOPED,
        max_notes: int | None = None,
    ) -> bool:
        """Conditional ``INSERT ... SELECT`` guarded by status and tenant, row share-locked.

        With ``max_notes`` the statement also requires ``COUNT(notes) < max_notes``.
        On Postgres a transaction-scoped advisory lock per run is taken first,
        so concurrent note submitters of one run count and insert one at a time
        (each statement sees the previous commit) without blocking the runner's
        own writes to the run row. SQLite serializes writers already.
        """
        import datetime as _dt

        from sqlalchemy import (
            JSON,
            DateTime,
            Integer,
            String,
            Text,
            func,
            insert,
            literal,
            null,
            select,
        )

        from dendrux.db.models import AgentRun, RunInbox
        from dendrux.errors import NoteLimitExceededError

        if max_notes is not None and (isinstance(max_notes, bool) or max_notes < 1):
            raise ValueError("max_notes must be a positive int or None.")

        def _note_count() -> Any:
            return select(func.count()).select_from(RunInbox).where(RunInbox.run_id == run_id)

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                if max_notes is not None and self._engine.dialect.name == "postgresql":
                    await session.execute(
                        select(
                            func.pg_advisory_xact_lock(
                                literal(_NOTE_LOCK_NAMESPACE, Integer()), func.hashtext(run_id)
                            )
                        )
                    )
                source = select(
                    literal(note_id, String(26)),
                    AgentRun.id,
                    literal(text, Text()),
                    null() if metadata is None else literal(metadata, JSON()),
                    literal(_dt.datetime.now(_dt.UTC), DateTime(timezone=True)),
                ).where(
                    AgentRun.id == run_id,
                    AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                )
                if max_notes is not None:
                    source = source.where(_note_count().scalar_subquery() < max_notes)
                # FOR SHARE (Postgres; a no-op on SQLite, whose writes are
                # serialized): a concurrent finalize waits for this insert, or
                # this insert re-checks the status the finalize committed.
                source = _apply_tenant_scope(source, tenant_id).with_for_update(read=True)
                result = await session.execute(
                    insert(RunInbox).from_select(
                        [
                            RunInbox.id,
                            RunInbox.run_id,
                            RunInbox.text,
                            RunInbox.metadata_,
                            RunInbox.created_at,
                        ],
                        source,
                    )
                )
                inserted = bool(result.rowcount and result.rowcount > 0)
                if not inserted and max_notes is not None:
                    # Still under the lock: tell "full" from missing/terminal/foreign.
                    status = await session.scalar(
                        _apply_tenant_scope(
                            select(AgentRun.status).where(AgentRun.id == run_id), tenant_id
                        )
                    )
                    full = (await session.scalar(_note_count()) or 0) >= max_notes
                    if status is not None and status not in _TERMINAL_STATUS_VALUES and full:
                        await session.rollback()
                        logger.warning(
                            "note limit reached: run_id=%s code=note_limit limit=%d",
                            run_id,
                            max_notes,
                            extra={"run_id": run_id, "code": "note_limit"},
                        )
                        raise NoteLimitExceededError(run_id, limit=max_notes)
                await session.commit()
                return inserted

        return await retry_transient_db(_attempt, label="add_note", run_id=run_id)

    async def get_notes(
        self,
        run_id: str,
        *,
        undelivered_only: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[NoteRecord]:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun, RunInbox

        async with self._session_factory() as session:
            stmt = select(RunInbox).where(RunInbox.run_id == run_id)
            if not isinstance(tenant_id, _Unscoped):
                owner = _apply_tenant_scope(
                    select(AgentRun.id).where(AgentRun.id == run_id), tenant_id
                )
                stmt = stmt.where(owner.exists())
            if undelivered_only:
                stmt = stmt.where(RunInbox.delivered_at.is_(None))
            rows = (
                (await session.execute(stmt.order_by(RunInbox.created_at, RunInbox.id)))
                .scalars()
                .all()
            )
            return [
                NoteRecord(
                    id=r.id,
                    run_id=r.run_id,
                    text=r.text,
                    metadata=r.metadata_,
                    created_at=_to_aware_utc(r.created_at),
                    delivered_at=_to_aware_utc(r.delivered_at),
                    delivered_iteration=r.delivered_iteration,
                )
                for r in rows
            ]

    async def mark_notes_delivered(
        self, run_id: str, note_ids: list[str], *, iteration: int
    ) -> int:
        import datetime as _dt

        from sqlalchemy import update

        from dendrux.db.models import RunInbox

        if not note_ids:
            return 0

        async def _attempt() -> int:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(RunInbox)
                    .where(
                        RunInbox.run_id == run_id,
                        RunInbox.id.in_(note_ids),
                        RunInbox.delivered_at.is_(None),
                    )
                    .values(
                        delivered_at=_dt.datetime.now(_dt.UTC),
                        delivered_iteration=iteration,
                    )
                )
                await session.commit()
                return int(result.rowcount or 0)

        return await retry_transient_db(_attempt, label="mark_notes_delivered", run_id=run_id)

    async def get_next_event_sequence(self, run_id: str) -> int:
        """Return ``max(sequence_index) + 1`` for the run, or 0 if none."""
        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        async def _attempt() -> int:
            async with self._session_factory() as session:
                stmt = select(func.max(RunEvent.sequence_index)).where(
                    RunEvent.agent_run_id == run_id
                )
                row = (await session.execute(stmt)).first()
                if row is None or row[0] is None:
                    return 0
                return int(row[0]) + 1

        return await retry_transient_db(_attempt, label="get_next_event_sequence", run_id=run_id)

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        """Persist pause state and set the paused status.

        ``total_usage`` writes the run's running totals (``total_*``).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                stmt = update(AgentRun).where(AgentRun.id == run_id).values(**values)
                if claim_epoch is not None:
                    stmt = stmt.where(
                        AgentRun.claim_epoch == claim_epoch, AgentRun.status == "running"
                    )
                await session.execute(stmt)
                await session.commit()

        await retry_transient_db(_attempt, label="pause_run", run_id=run_id)

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

    async def get_active_pause_sequence(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> int | None:
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun, RunEvent

        seq = (
            select(func.max(RunEvent.sequence_index))
            .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
            .scalar_subquery()
        )
        invalidated = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(
                    ["run.resumed", "run.completed", "run.error", "run.cancelled"]
                ),
            )
            .scalar_subquery()
        )
        async with self._session_factory() as session:
            value = (
                await session.execute(
                    _apply_tenant_scope(
                        select(seq).where(
                            AgentRun.id == run_id,
                            seq > invalidated,
                            AgentRun.status.in_(
                                [
                                    "waiting_client_tool",
                                    "waiting_human_input",
                                    "waiting_approval",
                                    "paused",
                                ]
                            ),
                        ),
                        tenant_id,
                    )
                )
            ).scalar_one_or_none()
            return None if value is None else int(value)

    async def get_approval_params(
        self, run_id: str, pause_sequence: int, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, dict[str, Any]]:
        """Latest recorded model args per call in this pause's event interval.

        Read at most 1000 approval requests. Missing/overflowed calls fail closed
        in the display layer when PII mapping exists.
        """
        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        boundary = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(["run.paused", "run.resumed"]),
                RunEvent.sequence_index < pause_sequence,
            )
            .scalar_subquery()
        )
        stmt = (
            select(RunEvent.data)
            .where(
                RunEvent.agent_run_id == run_id,
                _run_in_scope(run_id, tenant_id),
                RunEvent.event_type == "approval.requested",
                RunEvent.sequence_index > boundary,
                RunEvent.sequence_index < pause_sequence,
            )
            .order_by(RunEvent.sequence_index.desc())
            .limit(1000)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        result: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for data in rows:
            data = _approval_event_object(data)
            call_id = data.get("call_id")
            if isinstance(call_id, str) and call_id not in seen:
                seen.add(call_id)
                if isinstance(data.get("params"), dict):
                    result[call_id] = data["params"]
        return result

    async def get_pause_summary(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> PauseSummary | None:
        """What a run waits for, in one bounded statement; None when missing or out of scope.

        Only :data:`PAUSE_SUMMARY_KEYS` and the last step are extracted from
        ``pause_data`` in SQL (``jsonb`` operators on Postgres, ``json_extract``
        on SQLite), so the pause history never leaves the database.
        """
        from sqlalchemy import func, literal_column, select

        from dendrux.db.models import AgentRun, RunEvent

        dialect = self._engine.dialect.name
        parts: list[Any]
        paused_seq = (
            select(func.max(RunEvent.sequence_index))
            .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
            .scalar_subquery()
        )
        if dialect == "postgresql":
            doc = _readable_json_document(AgentRun.pause_data)
            parts = [doc[key] for key in PAUSE_SUMMARY_KEYS] + [doc["steps"][-1]]
        elif dialect == "sqlite":
            parts = [func.json_extract(AgentRun.pause_data, f"$.{k}") for k in PAUSE_SUMMARY_KEYS]
            parts.append(func.json_extract(AgentRun.pause_data, literal_column("'$.steps[#-1]'")))
        else:
            parts = [AgentRun.pause_data]
        stmt = _apply_tenant_scope(
            select(AgentRun.status, paused_seq, AgentRun.pii_mapping, *parts).where(
                AgentRun.id == run_id
            ),
            tenant_id,
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).first()
        if row is None:
            return None
        status, seq, mapping, *values = row
        mapping = dict(mapping) if isinstance(mapping, dict) else None
        if dialect not in ("postgresql", "sqlite"):
            data = dict(values[0] or {})
            steps = data.get("steps") or []
            pause = {k: data[k] for k in PAUSE_SUMMARY_KEYS if k in data}
            return PauseSummary(str(status), pause, steps[-1] if steps else None, seq, mapping)
        if dialect == "sqlite":
            # json_extract returns arrays/objects as JSON text and scalars as values.
            keys = (*PAUSE_SUMMARY_KEYS, "steps")
            decoded = [
                _json_value(v) if k not in _PAUSE_SCALAR_KEYS else v
                for k, v in zip(keys, values, strict=True)
            ]
        else:
            decoded = list(values)
        pause = {
            k: v for k, v in zip(PAUSE_SUMMARY_KEYS, decoded[:-1], strict=True) if v is not None
        }
        last = decoded[-1]
        last_step = last if isinstance(last, dict) else None
        return PauseSummary(str(status), pause, last_step, seq, mapping)

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

    async def claim_paused_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        expected_pause_id: str | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        """Atomically transition a paused run to RUNNING.

        Returns the new claim epoch if the claim succeeded.
        Returns False if someone else already claimed it or status didn't match.
        Uses UPDATE ... WHERE status=? for atomicity — no race window.
        The claim clears ``pause_requested`` in the same statement, so a
        resumed PAUSED run does not immediately re-pause on a stale flag.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> int:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .values(
                        **_status_values("running"),
                        last_progress_at=_dt.datetime.now(_dt.UTC),
                        pause_requested=False,
                        claim_epoch=AgentRun.claim_epoch + 1,
                    )
                )
                result = await session.execute(stmt)
                if result.rowcount:
                    epoch = (
                        await session.execute(
                            select(AgentRun.claim_epoch).where(AgentRun.id == run_id)
                        )
                    ).scalar_one()
                if result.rowcount and (validate is not None or expected_pause_id is not None):
                    pause = (
                        await session.execute(
                            select(AgentRun.pause_data).where(AgentRun.id == run_id)
                        )
                    ).scalar_one_or_none()
                    try:
                        if expected_pause_id is not None:
                            from dendrux.errors import PauseStatusMismatchError

                            current_id = (pause or {}).get("pause_id")
                            if current_id != expected_pause_id:
                                logger.warning(
                                    "Pause generation mismatch: run_id=%s code=pause_id_mismatch",
                                    run_id,
                                    extra={"run_id": run_id, "code": "pause_id_mismatch"},
                                )
                                raise PauseStatusMismatchError(
                                    run_id,
                                    RunStatus(expected_status),
                                    RunStatus(expected_status),
                                    expected_pause_id=expected_pause_id,
                                    current_pause_id=current_id,
                                    reason="pause_id",
                                )
                        if validate is not None:
                            validate(pause or {})
                    except BaseException:
                        await session.rollback()
                        raise
                await session.commit()
                return epoch if result.rowcount else False

        return await retry_transient_db(_attempt, label="claim_paused_run", run_id=run_id)

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
        validate: Callable[[dict[str, Any]], None] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> int:
        """Atomically save submitted data and claim the run.

        Implementation (one transaction):
          1. Conditional UPDATE status -> 'running' WHERE status still equals
             *expected_status*. Losing this CAS returns False. Winning it
             takes the row's write lock (Postgres row lock / SQLite write
             lock), so the pause_data read next cannot change underneath.
          2. Read pause_data and run *validate* on it. A failure rolls the
             whole transaction back (status stays *expected_status*) and
             re-raises.
          3. Merge submitted_data into pause_data.
          4. Write ``blob_refs`` (tenant asserted, blobs claimed) and commit.
             A failure rolls the claim back too.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> int:
            async with self._session_factory() as session:
                # 1. Claim first — the CAS on status is the concurrency
                # boundary. updated_at is filled by the model's onupdate hook.
                claim = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .values(
                        **_status_values("running"),
                        last_progress_at=_dt.datetime.now(_dt.UTC),
                        pause_requested=False,
                        claim_epoch=AgentRun.claim_epoch + 1,
                    )
                )
                if not claim.rowcount:
                    await session.rollback()
                    return False

                # 2. Read the pause being claimed, under the claim's lock.
                current_pause_data = (
                    await session.execute(select(AgentRun.pause_data).where(AgentRun.id == run_id))
                ).scalar_one_or_none()
                if current_pause_data is None:
                    await session.rollback()
                    return False
                if validate is not None:
                    try:
                        validate(current_pause_data)
                    except BaseException:
                        await session.rollback()
                        raise

                # 3. Merge submitted_data into pause_data.
                merged = dict(current_pause_data)
                merged.update(submitted_data)
                await session.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(pause_data=merged)
                )
                if blob_refs:
                    await write_blob_refs(
                        session,
                        run_id=run_id,
                        blob_refs=blob_refs,
                        blob_refs_scope=blob_refs_scope,
                    )
                epoch = (
                    await session.execute(select(AgentRun.claim_epoch).where(AgentRun.id == run_id))
                ).scalar_one()
                await session.commit()
                return int(epoch)

        return await retry_transient_db(_attempt, label="submit_and_claim", run_id=run_id)

    async def release_run_claim(self, run_id: str, *, status: str, claim_epoch: int) -> bool:
        """Return an owned claim to its pause after post-claim validation fails."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == "running",
                        AgentRun.claim_epoch == claim_epoch,
                    )
                    .values(**_status_values(status))
                )
                await session.commit()
                return bool(result.rowcount)

        return await retry_transient_db(attempt, label="release_run_claim", run_id=run_id)

    @_safe_database_call
    async def _get_result_record(self, run_id: str, *, tenant_id: TenantScope) -> RunRecord | None:
        """Project one browser-result record without prompts or pause history.

        Private optimization; custom stores keep the existing scoped get_run
        contract. All reads use the same tenant predicate as get_run.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        stop_key = "dendrux.stop_reason"

        fields = (
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
            AgentRun.meta["dendrux.error_info"].label("error_info"),
            AgentRun.meta[stop_key].as_string().label("stop_reason"),
        )
        stmt = _apply_tenant_scope(select(*fields).where(AgentRun.id == run_id), tenant_id)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).mappings().first()
        if row is None:
            return None
        values = dict(row)
        info, stop_reason = values.pop("error_info"), values.pop("stop_reason")
        values["total_cost_usd"] = (
            float(values["total_cost_usd"]) if values["total_cost_usd"] is not None else None
        )
        return RunRecord(
            **values,
            meta={"dendrux.error_info": info, stop_key: stop_reason},
        )

    async def get_run(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> RunRecord | None:
        """The run, or None when it does not exist or is outside ``tenant_id``.

        The tenant predicate is part of the query, so a foreign run reads
        exactly like a missing one.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(select(AgentRun).where(AgentRun.id == run_id), tenant_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _run_to_record(row)

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
                stmt = stmt.limit(max(1, min(limit, 1000)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                TraceRecord(
                    id=r.id,
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
                stmt = stmt.limit(max(1, min(limit, 1000)))
            if offset:
                stmt = stmt.offset(max(0, offset))
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
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def save_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> int:
        """Append an event with the next sequence index of its run; return the index.

        One ``INSERT ... SELECT COALESCE(MAX(sequence_index), -1) + 1``. On
        Postgres the run row is first locked ``FOR NO KEY UPDATE`` in the same
        transaction, so concurrent writers of one run (and the CAS paths in
        :func:`_insert_event_locked`, which hold the same row lock) allocate
        one at a time, in commit order; the next writer's statement sees the
        previous commit. SQLite serializes writers already. Cost: one indexed
        ``MAX`` lookup on ``(agent_run_id, sequence_index)``. There is no way to
        pass an index (0.4.1 callers that do get a ``TypeError``, not a silently
        wrong order).
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            if self._engine.dialect.name == "postgresql":
                await session.execute(
                    select(AgentRun.id).where(AgentRun.id == run_id).with_for_update(key_share=True)
                )
            seq = await _insert_event(
                session,
                run_id,
                event_type,
                data,
                iteration_index=iteration_index,
                correlation_id=correlation_id,
            )
            await session.commit()
            return seq

    async def get_run_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
        event_types: list[str] | None = None,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
        _pause_page: bool = False,
    ) -> list[RunEventRecord]:
        """Events of a run by ``sequence_index``; empty when outside ``tenant_id``."""
        from sqlalchemy import false, select

        from dendrux.db.models import RunEvent

        _require_aware_utc("before", before)

        async with self._session_factory() as session:
            stmt = (
                select(RunEvent)
                .where(RunEvent.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(RunEvent.sequence_index)
            )

            if after_sequence_index is not None:
                stmt = stmt.where(RunEvent.sequence_index > after_sequence_index)

            if before is not None:
                stmt = stmt.where(RunEvent.created_at < _to_aware_utc(before))

            if event_types is not None:
                if not event_types:
                    # Empty list = "match no event_type" — symmetric with status=[].
                    stmt = stmt.where(false())
                else:
                    stmt = stmt.where(RunEvent.event_type.in_(event_types))

            if limit is not None:
                # Clamp to 1..1000
                clamped = max(1, min(limit, 2002 if _pause_page else 1000))
                stmt = stmt.limit(clamped)

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
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Progress tracking & stale-run sweep
    # ------------------------------------------------------------------

    async def touch_progress(self, run_id: str) -> None:
        """Update progress only while running; paused/terminal rows are untouched."""
        import datetime as _dt

        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = (
                update(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.status == "running")
                .values(last_progress_at=_dt.datetime.now(_dt.UTC))
            )
            await session.execute(stmt)
            await session.commit()

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Inspect running candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000. Fresh rows and losing CAS attempts count
        toward it; hitting the cap emits sweep_cap_reached.
        """
        return await self._sweep_runs(
            older_than, waiting=False, batch_size=batch_size, max_runs=max_runs
        )

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Inspect waiting candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000 across all waiting statuses. The final CAS
        requires both status_changed_at and updated_at to be stale. Losing CAS
        attempts count toward the cap; hitting it emits sweep_cap_reached.
        """
        return await self._sweep_runs(
            older_than, waiting=True, batch_size=batch_size, max_runs=max_runs
        )

    async def _sweep_runs(
        self,
        older_than: timedelta,
        *,
        waiting: bool,
        batch_size: int,
        max_runs: int,
    ) -> list[SweptRun]:
        from sqlalchemy import case, func, literal_column, select, true, tuple_, update

        from dendrux.db.models import AgentRun, RunEvent

        for name, value, ceiling in (
            ("batch_size", batch_size, 1000),
            ("max_runs", max_runs, 100000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        _validate_sweep_age(older_than)
        now = _dt.datetime.now(_dt.UTC)
        cutoff = now - older_than
        age = (
            AgentRun.status_changed_at
            if waiting
            else func.coalesce(AgentRun.last_progress_at, AgentRun.created_at)
        )
        statuses = (
            ["waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"]
            if waiting
            else ["running"]
        )
        # A scalar LIMIT subquery prevents a hashed scan of every started event.
        started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column("'run.started'"),
            )
            .limit(1)
            .correlate(AgentRun)
            .scalar_subquery()
        )
        swept: list[SweptRun] = []
        visited = 0
        for status_filter in statuses:
            after: tuple[_dt.datetime, str] | str | None = None
            while visited < max_runs:
                statement = select(
                    AgentRun.id,
                    AgentRun.agent_name,
                    AgentRun.status,
                    AgentRun.last_progress_at,
                    AgentRun.iteration_count,
                    started.label("started"),
                    age.label("sweep_age"),
                ).limit(min(batch_size, max_runs - visited))
                if waiting:
                    statement = statement.where(
                        AgentRun.status == status_filter, age < cutoff
                    ).order_by(age, AgentRun.id)
                    if after is not None:
                        statement = statement.where(tuple_(age, AgentRun.id) > after)
                else:
                    # Equal range bounds preserve the (status, id) ordering in
                    # the planner. Equality alone lets it prefer the primary key
                    # and filter through terminal history on every page.
                    statement = statement.where(
                        AgentRun.status.between(status_filter, status_filter)
                    ).order_by(AgentRun.status, AgentRun.id)
                    if after is not None:
                        statement = statement.where(
                            tuple_(AgentRun.status, AgentRun.id) > (status_filter, after)
                        )
                async with self._session_factory() as session:
                    rows = (await session.execute(statement)).all()
                if not rows:
                    break
                visited += len(rows)
                after = (rows[-1].sweep_age, rows[-1].id) if waiting else rows[-1].id
                for row in rows:
                    # Running rows are bounded by live concurrency. Count every
                    # inspected row against the work cap, including fresh ones.
                    if not waiting and _to_aware_utc(row.sweep_age) >= cutoff:
                        continue
                    reason = (
                        "abandoned_waiting"
                        if waiting
                        else "stale_running"
                        if row.started
                        else "never_started"
                    )
                    flagged = AgentRun.cancel_requested.is_(True)
                    new_status = "error" if waiting else case((flagged, "cancelled"), else_="error")
                    async with self._session_factory() as session:
                        result = await session.execute(
                            update(AgentRun)
                            .where(
                                AgentRun.id == row.id,
                                AgentRun.status == row.status,
                                age < cutoff,
                                # Legacy workers update updated_at on pause but
                                # do not know the new status-transition clock.
                                AgentRun.updated_at < cutoff if waiting else true(),
                            )
                            .values(
                                **_status_values(new_status),
                                claim_epoch=AgentRun.claim_epoch + 1,
                                failure_reason=reason,
                                pause_data=None,
                                error=f"Run swept as {reason}"
                                if waiting
                                else case((flagged, None), else_=f"Run swept as {reason}"),
                                cancel_requested=False,
                                pause_requested=False,
                            )
                            .returning(AgentRun.status)
                        )
                        status = result.scalar_one_or_none()
                        if status is None:
                            continue
                        event_type = "run.abandoned" if waiting else "run.interrupted"
                        data = {"failure_reason": reason}
                        if status == "cancelled":
                            event_type = "run.cancelled"
                            data["reason"] = "cancel_requested"
                        await _insert_event_locked(
                            session,
                            row.id,
                            event_type,
                            data,
                            iteration_index=row.iteration_count or 0,
                        )
                        await session.commit()
                    swept.append(
                        SweptRun(
                            run_id=row.id,
                            agent_name=row.agent_name,
                            previous_status=row.status,
                            failure_reason=reason,
                            last_progress_at=_to_aware_utc(row.last_progress_at),
                            swept_at=_to_aware_utc(now),
                            status=status,
                        )
                    )
        if visited == max_runs:
            logger.warning(
                "Sweep candidate cap reached",
                extra={
                    "code": "sweep_cap_reached",
                    "category": "waiting" if waiting else "running",
                    "cap": max_runs,
                    "inspected": visited,
                },
            )
        return swept

    # ------------------------------------------------------------------
    # Delegation queries
    # ------------------------------------------------------------------

    async def get_delegation_info(
        self, run_id: str, *, max_rows: int = 1000, max_depth: int = 32
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
            or not 1 <= max_rows <= 10000
        ):
            raise ValueError("max_rows must be an int between 1 and 10000.")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 0 <= max_depth <= 100
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
                    rows = list((await session.execute(stmt)).scalars())
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

    async def _bfs_descendants(
        self,
        session: AsyncSession,
        root_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        light: bool = False,
        max_nodes: int | None = None,
    ) -> list[tuple[AgentRun, int]]:
        """Descendants of ``root_id`` as ``(row, depth)`` in BFS order, root excluded.

        ``depth`` is 1 for direct children. Cycle-safe via a seen set. With
        ``tenant_id`` set, other tenants' runs are skipped and so is
        everything below them. ``light`` loads only id / parent / tenant so
        usage-only callers don't pull large JSON columns.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from dendrux.db.models import AgentRun

        seen: set[str] = {root_id}
        current_level = [root_id]
        found: list[tuple[AgentRun, int]] = []
        depth = 0

        while current_level:
            depth += 1
            rows: list[AgentRun] = []
            for i in range(0, len(current_level), _IN_CHUNK):
                stmt = select(AgentRun).where(
                    AgentRun.parent_run_id.in_(current_level[i : i + _IN_CHUNK])
                )
                if light:
                    stmt = stmt.options(
                        load_only(AgentRun.id, AgentRun.parent_run_id, AgentRun.tenant_id)
                    )
                stmt = _apply_tenant_scope(stmt, tenant_id)
                if max_nodes is not None:
                    stmt = stmt.limit(max_nodes - len(found) - len(rows))
                result = await session.execute(stmt.order_by(AgentRun.created_at, AgentRun.id))
                rows.extend(result.scalars().all())
                if max_nodes is not None and len(found) + len(rows) >= max_nodes:
                    import logging

                    from dendrux.errors import BudgetConfigurationError

                    logging.getLogger(__name__).warning(
                        "Budget subtree bound reached: run_id=%s code=subtree_limit",
                        root_id,
                        extra={"run_id": root_id, "code": "subtree_limit"},
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
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> SubtreeRecord | None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(select(AgentRun).where(AgentRun.id == run_id), tenant_id)
            root = (await session.execute(stmt)).scalar_one_or_none()
            if root is None:
                return None

            rows = [(root, 0), *await self._bfs_descendants(session, root.id, tenant_id=tenant_id)]
            usage = await self._usage_by_run(session, [r.id for r, _ in rows])
            nodes = [
                SubtreeNodeRecord(
                    run=_run_to_record(r), depth=d, usage=usage.get(r.id, UsageRollup())
                )
                for r, d in rows
            ]
            return SubtreeRecord(nodes=nodes, usage=UsageRollup.combine(n.usage for n in nodes))

    async def get_descendant_page(
        self,
        run_id: str,
        *,
        after: str | None = None,
        limit: int = 500,
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
        stmt = stmt.order_by(tree.c.id).limit(max(1, min(limit, 500)))
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

    _MAX_LIST_LIMIT = 1000

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
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
                select(AgentRun)
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
            )
            stmt = stmt.limit(capped_limit).offset(clamped_offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_run_to_record(r) for r in rows]

    async def count_pauses_per_run(
        self,
        run_ids: list[str],
    ) -> dict[str, int]:
        if not run_ids:
            return {}

        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        async with self._session_factory() as session:
            stmt = (
                select(RunEvent.agent_run_id, func.count())
                .where(RunEvent.agent_run_id.in_(run_ids))
                .where(RunEvent.event_type == "run.paused")
                .group_by(RunEvent.agent_run_id)
            )
            result = await session.execute(stmt)
            return {row[0]: int(row[1]) for row in result.all()}

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
    ) -> Any:
        from sqlalchemy import false

        from dendrux.db.models import AgentRun

        # Naive datetimes here would silently break PG comparisons against
        # TIMESTAMPTZ. Reject at the boundary so future filter additions
        # inherit the guard automatically by routing through this helper.
        _require_aware_utc("started_after", started_after)
        _require_aware_utc("started_before", started_before)

        stmt = _apply_tenant_scope(stmt, tenant_id)
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
            stmt = stmt.where(AgentRun.parent_run_id == parent_run_id)
        # Normalized to UTC: SQLite stores UTC wall-clock text, so a non-UTC
        # offset would otherwise compare against the wrong instant.
        if started_after is not None:
            stmt = stmt.where(AgentRun.created_at >= _to_aware_utc(started_after))
        if started_before is not None:
            stmt = stmt.where(AgentRun.created_at < _to_aware_utc(started_before))
        if metadata_filter:
            stmt = self._apply_metadata_filter(stmt, metadata_filter)
        return stmt

    # JSON keys are limited to a conservative identifier shape so the
    # path string we splice into the SQL (``$.<key>`` for SQLite) cannot
    # smuggle path operators or quotes. Values are always parameter-bound,
    # so only the key is sensitive here.
    _METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

    def _apply_metadata_filter(self, stmt: Any, metadata_filter: dict[str, Any]) -> Any:
        """Filter by exact-match equality on top-level ``meta`` keys.

        AND across all keys. ``thread_id="abc"`` is the canonical chatbot
        case; arbitrary scalar values (str/int/bool) work too. Nested
        paths (``user.id``) and list-membership checks are intentionally
        unsupported in v1 — add them when a real use case arrives.
        """
        from sqlalchemy import func

        from dendrux.db.models import AgentRun
        from dendrux.types import _validate_persistable

        _validate_persistable(metadata_filter, "metadata_filter")
        for key in metadata_filter:
            if not isinstance(key, str) or not self._METADATA_KEY_RE.match(key):
                raise ValueError(
                    f"metadata_filter keys must match {self._METADATA_KEY_RE.pattern!r}; "
                    f"got {key!r}. Nested paths are not supported in v1."
                )

        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            from sqlalchemy import cast
            from sqlalchemy.dialects.postgresql import JSONB

            return stmt.where(_jsonb_or_null(AgentRun.meta).contains(cast(metadata_filter, JSONB)))

        # SQLite (and other backends): per-key json_extract equality.
        # json_extract returns native types, so equality works naturally
        # for str/int/bool. The key is validated above; the value is
        # parameter-bound by SQLAlchemy.
        for key, value in metadata_filter.items():
            stmt = stmt.where(func.json_extract(AgentRun.meta, f"$.{key}") == value)
        return stmt


def _approval_event_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _jsonb_or_null(column: Any) -> Any:
    """``column::jsonb``, or NULL for a stored document ``jsonb`` cannot hold.

    The ``json`` column accepts ``\\u0000`` and unpaired surrogate escapes;
    converting such a row raises and fails the whole statement, so one row
    would break a scan for every tenant. The text is checked first: escaped
    backslashes and valid surrogate pairs are removed, and any ``\\u0000`` or
    surrogate escape left means the row is not convertible (it matches no
    filter). Rows without any ``\\u`` escape skip the check.
    """
    from sqlalchemy import Text, case, cast, func
    from sqlalchemy.dialects.postgresql import JSONB

    raw = cast(column, Text)
    no_backslash_pairs = func.regexp_replace(raw, r"\\\\", "", "g")
    no_pairs = func.regexp_replace(
        no_backslash_pairs,
        r"\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}",
        "",
        "g",
    )
    unconvertible = no_pairs.op("~")(r"\\u(0000|[dD][89a-fA-F])")
    return case(
        (func.strpos(raw, "\\u") == 0, cast(column, JSONB)),
        (~unconvertible, cast(column, JSONB)),
        else_=None,
    )


async def read_run_control(store: Any, run_id: str) -> RunControl:
    """Checkpoint read that also works on pre-0.3 custom stores.

    A store without ``get_run_control`` gets the cancel flag from
    ``is_cancel_requested``; it has no pause flag and no inbox.
    """
    getter = getattr(store, "get_run_control", None)
    if getter is not None:
        control: RunControl = await getter(run_id)
        return control
    return RunControl(cancel_requested=bool(await store.is_cancel_requested(run_id)))


def require_store_method(store: Any, name: str) -> Any:
    """Return ``store.<name>`` or raise a clear error for a pre-0.3 custom store."""
    method = getattr(store, name, None)
    if method is None:
        raise NotImplementedError(
            f"{type(store).__name__} does not implement StateStore.{name}(), which this "
            "run-control API needs (added in Dendrux 0.3). Implement it on the custom "
            "store or use SQLAlchemyStateStore."
        )
    return method


async def _insert_event_locked(
    session: AsyncSession,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None,
    *,
    iteration_index: int = 0,
) -> int:
    """Append a run event with the run's next sequence index, in the caller's transaction.

    Called only after the caller's CAS UPDATE on the run row succeeded, so
    the row lock serializes it with every other allocating writer
    (:meth:`SQLAlchemyStateStore.save_run_event` takes the same lock).
    Returns the allocated index.
    """
    return await _insert_event(session, run_id, event_type, data, iteration_index=iteration_index)


async def _insert_event(
    session: AsyncSession,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None,
    *,
    iteration_index: int = 0,
    correlation_id: str | None = None,
) -> int:
    """``INSERT ... SELECT COALESCE(MAX(sequence_index), -1) + 1`` for one run; the index.

    The caller provides the serialization (the run row lock on Postgres).
    The ``MAX`` reads ``ix_run_events_agent_run_id_seq`` backwards.
    """
    import datetime as _dt

    from sqlalchemy import JSON, DateTime, Integer, String, func, insert, literal, select

    from dendrux.db.models import RunEvent

    next_index = (
        select(func.coalesce(func.max(RunEvent.sequence_index), -1) + 1)
        .where(RunEvent.agent_run_id == run_id)
        .scalar_subquery()
    )
    source = select(
        literal(generate_ulid(), String(26)),
        literal(run_id, String(26)),
        literal(event_type, String(50)),
        next_index,
        literal(iteration_index, Integer()),
        literal(correlation_id, String(26)),
        literal(data, JSON()),
        literal(_dt.datetime.now(_dt.UTC), DateTime(timezone=True)),
    )
    stmt = (
        insert(RunEvent)
        .from_select(
            [
                RunEvent.id,
                RunEvent.agent_run_id,
                RunEvent.event_type,
                RunEvent.sequence_index,
                RunEvent.iteration_index,
                RunEvent.correlation_id,
                RunEvent.data,
                RunEvent.created_at,
            ],
            source,
        )
        .returning(RunEvent.sequence_index)
    )
    return int((await session.execute(stmt)).scalar_one())


# Unique indexes that enforce idempotency keys: the 0.5 tenant-scoped one, and
# the pre-0.5 global ones (Alembic index, Postgres ``create_all`` constraint,
# SQLite inline UNIQUE) still present on databases that were not rebuilt.
_IDEMPOTENCY_VIOLATION_MARKERS = (
    "uq_agent_runs_tenant_idempotency_key",
    "ix_agent_runs_idempotency_key",
    "agent_runs_idempotency_key_key",
    "UNIQUE constraint failed: agent_runs.idempotency_key",
)


def _is_idempotency_violation(exc: Exception) -> bool:
    """True when ``exc`` is a unique violation of an idempotency-key index."""
    message = str(getattr(exc, "orig", None) or exc)
    return any(marker in message for marker in _IDEMPOTENCY_VIOLATION_MARKERS)


def _apply_tenant_scope(stmt: Any, tenant_id: TenantScope) -> Any:
    """``UNSCOPED`` adds nothing; ``None`` means untenanted only; a string means that tenant."""
    from dendrux.db.models import AgentRun

    if isinstance(tenant_id, _Unscoped):
        return stmt
    _check_scope(tenant_id)
    if tenant_id is None:
        return stmt.where(AgentRun.tenant_id.is_(None))
    return stmt.where(AgentRun.tenant_id == tenant_id)


def _finite_json_float(value: str) -> float:
    """Reject NaN/Infinity and overflow so tool results stay JSON-column safe."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _json_value(value: Any) -> Any:
    """A SQLite ``json_extract`` array/object result (JSON text) decoded."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _check_scope(tenant_id: object) -> None:
    """Fail closed on a scope that is not ``UNSCOPED``, ``None`` or a non-empty string."""
    if tenant_id is None or isinstance(tenant_id, _Unscoped):
        return
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError(
            "tenant_id must be a non-empty str, None (untenanted runs) or UNSCOPED (all)."
        )


def _run_in_scope(run_id: str, tenant_id: TenantScope) -> Any:
    """SQL predicate: the run exists within ``tenant_id`` (always true when ``UNSCOPED``).

    Added to a child-table query (events, traces, ...) so the scope check is
    part of the same statement.
    """
    from sqlalchemy import select, true

    from dendrux.db.models import AgentRun

    if isinstance(tenant_id, _Unscoped):
        return true()
    return _apply_tenant_scope(select(AgentRun.id).where(AgentRun.id == run_id), tenant_id).exists()


MAX_LIST_OFFSET = 10_000
"""Largest ``offset`` ``list_runs`` accepts. Keyset pagination is planned for 0.6."""


def _check_offset(offset: int) -> int:
    if offset > MAX_LIST_OFFSET:
        raise ValueError(
            f"offset {offset} is over the maximum of {MAX_LIST_OFFSET}. Narrow the query "
            "with filters (status, agent_name, started_before, ...) instead of paging deeper."
        )
    return max(0, offset)


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
        total_cache_read_tokens=row.total_cache_read_tokens,
        total_cache_creation_tokens=row.total_cache_creation_tokens,
        total_reasoning_tokens=row.total_reasoning_tokens,
        meta=row.meta,
        last_progress_at=_to_aware_utc(row.last_progress_at),
        failure_reason=row.failure_reason,
        retry_of_run_id=row.retry_of_run_id,
        cancel_requested=bool(row.cancel_requested),
        created_at=_to_aware_utc(row.created_at),
        updated_at=_to_aware_utc(row.updated_at),
        tenant_id=row.tenant_id,
        pause_requested=bool(row.pause_requested),
    )


async def _finalized_output_data(
    session: AsyncSession,
    run_id: str,
    answer: str | None,
    usage: UsageStats | None,
) -> dict[str, Any]:
    from sqlalchemy import select

    from dendrux.db.models import AgentRun

    if answer is not None:
        output: dict[str, Any] = {"answer": answer}
    else:
        existing = await session.scalar(select(AgentRun.output_data).where(AgentRun.id == run_id))
        output = dict(existing or {})
    if usage is not None:
        output["usage"] = _usage_to_dict(usage)
    return output


def validate_event_writer(store: Any) -> None:
    """Reject the 0.4 event API before a legacy writer can silently persist index zero."""
    import inspect

    if "sequence_index" in inspect.signature(store.save_run_event).parameters:
        raise TypeError(
            "StateStore.save_run_event still accepts sequence_index (0.4 API). "
            "Remove that parameter and allocate and return an integer index in the insert."
        )


def validate_event_index(index: Any) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        import warnings

        warnings.warn(
            "StateStore.save_run_event must return its allocated non-negative integer index; "
            "update this custom store before relying on event cursors.",
            RuntimeWarning,
            stacklevel=2,
        )


class _ClaimedStateStore:
    """Execution-local store view; only status writes and checkpoints are fenced.

    Inserts remain compatible with 0.4 workers during rolling upgrades. Their
    per-insert fencing is deferred to 0.6.
    """

    def __init__(self, store: StateStore, run_id: str, epoch: int) -> None:
        self._store = store
        self._run_id = run_id
        self.claim_epoch = epoch

    @property
    def store_identity(self) -> str | None:
        from dendrux.runtime.context import get_store_identity

        return get_store_identity(self._store)

    def __getattr__(self, name: str) -> Any:
        import inspect

        method = getattr(self._store, name)
        if (
            name
            in {
                "finalize_run",
                "finalize_run_if_status_in",
                "finalize_with_event",
                "try_pause_run",
                "pause_with_event",
                "pause_run",
            }
            and method is not None
            and "claim_epoch" in inspect.signature(method).parameters
        ):

            async def fenced(run_id: str, **kwargs: Any) -> Any:
                result = await method(run_id, claim_epoch=self.claim_epoch, **kwargs)
                if result is False or result is None:
                    control = await read_run_control(self._store, run_id)
                    if control.claim_epoch is not None and control.claim_epoch != self.claim_epoch:
                        logger.warning(
                            "Run fenced at status transition: run_id=%s code=run_fenced",
                            run_id,
                            extra={"run_id": run_id, "code": "run_fenced"},
                        )
                return result

            return fenced
        return method

    async def get_run_control(self, run_id: str) -> RunControl:
        control = await read_run_control(self._store, run_id)
        if isinstance(self._store, SQLAlchemyStateStore) and control.claim_epoch is None:
            # Older overrides may return flags only. Read the base projection
            # to distinguish that from a missing row and retain epoch fencing.
            from dataclasses import replace

            persisted = await SQLAlchemyStateStore.get_run_control(self._store, run_id)
            control = replace(control, claim_epoch=persisted.claim_epoch, status=persisted.status)
        if (
            (isinstance(self._store, SQLAlchemyStateStore) and control.claim_epoch is None)
            or (control.claim_epoch is not None and control.claim_epoch != self.claim_epoch)
            or (control.status is not None and control.status != RunStatus.RUNNING.value)
        ):
            from dendrux.errors import RunFencedError

            logger.warning(
                "Run fenced: run_id=%s code=run_fenced",
                run_id,
                extra={"run_id": run_id, "code": "run_fenced"},
            )
            raise RunFencedError(run_id, claim_epoch=self.claim_epoch)
        return control


def claimed_store(store: StateStore, run_id: str, epoch: int | bool) -> StateStore:
    """Bind a returned epoch; bool is the pre-0.5 custom-store claim contract."""
    if type(epoch) is bool:
        return store
    if isinstance(store, _ClaimedStateStore):
        store = store._store
    return cast("StateStore", _ClaimedStateStore(store, run_id, epoch))


def unclaimed_store(store: StateStore | None) -> StateStore | None:
    """The shared store for identity-sensitive capabilities such as blob catalogs."""
    return store._store if isinstance(store, _ClaimedStateStore) else store
