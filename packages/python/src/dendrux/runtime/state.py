"""StateStore — persistence interface for agent runs.

Protocol-based so developers can swap in Redis, DynamoDB, or any other
backend. The default implementation uses SQLAlchemy async (SQLite/Postgres).

Usage:
    store = store_for_engine(engine)
    run_id = await store.create_run(...)
    # ... loop runs with recorder writing traces/tool_calls/usage ...
    await store.finalize_run(run_id, status=..., answer=..., ...)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

from dendrux import _codes
from dendrux.types import (
    CreateRunResult,
    RunStatus,
    UsageStats,
    _usage_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncSession

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


@dataclass(frozen=True)
class StoreCapabilities:
    """Explicit, versioned promises made by a state-store implementation.

    Usage retry equality covers persisted token_usage columns (cost at six
    decimals); non-persisted UsageStats fields such as cost_source are excluded.
    """

    version: int = 1
    fenced_writes: bool = False
    retry_ids: bool = False
    checkpointing: bool = False
    atomic_completion: bool = False
    recovery: bool = False
    notify: bool = False
    deployment: Literal["single_process", "single_host", "multi_server"] | None = None


@dataclass(frozen=True)
class ClaimResult:
    """The immutable execution token returned by a successful claim."""

    claim_epoch: int
    storage_format: int
    loop_state: dict[str, Any] | None
    submission: dict[str, Any] | None


def _initial_loop_state(loop_name: str | None) -> dict[str, Any] | None:
    kind = {"ReActLoop": "react", "SingleCall": "single_call"}.get(loop_name or "")
    return {"v": 1, "loop": kind} if kind is not None else None


_creation_loop: ContextVar[tuple[object, str, str | None] | None] = ContextVar(
    "dendrux_creation_loop", default=None
)


@dataclass(frozen=True)
class _FinalizedRunRead:
    id: str
    tenant_id: str | None
    reclaim_attempts: int
    last_reclaimed_at: datetime | None


_finalized_run_read: ContextVar[tuple[object, str, list[_FinalizedRunRead]] | None] = ContextVar(
    "dendrux_finalized_run_read", default=None
)


@contextmanager
def _capture_finalized_run(store: object, run_id: str) -> Iterator[list[_FinalizedRunRead]]:
    records: list[_FinalizedRunRead] = []
    token = _finalized_run_read.set((store, run_id, records))
    try:
        yield records
    finally:
        _finalized_run_read.reset(token)


@contextmanager
def _creation_loop_scope(store: object, run_id: str, loop_name: str | None) -> Iterator[None]:
    token = _creation_loop.set((store, run_id, loop_name))
    try:
        yield
    finally:
        _creation_loop.reset(token)


def _loop_state_at_creation(
    store: object, run_id: str, meta: dict[str, Any] | None
) -> dict[str, Any] | None:
    current = _creation_loop.get()
    if current is not None and current[0] is store and current[1] == run_id:
        return _initial_loop_state(current[2])
    return _initial_loop_state((meta or {}).get("dendrux.loop"))


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


@dataclass(frozen=True)
class LoopStatePatch:
    """Changes applied to the current checkpoint under the run-row lock."""

    remove_pending: tuple[str, ...] = ()
    set_phase: str | None = None
    set_fields: Mapping[str, Any] = field(default_factory=dict)
    set_call_phase: Mapping[str, str] = field(default_factory=dict)


def _apply_loop_patch(current: dict[str, Any], patch: LoopStatePatch) -> dict[str, Any]:
    if "pending" in patch.set_fields:
        raise ValueError("A patch cannot replace pending; remove calls by identity")
    changed = {**current, **patch.set_fields}
    if patch.remove_pending:
        changed["pending"] = [
            p for p in current.get("pending", []) if p["call_id"] not in patch.remove_pending
        ]
    if patch.set_call_phase:
        changed["pending"] = [
            {**p, "phase": patch.set_call_phase[p["call_id"]]}
            if p["call_id"] in patch.set_call_phase
            else p
            for p in changed.get("pending", [])
        ]
    if patch.set_phase is not None:
        changed["phase"] = (
            current.get("phase", "dispatch")
            if patch.set_phase == "model_next" and patch.remove_pending and changed.get("pending")
            else patch.set_phase
        )
    return changed


@dataclass(frozen=True)
class TraceWrite:
    """Caller-owned trace identity for a transactional write."""

    trace_id: str
    role: str
    content: str
    order_index: int
    trace_kind: str = "turn"
    meta: dict[str, Any] | None = None
    blob_refs: Sequence[BlobEdge] | None = None
    blob_refs_scope: str | None = None
    content_digest: str | None = None


@dataclass(frozen=True)
class EventWrite:
    """Caller-owned event identity, independent of its allocated sequence."""

    event_id: str
    event_type: str
    iteration_index: int
    data: dict[str, Any] | None
    correlation_id: str | None
    content_digest: str | None


@dataclass(frozen=True)
class ConversationEntry:
    """One expanded conversation entry, at a monotonic virtual position."""

    position: int
    role: str
    content: str
    meta: dict[str, Any]
    source: str
    trace_id: str


@dataclass(frozen=True)
class ConversationPage:
    """Bounded projection page; next_position is absent only at the end."""

    entries: list[ConversationEntry]
    next_position: int | None


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
    storage_format: int = 1
    claim_epoch: int = 0
    status_changed_at: datetime | None = None
    reclaim_attempts: int = 0
    reclaim_limit: int | None = None
    lease_expired_at: datetime | None = None
    last_reclaimed_at: datetime | None = None
    trace_link: str | None = None


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
    trace_kind: str = "turn"


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
    outcome: str = "completed"
    idempotent: bool = False
    replayed_at: datetime | None = None
    observation: dict[str, Any] | None = None


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
class ReclaimableRun:
    """A durable lease-expiry mark and the epoch a caller observed."""

    run_id: str
    tenant_id: str | None
    agent_name: str
    last_progress_at: datetime | None
    lease_expired_at: datetime
    reclaim_attempts: int
    reclaim_limit: int
    claim_epoch: int


@dataclass(frozen=True)
class MarkResult:
    """A bounded recovery scan, including rows finalised instead of marked."""

    reclaimable: list[ReclaimableRun]
    finalised: list[SweptRun]
    legacy: list[SweptRun]
    inspected: int
    cursor: str | None


def _reclaimable(row: Any) -> ReclaimableRun:
    return ReclaimableRun(
        row.id,
        row.tenant_id,
        row.agent_name,
        _to_aware_utc(row.last_progress_at),
        _to_aware_utc(row.lease_expired_at),
        row.reclaim_attempts,
        row.reclaim_limit,
        row.claim_epoch,
    )


@dataclass(frozen=True)
class SweepResult:
    """Result of a sweep operation."""

    stale_running: list[SweptRun]
    abandoned_waiting: list[SweptRun] = field(default_factory=list)
    reclaimable: list[ReclaimableRun] = field(default_factory=list)
    cursor: str | None = None


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

    capabilities: StoreCapabilities

    async def update_run_meta(
        self, run_id: str, patch: dict[str, Any], *, claim_epoch: int | None = None
    ) -> None: ...

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
        seed_traces: Sequence[TraceWrite] | None = None,
        pii_mapping: dict[str, str] | None = None,
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
        claim_epoch: int | None = None,
        trace_id: str | None = None,
        trace_kind: str = "turn",
        content_digest: str | None = None,
    ) -> str: ...

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
    ) -> str: ...

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
        claim_epoch: int | None = None,
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
        trace: TraceWrite | None = None,
        event: EventWrite | None = None,
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
        claim_epoch: int | None = None,
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
        self, run_id: str, note_ids: list[str], *, iteration: int, claim_epoch: int | None = None
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
    ) -> ClaimResult | Literal[False]: ...

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
        submission: dict[str, Any] | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> ClaimResult | Literal[False]:
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
        claim_epoch: int | None = None,
        event_id: str | None = None,
        content_digest: str | None = None,
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

    async def touch_progress(self, run_id: str, *, claim_epoch: int | None = None) -> bool:
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

    async def checkpoint_step(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        loop_state: dict[str, Any],
        pii_mapping: dict[str, str] | None,
        trace: TraceWrite | None,
    ) -> None: ...

    async def conversation_traces(
        self,
        run_id: str,
        *,
        after_position: int | None,
        limit: int = 500,
        include_superseded: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> ConversationPage: ...

    async def get_traces_after(
        self,
        run_id: str,
        *,
        after_order_index: int | None,
        limit: int = 500,
        kinds: tuple[str, ...] = ("seed", "turn", "rewrite", "state"),
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceRecord]: ...

    async def get_trace(
        self, run_id: str, trace_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> TraceRecord | None: ...

    async def get_loop_state(self, run_id: str) -> dict[str, Any] | None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    async def consume_submission(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        submission_id: str,
        traces: Sequence[TraceWrite],
        patch: LoopStatePatch,
        pii_mapping: dict[str, str] | None = None,
    ) -> bool: ...

    async def mark_lease_expired(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        max_runs: int,
        max_returned: int,
        reclaim_limit: int,
        cursor: str | None,
        finalise_unrecoverable: bool = False,
    ) -> MarkResult: ...

    async def get_lease_expired(
        self, *, limit: int, cursor: str | None
    ) -> list[ReclaimableRun]: ...

    async def reclaim_run(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        agent_name: str,
        expected_claim_epoch: int | None,
        governance: dict[str, Any],
        governance_policy: Literal["warn", "raise"],
        skill_pins: dict[str, str],
        loop_type: str,
    ) -> ClaimResult | Literal[False]: ...

    async def set_trace_link(self, run_id: str, *, claim_epoch: int, trace_link: str) -> None: ...

    async def get_replayed_tool_calls(
        self, run_id: str, *, limit: int = 100, tenant_id: TenantScope = UNSCOPED
    ) -> tuple[list[str], bool]: ...

    async def mark_cancel_requested(
        self, run_ids: Sequence[str], *, tenant_id: TenantScope
    ) -> int: ...

    async def get_children_page(
        self, parent_ids: Sequence[str], *, after: str | None, limit: int, tenant_id: TenantScope
    ) -> list[str]: ...


_PERSISTED_VALUE_FIELDS = frozenset(
    {
        "content",
        "params",
        "result_payload",
        "error_message",
        "meta",
        "patch",
        "loop_state_patch",
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


def _approval_event_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
            "store or use a built-in SQL store."
        )
    return method


async def _write_event(session: AsyncSession, run_id: str, event: EventWrite) -> int:
    from dendrux.types import _validate_persistable

    _validate_persistable(event.data, "data")
    return await _insert_event(
        session,
        run_id,
        event.event_type,
        event.data,
        iteration_index=event.iteration_index,
        correlation_id=event.correlation_id,
        event_id=event.event_id,
        content_digest=event.content_digest,
    )


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
    (:meth:`StateStore.save_run_event` takes the same lock).
    Returns the allocated index.
    """
    return await _insert_event(session, run_id, event_type, data, iteration_index=iteration_index)


# Unique indexes that enforce idempotency keys: the 0.5 tenant-scoped one, and
# the pre-0.5 global ones (Alembic index, Postgres ``create_all`` constraint,
# SQLite inline UNIQUE) still present on databases that were not rebuilt.
_IDEMPOTENCY_VIOLATION_MARKERS = (
    "uq_agent_runs_tenant_idempotency_key",
    "ix_agent_runs_idempotency_key",
    "agent_runs_idempotency_key_key",
    "UNIQUE constraint failed: agent_runs.idempotency_key",
)


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
        storage_format=row.storage_format,
        claim_epoch=row.claim_epoch,
        status_changed_at=_to_aware_utc(row.status_changed_at),
        reclaim_attempts=row.reclaim_attempts,
        reclaim_limit=row.reclaim_limit,
        lease_expired_at=_to_aware_utc(row.lease_expired_at),
        last_reclaimed_at=_to_aware_utc(row.last_reclaimed_at),
        trace_link=row.trace_link,
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
    """Execution-local epoch binding, preserving undeclared stores' legacy signatures."""

    claimed_writes = frozenset(
        {
            "save_trace",
            "save_tool_call",
            "save_usage",
            "save_llm_interaction",
            "save_run_event",
            "touch_progress",
            "add_note",
            "mark_notes_delivered",
            "update_run_meta",
            "checkpoint_step",
            "set_trace_link",
            "start_tool_call",
            "complete_tool_call",
            "consume_submission",
        }
    )

    # These methods commit a fence on every successful return. Completion and
    # submission retries can roll back/no-op, so do not infer renewal from them.
    progress_writes = frozenset(
        {
            "checkpoint_step",
            "start_tool_call",
            "save_trace",
            "save_tool_call",
            "save_run_event",
            "save_usage",
            "save_llm_interaction",
            "touch_progress",
        }
    )

    def __init__(self, store: StateStore, run_id: str, epoch: int | ClaimResult) -> None:
        self._store = store
        self._run_id = run_id
        self.claim_epoch = epoch.claim_epoch if isinstance(epoch, ClaimResult) else epoch
        self.storage_format = epoch.storage_format if isinstance(epoch, ClaimResult) else 1
        self.loop_state = epoch.loop_state if isinstance(epoch, ClaimResult) else None
        self.submission = epoch.submission if isinstance(epoch, ClaimResult) else None

    @property
    def capabilities(self) -> StoreCapabilities:
        from dendrux.runtime._capabilities import capabilities

        return capabilities(self._store)

    @property
    def store_identity(self) -> str | None:
        from dendrux.runtime.context import get_store_identity

        return get_store_identity(self._store)

    def __getattr__(self, name: str) -> Any:
        import inspect

        from dendrux.errors import RunFencedError
        from dendrux.runtime._capabilities import capabilities

        method = getattr(self._store, name)
        status_writes = {
            "finalize_run",
            "finalize_run_if_status_in",
            "finalize_with_event",
            "try_pause_run",
            "pause_with_event",
            "pause_run",
            "release_run_claim",
        }

        enabled = name in status_writes or (
            capabilities(self._store).fenced_writes and name in self.claimed_writes
        )
        if enabled and callable(method) and "claim_epoch" in inspect.signature(method).parameters:

            async def fenced(run_id: str, *args: Any, **kwargs: Any) -> Any:
                kwargs["claim_epoch"] = self.claim_epoch
                started = time.monotonic()
                result = await method(run_id, *args, **kwargs)
                if name == "touch_progress" and result is False:
                    _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
                    raise RunFencedError(run_id, claim_epoch=self.claim_epoch)
                if self.storage_format == 2 and name in self.progress_writes:
                    from dendrux.runtime.context import get_delegation_context

                    ctx = get_delegation_context()
                    if (
                        ctx is not None
                        and ctx.persisted
                        and ctx.run_id == run_id
                        and ctx.state_store is self
                    ):
                        # Use request start, not return time: a slow commit must
                        # not make an old database timestamp appear fresh.
                        ctx._lease_at[0] = max(ctx._lease_at[0], started)
                        ctx._lease_failures[0] = 0
                if name in status_writes and (result is False or result is None):
                    control = await read_run_control(self._store, run_id)
                    if control.claim_epoch is not None and control.claim_epoch != self.claim_epoch:
                        _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
                return result

            return fenced
        return method

    async def get_run_control(self, run_id: str) -> RunControl:
        control = await read_run_control(self._store, run_id)
        fallback = getattr(self._store, "_complete_run_control", None)
        if callable(fallback) and control.claim_epoch is None:
            control = await fallback(run_id, control)
        if (
            (
                getattr(self._store, "_requires_persisted_claim", False)
                and control.claim_epoch is None
            )
            or (control.claim_epoch is not None and control.claim_epoch != self.claim_epoch)
            or (control.status is not None and control.status != RunStatus.RUNNING.value)
        ):
            from dendrux.errors import RunFencedError

            _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
            raise RunFencedError(run_id, claim_epoch=self.claim_epoch)
        return control


def claimed_store(store: StateStore, run_id: str, epoch: int | bool | ClaimResult) -> StateStore:
    """Bind a returned epoch; bool is the pre-0.5 custom-store claim contract."""
    if type(epoch) is bool:
        return store
    if isinstance(store, _ClaimedStateStore):
        if (
            not isinstance(epoch, ClaimResult)
            and run_id == store._run_id
            and epoch == store.claim_epoch
        ):
            return cast("StateStore", store)
        store = store._store
    return cast("StateStore", _ClaimedStateStore(store, run_id, epoch))


def unclaimed_store(store: StateStore | None) -> StateStore | None:
    """The shared store for identity-sensitive capabilities such as blob catalogs."""
    return store._store if isinstance(store, _ClaimedStateStore) else store


async def _insert_event(
    session: AsyncSession, run_id: str, event_type: str, data: dict[str, Any] | None, **kwargs: Any
) -> int:
    return cast(
        "int",
        await session.info["dendrux.store"]._insert_event(
            session, run_id, event_type, data, **kwargs
        ),
    )


def __getattr__(name: str) -> Any:
    if name == "SQLAlchemyStateStore":
        raise ImportError(
            "Removed in 0.6.0; use PostgresStateStore / SQLiteStateStore / store_for_engine."
        )
    raise AttributeError(name)
