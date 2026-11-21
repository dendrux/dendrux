"""Cancel a persisted run and its subtree through an unclaimed administrative store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import PAUSE_STATUS_VALUES, TERMINAL_STATUSES, CancelReason, StoredEventType
from dendrux.errors import RunNotFoundError
from dendrux.runtime import _managed_cancel
from dendrux.runtime._run_errors import _error_info_from_meta
from dendrux.runtime._run_events import _attach_undelivered_notes, _transition_with_event
from dendrux.runtime._run_pending import _recovery_info, _stop_reason_from_meta
from dendrux.runtime.state import UNSCOPED
from dendrux.types import RunResult, RunStatus, UsageStats

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.runtime.state import RunRecord, StateStore, TenantScope

    LocalSignal = Callable[[str], object]


async def cancel_run(
    store: StateStore,
    run_id: str,
    *,
    tenant_id: TenantScope,
    cascade: bool,
    max_nodes: int,
    signal: LocalSignal | None = None,
) -> RunResult:
    """Request cancellation of ``run_id`` and settle what can be settled now.

    ``store`` is the caller's unclaimed administrative store and ``tenant_id`` its resolved
    scope; ``max_nodes`` bounds the subtree traversal. Managed roots always traverse their
    managed subtree (``cascade`` adds legacy edges) and managed children are refused.
    Each run gets the durable request first, then a status CAS over pause statuses only,
    with its terminal event written atomically; a RUNNING row is never finalized here.
    ``signal`` interrupts same-process work for a run id: after a frontier is marked,
    when a run's CAS fails, and for every requested run once the call ends, success or not.
    """
    record = await store.get_run(run_id)
    if record is None or not (tenant_id is UNSCOPED or record.tenant_id == tenant_id):
        raise RunNotFoundError(run_id)
    cascade, pages = await _managed_cancel.cancel_scope(store, record, cascade=cascade)
    settlement = _Settlement(signal, cascade and _managed_cancel.can_mark_cancel(store))
    try:
        truncated = False
        if cascade:
            truncated = await _settle_descendants(store, record, pages, settlement, max_nodes)
        if record.status in TERMINAL_STATUSES:
            settlement.outcomes.append({"run_id": run_id, "outcome": "terminal"})
        else:
            await settlement.settle_root(store, run_id)
        if cascade or record.status not in TERMINAL_STATUSES:
            record = await store.get_run(run_id)
            if record is None:  # pragma: no cover — concurrent deletion
                raise RunNotFoundError(run_id)
        result = await cancel_result(store, record)
        if cascade:
            settlement.describe(result, truncated=truncated)
        return result
    finally:
        settlement.flush()


async def cancel_one(store: StateStore, run_id: str, *, signal: LocalSignal) -> str:
    """Cancel one run: set the flag, then CAS-finalize it only if it is paused.

    Returns ``"terminal"``, ``"cancelled"`` or ``"cancel_requested"``. ``signal`` runs only
    when the CAS write raises, so a committed request still interrupts local work.
    """
    # The cooperative flag is observed by runner checkpoints. If paused the CAS wins
    # instantly; if running it misses and the runner finalizes itself at a checkpoint.
    if not await store.request_cancel(run_id):
        return "terminal"
    # Paused-run cancel owns the terminal event, written in the CAS's transaction (sequence
    # allocated under the row lock) so it always sorts after the runner's run.paused.
    try:
        cancel_won = await _transition_with_event(
            store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=list(PAUSE_STATUS_VALUES),
            event_type=StoredEventType.RUN_CANCELLED.value,
            event_data={"reason": CancelReason.CANCEL_REQUESTED.value},
        )
    except BaseException:
        signal(run_id)
        raise
    return RunStatus.CANCELLED.value if cancel_won else CancelReason.CANCEL_REQUESTED.value


async def cancel_result(store: StateStore, record: RunRecord) -> RunResult:
    """Project the persisted state after a cancel attempt, with recovery and notes."""
    result = record_to_run_result(record)
    result.recovery = await _recovery_info(store, record)
    if result.status.value in TERMINAL_STATUSES:
        await _attach_undelivered_notes(store, record.id, result)
    return result


def record_to_run_result(record: Any) -> RunResult:
    """Project a RunRecord onto a RunResult for cancel/no-op return paths."""
    usage = UsageStats(
        input_tokens=record.total_input_tokens,
        output_tokens=record.total_output_tokens,
        total_tokens=record.total_input_tokens + record.total_output_tokens,
        cost_usd=record.total_cost_usd,
        cache_read_input_tokens=record.total_cache_read_tokens,
        cache_creation_input_tokens=record.total_cache_creation_tokens,
    )
    status = RunStatus(record.status)
    return RunResult(
        run_id=record.id,
        status=status,
        answer=record.answer,
        error=record.error,
        error_info=_error_info_from_meta(record.meta),
        iteration_count=record.iteration_count,
        usage=usage,
        meta={"cancel_requested": bool(record.cancel_requested)},
        stop_reason=(
            _stop_reason_from_meta(getattr(record, "meta", None))
            if status == RunStatus.SUCCESS
            else None
        ),
    )


async def _settle_descendants(
    store: StateStore,
    record: RunRecord,
    pages: _managed_cancel.CancelPageReader | None,
    settlement: _Settlement,
    max_nodes: int,
) -> bool:
    """Mark root first, frontier by frontier, then settle descendants deepest first."""
    levels, truncated = await _managed_cancel.mark_subtree(
        store,
        record.id,
        tenant_id=record.tenant_id,
        limit=max_nodes,
        signal=settlement.notify,
        pages=pages,
    )
    if pages is not None:
        settlement.outcomes.extend(
            {"run_id": rid, "outcome": "skipped"} for rid in pages.skipped_run_ids
        )
    for rid in (rid for level in reversed(levels) for rid in level if rid != record.id):
        await settlement.settle_descendant(store, rid)
    return truncated


@dataclass
class _Settlement:
    """Outcomes of one cancel call, and the runs to signal locally when it ends.

    When every frontier is marked (and signalled) before scanning, requested runs are
    already signalled; otherwise each requested run is signalled once the call ends.
    """

    signal: LocalSignal | None
    frontier_signals: bool
    outcomes: list[dict[str, str]] = field(default_factory=list)
    to_signal: list[str] = field(default_factory=list)
    count: int = 0

    async def settle_root(self, store: StateStore, run_id: str) -> None:
        self.outcomes.append({"run_id": run_id, "outcome": await self._cancel(store, run_id)})

    async def settle_descendant(self, store: StateStore, run_id: str) -> None:
        outcome = await self._cancel(store, run_id)
        self.count += 1
        if len(self.outcomes) < _managed_cancel._CANCEL_OUTCOME_LIMIT:
            self.outcomes.append({"run_id": run_id, "outcome": outcome})

    async def _cancel(self, store: StateStore, run_id: str) -> str:
        outcome = await cancel_one(store, run_id, signal=self.notify)
        if outcome != "terminal" and not self.frontier_signals:
            self.to_signal.append(run_id)
        return outcome

    def describe(self, result: RunResult, *, truncated: bool) -> None:
        result.meta["cascade"] = self.outcomes
        result.meta["cascade_count"] = self.count + 1
        result.meta["cascade_truncated"] = truncated or self.count + 1 > len(self.outcomes)

    def notify(self, run_id: str) -> None:
        if self.signal is not None:
            self.signal(run_id)

    def flush(self) -> None:
        for rid in self.to_signal:
            self.notify(rid)
