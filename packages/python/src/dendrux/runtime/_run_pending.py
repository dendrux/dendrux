"""Reconstruct caller-visible durable continuation state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUSES as _PAUSE_STATUSES
from dendrux._meta_keys import STOP_REASON_META_KEY as STOP_REASON_META_KEY
from dendrux.runtime import _run_errors
from dendrux.runtime._run_errors import logger
from dendrux.runtime.replay import REPLAY_META_KEY
from dendrux.runtime.submit import _claim_failure_mapping
from dendrux.types import (
    PauseState,
    ReplaySummary,
    RunStatus,
    StopReason,
    UsageStats,
    _usage_from_dict,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.runtime.state import StateStore
    from dendrux.types import RunResult

_CONTROL_EVENT_LIMIT = 100

_STOP_REASONS = frozenset(get_args(StopReason))


def _in_memory_pending(result: RunResult) -> Any:
    """Project an ephemeral pause without ever serializing its history."""
    from dendrux.runtime._pending import _KIND_BY_STATUS, _pending_from, approval_calls
    from dendrux.types import Clarification

    state = result.meta.get("pause_state")
    kind = _KIND_BY_STATUS.get(result.status.value)
    if state is None or kind is None:
        return None
    pause = {
        "pause_id": state.pause_id,
        "pending_tool_calls": [
            {"id": c.id, "name": c.name, "params": c.params} for c in state.pending_tool_calls
        ],
        "pending_targets": state.pending_targets,
        "approval_gates": state.approval_gates,
        "pause_reason": state.pause_reason,
        "pause_message": state.pause_message,
        "budget_exceeded": state.budget_exceeded,
    }
    if kind in ("approval", "input"):
        pause["pending_tool_calls"] = approval_calls(pause["pending_tool_calls"], None)
    last = None
    if state.steps and isinstance(state.steps[-1].action, Clarification):
        action = state.steps[-1].action
        last = {
            "action": {
                "type": "clarification",
                "question": action.question,
                "options": action.options,
            }
        }
    return _pending_from(result.run_id, result.status.value, kind, pause, last)


async def _recovery_info(store: StateStore, record: Any, *, best_effort: bool = True) -> Any:
    """Project recovery strictly for reads, or best effort after a durable outcome."""
    if record is None or getattr(record, "last_reclaimed_at", None) is None:
        return None
    return await _read_recovery_info(store, record, best_effort=best_effort)


async def _read_recovery_info(store: StateStore, record: Any, *, best_effort: bool) -> Any:
    from dendrux.types import RecoveryInfo

    try:
        ids, truncated = await store.get_replayed_tool_calls(
            record.id, limit=_CONTROL_EVENT_LIMIT, tenant_id=record.tenant_id
        )
    except Exception:
        if not best_effort:
            raise
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=record.id)
        return None
    return RecoveryInfo(record.reclaim_attempts, record.last_reclaimed_at, ids, truncated)


async def _attach_persisted_pending(store: StateStore, result: RunResult, record: Any) -> RunResult:
    result.recovery = await _recovery_info(store, record)
    if await _pending_moved(store, result, record):
        await _reread_moved(store, result)
    return result


async def _pending_moved(store: StateStore, result: RunResult, record: Any) -> bool:
    """Attach a paused run's pending; True when that read saw the run move on (0.7)."""
    if result.status not in _PAUSE_STATUSES or record is None:
        return False
    return await _load_persisted_pending(store, result, record)


async def _reread_moved(store: StateStore, result: RunResult) -> None:
    """The run moved between its record and pending reads: read it once more (0.7).

    As ``Agent.get_result`` does, so the result is one coherent later view: never
    a stale status with a fresh pending, nor a managed root's private wait.
    """
    try:
        fresh = await store.get_run(result.run_id)
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=result.run_id)
        return
    if fresh is None:
        return
    result.status = RunStatus(fresh.status)
    result.answer, result.error = fresh.answer, fresh.error
    result.error_info = _run_errors._error_info_from_meta(fresh.meta)
    if result.status in _PAUSE_STATUSES:
        await _load_persisted_pending(store, result, fresh)


async def _load_persisted_pending(store: StateStore, result: RunResult, record: Any) -> bool:
    """Attach the run's pending; True when that read saw the run move on (0.7).

    A moved read keeps the later view's status: it is never a private managed
    wait, and it is coherent with the pending it came with.
    """
    from dendrux.runtime._pending import _PendingReader

    result.pending = None
    result.pause_id = None
    reader = _PendingReader(store)
    try:
        result.pending = await reader.get_pending(
            result.run_id, tenant_id=getattr(record, "tenant_id", None)
        )
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=result.run_id)
        return False
    status, moved = _read_status(
        result.status.value, getattr(record, "status", None), reader.last_view
    )
    result.status = RunStatus(status)
    result.pause_id = result.pending.pause_id if result.pending else None
    return moved


def _read_status(status: str, record_status: str | None, view: Any) -> tuple[str, bool]:
    """The status a pending read shows, and whether the run moved since its record.

    A parent waiting on a subagent reports its active branch; a run that moved
    to another public status between the reads keeps that later status (0.7).
    """
    from dendrux.runtime._pending import facing_status

    moved = view is not None and not view.managed and view.status != record_status
    return (view.status if moved else facing_status(status, view)), moved


async def _committed_pause_result(
    store: StateStore, result: RunResult, persisted: Callable[[RunStatus], Awaitable[RunResult]]
) -> RunResult:
    try:
        record = await store.get_run(result.run_id)
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=result.run_id)
        result.pending = None
        result.pause_id = None
        return result
    if record is not None and record.status != result.status.value:
        return await persisted(RunStatus(record.status))
    return await _attach_persisted_pending(store, result, record)


def _ensure_pause_generation(
    result: RunResult, provider_kwargs: dict[str, Any] | None = None
) -> None:
    if result.status in _PAUSE_STATUSES:
        pause = result.meta.get("pause_state")
        if pause is not None:
            if pause.pause_id is None:
                pause.pause_id = generate_ulid()
            pause.provider_kwargs = {
                k: v
                for k, v in (provider_kwargs or pause.provider_kwargs).items()
                if k in {"model", "max_tokens", "temperature", "thinking", "effort"}
            }
            result.pause_id = pause.pause_id
            result.pending = _in_memory_pending(result)


def _pause_event_data(result: RunResult, pause_state: PauseState) -> dict[str, Any]:
    """Build the existing pause lifecycle event without changing its shape."""
    paused_data: dict[str, Any] = {
        "status": result.status.value,
        "pause_id": pause_state.pause_id,
        "pending_tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "target": pause_state.pending_targets.get(tc.id),
                "params": tc.params,
            }
            for tc in pause_state.pending_tool_calls
        ],
    }
    if pause_state.pause_reason is not None:
        paused_data["reason"] = pause_state.pause_reason
    if pause_state.pause_message:
        paused_data["message"] = pause_state.pause_message[: _run_errors.ERROR_TEXT_CHARS]
    if pause_state.budget_exceeded:
        paused_data["exceeded"] = pause_state.budget_exceeded
    return paused_data


async def _final_run_recovery(state_store: StateStore, run_id: str, records: Any) -> Any:
    from dendrux.runtime._capabilities import capabilities

    if not capabilities(state_store).recovery:
        return None
    try:
        record = records[-1] if records else await state_store.get_run(run_id)
    except Exception:
        _codes.log_code(logger, _codes.RESULT_READ_FAILED, run_id=run_id)
        return None
    return await _recovery_info(state_store, record)


async def _raise_resume_claim_failure(
    state_store: StateStore, run_id: str, expected_status: str
) -> None:
    """Raise a descriptive error when resume claim fails.

    Checks the actual run status to give a specific message for swept
    (abandoned/stale) runs instead of the generic "claimed by another caller."
    """
    run = await state_store.get_run(run_id)
    rule = _claim_failure_mapping(run, after_preflight=True)
    raise ValueError(
        rule.resume_message.format(
            run_id=run_id, expected=expected_status, reason=getattr(run, "failure_reason", None)
        )
    )


def _cached_output(answer: str | None, output_type: type | None, run_id: str) -> Any:
    """Recover only a configured structured schema, failing closed on old data."""
    # Rebuild typed output from persisted answer if output_type was requested
    output = None
    if output_type is not None and answer:
        from pydantic import ValidationError

        try:
            output = output_type.model_validate_json(answer)  # type: ignore[attr-defined]
        except (ValidationError, Exception):
            # If the persisted answer doesn't validate (schema evolved,
            # corruption, etc.), return None rather than crashing the
            # dedup path. The caller still gets answer as raw text.
            logger.warning(
                "Cached run %s answer failed validation against %s",
                run_id,
                output_type.__name__,
            )

    return output


async def _build_cached_result(
    state_store: StateStore,
    run_id: str,
    output_type: type | None = None,
) -> RunResult:
    """Rebuild a summary RunResult from persisted DB state for idempotent dedup.

    This is a *summary*, not a faithful replay of the original RunResult:
    - steps is always [] (AgentStep history is not persisted in recoverable form)
    - developer meta is omitted; known PII is retained for safe wire projection
    - stop_reason is restored from ``meta[STOP_REASON_META_KEY]`` on SUCCESS
      (None for runs finished before 0.4.1)
    - usage is reconstructed from aggregate columns (total_input_tokens, etc.),
      which may be zero for errored runs where finalize_run got total_usage=None

    When output_type is provided, re-validates the persisted answer against
    the schema to populate result.output. This ensures the idempotent path
    returns the same typed result as a fresh run.

    The contract is: "return the cached final outcome" — status, answer, error,
    iteration count, and aggregate usage. Not "return the identical object."
    """
    from dendrux.types import RunResult

    run = await state_store.get_run(run_id)
    if run is None:
        raise RuntimeError(f"Idempotent run {run_id} not found in database")

    output = _cached_output(run.answer, output_type, run_id)

    result = RunResult(
        run_id=run.id,
        status=RunStatus(run.status),
        answer=run.answer,
        output=output,
        steps=[],
        iteration_count=run.iteration_count,
        usage=(
            _usage_from_dict(run.output_data["usage"])
            if run.output_data and isinstance(run.output_data.get("usage"), dict)
            else UsageStats(
                input_tokens=run.total_input_tokens,
                output_tokens=run.total_output_tokens,
                total_tokens=run.total_input_tokens + run.total_output_tokens,
                cost_usd=run.total_cost_usd,
                reasoning_tokens=run.total_reasoning_tokens,
            )
        ),
        error=run.error,
        error_info=_run_errors._error_info_from_meta(run.meta),
        replay=_replay_from_meta(run.meta),
        stop_reason=(
            _stop_reason_from_meta(run.meta) if run.status == RunStatus.SUCCESS.value else None
        ),
    )

    mapping = await state_store.get_pii_mapping(run_id)
    if mapping:
        result.meta = _run_errors._public_meta({"pii_mapping": mapping})
    return await _attach_persisted_pending(state_store, result, run)


def _stop_reason_from_meta(meta: dict[str, Any] | None) -> StopReason | None:
    value = (meta or {}).get(STOP_REASON_META_KEY)
    return value if value in _STOP_REASONS else None


def _replay_from_meta(meta: dict[str, Any] | None) -> ReplaySummary | None:
    data = (meta or {}).get(REPLAY_META_KEY)
    return ReplaySummary.from_dict(data) if isinstance(data, dict) else None
