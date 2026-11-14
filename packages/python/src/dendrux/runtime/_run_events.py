"""Apply claim-fenced runner lifecycle transitions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._lifecycle import (
    CancelReason,
    StoredEventType,
)
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY,
)
from dendrux._sentinel import _OMITTED, _OmittedType
from dendrux.errors import (
    RunFencedError,
    safe_exception_info,
)
from dendrux.runtime import (
    _run_contracts,
    _run_pending,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.types import (
    ErrorInfo,
    RunStatus,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.runtime.tasks import RunTaskManager
    from dendrux.types import (
        RunEvent,
        RunResult,
    )


import dendrux.loops._helpers as _hooks


async def _finalize_running_event(
    store: StateStore,
    run_id: str,
    status: str,
    event_type: str,
    event_data: dict[str, Any],
    *,
    emit_event: bool = True,
    **kwargs: Any,
) -> tuple[bool, bool]:
    atomic = getattr(store, "finalize_with_event", None)
    if atomic is not None and emit_event:
        kwargs.pop("error_info", None)  # Carried by the atomic lifecycle event.
        seq = await atomic(
            run_id,
            status=status,
            allowed_current_statuses=[RunStatus.RUNNING.value],
            event_type=event_type,
            event_data=event_data,
            **kwargs,
        )
        return seq is not None, False
    won = await store.finalize_run(
        run_id, status=status, expected_current_status=RunStatus.RUNNING.value, **kwargs
    )
    return bool(won), bool(won and emit_event)


def _take_task_interrupt(manager: RunTaskManager | None, run_id: str) -> tuple[bool, bool]:
    if manager is None:
        return False, False
    interrupted, others = manager.take_interrupt(run_id)
    task = asyncio.current_task()
    swallow = interrupted and not others and task is not None and task.uncancel() == 0
    return interrupted, swallow


def _may_swallow_cancellation(swallow: bool) -> bool:
    task = asyncio.current_task()
    return swallow and (task is None or task.cancelling() == 0)


async def _drain_task_cancellation(
    store: StateStore | None, run_id: str, interrupted: bool
) -> RunResult:
    from dendrux.runtime.tasks import _drain_cleanup

    result, _ = await _drain_cleanup(
        _persist_task_cancellation(store, run_id, interrupted=interrupted),
        name=f"cancel-run:{run_id}",
        run_id=run_id,
    )
    return result


async def _emit_event(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Record a durable run-level lifecycle event. Fail-closed with retry.

    Runner lifecycle events (run.started, run.paused, run.completed,
    run.error, run.resumed, run.cancelled) are authoritative truth —
    they get the same retry/propagate contract as PersistenceRecorder's
    fail-closed writes. The store allocates the event's sequence index in
    the insert, so every writer of a run (runner, recorder, ``cancel_run``,
    another process) gets the next index in commit order.
    """
    if state_store is None:
        return

    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.durability import retry_critical

    identity: dict[str, Any] = (
        {"event_id": generate_ulid()} if capabilities(state_store).retry_ids else {}
    )

    async def _write() -> None:
        from dendrux.runtime.state import validate_event_index, validate_event_writer

        validate_event_writer(state_store)
        index = await state_store.save_run_event(
            run_id,
            event_type=event_type,
            correlation_id=correlation_id,
            data=data,
            **identity,
        )
        validate_event_index(index)

    await retry_critical(_write, label=f"emit_{event_type}", run_id=run_id)


async def _emit_event_safe(
    state_store: StateStore | None,
    run_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Best-effort evidence or failure event; ownership fencing still propagates."""
    if state_store is None:
        return
    try:
        await _emit_event(state_store, run_id, event_type, data, correlation_id)
    except RunFencedError:
        raise
    except Exception:
        logger.warning(
            "Failed to record %s for run %s",
            event_type,
            run_id,
            exc_info=safe_exception_info(),
        )


async def _attach_undelivered_notes(
    state_store: StateStore, run_id: str, result: RunResult
) -> None:
    """List inbox notes the run ended without delivering in ``meta["undelivered_notes"]``.

    Only set when there are any, so runs that never get notes are unchanged.
    """
    get_notes = getattr(state_store, "get_notes", None)
    if get_notes is None:  # pre-0.3 store: no inbox
        return
    try:
        notes = await get_notes(run_id, undelivered_only=True)
    except Exception:
        logger.warning(
            "Failed to read undelivered notes for run %s", run_id, exc_info=safe_exception_info()
        )
        return
    if notes:
        result.meta["undelivered_notes"] = [n.id for n in notes]


async def _pause_request_message(state_store: StateStore, run_id: str) -> str | None:
    """The ``reason=`` text of the latest ``request_pause`` on this run, if any."""
    from dendrux.runtime.state import PAUSE_REQUEST_META_KEY

    try:
        record = await state_store.get_run(run_id)
    except Exception:
        logger.warning(
            "Failed to read pause request for run %s", run_id, exc_info=safe_exception_info()
        )
        return None
    request = ((record.meta if record else None) or {}).get(PAUSE_REQUEST_META_KEY)
    reason = request.get("reason") if isinstance(request, dict) else None
    return reason if isinstance(reason, str) else None


async def _transition_with_event(
    state_store: StateStore,
    run_id: str,
    *,
    status: str,
    allowed_current_statuses: list[str],
    event_type: str,
    event_data: dict[str, Any],
    **finalize_kwargs: Any,
) -> bool:
    """CAS-finalize and record the transition event atomically when the store can.

    Pre-0.3 stores lack ``finalize_with_event``: fall back to the CAS
    followed by a separate event write. Their CAS cannot store a
    ``failure_reason``; the transition event still carries the reason.
    """
    atomic = getattr(state_store, "finalize_with_event", None)
    # Stores predating a keyword (e.g. failure_reason) cannot persist it;
    # the transition and its event (which carries the reason) still happen.
    if finalize_kwargs.get("failure_reason") is None or not _run_contracts._accepts_kwarg(
        atomic or state_store.finalize_run_if_status_in, "failure_reason"
    ):
        finalize_kwargs.pop("failure_reason", None)
    if atomic is not None:
        seq = await atomic(
            run_id,
            status=status,
            allowed_current_statuses=allowed_current_statuses,
            event_type=event_type,
            event_data=event_data,
            **finalize_kwargs,
        )
        return seq is not None
    won = await state_store.finalize_run_if_status_in(
        run_id,
        status=status,
        allowed_current_statuses=allowed_current_statuses,
        **finalize_kwargs,
    )
    if won:
        await _emit_event(state_store, run_id, event_type, event_data)
    return bool(won)


async def _pause_with_event(
    state_store: StateStore,
    run_id: str,
    *,
    status: str,
    pause_data: dict[str, Any],
    event_data: dict[str, Any],
    iteration_count: int | None,
    pii_mapping: dict[str, str] | None,
    total_usage: UsageStats | None = None,
) -> bool:
    """Persist a pause and its ``run.paused`` event; False when cancellation won.

    ``total_usage`` is written as the run's running totals so a paused run
    does not read as zero spend. Pre-0.3 stores lack ``pause_with_event``:
    fall back to the legacy cancel read, unconditional ``pause_run`` (with
    totals only when it accepts them) and a separate event write.
    """
    usage_kwargs: dict[str, Any] = {"total_usage": total_usage} if total_usage else {}
    atomic = getattr(state_store, "pause_with_event", None)
    if not _run_contracts._accepts_kwarg(atomic or state_store.pause_run, "total_usage"):
        usage_kwargs = {}
    if atomic is not None:
        seq = await atomic(
            run_id,
            status=status,
            pause_data=pause_data,
            event_data=event_data,
            iteration_count=iteration_count,
            pii_mapping=pii_mapping,
            **usage_kwargs,
        )
        return seq is not None
    if await state_store.is_cancel_requested(run_id):
        return False
    await state_store.pause_run(
        run_id,
        status=status,
        pause_data=pause_data,
        iteration_count=iteration_count,
        pii_mapping=pii_mapping,
        **usage_kwargs,
    )
    await _emit_event(state_store, run_id, StoredEventType.RUN_PAUSED.value, event_data)
    # The read and the pause are separate here: a cancel committed in between
    # left a paused row with cancel_requested set. Finish it now; the caller
    # reloads and returns the persisted winner.
    if await state_store.is_cancel_requested(run_id):
        await _transition_with_event(
            state_store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[status],
            event_type=StoredEventType.RUN_CANCELLED.value,
            event_data={"reason": CancelReason.CANCEL_REQUESTED.value},
        )
    return True


async def _read_cancel_winner(
    store: StateStore, run_id: str, *, safe: bool = False
) -> RunRecord | None:
    try:
        return await store.get_run(run_id)
    except Exception:
        if not safe:
            raise
        return None


async def _persist_run_cancellation(
    store: StateStore,
    run_id: str,
    payload: dict[str, Any],
    *,
    on_won: Callable[[], None],
    answer: str | None | _OmittedType = _OMITTED,
    emit_event: bool = True,
    read_on_loss: Callable[[], bool] | None = None,
    safe_read: bool = False,
) -> tuple[bool, RunRecord | None]:
    kwargs: dict[str, Any] = {} if isinstance(answer, _OmittedType) else {"answer": answer}
    won, needs_event = await _finalize_running_event(
        store,
        run_id,
        RunStatus.CANCELLED.value,
        StoredEventType.RUN_CANCELLED.value,
        payload,
        emit_event=emit_event,
        **kwargs,
    )
    if won:
        # Remember the committed CAS even if the separate event write raises.
        on_won()
        if needs_event:
            await _emit_event(store, run_id, StoredEventType.RUN_CANCELLED.value, payload)
        return True, None
    record = None
    if read_on_loss is None or read_on_loss():
        record = await _read_cancel_winner(store, run_id, safe=safe_read)
    return False, record


async def _persist_task_cancellation(
    state_store: StateStore | None,
    run_id: str,
    *,
    interrupted: bool = False,
) -> RunResult:
    """Finalize a still-RUNNING run as CANCELLED after its task was cancelled.

    ``interrupted`` marks a ``cancel_run`` interrupt of a batch ``run()``:
    the result and the ``run.cancelled`` event carry ``interrupted=True`` and
    ``partial_output=False`` (a batch provider call has no partial text).
    """
    from dendrux.types import RunResult

    interrupt_meta = {"interrupted": True, "partial_output": False} if interrupted else {}
    result = RunResult(
        run_id=run_id,
        status=RunStatus.CANCELLED,
        meta={**interrupt_meta, CANCEL_PERSISTED_META_KEY: False},
    )
    if state_store is None:
        return result
    try:
        won, record = await _persist_run_cancellation(
            state_store,
            run_id,
            {"reason": CancelReason.CANCEL_REQUESTED.value, **interrupt_meta}
            if interrupted
            else {"reason": CancelReason.TASK_CANCELLED.value},
            on_won=lambda: result.meta.update({CANCEL_PERSISTED_META_KEY: True}),
        )
        if won:
            await _attach_undelivered_notes(state_store, run_id, result)
        elif record is not None:
            result = await _run_pending._build_cached_result(state_store, run_id)
            if result.status == RunStatus.CANCELLED:
                result.meta[CANCEL_PERSISTED_META_KEY] = True
        # No row: cancelled before create_run committed, or while resolving an
        # idempotent hit under the provisional id. Nothing to finalize.
    except Exception:
        _codes.log_code(logger, _codes.CANCEL_PERSIST_FAILED, run_id=run_id)
    return result


def _fenced_stream_event(run_id: str, result: RunResult) -> RunEvent:
    from dataclasses import replace

    from dendrux.types import RunEvent, RunEventType

    types = {
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
        RunStatus.SUCCESS: RunEventType.RUN_COMPLETED,
        RunStatus.MAX_ITERATIONS: RunEventType.RUN_COMPLETED,
        RunStatus.PAUSED: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_CLIENT_TOOL: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_APPROVAL: RunEventType.RUN_PAUSED,
        RunStatus.WAITING_HUMAN_INPUT: RunEventType.RUN_PAUSED,
    }
    if result.status not in types:
        info = ErrorInfo(
            category="run_fenced",
            retryable=False,
            safe_message="Execution ownership changed; read the run's current state.",
            error_type="RunFencedError",
        )
        return RunEvent(
            type=RunEventType.RUN_ERROR,
            run_id=run_id,
            run_result=replace(result, error_info=info),
            error=info.safe_message,
        )
    return RunEvent(
        type=types[result.status],
        run_id=run_id,
        run_result=result,
        error=result.error if result.status == RunStatus.ERROR else None,
    )


async def _finish_fenced_lifecycle(
    run_id: str,
    result: RunResult,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
) -> None:
    """Close local lifecycle sinks with the persisted winner, without a status write."""
    try:
        await _hooks.record_run_finished(recorder, run_id, result)
    except Exception:
        _codes.log_code(logger, _codes.FENCED_FINISH_FAILED, run_id=run_id)
    await _hooks.notify_run_finished(notifier, run_id, result)


async def _check_claim(store: StateStore, run_id: str) -> None:
    getter = getattr(store, "get_run_control", None)
    if getter is not None:
        await getter(run_id)


async def _shielded_cleanup(cleanup: Callable[[], Any], *, run_id: str | None = None) -> Any:
    import anyio

    from dendrux.runtime.tasks import _drain_cleanup

    result, cancellation = await _drain_cleanup(cleanup(), run_id=run_id)
    if cancellation is not None:
        raise cancellation
    await anyio.lowlevel.checkpoint_if_cancelled()
    return result
