"""Project and persist loop results/events as terminal, paused or managed-handoff outcomes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import (
    CANCEL_FINALIZE_STATUS_VALUES,
    STORAGE_FORMAT_CHECKPOINTED,
    CancelReason,
    FailureReason,
    PauseReason,
    StoredEventType,
)
from dendrux._lifecycle import PAUSE_STATUSES as _PAUSE_STATUSES
from dendrux._meta_keys import (
    STOP_REASON_META_KEY,
)
from dendrux.errors import (
    safe_exception_info,
)
from dendrux.loops._budget import BUDGET_EXCEEDED
from dendrux.loops._helpers import (
    project_message,
)
from dendrux.loops.react import MANAGED_HANDOFF_KEY
from dendrux.runtime import (
    _managed,
    _managed_budget,
    _run_contracts,
    _run_errors,
    _run_events,
    _run_observers,
    _run_pending,
    _run_start,
)
from dendrux.runtime._run_errors import logger
from dendrux.runtime.state import (
    unclaimed_store,
)
from dendrux.runtime.submit import (
    GOVERNANCE_KEY,
)
from dendrux.types import (
    GovernanceEventType,
    PauseState,
    RunEventType,
    RunStatus,
    StopReason,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
    )

    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.replay import ReplayPlan
    from dendrux.runtime.state import StateStore
    from dendrux.types import (
        RunEvent,
        RunResult,
    )


@asynccontextmanager
async def _closing_loop_stream(
    loop_stream: AsyncGenerator[RunEvent, None], run_id: str
) -> AsyncIterator[AsyncGenerator[RunEvent, None]]:
    """Close the loop generator in this task, never raising out of the close.

    When the consumer abandons the stream, ``GeneratorExit`` is travelling
    through the runner generator. An exception escaping the loop's cleanup
    (a provider's close failing, say) would land in the runner's
    ``except Exception`` and make it *yield* while closing — which Python
    reports to the consumer as ``async generator ignored GeneratorExit``.
    Cleanup failures are logged instead; the run is still finalized by the
    stream's cleanup callback.
    """
    try:
        yield loop_stream
    finally:
        try:
            await loop_stream.aclose()
        except Exception:
            logger.warning(
                "Loop stream cleanup raised for run %s", run_id, exc_info=safe_exception_info()
            )


def _terminal_stream_result(event: RunEvent) -> RunResult | None:
    if event.type in (
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_ERROR,
        RunEventType.RUN_PAUSED,
        RunEventType.RUN_CANCELLED,
    ):
        return event.run_result
    return None


async def _finish_stream_event(
    event: RunEvent,
    result: RunResult,
    *,
    state_store: StateStore | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    provider_kwargs: dict[str, Any] | None,
) -> RunEvent:
    from dendrux.types import RunEvent

    if state_store is not None:
        result = await _persist_loop_outcome(
            recorder=recorder,
            provider_kwargs=provider_kwargs,
            state_store=state_store,
            run_id=run_id,
            result=result,
        )
        # A pre-pause cancellation changes the terminal event delivered to the consumer.
        if result.status == RunStatus.CANCELLED:
            event = RunEvent(type=RunEventType.RUN_CANCELLED, run_result=result)
        else:
            event = RunEvent(type=event.type, run_result=result, phase=event.phase)
    # A managed handoff is not finished: the parent continues after adoption.
    await _run_start._record_finished(recorder, notifier, run_id, result)
    if state_store is None:
        result.meta = _run_errors._public_meta(result.meta)
    return event


def _prepare_stream_result(
    result: RunResult,
    provider_kwargs: dict[str, Any] | None,
    mcp_meta: dict[str, Any],
    replay_plan: ReplayPlan | None,
    new_run: bool,
) -> None:
    if new_run:
        _run_pending._ensure_pause_generation(result, provider_kwargs)
        if replay_plan is not None:
            result.replay = replay_plan.summary
    _run_observers._attach_mcp_meta(result, mcp_meta)


async def _stream_loop_events(
    loop_stream: AsyncGenerator[RunEvent, None],
    *,
    state_store: StateStore | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    provider_kwargs: dict[str, Any] | None,
    shared: dict[str, Any],
    mcp_meta: dict[str, Any],
    replay_plan: ReplayPlan | None = None,
    new_run: bool = False,
) -> AsyncGenerator[RunEvent, None]:
    async with _closing_loop_stream(loop_stream, run_id):
        async for event in loop_stream:
            result = _terminal_stream_result(event)
            if result is None or (not new_run and not result):
                _track_partial_text(shared, event)
                yield event
                continue
            _prepare_stream_result(result, provider_kwargs, mcp_meta, replay_plan, new_run)
            yield await _finish_stream_event(
                event,
                result,
                state_store=state_store,
                recorder=recorder,
                notifier=notifier,
                run_id=run_id,
                provider_kwargs=provider_kwargs,
            )


def _track_partial_text(shared: dict[str, Any], event: RunEvent) -> None:
    """Buffer streamed text so an abandoned stream can persist a partial answer.

    The buffer holds the current LLM call's deltas; it resets when a tool
    result passes, since a new call follows.
    """
    if event.type == RunEventType.TEXT_DELTA:
        shared.setdefault("partial_text", []).append(event.text or "")
    elif event.type == RunEventType.TOOL_RESULT:
        shared.pop("partial_text", None)


def _partial_answer(shared: dict[str, Any]) -> str | None:
    parts = shared.get("partial_text")
    return "".join(parts) if parts else None


def _terminal_writer(store: StateStore) -> Any:
    """The method that will write a loop's terminal status on ``store``."""
    return getattr(store, "finalize_with_event", None) or store.finalize_run


def _terminal_failure_reason(result: RunResult) -> str | None:
    """The reason a returned error ended the run, for the reasons a store may keep."""
    if result.status != RunStatus.ERROR:
        return None
    reason = result.meta.get("failure_reason")
    known = (FailureReason.GUARDRAIL_BLOCKED.value, FailureReason.PENDING_LIMIT.value)
    return reason if reason in known else None


def _terminal_failure_kwargs(state_store: StateStore, result: RunResult) -> dict[str, Any]:
    """Keep guardrail reason support optional for legacy custom stores."""
    base = unclaimed_store(state_store)
    reason = _terminal_failure_reason(result)
    if (
        base is None
        or reason is None
        or not _run_contracts._accepts_kwarg(_terminal_writer(base), "failure_reason")
    ):
        return {}
    return {"failure_reason": reason}


def _unstored_failure_reason(result: RunResult, stored: dict[str, Any]) -> dict[str, Any]:
    """The reason for the event, when the store's write cannot keep it on the row."""
    reason = _terminal_failure_reason(result)
    return {"failure_reason": reason} if reason and "failure_reason" not in stored else {}


def _wrap_loop_result(
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    *,
    status: RunStatus,
    won: bool,
    recovery: Any = None,
) -> RunResult:
    from dendrux.guardrails._engine import deanonymize_text
    from dendrux.types import RunResult

    pause_id, pending = (
        (result.pause_id, result.pending) if status in _PAUSE_STATUSES else (None, None)
    )
    return RunResult(
        run_id=run_id,
        status=status,
        answer=user_answer,
        output=result.output,
        error_info=result.error_info,
        error=result.error,
        steps=result.steps,
        iteration_count=result.iteration_count,
        usage=result.usage,
        meta={**_run_errors._public_meta(result.meta), "_finalize_won": won},
        commentary=[deanonymize_text(c, pii_mapping)[0] or c for c in result.commentary],
        replay=result.replay,
        stop_reason=result.stop_reason if status == RunStatus.SUCCESS else None,
        pause_id=pause_id,
        pending=pending,
        recovery=recovery,
    )


async def _persisted_loop_result(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    status_if_missing: RunStatus,
) -> RunResult:
    record = await state_store.get_run(run_id)
    status = RunStatus(record.status) if record is not None else status_if_missing
    persisted = _wrap_loop_result(
        run_id, result, user_answer, pii_mapping, status=status, won=False
    )
    if record is not None:
        persisted.answer = record.answer
        persisted.output = None
        persisted.error = record.error
        persisted.error_info = _run_errors._error_info_from_meta(record.meta)
    if status not in _PAUSE_STATUSES:
        persisted.meta.pop("pause_state", None)
        await _run_events._attach_undelivered_notes(state_store, run_id, persisted)
    return await _run_pending._attach_persisted_pending(state_store, persisted, record)


async def _finalize_loop_cancelled(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    user_answer: str | None,
    pii_mapping: dict[str, Any],
    allowed: list[str],
    payload: dict[str, Any],
    failure_reason: str | None = None,
) -> RunResult:
    from dendrux.runtime.state import _capture_finalized_run

    with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
        won = await _run_events._transition_with_event(
            state_store,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=allowed,
            event_type=StoredEventType.RUN_CANCELLED.value,
            event_data=payload,
            answer=user_answer,
            iteration_count=result.iteration_count,
            total_usage=result.usage,
            pii_mapping=result.meta.get("pii_mapping"),
            failure_reason=failure_reason,
        )
    if not won:
        return await _persisted_loop_result(
            state_store, run_id, result, user_answer, pii_mapping, RunStatus.CANCELLED
        )
    recovery = await _run_pending._final_run_recovery(state_store, run_id, records)
    cancelled = _wrap_loop_result(
        run_id,
        result,
        user_answer,
        pii_mapping,
        status=RunStatus.CANCELLED,
        won=True,
        recovery=recovery,
    )
    cancelled.meta.pop("pause_state", None)
    await _run_events._attach_undelivered_notes(state_store, run_id, cancelled)
    return cancelled


def _loop_cancel_request(result: RunResult) -> tuple[list[str], dict[str, Any], str | None] | None:
    if result.status != RunStatus.CANCELLED:
        return None
    if result.meta.get("budget_exceeded"):
        return (
            [RunStatus.RUNNING.value],
            {"reason": BUDGET_EXCEEDED, "exceeded": result.meta["budget_exceeded"]},
            BUDGET_EXCEEDED,
        )
    payload: dict[str, Any] = {"reason": CancelReason.CANCEL_REQUESTED.value}
    if result.meta.get("interrupted"):
        payload["interrupted"] = True
        payload["partial_output"] = bool(result.meta.get("partial_output"))
    return list(CANCEL_FINALIZE_STATUS_VALUES), payload, None


def _terminal_loop_event(result: RunResult) -> tuple[StopReason | None, str, dict[str, Any]]:
    stop_reason = result.stop_reason if result.status == RunStatus.SUCCESS else None
    event_type = (
        StoredEventType.RUN_ERROR.value
        if result.status == RunStatus.ERROR
        else StoredEventType.RUN_COMPLETED.value
    )
    event_data = {
        **({"error": result.error} if result.status == RunStatus.ERROR else {}),
        "status": result.status.value,
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_tokens",
    }
    return stop_reason, event_type, event_data


def _completion_evidence(
    state_store: StateStore,
    result: RunResult,
    recorder: LoopRecorder | None,
    event_type: str,
    event_data: dict[str, Any],
) -> dict[str, Any]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.state import EventWrite

    if getattr(state_store, "storage_format", 1) != STORAGE_FORMAT_CHECKPOINTED:
        return {}
    if not capabilities(state_store).atomic_completion:
        return {}
    return {
        "trace": getattr(recorder, "finalization_trace", lambda: None)(),
        "event": EventWrite(
            generate_ulid(), event_type, result.iteration_count, event_data, None, None
        ),
    }


async def _persist_loop_outcome(
    *,
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    provider_kwargs: dict[str, Any] | None = None,
    recorder: LoopRecorder | None = None,
) -> RunResult:
    """Persist a loop's terminal/pause result and emit the lifecycle event.

    Centralizes pause-vs-finalize handling across initial run, retry,
    resume, and their streaming variants. Implements the cooperative
    cancellation pre-pause checkpoint: if ``cancel_requested`` was set
    during the iteration that just returned WAITING_*, we finalize as
    ``cancelled`` instead of pausing — otherwise the run would settle
    in a paused state holding pause_data the user already asked us to
    abandon.

    Always returns a fresh ``RunResult``; never mutates the input.
    ``meta['_finalize_won']`` records whether the CAS finalize won
    (``True`` for pauses since pause is unconditional).

    ``RunResult.answer`` returned to the caller is deanonymized using
    ``meta['pii_mapping']`` when one is present — the dev sees real values
    ("Nice to meet you, anmol!") instead of placeholders. The DB row keeps
    the raw LLM output so the dashboard's audit view stays placeholder-faithful.
    """
    from dendrux.guardrails._engine import deanonymize_text
    from dendrux.runtime.state import _capture_finalized_run

    pii_mapping = result.meta.get("pii_mapping") or {}
    user_answer, unmapped_placeholders = deanonymize_text(result.answer, pii_mapping)
    _wrap = partial(_wrap_loop_result, run_id, result, user_answer, pii_mapping)
    _persisted = partial(
        _persisted_loop_result, state_store, run_id, result, user_answer, pii_mapping
    )
    _finalize_cancelled = partial(
        _finalize_loop_cancelled, state_store, run_id, result, user_answer, pii_mapping
    )

    cancel_request = _loop_cancel_request(result)
    if cancel_request is not None:
        return await _finalize_cancelled(*cancel_request)

    if result.meta.get(MANAGED_HANDOFF_KEY) is not None:
        return await _persist_managed_handoff(
            state_store, run_id, result, recorder, provider_kwargs, _finalize_cancelled
        )

    if result.status in _PAUSE_STATUSES:
        return await _persist_pause(
            state_store,
            run_id,
            result,
            provider_kwargs,
            recorder,
            _wrap,
            _persisted,
            _finalize_cancelled,
        )

    # Terminal: success / max_iterations / error. Cooperative cancellation:
    # a cancel requested after the last checkpoint does not override a run
    # that finished — its work and side effects happened.
    _run_pii = result.meta.get("pii_mapping")
    failure_kwargs = _terminal_failure_kwargs(state_store, result)
    stop_reason, event_type, event_data = _terminal_loop_event(result)
    event_data.update(_unstored_failure_reason(result, failure_kwargs))
    atomic_final = getattr(state_store, "finalize_with_event", None) is not None
    if unmapped_placeholders:
        await _run_events._emit_event_safe(
            state_store,
            run_id,
            GovernanceEventType.GUARDRAIL_UNMAPPED_PLACEHOLDER.value,
            {"placeholders": unmapped_placeholders},
        )
    with _capture_finalized_run(unclaimed_store(state_store), run_id) as records:
        if atomic_final:
            finalize_won = (
                await state_store.finalize_with_event(
                    run_id,
                    status=result.status.value,
                    allowed_current_statuses=[RunStatus.RUNNING.value],
                    answer=result.answer,
                    iteration_count=result.iteration_count,
                    total_usage=result.usage,
                    pii_mapping=_run_pii,
                    error=result.error,
                    event_type=event_type,
                    event_data=event_data,
                    **_completion_evidence(state_store, result, recorder, event_type, event_data),
                    **failure_kwargs,
                )
                is not None
            )
        else:
            finalize_won = await state_store.finalize_run(
                run_id,
                status=result.status.value,
                answer=result.answer,
                iteration_count=result.iteration_count,
                total_usage=result.usage,
                expected_current_status=RunStatus.RUNNING.value,
                pii_mapping=_run_pii,
                error=result.error,
                **failure_kwargs,
            )
    if not finalize_won:
        return await _persisted(result.status)
    recovery = await _run_pending._final_run_recovery(state_store, run_id, records)
    if stop_reason is not None:
        await _record_stop_reason(state_store, run_id, stop_reason)
    if not atomic_final:
        await _run_events._emit_event(state_store, run_id, event_type, event_data)
    finished = _wrap(status=result.status, won=True, recovery=recovery)
    await _run_events._attach_undelivered_notes(state_store, run_id, finished)
    return finished


async def _persist_pause(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    provider_kwargs: dict[str, Any] | None,
    recorder: LoopRecorder | None,
    _wrap: Callable[..., RunResult],
    _persisted: Callable[..., Awaitable[RunResult]],
    _finalize_cancelled: Callable[..., Awaitable[RunResult]],
) -> RunResult:
    """Persist a pause and its event as one CAS; cancellation wins over pause."""

    # Pre-pause checkpoint as one CAS: the pause persists only while the
    # row is RUNNING with no pending cancel. Cancellation wins over pause;
    # any durable pause (incl. WAITING_*) satisfies a pause request. The
    # pause and its run.paused event commit together, so a finalizer of
    # paused runs (cancel_run, sweep) always acts after the event.
    await _run_events._check_claim(state_store, run_id)
    _run_pending._ensure_pause_generation(result, provider_kwargs)
    pause_state: PauseState = result.meta["pause_state"]
    # A pause generation belongs to this transition, never to the prior
    # pause from which this execution was resumed.
    pause_state = replace(pause_state, pause_id=generate_ulid())
    result.meta["pause_state"] = pause_state
    if (
        pause_state.pause_reason == PauseReason.REQUESTED.value
        and pause_state.pause_message is None
    ):
        pause_state = replace(
            pause_state, pause_message=await _run_events._pause_request_message(state_store, run_id)
        )
        result.meta["pause_state"] = pause_state
    paused_data = _run_pending._pause_event_data(result, pause_state)
    record = await state_store.get_run(run_id)
    checkpoint = getattr(recorder, "checkpoint_outcome", None)
    if checkpoint is not None:
        pause_state.history[:] = [await project_message(recorder, m) for m in pause_state.history]
        await checkpoint(result, provider_kwargs)
    pause_data = {**pause_state.to_dict(), "status": result.status.value}
    if record is not None and GOVERNANCE_KEY in (getattr(record, "meta", None) or {}):
        pause_data[GOVERNANCE_KEY] = (record.meta or {})[GOVERNANCE_KEY]
    paused = await _run_events._pause_with_event(
        state_store,
        run_id,
        status=result.status.value,
        pause_data=pause_data,
        event_data=paused_data,
        iteration_count=result.iteration_count,
        pii_mapping=result.meta.get("pii_mapping"),
        total_usage=result.usage,
    )
    if not paused:
        # CAS lost: honor the winner. On a still-RUNNING row only a
        # pending cancel makes the CAS miss, so finalize cancelled.
        record = await state_store.get_run(run_id)
        if record is not None and record.status != RunStatus.RUNNING.value:
            return await _persisted(RunStatus(record.status))
        return await _finalize_cancelled(
            [RunStatus.RUNNING.value], {"reason": CancelReason.CANCEL_REQUESTED.value}
        )
    result.pause_id = pause_state.pause_id
    # Return the persisted state: a finalizer may already have acted on
    # the durable pause (e.g. cancel_run right after it committed).
    return await _run_pending._committed_pause_result(
        state_store, _wrap(status=result.status, won=True), _persisted
    )


async def _persist_managed_handoff(
    state_store: StateStore,
    run_id: str,
    result: RunResult,
    recorder: LoopRecorder | None,
    provider_kwargs: dict[str, Any] | None,
    finalize_cancelled: Callable[..., Awaitable[RunResult]],
) -> RunResult:
    """A managed handoff: a durable checkpoint and the descend marker (0.7).

    No pause is written; the child's atomic bind parks this run. A cancel
    requested during the iteration finalizes it as for any pause.
    """
    await _run_events._check_claim(state_store, run_id)
    descend = await _managed.prepare_handoff(state_store, run_id, result, recorder, provider_kwargs)
    if descend is not None:
        return await _managed_budget.finish(descend, result, recorder, _persist_loop_outcome)
    return await finalize_cancelled(
        [RunStatus.RUNNING.value], {"reason": CancelReason.CANCEL_REQUESTED.value}
    )


async def _record_stop_reason(state_store: StateStore, run_id: str, stop_reason: str) -> None:
    """Best-effort ``meta[STOP_REASON_META_KEY]`` write after a won finalize.

    A store without ``update_run_meta`` or a failed write only means an
    idempotent replay of this run reports ``stop_reason=None``.
    """
    update = getattr(state_store, "update_run_meta", None)
    if update is None:
        return
    try:
        await update(run_id, {STOP_REASON_META_KEY: stop_reason})
    except Exception:
        logger.warning(
            "Failed to record stop_reason for run %s", run_id, exc_info=safe_exception_info()
        )


def _stream_error_meta(mcp_meta: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    """``RUN_ERROR`` result meta; an image rejection also lists the rejected image ids."""
    from dendrux.errors import ImageRejectedError

    meta = {**deepcopy(mcp_meta), "error_type": type(exc).__name__}
    if isinstance(exc, ImageRejectedError):
        meta["image_rejected"] = list(exc.image_ids)
    return meta
