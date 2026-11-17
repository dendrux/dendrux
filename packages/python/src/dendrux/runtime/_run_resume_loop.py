"""Execute and complete or release a claimed resumed loop."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import dendrux.loops._helpers as _hooks
from dendrux import _codes
from dendrux._lifecycle import (
    FailureReason,
)
from dendrux._meta_keys import (
    MAX_DELEGATION_DEPTH_META_KEY,
)
from dendrux.errors import (
    RunFencedError,
    safe_exception_info,
)
from dendrux.loops._budget import BudgetCarry
from dendrux.runtime import (
    _run_contracts,
    _run_events,
    _run_images,
    _run_observers,
    _run_outcome,
    _run_pending,
    _run_resume_prep,
    _run_start,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime.context import (
    DelegationContext,
    get_store_identity,
)
from dendrux.runtime.submit import (
    _release_supported,
)
from dendrux.types import (
    GovernanceEventType,
    PauseState,
    RunStatus,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
    )

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopRecorder
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.types import (
        RunEvent,
        RunResult,
        ToolResult,
    )


def _budget_carry_kwargs(loop_method: Any, pause_state: PauseState) -> dict[str, Any]:
    """``budget_carry=`` for a loop that accepts it (built-in loops); else nothing.

    Custom loops written against the pre-0.3 ``Loop`` signature keep working;
    they simply cannot restore fired budget keys.
    """
    params = _run_contracts._signature_params(loop_method)
    if params is None or "budget_carry" not in params:
        return {}
    return {"budget_carry": BudgetCarry.from_pause(pause_state)}


def _resume_delegation(
    agent: Agent,
    store: StateStore,
    run_id: str,
    record: RunRecord | None,
    trace_link: Any,
    tenant_explicit: bool | None,
) -> DelegationContext:
    depth = (record.meta or {}).get(MAX_DELEGATION_DEPTH_META_KEY) if record else None
    return DelegationContext(
        run_id=run_id,
        trace_link=trace_link,
        delegation_level=record.delegation_level if record else 0,
        persisted=True,
        store_identity=get_store_identity(store),
        max_delegation_depth=depth if isinstance(depth, int) else None,
        tenant_id=record.tenant_id if record else None,
        tenant_explicit=True if tenant_explicit is None else tenant_explicit,
        state_store=store,
        lease_interval_s=agent.lease_interval.total_seconds(),
        loop=asyncio.get_running_loop(),
    )


def _resume_warning_meta(
    ctx: _run_contracts._ResumeContext | None, meta: dict[str, Any]
) -> dict[str, Any]:
    warnings = getattr(ctx, "notifier_warnings", None)
    if warnings:
        meta["notifier_warnings"] = warnings + meta.get("notifier_warnings", [])
    return meta


async def _resume_loop_stream(
    ctx: _run_contracts._ResumeContext, events: AsyncGenerator[RunEvent, None]
) -> AsyncGenerator[RunEvent, None]:
    async with aclosing(events):
        async for event in events:
            if event.run_result is not None:
                _resume_warning_meta(ctx, event.run_result.meta)
            yield event


async def _finish_resumed_result(
    store: StateStore, run_id: str, ctx: _run_contracts._ResumeContext, result: RunResult
) -> RunResult:
    _resume_warning_meta(ctx, result.meta)
    result = await _run_outcome._persist_loop_outcome(
        recorder=ctx.recorder,
        provider_kwargs=ctx.pause_state.provider_kwargs,
        state_store=store,
        run_id=run_id,
        result=result,
    )
    await _run_start._record_finished(ctx.recorder, ctx.notifier, run_id, result)
    return result


def _initial_call_counts_kwargs(loop_method: Any, recorder: LoopRecorder) -> dict[str, Any]:
    if not _run_contracts._accepts_kwarg(loop_method, "_initial_call_counts"):
        return {}
    counts = getattr(recorder, "execution_counts", lambda: None)()
    return {"_initial_call_counts": counts}


def _resumed_loop_kwargs(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    image_runtime: ImageRuntime | None,
    loop_method: Any,
) -> dict[str, Any]:
    carry = _budget_carry_kwargs(loop_method, ctx.pause_state)
    images = _run_images._image_loop_kwargs(
        loop_method, image_runtime, _run_images._resume_image_view(ctx.pause_state, ctx.history)
    )
    return dict(
        agent=agent,
        provider=provider,
        strategy=ctx.resolved_strategy,
        user_input="",
        run_id=run_id,
        recorder=ctx.recorder,
        notifier=ctx.notifier,
        initial_history=ctx.history,
        initial_steps=ctx.pause_state.steps,
        iteration_offset=ctx.pause_state.iteration,
        initial_usage=ctx.pause_state.usage,
        provider_kwargs=ctx.pause_state.provider_kwargs or None,
        state_store=store,
        **_initial_call_counts_kwargs(loop_method, ctx.recorder),
        **carry,
        **images,
    )


async def _run_resumed_loop(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    pii: Any,
    image_runtime: ImageRuntime | None,
    on_work: Callable[[], None],
) -> RunResult:
    kwargs = _resumed_loop_kwargs(
        agent, provider, store, run_id, ctx, image_runtime, ctx.resolved_loop.run
    )
    on_work()
    return await ctx.resolved_loop.run(**kwargs, initial_pii_mapping=pii)


async def _cancel_resumed_run(
    agent: Agent, store: StateStore, run_id: str, ctx: _run_contracts._ResumeContext | None
) -> tuple[RunResult, bool]:
    interrupted, swallow = _run_events._take_task_interrupt(agent._task_manager, run_id)
    result = await _run_events._drain_task_cancellation(store, run_id, interrupted)
    if ctx is not None:
        try:
            await _hooks.record_run_finished(ctx.recorder, run_id, result)
        except Exception:
            logger.warning(
                "Recorder finish failed for run %s", run_id, exc_info=safe_exception_info()
            )
        await _hooks.notify_run_finished(ctx.notifier, run_id, result)
    return result, _run_events._may_swallow_cancellation(swallow)


async def _resume_failure_hooks(
    ctx: _run_contracts._ResumeContext | None, run_id: str, exc: Exception
) -> None:
    if ctx is None:
        return
    try:
        await _hooks.record_run_failed(ctx.recorder, run_id, exc)
        await _hooks.notify_run_failed(ctx.notifier, run_id, exc)
    except Exception:
        logger.warning(
            "Lifecycle hooks failed during resume error handler for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )


async def _released_resume_status(store: StateStore, run_id: str) -> RunStatus:
    try:
        record = await store.get_run(run_id)
        if record is not None:
            return RunStatus(record.status)
    except Exception:
        logger.warning(
            "Released run status read failed for %s", run_id, exc_info=safe_exception_info()
        )
    return RunStatus.RUNNING


async def _released_resume_hooks(
    store: StateStore, run_id: str, ctx: _run_contracts._ResumeContext | None, exc: Exception
) -> RunStatus:
    from dendrux.types import RunResult

    outcome = await _released_resume_status(store, run_id)
    if ctx is None:
        return outcome
    if outcome == RunStatus.RUNNING:
        await _resume_failure_hooks(ctx, run_id, exc)
        return outcome
    result = RunResult(
        run_id=run_id,
        status=outcome,
        steps=ctx.pause_state.steps,
        iteration_count=ctx.pause_state.iteration,
        usage=ctx.pause_state.usage,
    )
    if ctx.recorder_started:
        try:
            await _hooks.record_run_finished(ctx.recorder, run_id, result)
        except Exception:
            logger.warning(
                "Recorder finish failed for run %s", run_id, exc_info=safe_exception_info()
            )
    if ctx.notifier_started:
        await _hooks.notify_run_finished(ctx.notifier, run_id, result)
    return outcome


async def _try_resume_release(
    store: StateStore,
    run_id: str,
    status: str,
    ctx: _run_contracts._ResumeContext | None,
    exc: Exception,
) -> RunStatus:
    released = await _run_resume_prep._release_claim(store, run_id, status)
    if released:
        return await _released_resume_hooks(store, run_id, ctx, exc)
    await _resume_failure_hooks(ctx, run_id, exc)
    return await _released_resume_status(store, run_id)


def _resume_release_supported(store: StateStore, status: str) -> bool:
    return status == RunStatus.RUNNING.value or _release_supported(store)


async def _handle_resume_failure(
    store: StateStore,
    run_id: str,
    status: str,
    ctx: _run_contracts._ResumeContext | None,
    exc: Exception,
    progress: _run_contracts._ResumeProgress,
    meta: dict[str, Any],
) -> None:
    if progress._can_release(ctx, exc) and _resume_release_supported(store, status):
        await _try_resume_release(store, run_id, status, ctx, exc)
        return
    _run_observers._attach_run_error_meta(exc, run_id, meta)
    await _run_start._persist_run_failure(store, run_id, exc)
    await _resume_failure_hooks(ctx, run_id, exc)


async def _fenced_resume_result(
    store: StateStore, run_id: str, ctx: _run_contracts._ResumeContext | None
) -> RunResult:
    result = await _run_pending._build_cached_result(store, run_id)
    if ctx is not None:
        await _run_events._finish_fenced_lifecycle(run_id, result, ctx.recorder, ctx.notifier)
    return result


_CONTINUATION_BLOCKED = "Run stopped because it contains blocked content."


def _continuation_block_error(
    store: StateStore, run_id: str, pause: PauseState | None
) -> tuple[str, Any, str, int | None]:
    """Error, findings, location and iteration for a marked continuation stop.

    A block found in this call keeps the guardrail's own safe text (name and finding
    kinds, never matched text), located where the resumed history would have put
    the message. A mark left by an earlier call gets the generic message.
    """
    from dendrux.guardrails._engine import _message_location
    from dendrux.runtime._blocked_content import _claimed_block_detail

    iteration = pause.iteration if pause is not None else None
    block = _claimed_block_detail(store, run_id)
    if block is None or pause is None:
        return _CONTINUATION_BLOCKED, [], "continuation", iteration
    # Request messages are the system prompt, the paused history, then the submission.
    message = block.location_message()
    located = _message_location(message, 1 + len(pause.history) + block.offset)
    error = block.error.replace(_message_location(message, None), located, 1)
    from types import SimpleNamespace

    counted = SimpleNamespace(entity_counts=dict(block.counts))
    return error, counted, located, pause.iteration + 1


async def _stop_marked_continuation(
    store: StateStore,
    run_id: str,
    record: RunRecord | None,
    pause: PauseState | None = None,
    *,
    agent: Agent | None = None,
    claimed_here: bool = False,
    admitted: bool = False,
    iteration: int | None = None,
    usage: UsageStats | None = None,
) -> RunResult | None:
    from dendrux.runtime._blocked_content import _continuation_blocked
    from dendrux.types import RunResult

    if admitted or not await _continuation_blocked(
        store, run_id, record, claimed_here=claimed_here
    ):
        return None
    _codes.log_code(logger, _codes.BLOCKED_CONTENT, run_id=run_id)
    from dendrux.runtime._continuation_guardrails import _continuation_mapping

    mapping = await _continuation_mapping(store, run_id)
    error, findings, location, pause_iteration = _continuation_block_error(store, run_id, pause)
    if pause is not None:
        iteration, usage = pause_iteration, pause.usage
    meta: dict[str, Any] = {
        "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value,
        "pii_mapping": mapping or {},
    }
    if agent is not None and not agent._tool_sources:
        # Same as any resumed result of an agent without tool sources.
        meta.update(mcp_skipped_sources=[], mcp_skipped_tools=[])
    result = RunResult(run_id=run_id, status=RunStatus.ERROR, error=error, meta=meta)
    data = {"error": error, "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value}
    from dendrux.loops.react import _blocked_data

    terminal: dict[str, Any] = {} if iteration is None else {"iteration_count": iteration}
    try:
        await _run_events._emit_event(
            store,
            run_id,
            GovernanceEventType.GUARDRAIL_BLOCKED.value,
            _blocked_data("incoming", error, findings, location),
        )
        won = await _run_start._persist_run_failure(
            store,
            run_id,
            None,
            payload=data,
            finalize={
                "error": error,
                "pii_mapping": mapping,
                "total_usage": usage,
                **terminal,
                **_run_outcome._terminal_failure_kwargs(store, result),
            },
            strict=True,
        )
    except RunFencedError:
        return await _run_pending._build_cached_result(store, run_id)
    if not won:
        return await _run_pending._build_cached_result(store, run_id)
    cached = await _run_pending._build_cached_result(store, run_id)
    if usage is not None:
        cached.usage = usage
    if pause is not None:
        cached.steps = list(pause.steps)
    cached.meta.update(result.meta)
    cached.meta["_finalize_won"] = True
    return cached


def _resume_history_inputs(
    recovering: bool, tool_results: list[ToolResult] | None, user_input: str | None
) -> tuple[list[ToolResult] | None, str | None]:
    return (None, None) if recovering else (tool_results, user_input)
