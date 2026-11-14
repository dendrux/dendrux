"""Apply adopted/injected submissions before resumed execution and mark durable work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from dendrux.loops._helpers import (
    notify_message,
    order_observations,
    project_message,
    record_tool,
)
from dendrux.runtime import (
    _reclaim,
    _run_contracts,
    _run_entry,
    _run_images,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.runtime._continuation_guardrails import _SubmissionProjection
    from dendrux.runtime.state import StateStore
    from dendrux.types import (
        ApprovalDecision,
        ContentPart,
        Message,
        RunResult,
        ToolCall,
        ToolResult,
    )


from dendrux.runtime import _run_resume_prep


def _mark_work(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


async def _consume_injected_history(
    ctx: _run_contracts._ResumeContext, on_consumed: Callable[[], None] | None
) -> None:
    consume = getattr(ctx.recorder, "consume_resume", None)
    if consume is None:
        return
    state = getattr(ctx.recorder, "_loop_state", {})
    had_submission = state.get("submission") is not None
    start = len(ctx.pause_state.history)
    for index in range(start, len(ctx.history)):
        ctx.history[index] = await project_message(ctx.recorder, ctx.history[index])
    await consume(ctx.history[start:])
    if had_submission and getattr(ctx.recorder, "_loop_state", {}).get("submission") is None:
        _mark_work(on_consumed)


async def _replay_tool_result(
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    tc: ToolCall,
    tr: ToolResult,
    msg: Message,
    on_consumed: Callable[[], None] | None,
) -> None:
    from dendrux.loops._helpers import notify_tool as _notify_tool
    from dendrux.loops._helpers import notify_tool_started as _notify_tool_started

    iteration = ctx.pause_state.iteration
    blocked = await _replayed_result_blocked(ctx, run_id, tr, msg)
    if blocked is not None:
        # A claim made without projection: record only the projected evidence and stop,
        # before the raw tool message reaches any trace or callback.
        _mark_work(on_consumed)
        await _record_blocked_replay(ctx, run_id, tc, blocked, iteration)
    if ctx.pause_state.schema_version != 2:
        await ctx.recorder.on_tool_started(run_id, tc, iteration)
    await _notify_tool_started(ctx.notifier, run_id, tc, iteration)
    await ctx.recorder.on_message_appended(run_id, msg, iteration)
    _mark_work(on_consumed)
    await record_tool(ctx.recorder, run_id, tc, tr, iteration)
    await notify_message(ctx.notifier, run_id, msg, iteration)
    await _notify_tool(ctx.notifier, run_id, tc, tr, iteration)


async def _replayed_result_blocked(
    ctx: _run_contracts._ResumeContext, run_id: str, tr: ToolResult, message: Message
) -> Any:
    from dataclasses import replace

    from dendrux.guardrails._engine import _entry_engine
    from dendrux.loops._helpers import _BlockedObservationError, _project_tool_message

    engine = getattr(ctx.recorder, "guardrail_engine", None) or _entry_engine(run_id)
    if engine is None:
        return None
    projected, findings, error = await _project_tool_message(engine, tr, message)
    if error is None:
        return None
    clean = replace(tr, payload=projected.content, error=None, meta={}, notes=())
    return _BlockedObservationError(projected, findings, error, clean)


async def _record_blocked_replay(
    ctx: _run_contracts._ResumeContext, run_id: str, tc: ToolCall, blocked: Any, iteration: int
) -> NoReturn:

    from contextlib import nullcontext

    from dendrux.loops._helpers import _record_tool_projection
    from dendrux.loops._helpers import notify_tool as _notify_tool

    scope = getattr(ctx.recorder, "observation_scope", None)
    async with scope() if scope is not None else nullcontext():
        await _record_tool_projection(
            ctx.recorder, run_id, tc, blocked.result, iteration, blocked.message, blocked=True
        )
    await _notify_tool(ctx.notifier, run_id, tc, blocked.result, iteration)
    raise blocked


async def _project_adopted_submission(
    store: StateStore,
    run_id: str,
    raw: dict[str, Any] | None,
    status: str,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
    decisions: list[tuple[str, ApprovalDecision]] | None,
    *,
    skip_claim: bool,
    stored: dict[str, Any] | None = None,
    pending: Sequence[ToolCall] = (),
) -> tuple[
    list[ToolResult] | None,
    str | None,
    tuple[ContentPart, ...] | None,
    list[tuple[str, ApprovalDecision]] | None,
    _SubmissionProjection | None,
]:
    """Project a submission this call did not claim, once, before anything stores it.

    ``stored`` is the unconsumed payload reclaim found in the pause row: only its verdict
    is retained (recovery consumes the stored values itself), so a block stops the run
    before recovery resolution writes any decision, reason or trace. Its texts are
    projected in the forms recovery sends (a denial as the ``pending`` call's tool
    result), so recovery reuses each verdict rather than scanning the text again; a denial
    reason that JSON escaping changes is also scanned once as its raw text.
    """
    from dendrux.guardrails._engine import _entry_engine
    from dendrux.runtime._continuation_guardrails import (
        _project_submission,
        _SubmissionProjection,
    )
    from dendrux.runtime.submit import deserialize_decisions, deserialize_tool_results

    if _entry_engine(run_id) is None or _claimed_in_call(store, run_id, skip_claim=skip_claim):
        return tool_results, user_input, user_parts, decisions, None
    payload = (
        stored
        if stored is not None
        else _run_resume_prep._resume_claim_payload(
            run_id, raw, status, tool_results, user_input, user_parts, decisions
        )
        if raw is not None
        else None
    )
    projection = _SubmissionProjection()
    clean, _ = await _project_submission(
        run_id,
        payload,
        projection=projection,
        denied_calls={tc.id: tc for tc in pending} if stored is not None else None,
    )
    if clean is None or stored is not None:
        kept = projection if stored is not None else None
        return tool_results, user_input, user_parts, decisions, kept
    return (
        deserialize_tool_results(clean["submitted_tool_results"])
        if tool_results is not None
        else None,
        clean.get("submitted_user_input"),
        _run_images._stored_parts(clean.get("submitted_user_parts")),
        deserialize_decisions(clean["submitted_decisions"]) if decisions is not None else None,
        projection,
    )


def _claimed_in_call(store: StateStore, run_id: str, *, skip_claim: bool) -> bool:
    """Whether this call won the claim it continues (its own resume claim, or a submission
    claim it made). Such a continuation checks only the already-loaded row; adoption and
    reclaim also run the scoped lookup, which sees a legacy blocked event."""
    from dendrux.runtime._blocked_content import _claimed_here

    return not skip_claim or _claimed_here(store, run_id)


async def _replay_submission_or_stop(
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    *,
    recovering: bool,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    on_consumed: Callable[[], None] | None,
    input_block: Any = None,
) -> RunResult | None:
    """Replay the submitted history (not when recovering); a blocked result stops."""
    from dendrux.loops._helpers import _BlockedObservationError

    if recovering:
        return (
            await _resume_blocked_result(ctx, run_id, input_block)
            if input_block is not None
            else None
        )
    try:
        await _replay_resume_injected_history(
            ctx,
            run_id,
            tool_results=tool_results,
            user_input=user_input,
            on_consumed=on_consumed,
            input_block=input_block,
        )
        if input_block is not None:
            return await _resume_blocked_result(ctx, run_id, input_block)
    except _BlockedObservationError as blocked:
        return await _resume_blocked_result(ctx, run_id, blocked)
    return None


async def _resolve_recovery(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    *,
    image_runtime: Any,
    tenant_id: str | None,
    on_work: Callable[[], None] | None,
) -> RunResult | None:
    """Recovery resolution; a submitted message it would send that projects as blocked
    stops the run before consumption, with the mark committed with its first evidence."""
    try:
        return await _reclaim.resolve(
            agent,
            store,
            run_id,
            ctx,
            image_runtime=image_runtime,
            tenant_id=tenant_id,
            on_work=on_work,
        )
    except _reclaim._RecoveryBlockedError as stopped:
        return await _resume_blocked_result(ctx, run_id, stopped.blocked)


async def _resume_blocked_result(
    ctx: _run_contracts._ResumeContext, run_id: str, blocked: Any
) -> RunResult:
    """The normal guardrail outcome for a resumed submission stopped by blocked content."""
    from dendrux.guardrails._engine import _entry_engine
    from dendrux.runtime._entry_guardrails import _EntryProjection

    projection = _EntryProjection(_entry_engine(run_id), blocked.error, blocked.findings)
    result = await _run_entry._blocked_entry_result(
        projection, ctx.recorder, ctx.notifier, run_id, "", None, []
    )
    result.iteration_count = ctx.pause_state.iteration + 1
    result.steps = list(ctx.pause_state.steps)
    return result


async def _replay_resume_injected_history(
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    *,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    on_consumed: Callable[[], None] | None = None,
    input_block: Any = None,
) -> None:
    await _consume_injected_history(ctx, on_consumed)
    if tool_results is not None:
        pending = {tc.id: tc for tc in ctx.pause_state.pending_tool_calls}
        start = len(ctx.pause_state.history)
        for index, result in enumerate(tool_results):
            await _replay_tool_result(
                ctx,
                run_id,
                pending[result.call_id],
                result,
                ctx.history[start + index],
                on_consumed,
            )
    elif user_input is not None:
        await _replay_user_input(ctx, run_id, on_consumed, input_block)
    order_observations(ctx.recorder, ctx.history)


async def _replay_user_input(
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    on_consumed: Callable[[], None] | None,
    input_block: Any,
) -> None:
    """Record the resumed user turn; an adopted input that projected as blocked is
    recorded as its placeholder together with the run mark, then stops the run."""
    from dendrux.loops._helpers import _evidence_scope

    msg = ctx.history[-1]
    async with _evidence_scope(ctx.recorder, run_id, input_block is not None):
        await ctx.recorder.on_message_appended(run_id, msg, ctx.pause_state.iteration)
    _mark_work(on_consumed)
    await notify_message(ctx.notifier, run_id, msg, ctx.pause_state.iteration)
    if input_block is not None:
        raise input_block
