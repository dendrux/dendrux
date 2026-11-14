"""Expose non-streaming continuation entry points over owned claims."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import dendrux.runtime.state as _state
import dendrux.runtime.submit as _submit
from dendrux.errors import (
    RunFencedError,
    RunNotFoundError,
)
from dendrux.loops._helpers import (
    record_run_started,
)
from dendrux.runtime import (
    _run_contracts,
    _run_events,
    _run_images,
    _run_observers,
    _run_pending,
    _run_resume_prep,
    _run_resume_replay,
    _run_tools,
)
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError,
)
from dendrux.runtime._managed_cancel import (
    stopped_resume_result as _stopped_resume_result,
)
from dendrux.runtime.context import (
    reset_delegation_context,
    set_delegation_context,
)
from dendrux.types import (
    ApprovalDecision,
    PauseState,
    RunStatus,
)

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.media._image import UserContent
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        RunResult,
        ToolResult,
    )


from dendrux.runtime import _run_approval, _run_resume_loop


async def resume(
    run_id: str,
    tool_results: list[ToolResult] | None,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a paused run.

    Three shapes:
      - tool_results provided → client-tool results or approval rejection
      - No args (tool_results=None) → approval approve, or continue a PAUSED run

    Uses an atomic claim to prevent double-resume races.

    Args:
        run_id: The paused run's ID.
        tool_results: Results for the pending tool calls, or None for
            approval-approve (no-arg resume). Each result must have
            a call_id matching one of the pending_tool_calls.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        extra_notifier: Optional additional notifier for SSE streaming.
    """
    if tool_results is None:
        # No-arg = approval-approve. Status gate in _resume_core.
        expected = RunStatus.WAITING_APPROVAL.value
    else:
        # Tool results — try WAITING_CLIENT_TOOL first.
        # _resume_core will fall back to WAITING_APPROVAL on claim failure.
        expected = RunStatus.WAITING_CLIENT_TOOL.value

    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        expected_status=expected,
        tool_results=tool_results,
        extra_notifier=extra_notifier,
        expected_pause_id=expected_pause_id,
        _tenant_explicit=_tenant_explicit,
    )


async def resume_with_input(
    run_id: str,
    user_input: UserContent,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a paused run by providing clarification input.

    Only works on runs with status WAITING_HUMAN_INPUT. Appends the
    user's response as a normal USER message and re-enters the loop.

    Args:
        run_id: The paused run's ID.
        user_input: Free-text response to the agent's clarification question.
        state_store: Persistence backend (required for resume).
        agent: Agent definition (must match the paused run's agent).
        provider: LLM provider for continuing the run.
        strategy: Strategy override. Defaults to NativeToolCalling.
        loop: Loop override. Defaults to ReActLoop.
        extra_notifier: Optional additional notifier for SSE streaming.
    """
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        strategy=strategy,
        loop=loop,
        expected_status=RunStatus.WAITING_HUMAN_INPUT.value,
        user_input=user_input,
        extra_notifier=extra_notifier,
        expected_pause_id=expected_pause_id,
        _tenant_explicit=_tenant_explicit,
    )


async def _resume_core(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    expected_status: str,
    tool_results: list[ToolResult] | None = None,
    user_input: UserContent | None = None,
    extra_notifier: LoopNotifier | None = None,
    _skip_claim: bool = False,
    decisions: list[tuple[str, ApprovalDecision]] | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    expected_pause_id: str | None = None,
    _reclaim_pause: PauseState | None = None,
    _reclaim_history: Any = None,
    _reclaim_record: RunRecord | None = None,
    _reclaim_submission: dict[str, Any] | None = None,
    _tenant_explicit: bool | None = None,
    _raw_pause: dict[str, Any] | None = None,
) -> RunResult:
    """Shared resume logic for tool results and clarification input.

    Args:
        decisions: Per-call approval decisions (call_id, decision) for an
            approval resume; validated against the pause's persisted
            ``approval_gates`` snapshot. None = legacy batch approve.
        extra_notifier: Optional additional LoopNotifier. Composed with
            PersistenceRecorder — recorder handles persistence, notifier
            handles observation side-effects.
        _skip_claim: Internal flag — skip the atomic claim step when the caller
            has already claimed via submit_and_claim(). Used by resume_claimed().
        user_parts: Already-ingested parts of a multimodal user input (the
            claimed path); ``user_input`` is then their text projection.
    """
    async with _run_resume_prep._release_setup_failure(
        state_store, run_id, expected_status, enabled=_skip_claim
    ):
        skill_session, user_input, input_items = await _run_resume_prep._resume_session(
            agent, state_store, run_id, loop, user_input, tool_results, decisions, expected_status
        )
    from dendrux.runtime._continuation_guardrails import _continuation_projection

    with skill_session.bind(), await _continuation_projection(agent, state_store, run_id):
        async with _run_resume_prep._release_setup_failure(
            state_store, run_id, expected_status, enabled=_skip_claim
        ):
            (
                run_record,
                reads,
                raw_pause,
                pause_state,
                recovering,
                _saved_pii_mapping,
            ) = await _run_resume_prep._read_resume_state(
                state_store,
                agent,
                provider,
                run_id,
                _skip_claim,
                expected_pause_id,
                _raw_pause,
                _reclaim_pause,
                reclaimed_record=_reclaim_record,
            )
            (
                state_store,
                expected_status,
                raw_pause,
                tool_results,
                user_parts,
            ) = await _run_resume_prep._resume_submission(
                state_store,
                agent,
                provider,
                run_id,
                pause_state,
                raw_pause,
                reads,
                loop,
                _skip_claim,
                expected_status,
                expected_pause_id,
                tool_results,
                user_input,
                user_parts,
                input_items,
                decisions,
            )
        # 4-10. Everything that runs after on_run_started fires must live
        #       inside the try block so a closing hook can balance the lifecycle
        #       pair on any failure (including injected-history replay errors).
        ctx: _run_contracts._ResumeContext | None = None
        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        progress = _run_contracts._ResumeProgress()
        agent._task_manager.register_current(run_id)
        try:
            stopped = await _run_resume_loop._stop_marked_continuation(
                state_store,
                run_id,
                run_record,
                pause_state,
                agent=agent,
                claimed_here=_run_resume_replay._claimed_in_call(
                    state_store, run_id, skip_claim=_skip_claim
                ),
                admitted=_reclaim_record is not None,
            )
            if stopped is not None:
                return stopped
            (
                tool_results,
                user_input,
                user_parts,
                decisions,
                adopted,
            ) = await _run_resume_replay._project_adopted_submission(
                state_store,
                run_id,
                raw_pause,
                expected_status,
                tool_results,
                user_input,
                user_parts,
                decisions,
                skip_claim=_skip_claim,
                stored=_reclaim_submission,
                pending=pause_state.pending_tool_calls,
            )
            run_tenant = run_record.tenant_id if run_record else None
            # 4-8. Prepare history and notifier (shared with resume_stream).
            #      Does NOT fire any lifecycle hooks — that's the caller's job.
            history_results, history_input = _run_resume_loop._resume_history_inputs(
                recovering, tool_results, user_input
            )
            ctx = await _run_resume_prep._prepare_resume(
                run_id,
                pause_state,
                state_store=state_store,
                agent=agent,
                provider=provider,
                strategy=strategy,
                loop=loop,
                expected_status=expected_status,
                tool_results=history_results,
                user_input=history_input,
                extra_notifier=extra_notifier,
                user_parts=user_parts,
                tenant_id=run_tenant,
                pii_mapping=_saved_pii_mapping,
                history_records=(
                    _reclaim_history if _reclaim_history is not None else reads.history_records
                ),
                pause_data=raw_pause,
                submission_projection=adopted,
                recovering=recovering,
            )
            progress.submission = getattr(ctx.recorder, "_loop_state", {}).get("submission")
            image_runtime = _run_images._image_runtime(agent, state_store, run_tenant, run_id)
            # 8a. Fire run-level lifecycle hooks. AFTER this point, any failure
            #     must close the lifecycle pair in the except handler.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            ctx.recorder_started = True
            trace_link, ctx.notifier = await _run_observers._start_trace(
                ctx.notifier,
                state_store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
                on_started=ctx._mark_notifier_started,
            )
            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_run_tools._snapshot_mcp_meta(agent))
            # 8b. tool/user-input history. Inside the try so
            #     a recorder failure here fires on_run_failed.
            blocked_stop = await _run_resume_replay._replay_submission_or_stop(
                ctx,
                run_id,
                recovering=recovering,
                tool_results=tool_results,
                user_input=user_input,
                on_consumed=progress._start,
                input_block=adopted.block if adopted is not None else None,
            )
            if blocked_stop is not None:
                _run_observers._attach_mcp_meta(blocked_stop, mcp_meta)
                return await _run_resume_loop._finish_resumed_result(
                    state_store, run_id, ctx, blocked_stop
                )

            ctx_token = set_delegation_context(
                _run_resume_loop._resume_delegation(
                    agent, state_store, run_id, run_record, trace_link, _tenant_explicit
                )
            )

            if recovering:
                stop = await _run_resume_replay._resolve_recovery(
                    agent,
                    state_store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_tenant,
                    on_work=progress._start,
                )
                if stop is not None:
                    return await _run_resume_loop._finish_resumed_result(
                        state_store, run_id, ctx, stop
                    )

            if not recovering:
                stop = await _run_approval._resume_approval_work(
                    agent,
                    state_store,
                    run_id,
                    ctx,
                    expected_status,
                    tool_results,
                    user_input,
                    decisions,
                    image_runtime,
                    progress._start,
                )
                if stop is not None:
                    _run_observers._attach_mcp_meta(stop, mcp_meta)
                    return await _run_resume_loop._finish_resumed_result(
                        state_store, run_id, ctx, stop
                    )

            result = await _run_resume_loop._run_resumed_loop(
                agent,
                provider,
                state_store,
                run_id,
                ctx,
                _saved_pii_mapping,
                image_runtime,
                progress._start,
            )
            _run_observers._attach_mcp_meta(result, mcp_meta)
            return await _run_resume_loop._finish_resumed_result(state_store, run_id, ctx, result)
        except asyncio.CancelledError:
            cancelled_result, swallow = await _run_resume_loop._cancel_resumed_run(
                agent, state_store, run_id, ctx
            )
            if swallow:
                return cancelled_result
            raise
        except (ManagedRootStoppedError, RunFencedError) as stop:
            return await _stopped_resume_result(stop, state_store, run_id, ctx)
        except Exception as exc:
            await _run_resume_loop._handle_resume_failure(
                state_store,
                run_id,
                expected_status,
                ctx,
                exc,
                progress,
                _run_resume_loop._resume_warning_meta(ctx, mcp_meta),
            )
            raise
        finally:
            agent._task_manager.release_current(run_id)
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


async def _owned_claim_view(
    store: StateStore, run_id: str, claim_epoch: int | None = None
) -> StateStore:
    if claim_epoch is not None:
        return _state.claimed_store(store, run_id, claim_epoch)
    # Legacy internal callers without a token adopt the current epoch best effort.
    if getattr(store, "claim_epoch", None) is None:
        getter = getattr(store, "get_run_control", None)
        if getter is not None:
            control = await getter(run_id)
            if control.claim_epoch is not None:
                return _state.claimed_store(store, run_id, control.claim_epoch)
    return store


async def _validate_owned_pause(
    store: StateStore,
    run_id: str,
    agent: Agent,
    status: str,
    *,
    require_submission: bool = False,
) -> dict[str, Any]:
    from dendrux.runtime.submit import (
        deserialize_decisions,
        deserialize_tool_results,
        validate_approval_decisions,
    )

    async with _run_resume_prep._release_setup_failure(store, run_id, status):
        await _run_events._check_claim(store, run_id)
        raw = await store.get_pause_state(run_id)
        if raw is None:
            raise ValueError("Claimed run has no pause state")
        _submit.submission_validator(run_id, raw, agent=agent)
        from dendrux.runtime.submit import submitted_view

        raw = submitted_view(raw)
        submitted = raw.get("submitted_tool_results")
        if submitted is not None:
            _submit.validate_result_names(
                run_id, PauseState.from_dict(raw), deserialize_tool_results(submitted)
            )
        elif raw.get("submitted_decisions") is not None:
            lookups = await agent.get_tool_lookups()
            validate_approval_decisions(
                run_id,
                PauseState.from_dict(raw),
                deserialize_decisions(raw["submitted_decisions"]),
                lookups.target,
            )
        elif require_submission and raw.get("submitted_user_input") is None:
            raise ValueError("Claimed run has no submitted input")
        return raw


async def _resume_claim_status(store: StateStore, run_id: str, prior: str | None) -> str:
    if prior is not None:
        return prior
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    return record.status


async def resume_claimed(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    pre_claim_status: str | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume a run that was already claimed via submit_and_claim().

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    Internal helper for the persist-first handoff. NOT a public API.

    Agent submit methods call submit_and_claim() to atomically persist the
    submitted payload and transition WAITING → RUNNING, then call this
    helper to re-enter the loop.

    Validates before proceeding:
      - pause_data exists (run was paused)
      - submitted_tool_results or submitted_user_input present in blob
      - for tool results: call_ids match pending_tool_calls

    Args:
        run_id: The run ID (already in RUNNING status after submit_and_claim).
        state_store: Persistence backend.
        agent: Agent definition.
        provider: LLM provider.
        extra_notifier: Optional additional LoopNotifier.
        pre_claim_status: The pause status the run was in before
            submit_and_claim flipped it to RUNNING. Forwarded to
            ``_resume_core`` as ``expected_status`` so downstream
            governance gates (e.g. ``approval.decided=rejected`` for
            ``submit_approval(approved=False)``) can fire correctly.
            When omitted, uses the current run-record status for release.
            Pass the original pause status to restore that pause on setup failure.

    Returns:
        RunResult from the resumed loop.

    Raises:
        ValueError: If pause_data is missing, no submitted data found,
            or submitted call_ids don't match pending tool calls.
    """
    from dendrux.runtime.submit import deserialize_tool_results

    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        pre_claim_status = await _resume_claim_status(state_store, run_id, pre_claim_status)
        raw_pause = await _validate_owned_pause(
            state_store,
            run_id,
            agent,
            pre_claim_status,
            require_submission=True,
        )
    except RunFencedError:
        return await _run_pending._build_cached_result(state_store, run_id)

    # 2. Detect submission type and extract data
    submitted_results = raw_pause.get("submitted_tool_results")
    submitted_input = raw_pause.get("submitted_user_input")

    if submitted_results is not None:
        # 3a. Convert dicts to typed ToolResult objects (images as stored refs)
        tool_results = deserialize_tool_results(submitted_results)

        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=pre_claim_status,
            tool_results=tool_results,
            extra_notifier=extra_notifier,
            _skip_claim=True,
            _raw_pause=raw_pause,
            _tenant_explicit=_tenant_explicit,
        )

    submitted_decisions = raw_pause.get("submitted_decisions")
    if submitted_decisions is not None:
        from dendrux.runtime.submit import deserialize_decisions

        # Re-validated against the approval_gates snapshot inside _resume_core,
        # where a mismatch releases an unconsumed claim; after consumption it
        # finalizes the run as ERROR.
        return await _resume_core(
            run_id,
            state_store=state_store,
            agent=agent,
            provider=provider,
            expected_status=RunStatus.WAITING_APPROVAL.value,
            extra_notifier=extra_notifier,
            _skip_claim=True,
            decisions=deserialize_decisions(submitted_decisions),
            _raw_pause=raw_pause,
            _tenant_explicit=_tenant_explicit,
        )

    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=pre_claim_status,
        user_input=submitted_input,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        user_parts=_run_images._stored_parts(raw_pause.get("submitted_user_parts")),
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
    )


async def resume_approval_approved_claim(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Resume an approval-approved run that was already claimed.

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    Paired with :func:`resume_claimed`: ``resume_claimed`` handles the
    post-claim path for tool results and user input, while this function
    handles approval-approve (no submitted payload — just re-enter the
    loop with the pending batch executed server-side).

    The caller (e.g. :meth:`Agent.submit_approval`) must have already
    transitioned the run to ``RUNNING`` via
    :meth:`StateStore.claim_paused_run` with
    ``expected_status="waiting_approval"``.
    """
    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        raw_pause = await _validate_owned_pause(
            state_store, run_id, agent, RunStatus.WAITING_APPROVAL.value
        )
    except RunFencedError:
        return await _run_pending._build_cached_result(state_store, run_id)
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=RunStatus.WAITING_APPROVAL.value,
        tool_results=None,
        user_input=None,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
    )


async def resume_paused_claim(
    run_id: str,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    extra_notifier: LoopNotifier | None = None,
    claim_epoch: int | None = None,
    _tenant_explicit: bool | None = None,
) -> RunResult:
    """Continue a PAUSED run that :meth:`Agent.submit_resume` already claimed.

    Raw-store callers should pass the epoch returned by their claim as
    ``claim_epoch=``. Omitting it retains the legacy best-effort current-epoch
    lookup. SDK submissions pass an already-bound store view.


    The claim cleared ``pause_requested``. The loop resumes with the full
    history; a pause that interrupted an approval batch first executes the
    batch's remainder with its persisted decisions.
    """
    try:
        state_store = await _owned_claim_view(state_store, run_id, claim_epoch)
        raw_pause = await _validate_owned_pause(state_store, run_id, agent, RunStatus.PAUSED.value)
    except RunFencedError:
        return await _run_pending._build_cached_result(state_store, run_id)
    return await _resume_core(
        run_id,
        state_store=state_store,
        agent=agent,
        provider=provider,
        expected_status=RunStatus.PAUSED.value,
        extra_notifier=extra_notifier,
        _skip_claim=True,
        _raw_pause=raw_pause,
        _tenant_explicit=_tenant_explicit,
    )


resume.__module__ = "dendrux.runtime.runner"
resume_with_input.__module__ = "dendrux.runtime.runner"
resume_claimed.__module__ = "dendrux.runtime.runner"
resume_approval_approved_claim.__module__ = "dendrux.runtime.runner"
resume_paused_claim.__module__ = "dendrux.runtime.runner"
