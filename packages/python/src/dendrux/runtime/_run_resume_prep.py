"""Prepare the claimed continuation context for resumed execution."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import dendrux.chat as _chat
import dendrux.runtime.state as _state
import dendrux.strategies.native as _native
from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    FailureReason,
    StoredEventType,
)
from dendrux.errors import (
    RunFencedError,
)
from dendrux.loops.react import ReActLoop
from dendrux.runtime import (
    _managed_budget,
    _reclaim,
    _run_contracts,
    _run_entry,
    _run_events,
    _run_images,
    _run_pending,
    _run_tools,
)
from dendrux.runtime._run_errors import (
    logger,
)
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.runtime.submit import (
    _claim_request,
    _claim_submission,
    _claimed_submission_view,
    _release_supported,
    legacy_approval_denial,
)
from dendrux.skills._session import SkillSession, current
from dendrux.types import (
    AgentStep,
    ApprovalDecision,
    ImagePart,
    ImageViewState,
    Message,
    PauseState,
    Role,
    RunStatus,
    tool_result_message,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
    )

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.media._image import UserContent
    from dendrux.runtime._continuation_guardrails import _SubmissionProjection
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        ToolResult,
    )


import dendrux.runtime.submit as _submit
import dendrux.skills._session as _skills


async def _resolve_resume_stream_store(
    store: StateStore | None, resolver: Callable[[], Any] | None
) -> StateStore:
    if store is None and resolver is not None:
        store = await resolver()
    if store is None:
        raise ValueError(
            "resume_stream() requires persistence. Pass database_url or state_store to the agent."
        )
    return store


async def _load_pause_state(
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    agent: Agent,
    provider: LLMProvider,
    *,
    reads: _run_contracts._PauseReads | None = None,
) -> PauseState:
    from dendrux.errors import PersistenceError, RunNotReclaimableError
    from dendrux.runtime.replay import rebuild_history
    from dendrux.types import InputValidationError

    pause = PauseState.from_dict(raw)
    if pause.schema_version != 2:
        return pause
    record = reads.record if reads is not None else await store.get_run(run_id)
    if record is None or record.storage_format != STORAGE_FORMAT_CHECKPOINTED:
        _codes.log_code(logger, _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise PersistenceError(code="pause_format_unsupported")
    checkpoint = (
        getattr(store, "loop_state", None)
        if getattr(store, "storage_format", None) == STORAGE_FORMAT_CHECKPOINTED
        else None
    )
    if checkpoint is None:
        checkpoint = await store.get_loop_state(run_id)
    if not checkpoint:
        _codes.log_code(logger, _codes.PAUSE_FORMAT_UNSUPPORTED, run_id=run_id)
        raise PersistenceError(code="pause_format_unsupported")
    try:
        messages = await rebuild_history(
            store,
            run_id,
            limits=agent._limits,
            reasoning_for=pause.provider_kwargs.get("model") or provider.model,
        )
    except RunNotReclaimableError as exc:
        raise InputValidationError("history", FailureReason.HISTORY_TOO_LARGE.value) from exc
    # Strategy owns the system prompt; seeds retain it for audit/reconstruction.
    pause.history = [m for m in messages.messages if m.role != Role.SYSTEM]
    if reads is not None:
        reads.history_records = messages.records
        reads.checkpoint = checkpoint
    await _overlay_pause_checkpoint(pause, store, run_id, agent, checkpoint, messages.steps)
    return pause


async def _overlay_pause_checkpoint(
    pause: PauseState,
    store: StateStore,
    run_id: str,
    agent: Agent,
    checkpoint: dict[str, Any],
    steps: list[AgentStep],
    *,
    recovery: bool = False,
) -> None:
    from dendrux.types import _usage_from_dict

    pause.steps = steps
    # Reclaim tolerates absent counters; resume rejects incomplete checkpoints.
    if recovery:
        pause.iteration = checkpoint.get("iteration", 0)
        pause.trace_order_offset = checkpoint.get("trace_order_offset", 0)
        pause.usage = _usage_from_dict(checkpoint.get("usage", {}))
        pause.provider_kwargs = dict(checkpoint.get("provider_kwargs", {}))
    else:
        pause.usage = _usage_from_dict(checkpoint["usage"])
        pause.iteration = checkpoint["iteration"]
        pause.trace_order_offset = checkpoint["trace_order_offset"]
        pause.provider_kwargs = dict(checkpoint["provider_kwargs"])
    await _restore_pause_refs(
        pause,
        store,
        run_id,
        agent,
        checkpoint,
        missing_code=FailureReason.RECOVERY_STATE_MISSING.value
        if recovery
        else "pause_format_unsupported",
    )


async def _restore_pause_refs(
    pause: PauseState,
    store: StateStore,
    run_id: str,
    agent: Agent,
    checkpoint: dict[str, Any],
    *,
    missing_code: str,
) -> None:
    from dendrux.errors import PersistenceError

    for kind, ref in (
        ("image_view", checkpoint.get("image_view_ref")),
        ("skills_session", checkpoint.get("skills", {}).get("session_ref")),
        ("clarification", checkpoint.get("clarification_ref")),
    ):
        if ref is None:
            continue
        state_trace = await store.get_trace(run_id, ref)
        if state_trace is None:
            raise PersistenceError(code=missing_code)
        payload = json.loads(state_trace.content)
        if kind == "image_view":
            pause.image_view = ImageViewState.from_dict(payload)
        elif kind == "skills_session":
            session = _run_entry.current_skill_session(agent)
            if session is not None:
                session.restore_snapshot(payload)
        else:
            pause.pause_message = payload.get("pause_message")


def _resumed_payload(
    expected_status: str,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
) -> dict[str, Any]:
    resume_data: dict[str, Any] = {"resumed_from": expected_status}
    if tool_results is not None:
        submitted: list[dict[str, Any]] = []
        for tr in tool_results:
            entry: dict[str, Any] = {"call_id": tr.call_id, "name": tr.name, "success": tr.success}
            if tr.images:
                entry["image_count"] = len(tr.images)
            submitted.append(entry)
        resume_data["submitted_results"] = submitted
    elif user_input is not None:
        resume_data["user_input"] = user_input
        if user_parts is not None:
            resume_data["image_count"] = sum(1 for p in user_parts if isinstance(p, ImagePart))
    return resume_data


def _resume_history(
    pause_state: PauseState,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
    submission_projection: _SubmissionProjection | None,
) -> list[Message]:
    history = list(pause_state.history)

    if submission_projection is not None:
        history.extend(submission_projection.messages)
    elif tool_results is not None:
        for tr in tool_results:
            history.append(tool_result_message(tr))  # noqa: PERF401 - validation can raise; keep incremental appends
    elif user_parts is not None:
        history.append(Message.with_parts(Role.USER, user_parts))
    elif user_input is not None:
        history.append(Message(role=Role.USER, content=user_input))
    return history


async def _prepare_resume(
    run_id: str,
    pause_state: PauseState,
    *,
    state_store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    expected_status: str,
    tool_results: list[ToolResult] | None = None,
    user_input: str | None = None,
    extra_notifier: LoopNotifier | None = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    tenant_id: str | None = None,
    pii_mapping: dict[str, str] | None = None,
    history_records: Any = None,
    pause_data: dict[str, Any] | None = None,
    submission_projection: _SubmissionProjection | None = None,
    recovering: bool = False,
) -> _run_contracts._ResumeContext:
    """Build everything needed to re-enter the loop after a resume.

    Receives an already-loaded PauseState — does NOT load pause data or
    claim the run. Those steps are owned by the caller (_resume_core for
    direct resume, resume_stream for streaming, resume_claimed for
    post-submit_and_claim resume).

    Steps performed:
      1. Record durable run.resumed event (the store allocates its index)
      2. Build resume history (inject tool results or user input)
      3. Create recorder with correct trace offset
      4. Notify recorder of injected messages
    """
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.tool import get_tool_def

    resolved_strategy = strategy or _native.NativeToolCalling()
    resolved_loop = loop or agent.loop or ReActLoop()
    _run_tools._validate_loop_skill_compat(agent, resolved_loop)

    from dendrux.runtime._continuation_guardrails import _submission_evidence

    async with _submission_evidence(
        state_store,
        run_id,
        submission_projection is not None and submission_projection.block is not None,
    ):
        await _run_events._emit_event(
            state_store,
            run_id,
            StoredEventType.RUN_RESUMED.value,
            _resumed_payload(expected_status, tool_results, user_input, user_parts),
        )

    history = (
        list(pause_state.history)
        if recovering
        else _resume_history(
            pause_state, tool_results, user_input, user_parts, submission_projection
        )
    )

    # 4. Create recorder with correct trace offset
    next_index = getattr(state_store, "get_next_trace_index", None)
    if next_index is not None:
        trace_order_offset = await next_index(run_id)
    else:
        # Compatibility for custom stores predating the scalar index read.
        traces = await state_store.get_traces(run_id)
        trace_order_offset = max((t.order_index for t in traces), default=-1) + 1

    if pause_state.schema_version == 2:
        trace_order_offset = max(trace_order_offset, pause_state.trace_order_offset)

    from dendrux.runtime._subagents import subagent_tool_def

    target_lookup = {}
    for fn in agent.tools:
        td = get_tool_def(fn)
        target_lookup[td.name] = td.target
    for child in agent.subagents:
        target_lookup[child.name] = subagent_tool_def(child).target
    from dendrux.guardrails import GuardrailEngine
    from dendrux.guardrails._engine import _entry_engine

    run_guardrails = (
        _entry_engine(run_id)
        or GuardrailEngine(agent.guardrails, pii_mapping=pii_mapping, run_id=run_id)
        if pause_state.schema_version == 2 and agent.guardrails
        else None
    )
    recorder = PersistenceRecorder(
        state_store,
        run_id,
        model=provider.model,
        provider_name=type(provider).__name__,
        target_lookup=target_lookup,
        initial_order_index=trace_order_offset,
        guardrail_engine=run_guardrails,
        **_run_images._recorder_image_kwargs(tenant_id, provider),
    )

    await _managed_budget.restore_recorder(recorder, agent, history_records, recovering)
    recorder._resume_pause_data = pause_data

    # NOTE: lifecycle hooks (record_run_started/notify_run_started) and
    # injected-history replay are intentionally NOT done here. The caller
    # fires them inside its own try/except so a failure during replay still
    # pairs with on_run_failed. See _replay_resume_injected_history below.

    await _run_tools._emit_declared_tools(agent, recorder, extra_notifier, run_id)
    return _run_contracts._ResumeContext(
        history=history,
        recorder=recorder,
        notifier=extra_notifier,
        pause_state=pause_state,
        resolved_loop=resolved_loop,
        resolved_strategy=resolved_strategy,
    )


def _canonicalize_legacy_denials(
    status: str, run_id: str, raw: dict[str, Any], results: list[ToolResult] | None
) -> None:
    if status == RunStatus.WAITING_APPROVAL.value and results is not None:
        results[:] = legacy_approval_denial(run_id, PauseState.from_dict(raw), results)


def _resume_claim_fallbacks(
    expected: str, results: list[ToolResult] | None, user_input: Any
) -> tuple[str, ...]:
    if results is not None and expected != RunStatus.WAITING_APPROVAL.value:
        return (RunStatus.WAITING_APPROVAL.value,)
    if results is None and user_input is None and expected == RunStatus.WAITING_APPROVAL.value:
        return (RunStatus.PAUSED.value,)
    return ()


def _resume_claim_payload(
    run_id: str,
    raw: dict[str, Any],
    status: str,
    results: list[ToolResult] | None,
    user_input: Any,
    user_parts: tuple[ContentPart, ...] | None,
    decisions: list[tuple[str, ApprovalDecision]] | None,
) -> dict[str, Any] | None:
    from dendrux.guardrails._engine import _entry_engine

    if raw.get("schema_version") != 2 and _entry_engine(run_id) is None:
        return None
    if results is None and user_input is None and status != RunStatus.WAITING_APPROVAL.value:
        return None
    from dendrux.runtime.submit import serialize_decisions, serialize_tool_results
    from dendrux.types import content_part_to_dict

    payload: dict[str, Any] = (
        {"submitted_decisions": serialize_decisions(PauseState.from_dict(raw), dict(decisions))}
        if decisions is not None
        else {"submitted_tool_results": serialize_tool_results(results)}
        if results is not None
        else {"submitted_user_input": user_input}
        if user_input is not None
        else {"approved": True}
    )
    if user_parts is not None:
        payload["submitted_user_parts"] = [content_part_to_dict(p) for p in user_parts]
    return payload


async def _claim_resume(
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    agent: Agent,
    expected: str,
    *,
    expected_pause_id: str | None,
    results: list[ToolResult] | None,
    user_input: Any = None,
    user_parts: tuple[ContentPart, ...] | None = None,
    decisions: list[tuple[str, ApprovalDecision]] | None = None,
    checkpoint: dict[str, Any] | None = None,
    on_release: Callable[[StateStore], None] | None = None,
) -> tuple[StateStore, str, dict[str, Any]]:
    request = _claim_request(
        raw,
        expected,
        expected_pause_id,
        fallbacks=_resume_claim_fallbacks(expected, results, user_input),
    )
    validate = _submit.submission_validator(
        run_id, raw, agent=agent, expected_pause_id=request.generation, call_results=results
    )
    record = await store.get_run(run_id)
    _submit.require_pause_id(
        agent,
        expected_pause_id,
        pause_id=raw.get("pause_id"),
        store=store,
        approval=(
            results is None
            and user_input is None
            and record is not None
            and record.status == RunStatus.WAITING_APPROVAL.value
        ),
        run_id=run_id,
    )
    for status in request.statuses:
        _canonicalize_legacy_denials(status, run_id, raw, results)
        payload = _resume_claim_payload(
            run_id, raw, status, results, user_input, user_parts, decisions
        )
        epoch, descriptor, payload = await _claim_submission(
            store,
            run_id,
            request,
            status,
            validate=_attempt_validator(validate, run_id, status, results),
            payload=payload,
            checkpointing=True,
            approval=False
            if status == RunStatus.WAITING_APPROVAL.value and results is not None
            else None,
            checkpoint=checkpoint,
        )
        if epoch:
            view = _state.claimed_store(store, run_id, epoch)
            await _report_claim_governance(agent, view, run_id, raw, status, on_release)
            return view, status, _claimed_submission_view(raw, descriptor, payload)
    await _run_pending._raise_resume_claim_failure(store, run_id, expected)
    raise AssertionError("unreachable")


def _attempt_validator(
    validate: Callable[[dict[str, Any]], None],
    run_id: str,
    status: str,
    results: list[ToolResult] | None,
) -> Callable[[dict[str, Any]], None]:
    """The claim attempt's locked validator (0.7).

    Results claimed for a client pause are the caller's: none may answer a
    generated subagent call. The approval attempt claims canonical runtime
    denials (``legacy_approval_denial``) for that exact generation, which stay
    valid. The status CAS precedes this check, so it sees the claimed pause.
    """
    from dendrux.runtime.submit import refuse_managed_results

    if results is None or status != RunStatus.WAITING_CLIENT_TOOL.value:
        return validate

    def client_attempt(current: dict[str, Any]) -> None:
        validate(current)
        refuse_managed_results(run_id, current.get("pending_targets"), results)

    return client_attempt


async def _report_claim_governance(
    agent: Agent,
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    status: str,
    on_release: Callable[[StateStore], None] | None,
) -> None:
    try:
        await _submit.report_governance_change(agent, store, run_id, raw)
    except BaseException as exc:
        await _release_claim(store, run_id, status)
        if on_release is not None and isinstance(exc, Exception) and _release_supported(store):
            on_release(store)
        raise


async def _release_claim(store: StateStore, run_id: str, status: str) -> bool:
    import sys

    from dendrux.runtime.submit import _release_claim_result

    epoch = getattr(store, "claim_epoch", True)
    try:
        return await _release_claim_result(store, run_id, status, epoch, sys.exception())
    except asyncio.CancelledError as cancellation:
        await _release_claim_result(store, run_id, status, epoch, cancellation)
        raise


@asynccontextmanager
async def _release_setup_failure(
    store: StateStore, run_id: str, status: str, *, enabled: bool = True
) -> AsyncIterator[None]:
    try:
        yield
    except BaseException as exc:
        if enabled and not isinstance(exc, RunFencedError):
            await _release_claim(store, run_id, status)
        raise


async def _resume_session(
    agent: Agent,
    store: StateStore,
    run_id: str,
    loop: Loop | None,
    user_input: UserContent | None,
    tool_results: list[ToolResult] | None,
    decisions: Any,
    expected_status: str,
) -> tuple[SkillSession, str | None, tuple[Any, ...]]:
    await ensure_agent_limits_fit(agent, store)
    items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, items = _chat.split_user_content(user_input)
    if decisions is not None and (
        tool_results is not None
        or user_input is not None
        or expected_status != RunStatus.WAITING_APPROVAL.value
    ):
        raise ValueError("Approval decisions only resume a run awaiting approval.")
    session = current(agent)
    if session is None or session.run_id != run_id:
        session = await _skills.continuation(agent, store, run_id, loop=loop)
    return session, user_input, items


async def _read_resume_pause(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    raw: dict[str, Any] | None,
    reads: _run_contracts._PauseReads,
    skip_claim: bool,
    expected_pause_id: str | None,
    *,
    read_record: bool = False,
) -> PauseState:
    if raw is None:
        raise ValueError(f"Run '{run_id}' has no pause state — cannot resume.")
    if not skip_claim:
        _submit.require_pause_id(
            agent, expected_pause_id, pause_id=raw.get("pause_id"), store=store, run_id=run_id
        )
    return await _load_resume_pause(
        store, agent, provider, run_id, raw, reads, read_record=read_record
    )


async def _load_resume_pause(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    raw: dict[str, Any],
    reads: _run_contracts._PauseReads,
    *,
    read_record: bool,
) -> PauseState:
    if read_record:
        reads.record = await store.get_run(run_id)
    return await _load_pause_state(store, run_id, raw, agent, provider, reads=reads)


def _require_pause_agent(pause: PauseState, agent: Agent, run_id: str, streaming: bool) -> None:
    if pause.agent_name and pause.agent_name != agent.name:
        raise ValueError(
            f"Agent name mismatch: run '{run_id}' was paused by "
            f"'{pause.agent_name}', but resume called with '{agent.name}'."
            + ("" if streaming else " This can cause state corruption.")
        )


async def _read_resume_state(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    skip_claim: bool,
    expected_pause_id: str | None,
    supplied_raw: dict[str, Any] | None,
    reclaimed: PauseState | None,
    *,
    streaming: bool = False,
    reclaimed_record: RunRecord | None = None,
) -> tuple[RunRecord | None, _run_contracts._PauseReads, Any, PauseState, bool, Any]:
    record = reclaimed_record
    if record is None and not streaming:
        record = await store.get_run(run_id)
    reads = _run_contracts._PauseReads(record)
    raw = None if reclaimed is not None else supplied_raw
    if reclaimed is None and raw is None:
        raw = await store.get_pause_state(run_id)
    pause = (
        reclaimed
        if reclaimed is not None
        else await _read_resume_pause(
            store,
            agent,
            provider,
            run_id,
            raw,
            reads,
            skip_claim,
            expected_pause_id,
            read_record=streaming,
        )
    )
    recovering = reclaimed is not None or await _reclaim.needs_resolution(
        store, run_id, pause, checkpoint=reads.checkpoint
    )
    from dendrux.runtime._continuation_guardrails import _continuation_mapping

    pii = None if streaming else await _continuation_mapping(store, run_id)
    _require_pause_agent(pause, agent, run_id, streaming)
    return reads.record, reads, raw, pause, recovering, pii


async def _resume_submission(
    store: StateStore,
    agent: Agent,
    provider: LLMProvider,
    run_id: str,
    pause: PauseState,
    raw: Any,
    reads: _run_contracts._PauseReads,
    loop: Loop | None,
    skip_claim: bool,
    expected_status: str,
    expected_pause_id: str | None,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
    input_items: tuple[Any, ...],
    decisions: Any,
) -> tuple[StateStore, str, Any, list[ToolResult] | None, tuple[ContentPart, ...] | None]:
    if tool_results is not None:
        _submit.validate_result_names(run_id, pause, tool_results)
        if not skip_claim:
            tool_results = await _submit.prepare_resume_results(
                agent, store, run_id, pause, tool_results
            )
    if not skip_claim and (input_items or any(tr.images for tr in tool_results or ())):
        tool_results, user_parts = await _run_images._ingest_resume_images(
            agent,
            provider,
            store,
            run_id,
            loop=loop,
            tool_results=tool_results,
            input_items=input_items,
        )
    if not skip_claim:
        assert raw is not None
        store, expected_status, raw = await _claim_resume(
            store,
            run_id,
            raw,
            agent,
            expected_status,
            expected_pause_id=expected_pause_id,
            results=tool_results,
            user_input=user_input,
            user_parts=user_parts,
            decisions=decisions,
            checkpoint=reads.checkpoint,
        )
    return store, expected_status, raw, tool_results, user_parts
