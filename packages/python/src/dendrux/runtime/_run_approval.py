"""Decide and execute approval batches through policy, pause and managed-handoff barriers."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import dendrux.strategies.native as _native
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    DenialReason,
    PauseReason,
)
from dendrux._meta_keys import (
    NOT_EXECUTED_META_KEY,
)
from dendrux.loops._helpers import (
    order_observations,
    record_message,
    record_tool,
)
from dendrux.loops._policy_context import (
    dispatch_policy,
    history_call_counts,
    static_denial,
)
from dendrux.runtime import (
    _run_contracts,
    _run_entry,
    _run_images,
    _run_resume_replay,
)
from dendrux.skills._session import current
from dendrux.types import (
    ApprovalDecision,
    GovernanceEventType,
    Message,
    PauseState,
    RunStatus,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from dendrux.agent import Agent
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        RunResult,
        ToolCall,
        ToolResult,
    )


def _carry_budget_fields(new: PauseState, old: PauseState) -> PauseState:
    """Copy fired budget keys from the pause being resumed onto a runner-built pause."""
    return replace(
        new,
        budget_fired=old.budget_fired,
        budget_unpriced_reported=old.budget_unpriced_reported,
        budget_known_cost_usd=old.budget_known_cost_usd,
        budget_unpriced_calls=old.budget_unpriced_calls,
    )


async def _resume_pause_barrier(state_store: StateStore, run_id: str) -> RunStatus | None:
    """Cancel-then-pause read of the run's control flags (one query).

    Returns ``CANCELLED`` when cancellation was requested, else ``PAUSED``
    when a pause was requested, else None. Cancellation wins over pause.
    """
    from dendrux.runtime.state import read_run_control

    control = await read_run_control(state_store, run_id)
    if control.cancel_requested:
        return RunStatus.CANCELLED
    if control.pause_requested:
        return RunStatus.PAUSED
    return None


async def _resume_barrier(state_store: StateStore | None, run_id: str) -> RunStatus | None:
    """Cancel-then-pause check before an approved side effect.

    Runs after a resume claim, before each approved/held server call.
    Cancellation wins over pause.
    """
    if state_store is None:
        return None
    return await _resume_pause_barrier(state_store, run_id)


def _approval_outcomes(
    pause_state: PauseState,
    target_lookup: dict[str, Any],
    decisions: dict[str, ApprovalDecision] | None,
) -> list[tuple[ToolCall, str, str | None]]:
    """Classify each pending call: (call, outcome, denial reason) in pending order.

    ``decisions=None`` is the legacy batch approve: every gate approved.
    """
    from dendrux.runtime.submit import DEFAULT_DENIAL_REASON, approval_gates_of, is_client_call

    gates = approval_gates_of(pause_state, target_lookup)
    out: list[tuple[ToolCall, str, str | None]] = []
    for tc in pause_state.pending_tool_calls:
        if is_client_call(pause_state, tc, target_lookup):
            out.append((tc, "client_pending", None))
        elif tc.id not in gates:
            out.append((tc, "not_gated_executed", None))
        else:
            decision = decisions.get(tc.id) if decisions is not None else None
            if decision is None or decision.approved:
                out.append((tc, "approved", None))
            else:
                out.append((tc, "denied", decision.reason or DEFAULT_DENIAL_REASON))
    return out


def _decided_payload(
    run_id: str,
    outcomes: list[tuple[ToolCall, str, str | None]],
    *,
    scalar: str | None = None,
) -> dict[str, Any]:
    """``approval.decided`` payload: scalar decision + every pending call's outcome."""
    if scalar is None:
        gated = [o for _, o, _ in outcomes if o in ("approved", "denied")]
        if all(o == "approved" for o in gated):
            scalar = "approved"
        elif all(o == "denied" for o in gated):
            scalar = "rejected"
        else:
            scalar = "mixed"
    return {
        "decision": scalar,
        "run_id": run_id,
        "decisions": [
            {"call_id": tc.id, "tool_name": tc.name, "decision": outcome, "reason": reason}
            for tc, outcome, reason in outcomes
        ],
    }


async def _record_approval_decided(
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    data: dict[str, Any],
    *,
    event_id: str | None = None,
) -> None:
    from dendrux.loops._helpers import notify_governance, record_governance
    from dendrux.runtime.persistence import PersistenceRecorder

    if event_id is not None and isinstance(recorder, PersistenceRecorder):
        await recorder._emit_event(
            GovernanceEventType.APPROVAL_DECIDED.value, iteration, data, event_id=event_id
        )
    else:
        await record_governance(
            recorder, run_id, GovernanceEventType.APPROVAL_DECIDED.value, iteration, data
        )
    await notify_governance(
        notifier, run_id, GovernanceEventType.APPROVAL_DECIDED.value, iteration, data
    )


def _rejection_decided_payload(
    run_id: str, pause_state: PauseState, tool_results: list[ToolResult]
) -> dict[str, Any]:
    """Legacy batch rejection: every pending call received a synthetic denial."""
    errors = {tr.call_id: tr.error for tr in tool_results}
    outcomes = [(tc, "denied", errors.get(tc.id)) for tc in pause_state.pending_tool_calls]
    return _decided_payload(run_id, outcomes, scalar="rejected")


def _approval_call_counts(recorder: LoopRecorder, pause_state: PauseState) -> dict[str, int]:
    restored_counts: dict[str, int] | None = getattr(recorder, "execution_counts", lambda: None)()
    if restored_counts is not None:
        return restored_counts
    counts = history_call_counts(pause_state.history)
    # The approved fallback historically excludes empty names, too.
    counts.pop("", None)
    return counts


def _limit_approval_clients(
    agent: Agent, outcomes: Any, lookups: Any, counts: dict[str, int]
) -> tuple[Any, Any]:
    from dendrux.tools import enforce_call_limits

    client_calls = [
        tc
        for tc, outcome, _ in outcomes
        if outcome == "client_pending" and tc.name not in agent.deny
    ]
    _, limited = enforce_call_limits(client_calls, max_calls=lookups.max_calls, call_counts=counts)
    client_rejections = {tc.id: result for tc, result in limited}
    outcomes = [
        (tc, "policy_denied", DenialReason.DENIED_BY_POLICY.value)
        if outcome == "client_pending" and tc.name in agent.deny
        else (tc, "call_limited", None)
        if tc.id in client_rejections
        else (tc, outcome, reason)
        for tc, outcome, reason in outcomes
    ]
    return outcomes, client_rejections


def _inject_approved_skill(agent: Agent, pause_state: PauseState, lookups: Any) -> None:
    from dendrux.skills._session import SkillTool

    session = current(agent)
    # An absent executor is a specified error only for a pinned approved script.
    # Inject an execution closure here without advertising the unavailable tool.
    if session is not None and session.configured and session.executor is None:
        for tc in pause_state.pending_tool_calls:
            if tc.name == "run_skill_script" and tc.name not in lookups.fn:
                lookups.fn[tc.name] = SkillTool(session, tc.name)


async def _approval_policy_denial(
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    tc: ToolCall,
    iteration: int,
    on_work: Callable[[], None] | None,
) -> ToolResult:
    from dendrux.loops._helpers import notify_governance, record_governance

    result, data = static_denial(tc.name, tc.id)
    await record_governance(
        recorder,
        run_id,
        GovernanceEventType.POLICY_DENIED.value,
        iteration,
        data,
        correlation_id=tc.id,
    )
    _run_resume_replay._mark_work(on_work)
    await notify_governance(
        notifier,
        run_id,
        GovernanceEventType.POLICY_DENIED.value,
        iteration,
        data,
        correlation_id=tc.id,
    )
    return result


def _approval_dispatch_inputs(
    strategy: Strategy | None, warnings: list[str] | None
) -> tuple[Strategy, list[str]]:
    return (
        strategy if strategy is not None else _native.NativeToolCalling(),
        warnings if warnings is not None else [],
    )


async def _execute_approved_tools(
    agent: Agent,
    pause_state: PauseState,
    recorder: LoopRecorder,
    notifier: LoopNotifier | None,
    run_id: str,
    *,
    history: list[Message] | None = None,
    resolved_loop: Loop | None = None,
    strategy: Strategy | None = None,
    notifier_warnings: list[str] | None = None,
    decisions: dict[str, ApprovalDecision] | None = None,
    state_store: StateStore | None = None,
    image_runtime: ImageRuntime | None = None,
    on_work: Callable[[], None] | None = None,
) -> _run_contracts._ApprovalResume:
    """Resolve an approval batch on resume, routing each call by target.

    Shared by _resume_core (batch) and resume_stream (streaming). Walks
    ``pending_tool_calls`` in order:

    - client-target calls are never executed; they are returned in
      ``client_pending`` and the caller pauses the run for client results;
    - denied gated calls get a failed result ``{"denied": reason}``;
    - approved gated calls and held non-gated server calls execute through
      the approved path (no ``before_tool`` re-run: the human decision wins
      and held calls were already evaluated before the pause).

    ``decisions=None`` is the legacy batch approve. Before every server
    execution a resume barrier checks cancel (then pause); on a hit the
    remaining calls are skipped and ``barrier`` is set.

    Tool messages are appended to ``history`` (the live ctx.history list,
    not pause_state.history which is a snapshot). ``resolved_loop`` is
    forwarded to ``agent.get_tool_lookups()`` so a runtime loop override
    stays consistent across resume. Caller emits ``approval.decided`` and
    handles loop re-entry.
    """
    import json

    from dendrux.loops._helpers import notify_message as _ntfy_msg
    from dendrux.loops._helpers import notify_tool as _ntfy_tool
    from dendrux.loops._helpers import notify_tool_started as _ntfy_tool_started
    from dendrux.loops._helpers import record_tool_started as _rec_tool_started
    from dendrux.loops.react import _execute_tool, _format_result, _process_tool_calls
    from dendrux.skills._session import SkillTool
    from dendrux.tools import enforce_call_limits
    from dendrux.tools._validate import check_tool_call
    from dendrux.types import AgentStep, ToolResult

    strategy, notifier_warnings = _approval_dispatch_inputs(strategy, notifier_warnings)
    target_history = history if history is not None else pause_state.history
    lookups = await agent.get_tool_lookups(loop=resolved_loop)
    configure = getattr(recorder, "configure_tools", None)
    if configure is not None:
        configure(list(lookups.defs.values()))
    outcomes = _approval_outcomes(pause_state, lookups.target, decisions)
    results: list[tuple[ToolCall, ToolResult]] = []
    counts = _approval_call_counts(recorder, pause_state)
    outcomes, client_rejections = _limit_approval_clients(agent, outcomes, lookups, counts)
    client_pending = [tc for tc, outcome, _ in outcomes if outcome == "client_pending"]
    decided = _decided_payload(
        run_id,
        [
            (tc, "denied" if outcome in {"policy_denied", "call_limited"} else outcome, reason)
            for tc, outcome, reason in outcomes
        ],
        scalar="approved" if decisions is None else None,
    )
    _inject_approved_skill(agent, pause_state, lookups)

    iteration = pause_state.iteration

    async def _append(tc: ToolCall, tr: ToolResult, *, executed: bool = True) -> None:
        if executed:
            await record_tool(recorder, run_id, tc, tr, iteration)
            _run_resume_replay._mark_work(on_work)
            await _ntfy_tool(notifier, run_id, tc, tr, iteration, warnings=notifier_warnings)
        result_msg = _format_result(strategy, tr)
        if not executed or tr.meta.get("code") == "max_calls":
            result_msg.meta[NOT_EXECUTED_META_KEY] = True
        target_history.append(result_msg)
        result_msg = await record_message(recorder, run_id, result_msg, iteration)
        _run_resume_replay._mark_work(on_work)
        target_history[-1] = result_msg
        await _ntfy_msg(notifier, run_id, result_msg, iteration, warnings=notifier_warnings)
        results.append((tc, tr))

    for index, (tc, outcome, reason) in enumerate(outcomes):
        if outcome == "client_pending":
            continue
        if outcome == "policy_denied":
            denied = await _approval_policy_denial(
                recorder, notifier, run_id, tc, iteration, on_work
            )
            await _append(tc, denied, executed=False)
            continue
        if outcome == "call_limited":
            await _append(tc, client_rejections[tc.id])
            continue
        if outcome == "denied":
            assert reason is not None
            denied = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=json.dumps({"denied": reason}),
                success=False,
                error=reason,
            )
            if getattr(state_store, "storage_format", 1) != STORAGE_FORMAT_CHECKPOINTED:
                await _rec_tool_started(recorder, run_id, tc, iteration)
            await _ntfy_tool_started(notifier, run_id, tc, iteration, warnings=notifier_warnings)
            await _append(tc, denied)
            continue

        barrier = await _resume_barrier(state_store, run_id)
        if barrier is not None:
            remaining = tuple(o for o in outcomes[index:] if o[1] != "client_pending")
            order_observations(recorder, target_history)
            return _run_contracts._ApprovalResume(
                results, client_pending, decided, barrier, remaining
            )

        if isinstance(lookups.fn.get(tc.name), SkillTool):
            _run_resume_replay._mark_work(on_work)
            outcome_s = await _process_tool_calls(
                step=AgentStep(reasoning=None, action=tc),
                call_counts=counts,
                lookups=lookups,
                strategy=strategy,
                recorder=recorder,
                notifier=notifier,
                run_id=run_id,
                iteration=iteration,
                history=target_history,
                warnings=notifier_warnings,
                deny=agent._deny,
                tool_result_transform=agent.tool_result_transform,
                tenant_id=None,
                **dispatch_policy(name=tc.name, phase="approved"),
                **_run_images._tool_image_kwargs(_process_tool_calls, image_runtime),
            )
            results.extend(outcome_s.all_results)
            continue
        if tc.name in agent.deny:
            denied = await _approval_policy_denial(
                recorder, notifier, run_id, tc, iteration, on_work
            )
            await _append(tc, denied, executed=False)
            continue
        _, rejected = enforce_call_limits([tc], max_calls=lookups.max_calls, call_counts=counts)
        if rejected:
            await _append(tc, rejected[0][1])
            continue
        counts[tc.name] = counts.get(tc.name, 0) + 1
        invalid_result = check_tool_call(tc, lookups.defs.get(tc.name))
        if invalid_result is None:
            await _rec_tool_started(recorder, run_id, tc, iteration)
            _run_resume_replay._mark_work(on_work)
            await _ntfy_tool_started(notifier, run_id, tc, iteration, warnings=notifier_warnings)
            from dendrux.loops._helpers import lease_scope

            async with lease_scope(recorder):
                tr = await _execute_tool(
                    tc,
                    lookups,
                    agent.tool_result_transform,
                    run_id=run_id,
                    **_run_images._tool_image_kwargs(_execute_tool, image_runtime),
                )
        else:
            tr = invalid_result
        await _append(tc, tr)
    order_observations(recorder, target_history)
    return _run_contracts._ApprovalResume(results, client_pending, decided, None)


async def _validated_decisions(
    agent: Agent,
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    pause_state: PauseState,
    decisions: list[tuple[str, ApprovalDecision]] | None,
) -> dict[str, ApprovalDecision] | None:
    """Re-validate claimed decisions against the persisted gate snapshot."""
    if decisions is None:
        return None
    from dendrux.runtime.submit import validate_approval_decisions

    lookups = await agent.get_tool_lookups(loop=ctx.resolved_loop)
    return validate_approval_decisions(run_id, pause_state, decisions, lookups.target)


def _approval_stop_result(
    run_id: str,
    pause_state: PauseState,
    history: list[Message],
    approval: _run_contracts._ApprovalResume,
) -> RunResult | None:
    """Result for an approval resume that must not re-enter the loop.

    Barrier hit → CANCELLED. Held client calls → a fresh WAITING_CLIENT_TOOL
    pause over the updated history. Otherwise None (re-enter the loop).
    """
    from dendrux.loops.react import _commentary, _snapshot_usage
    from dendrux.types import RunResult

    if approval.barrier == RunStatus.PAUSED:
        return _barrier_pause_result(run_id, pause_state, history, approval)
    if approval.barrier is not None:
        return RunResult(
            run_id=run_id,
            status=approval.barrier,
            steps=pause_state.steps,
            commentary=_commentary(pause_state.steps),
            iteration_count=pause_state.iteration,
            usage=pause_state.usage,
        )
    if not approval.client_pending:
        return None
    client_pause = PauseState(
        agent_name=pause_state.agent_name,
        pending_tool_calls=list(approval.client_pending),
        pending_targets={
            tc.id: pause_state.pending_targets.get(tc.id) or "client"
            for tc in approval.client_pending
        },
        history=list(history),
        steps=list(pause_state.steps),
        iteration=pause_state.iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(pause_state.usage),
        image_view=pause_state.image_view,
    )
    client_pause = _carry_budget_fields(client_pause, pause_state)
    client_pause = _run_images._carry_image_view(client_pause, pause_state)
    return RunResult(
        run_id=run_id,
        status=RunStatus.WAITING_CLIENT_TOOL,
        steps=pause_state.steps,
        commentary=_commentary(pause_state.steps),
        iteration_count=pause_state.iteration,
        usage=pause_state.usage,
        meta={"pause_state": client_pause},
    )


def _approval_stop(
    run_id: str,
    pause_state: PauseState,
    history: list[Message],
    approval: _run_contracts._ApprovalResume,
) -> RunResult | None:
    """The approved subagent call's handoff, else the ordinary approval stop (0.7)."""
    if approval.handoff is not None:
        return _approved_handoff(run_id, pause_state, history, approval.handoff)
    return _approval_stop_result(run_id, pause_state, history, approval)


def _approved_handoff(
    run_id: str, pause_state: PauseState, history: list[Message], handoff: Any
) -> RunResult:
    """The approved batch's subagent call, handed to the durable bind (0.7)."""
    from dataclasses import replace as _replace

    from dendrux.loops.react import MANAGED_HANDOFF_KEY
    from dendrux.types import RunResult

    call = handoff.call
    pause = _replace(
        pause_state,
        history=list(history),
        pending_tool_calls=[call],
        pending_targets={call.id: "agent"},
        approval_gates=None,
        pending_decisions=None,
        pause_reason=None,
    )
    return RunResult(
        run_id=run_id,
        status=RunStatus.RUNNING,
        steps=pause.steps,
        iteration_count=pause.iteration,
        usage=pause.usage,
        meta={"pause_state": pause, MANAGED_HANDOFF_KEY: handoff},
    )


def _barrier_pause_result(
    run_id: str,
    pause_state: PauseState,
    history: list[Message],
    approval: _run_contracts._ApprovalResume,
) -> RunResult:
    """PAUSED result for a pause observed between approved calls.

    Holds every call that has no result yet (the unexecuted remainder plus
    held client calls) in pending order, with the decisions already made for
    the gated ones; resume executes them without asking again.
    """
    from dendrux.loops.react import _commentary, _snapshot_usage
    from dendrux.types import RunResult

    old_gates = pause_state.approval_gates or {}
    gates: dict[str, dict[str, str | None]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for tc, outcome, reason in approval.remaining:
        if outcome in ("approved", "denied"):
            gates[tc.id] = dict(old_gates.get(tc.id) or {"source": "legacy", "reason": None})
            decisions[tc.id] = {"approved": outcome == "approved", "reason": reason}
    client_ids = {tc.id for tc in approval.client_pending}
    open_ids = {tc.id for tc, _, _ in approval.remaining} | client_ids
    pending = [tc for tc in pause_state.pending_tool_calls if tc.id in open_ids]
    targets: dict[str, str] = {}
    for tc in pending:
        target = pause_state.pending_targets.get(tc.id)
        if target is None and tc.id in client_ids:
            target = "client"
        if target is not None:
            targets[tc.id] = target
    pause = PauseState(
        agent_name=pause_state.agent_name,
        pending_tool_calls=pending,
        pending_targets=targets,
        history=list(history),
        steps=list(pause_state.steps),
        iteration=pause_state.iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(pause_state.usage),
        approval_gates=gates,
        pause_reason=PauseReason.REQUESTED.value,
        pending_decisions=decisions,
        image_view=pause_state.image_view,
    )
    pause = _carry_budget_fields(pause, pause_state)
    pause = _run_images._carry_image_view(pause, pause_state)
    return RunResult(
        run_id=run_id,
        status=RunStatus.PAUSED,
        steps=pause_state.steps,
        commentary=_commentary(pause_state.steps),
        iteration_count=pause_state.iteration,
        usage=pause_state.usage,
        meta={"pause_state": pause},
    )


def _pending_decision_map(pause_state: PauseState) -> dict[str, ApprovalDecision] | None:
    """Decisions persisted with a pause that interrupted an approval batch."""
    if pause_state.pending_decisions is None:
        return None
    return {
        call_id: ApprovalDecision(
            approved=bool(d.get("approved")),
            reason=d.get("reason") if isinstance(d.get("reason"), str) else None,
        )
        for call_id, d in pause_state.pending_decisions.items()
    }


async def _resume_approval_batch(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    user_input: str | None,
    image_runtime: ImageRuntime | None,
    *,
    decisions: Any = None,
    on_work: Callable[[], None] | None = None,
    streaming: bool = False,
) -> tuple[_run_contracts._ApprovalResume | None, bool]:
    pause = ctx.pause_state
    approving = (
        results is None and user_input is None and expected == RunStatus.WAITING_APPROVAL.value
    )
    paused_batch = expected == RunStatus.PAUSED.value and bool(pause.pending_tool_calls)
    if not (approving or paused_batch):
        return None, approving
    decision_map = None
    if not approving:
        decision_map = _pending_decision_map(pause)
    elif not streaming:
        decision_map = await _validated_decisions(agent, ctx, run_id, pause, decisions)
    approval = await _execute_approved_tools(
        agent,
        pause,
        ctx.recorder,
        ctx.notifier,
        run_id,
        history=ctx.history,
        resolved_loop=ctx.resolved_loop,
        strategy=ctx.resolved_strategy,
        notifier_warnings=ctx.notifier_warnings,
        decisions=decision_map,
        state_store=store,
        image_runtime=image_runtime,
        on_work=on_work,
    )
    return await _hand_off_approved_subagent(agent, ctx, run_id, approval), approving


async def _hand_off_approved_subagent(
    agent: Agent,
    ctx: _run_contracts._ResumeContext,
    run_id: str,
    approval: _run_contracts._ApprovalResume,
) -> _run_contracts._ApprovalResume:
    """A held generated subagent call goes to the bind, never to a client pause (0.7).

    The approved batch's server outcomes are already recorded. The subagent
    call (never gated: an approval decision on it is a denial) passes the
    child's brief limit, as in a live batch, and is handed off.
    """
    from dendrux.loops.react import ManagedHandoff, SubagentTool, _admit_managed_briefs

    if approval.barrier is not None or not approval.client_pending:
        return approval
    lookups = await agent.get_tool_lookups(loop=ctx.resolved_loop)
    managed = [
        tc for tc in approval.client_pending if isinstance(lookups.fn.get(tc.name), SubagentTool)
    ]
    if not managed:
        return approval
    if len(approval.client_pending) != 1:
        from dendrux.errors import ManagedUnsupportedError

        raise ManagedUnsupportedError("approval_mixed_batch", run_id)
    from dendrux.runtime._reclaim import _recovery_batch

    admitted = await _admit_managed_briefs(managed, lookups, _recovery_batch(ctx, run_id))
    entry = lookups.fn[managed[0].name]
    assert isinstance(entry, SubagentTool)
    handoff = (
        ManagedHandoff(call=admitted[0], child=entry.agent, parent=agent) if admitted else None
    )
    return approval._replace(client_pending=[], handoff=handoff)


async def _finish_approval_batch(
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    approval: _run_contracts._ApprovalResume | None,
    approving: bool,
) -> RunResult | None:
    pause = ctx.pause_state
    if approval is not None:
        if approving:
            await _record_approval_decided(
                ctx.recorder, ctx.notifier, run_id, pause.iteration, approval.decided
            )
        stop = _approval_stop(run_id, pause, ctx.history, approval)
        if stop is not None:
            return stop
    if results is not None and expected == RunStatus.WAITING_APPROVAL.value:
        data = _rejection_decided_payload(run_id, pause, results)
        await _record_approval_decided(ctx.recorder, ctx.notifier, run_id, pause.iteration, data)
    return None


async def _resume_approval_work(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    expected: str,
    tool_results: list[ToolResult] | None,
    user_input: str | None,
    decisions: Any,
    image_runtime: ImageRuntime | None,
    on_work: Callable[[], None],
) -> RunResult | None:
    from dendrux.loops._helpers import _BlockedObservationError

    try:
        approval, approving = await _resume_approval_batch(
            agent,
            store,
            run_id,
            ctx,
            expected,
            tool_results,
            user_input,
            image_runtime,
            decisions=decisions,
            on_work=on_work,
        )
        return await _finish_approval_batch(
            run_id, ctx, expected, tool_results, approval, approving
        )
    except _BlockedObservationError as blocked:
        from dendrux.guardrails._engine import _entry_engine
        from dendrux.runtime._entry_guardrails import _EntryProjection

        projection = _EntryProjection(_entry_engine(run_id), blocked.error, blocked.findings)
        return await _run_entry._blocked_entry_result(
            projection, ctx.recorder, ctx.notifier, run_id, "", None, []
        )
