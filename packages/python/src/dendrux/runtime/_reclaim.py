"""Explicit recovery of a fenced ReAct checkpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from dendrux import _codes
from dendrux._lifecycle import (
    TERMINAL_STATUSES,
    ApprovalReason,
    FailureReason,
    GovernanceEventType,
    PauseReason,
    StoredEventType,
)
from dendrux.errors import PersistenceError, RunNotFoundError, RunNotReclaimableError
from dendrux.loops._helpers import order_observations, project_message, record_message, record_tool
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime.persistence import PersistenceRecorder, resolve_pending
from dendrux.runtime.state import claimed_store
from dendrux.types import (
    Message,
    PauseState,
    Role,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
    _usage_from_dict,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from dendrux.agent import Agent
    from dendrux.loops.base import LoopNotifier
    from dendrux.runtime.runner import _ResumeContext
    from dendrux.runtime.state import ReclaimableRun, StateStore, TenantScope

logger = logging.getLogger(__name__)


async def reclaim(
    agent: Agent,
    store: StateStore,
    run_id: str,
    *,
    tenant_id: TenantScope,
    item: ReclaimableRun | None,
    notifier: LoopNotifier | None,
) -> RunResult:
    from dendrux.loops.react import ReActLoop
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.durability import _ownership_handoff
    from dendrux.runtime.runner import _resume_core, _stop_marked_continuation
    from dendrux.runtime.submit import _cancel_claim, governance_snapshot
    from dendrux.skills._session import SkillSession

    if not capabilities(store).recovery:
        _codes.log_code(logger, _codes.STORE_CAPABILITY, run_id=run_id)
        raise NotImplementedError(
            f"StateStore {type(store).__name__} does not declare the 'recovery' capability"
        )
    session = SkillSession.prepare(agent)
    session.run_id = run_id
    try:
        claim = await _ownership_handoff(
            store,
            lambda: store.reclaim_run(
                run_id,
                tenant_id=tenant_id,
                agent_name=agent.name,
                expected_claim_epoch=item.claim_epoch if item else None,
                governance=governance_snapshot(agent),
                governance_policy=agent._on_governance_change,
                skill_pins={s.name: s.digest for s in session.available.values()},
                loop_type="react"
                if isinstance(agent._effective_loop(None) or ReActLoop(), ReActLoop)
                else "single_call",
            ),
            partial(_cancel_claim, store, run_id),
            run_id=run_id,
        )
        if claim is False:
            raise RunNotReclaimableError(run_id, "already_reclaimed")
    except (RunNotFoundError, RunNotReclaimableError) as exc:
        code = exc.reason if isinstance(exc, RunNotReclaimableError) else "run_not_found"
        _codes.log_code(logger, _codes.REGISTRY[code], run_id=run_id)
        if code == "cancelled":
            from dendrux.runtime.runner import _build_cached_result

            return await _build_cached_result(store, run_id)
        raise
    owned = claimed_store(store, run_id, claim)
    with session.bind():
        history_records: list[Any] = []
        async with _reclaim_setup(owned, run_id):
            record = await owned.get_run(run_id)
            state = claim.loop_state or {}
            stopped = await _stop_marked_continuation(
                owned,
                run_id,
                record,
                agent=agent,
                iteration=state.get("iteration", 0),
                usage=_usage_from_dict(state.get("usage", {})),
            )
            if stopped is not None:
                return _reclaimed_result(run_id, stopped)
            reader = _PauseReadMemo(owned)
            pause = await rebuild_pause(
                agent, cast("StateStore", reader), run_id, state, history_records=history_records
            )
            provider = agent._require_provider()
        result = await _resume_core(
            run_id,
            state_store=owned,
            agent=agent,
            provider=provider,
            expected_status=RunStatus.RUNNING.value,
            extra_notifier=notifier,
            _skip_claim=True,
            _reclaim_pause=pause,
            _reclaim_history=history_records,
            _reclaim_record=record,
            _reclaim_submission=_unconsumed_submission(state, reader.raw),
        )
        return _reclaimed_result(run_id, result)


class _PauseReadMemo:
    """Forward to the claimed store, remembering the pause read the rebuild already makes.

    The unconsumed submission is then available without another read of the row.
    """

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self.raw: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        self.raw = await self._store.get_pause_state(run_id)
        return self.raw


def _unconsumed_submission(
    state: dict[str, Any], raw: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The stored payload of a claimed submission that recovery has not consumed yet.

    An invalid envelope returns None here; recovery reports it as before.
    """
    from dendrux.runtime.submit import _submission_payload

    if not state.get("submission") or not raw:
        return None
    try:
        return _submission_payload(raw) or None
    except ValueError:
        return None


def _reclaimed_result(run_id: str, result: RunResult) -> RunResult:
    if result.status in TERMINAL_STATUSES:
        _codes.log_code(logger, _codes.RUN_FINALIZED, run_id=run_id)
    return result


@asynccontextmanager
async def _reclaim_setup(store: StateStore, run_id: str) -> AsyncIterator[None]:
    from dendrux.runtime.runner import _drain_task_cancellation

    try:
        try:
            yield
        except Exception as exc:
            await _handle_rebuild_failure(store, run_id, exc)
            raise
    except asyncio.CancelledError:
        await _drain_task_cancellation(store, run_id, False)
        raise


def _rebuild_failure_reason(exc: Exception) -> str | None:
    if isinstance(exc, RunNotReclaimableError):
        reason = exc.reason
    elif isinstance(exc, PersistenceError):
        reason = exc.code
    elif isinstance(exc, (ValueError, KeyError, TypeError)):
        reason = FailureReason.RECOVERY_STATE_INVALID.value
    else:
        return None
    reasons = {
        FailureReason.HISTORY_TOO_LARGE.value,
        FailureReason.RECOVERY_CALL_MISSING.value,
        FailureReason.RECOVERY_STATE_MISSING.value,
        FailureReason.RECOVERY_STATE_INVALID.value,
    }
    return reason if reason in reasons else None


async def _handle_rebuild_failure(store: StateStore, run_id: str, exc: Exception) -> None:
    from dendrux.runtime.runner import _release_claim

    reason = _rebuild_failure_reason(exc)
    if reason is None:
        await _release_claim(store, run_id, RunStatus.RUNNING.value)
        return
    await store.finalize_with_event(
        run_id,
        status=RunStatus.ERROR.value,
        allowed_current_statuses=[RunStatus.RUNNING.value],
        event_type=StoredEventType.RUN_ERROR.value,
        event_data={"failure_reason": reason},
        failure_reason=reason,
        error=reason,
    )
    _codes.log_code(logger, _codes.REGISTRY[reason], run_id=run_id)
    raise RunNotReclaimableError(run_id, reason) from exc


async def rebuild_pause(
    agent: Agent,
    store: StateStore,
    run_id: str,
    state: dict[str, Any],
    *,
    history_records: list[Any] | None = None,
) -> PauseState:
    from dendrux.runtime.replay import rebuild_history
    from dendrux.runtime.runner import _overlay_pause_checkpoint
    from dendrux.runtime.state import TraceRecord

    class HistoryReader:
        def __getattr__(self, name: str) -> Any:
            return getattr(store, name)

        async def conversation_traces(self, *args: Any, **kwargs: Any) -> Any:
            page = await store.conversation_traces(*args, **kwargs)
            if history_records is not None:
                history_records.extend(
                    TraceRecord(
                        id=e.trace_id,
                        role=e.role,
                        content=e.content,
                        order_index=e.position,
                        meta=e.meta,
                        trace_kind=e.source,
                    )
                    for e in page.entries
                )
            return page

    rebuilt = await rebuild_history(
        cast("StateStore", HistoryReader()),
        run_id,
        limits=agent._limits,
        reasoning_for=state.get("provider_kwargs", {}).get("model")
        or agent._require_provider().model,
    )
    pending = {p["call_id"]: p for p in state.get("pending", [])}
    calls = {tc.id: tc for message in rebuilt.messages for tc in message.tool_calls or []}
    if any(call_id not in calls for call_id in pending):
        raise PersistenceError(code=FailureReason.RECOVERY_CALL_MISSING.value)
    raw = await store.get_pause_state(run_id)
    pause = (
        PauseState.from_dict(raw)
        if raw
        else PauseState(
            agent.name,
            [],
            [],
            [],
            0,
            0,
            _usage_from_dict({}),
            schema_version=2,
            loop_state_ref=True,
            image_view=None,
        )
    )
    pause.history = [m for m in rebuilt.messages if m.role != Role.SYSTEM]
    pause.steps = rebuilt.steps
    pause.pending_tool_calls = [calls[call_id] for call_id in pending]
    pause.pending_targets = {p["call_id"]: p["target"] for p in pending.values()}
    # The checkpoint decision supersedes gates in an older approval snapshot.
    pause.approval_gates = {
        call_id: gate
        for call_id, gate in (pause.approval_gates or {}).items()
        if pending.get(call_id, {}).get("phase") not in {"approved", "rejected"}
    } or None
    await _overlay_pause_checkpoint(
        pause, store, run_id, agent, state, rebuilt.steps, recovery=True
    )
    return pause


async def needs_resolution(
    store: StateStore, run_id: str, pause: PauseState, *, checkpoint: dict[str, Any] | None = None
) -> bool:
    if pause.schema_version != 2:
        return False
    if any(g.get("source") == "recovery" for g in (pause.approval_gates or {}).values()):
        return True
    if pause.pause_reason == PauseReason.REQUESTED.value and pause.pending_decisions is not None:
        return False
    if not pause.pending_tool_calls:
        return False
    calls = {tc.id for tc in pause.pending_tool_calls}
    if checkpoint is not None:
        resolution = _checkpoint_resolution(pause, calls, checkpoint)
        if resolution is not None:
            return resolution
    return any(
        row.tool_call_id in calls and row.outcome == "started"
        for row in await store.get_tool_calls(
            run_id, iteration_index=pause.iteration, limit=MAX_PAGE
        )
    )


def _checkpoint_resolution(
    pause: PauseState, calls: set[str], checkpoint: dict[str, Any]
) -> bool | None:
    pending = {p["call_id"]: p for p in checkpoint.get("pending", [])}
    remaining = [pending.get(call_id, {}) for call_id in calls]
    # Client calls never write server intents. Unknown entries still need
    # the defensive evidence read.
    if all(p.get("target") == "client" for p in remaining):
        return False
    # A requested pause is not an approval decision. The checkpoint retains
    # unresolved governance even before a hook, intent or approval pause exists.
    if pause.pause_reason == PauseReason.REQUESTED.value and any(
        p.get("target") == "server" and p.get("phase") in {"gated", "dispatchable"}
        for p in remaining
    ):
        return True
    return None


def recovery_gate() -> dict[str, str | None]:
    return {"source": "recovery", "reason": ApprovalReason.REPLAY_AFTER_CRASH.value}


def _recovery_decisions(payload: dict[str, Any]) -> Any:
    from dendrux.runtime.submit import deserialize_decisions, deserialize_tool_results
    from dendrux.types import ApprovalDecision

    if "submitted_decisions" in payload:
        return dict(deserialize_decisions(payload["submitted_decisions"]))
    if payload.get("approved") is False:
        return {
            tr.call_id: ApprovalDecision(approved=False, reason=tr.error)
            for tr in deserialize_tool_results(payload.get("submitted_tool_results", []))
        }
    return None


def _recovery_input(payload: dict[str, Any], kind: str) -> list[Message]:
    from dendrux.runtime.runner import _stored_parts
    from dendrux.runtime.submit import deserialize_tool_results
    from dendrux.types import tool_result_message

    if kind == "tool_results":
        return [
            tool_result_message(r)
            for r in deserialize_tool_results(payload["submitted_tool_results"])
        ]
    if kind == "input":
        parts = _stored_parts(payload.get("submitted_user_parts"))
        return [
            Message.with_parts(Role.USER, parts)
            if parts
            else Message(Role.USER, payload["submitted_user_input"])
        ]
    return []


def _recovery_denial(tc: ToolCall, reason: str | None) -> Message:
    from dendrux.types import tool_result_message

    return tool_result_message(
        ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=json.dumps({"denied": reason}),
            success=False,
            error=reason,
        )
    )


async def _recovery_approval(
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    lookups: Any,
    descriptor: dict[str, Any],
    payload: dict[str, Any],
    pause: PauseState,
) -> list[Message]:
    from dendrux.runtime.runner import (
        _approval_outcomes,
        _decided_payload,
        _record_approval_decided,
    )

    decisions = _recovery_decisions(payload)
    outcomes = _approval_outcomes(pause, lookups.target, decisions)
    if payload.get("approved") is False and "submitted_decisions" not in payload:
        outcomes = [(tc, "denied", decisions[tc.id].reason) for tc in pause.pending_tool_calls]
    await _record_approval_decided(
        ctx.recorder,
        ctx.notifier,
        run_id,
        pause.iteration,
        _decided_payload(run_id, outcomes),
        event_id=descriptor["submission_id"],
    )
    rows = await store.get_tool_calls(run_id, iteration_index=pause.iteration, limit=MAX_PAGE)
    submitted = (
        _recovery_input(payload, "tool_results") if "submitted_tool_results" in payload else []
    )
    denials = {message.call_id: message for message in submitted if message.call_id is not None}
    return await _recovery_denied_calls(ctx, run_id, outcomes, rows, denials)


async def _recovery_denied_calls(
    ctx: _ResumeContext, run_id: str, outcomes: Any, rows: Any, denials: dict[str, Message]
) -> list[Message]:
    recorder = cast("PersistenceRecorder", ctx.recorder)
    by_id = {row.tool_call_id: row for row in rows}
    messages = []
    for tc, outcome, reason in outcomes:
        if outcome != "denied":
            continue
        row = by_id.get(tc.id)
        if row is None:
            messages.append(denials.get(tc.id) or _recovery_denial(tc, reason))
        elif row.outcome == "started" and tc.id not in (ctx.pause_state.approval_gates or {}):
            # Consumption changes phases only for gated calls. Resolve a denied
            # held intent first so a second crash cannot revive its old approval.
            recorder._started_calls.add(tc.id)
            await _observe_recovery_call(ctx, run_id, tc, row, "resolve")
    return messages


async def _consume_recovery_submission(
    store: StateStore, run_id: str, ctx: _ResumeContext, lookups: Any
) -> tuple[dict[str, Any], Any]:
    from dendrux.runtime.submit import _submission_payload

    recorder = cast("PersistenceRecorder", ctx.recorder)
    descriptor = recorder._loop_state.get("submission")
    if descriptor:
        raw = await store.get_pause_state(run_id) or {}
        payload = _submission_payload(raw)
        injected = _recovery_input(payload, descriptor["kind"])
        if descriptor["kind"] == "approval":
            injected = await _recovery_approval(
                store, run_id, ctx, lookups, descriptor, payload, PauseState.from_dict(raw)
            )
        injected = [await project_message(recorder, message) for message in injected]
        await recorder.consume_resume(injected)
        ctx.history.extend(injected)
    return recorder._loop_state, descriptor


async def _recovery_barrier(
    store: StateStore,
    run_id: str,
    pause: PauseState,
    history: list[Message],
    actions: Any,
    index: int,
    calls: dict[str, ToolCall],
    held: list[ToolCall],
    rows: dict[str, Any],
    gates: dict[str, Any],
) -> RunResult | None:
    from dendrux.runtime.runner import _resume_barrier

    status = await _resume_barrier(store, run_id)
    if status is None:
        return None
    remaining = held + [
        calls[action.call_id] for action in actions[index:] if action.call_id is not None
    ]
    started_gates = {
        tc.id: recovery_gate()
        for tc in remaining
        if tc.id in rows and rows[tc.id].outcome == "started"
    }
    paused = replace(
        pause,
        history=history,
        pending_tool_calls=remaining,
        approval_gates={**(pause.approval_gates or {}), **gates, **started_gates} or None,
        pause_reason=PauseReason.REQUESTED.value,
        # Started calls retain their durable decisions when the pause is saved;
        # their recovery gates still select resolution before the evaluated path.
        pending_decisions={
            call_id: decision
            for call_id, decision in (pause.pending_decisions or {}).items()
            if call_id in started_gates
        }
        or None,
        pause_id=None,
    )
    return RunResult(
        run_id=run_id,
        status=status,
        steps=pause.steps,
        iteration_count=pause.iteration,
        usage=pause.usage,
        meta={"pause_state": paused} if status == RunStatus.PAUSED else {},
    )


async def _hold_recovery_call(
    ctx: _ResumeContext,
    run_id: str,
    tc: ToolCall,
    action: Any,
    held: list[ToolCall],
    gates: dict[str, Any],
) -> bool:
    if action.kind not in ("approval_replay", "wait_client"):
        return False
    held.append(tc)
    if action.kind != "wait_client":
        gates[tc.id] = recovery_gate()
        await ctx.recorder.on_governance_event(
            run_id,
            GovernanceEventType.APPROVAL_REQUESTED.value,
            ctx.pause_state.iteration,
            {"call_id": tc.id, "tool_name": tc.name, **gates[tc.id]},
            correlation_id=tc.id,
        )
    return True


def _saved_recovery_payload(result: Any) -> str:
    if isinstance(result, dict) and set(result) == {"raw"} and isinstance(result["raw"], str):
        return result["raw"]
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _recovery_tool_result(tc: ToolCall, row: Any, kind: str) -> ToolResult:
    if kind == "observe" and row is not None:
        return ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=_saved_recovery_payload(row.result),
            success=row.success,
            error=row.error_message,
        )
    unknown = row is not None and row.outcome == "started"
    code = "outcome_unknown" if unknown else "rejected"
    return ToolResult(
        name=tc.name,
        call_id=tc.id,
        payload=json.dumps({"code": code}, separators=(",", ":")),
        success=False,
        error=code,
        meta={"code": code} if unknown else {},
    )


async def _observe_recovery_call(
    ctx: _ResumeContext, run_id: str, tc: ToolCall, row: Any, kind: str
) -> bool:
    from dendrux.types import _message_from_dict, tool_result_message

    recorder = cast("PersistenceRecorder", ctx.recorder)
    if kind == "observe" and any(m.role == Role.TOOL and m.call_id == tc.id for m in ctx.history):
        await recorder.resolve_observation(tc.id, None)
        return True
    if kind not in ("observe", "resolve"):
        return False
    if kind == "observe" and row is not None and row.observation is not None:
        message = _message_from_dict(row.observation)
        await recorder.resolve_observation(tc.id, message)
        ctx.history.append(message)
        return True
    tr = _recovery_tool_result(tc, row, kind)
    if kind == "observe":
        message = await project_message(recorder, tool_result_message(tr))
        await recorder.resolve_observation(tc.id, message)
    else:
        await record_tool(recorder, run_id, tc, tr, ctx.pause_state.iteration)
        message = await record_message(
            recorder, run_id, tool_result_message(tr), ctx.pause_state.iteration
        )
    ctx.history.append(message)
    return True


async def _dispatch_recovery_call(
    agent: Agent,
    ctx: _ResumeContext,
    run_id: str,
    tc: ToolCall,
    kind: str,
    phase: str,
    lookups: Any,
    counts: dict[str, int],
    image_runtime: Any,
    tenant_id: str | None,
    on_work: Callable[[], None] | None,
    *,
    prepare_only: bool = False,
    anticipated_skills: set[str] | None = None,
) -> Any:
    from dendrux.loops._policy_context import dispatch_policy
    from dendrux.loops.react import _prepare_tool_calls, _ToolCallOutcome
    from dendrux.runtime.runner import _mark_work

    recorder = cast("PersistenceRecorder", ctx.recorder)
    if kind == "replay":
        recorder._replay_calls.add(tc.id)
    policy = dispatch_policy(
        name=tc.name,
        phase=phase,
        kind="" if kind == "approval" else kind,
        require_approval=agent.require_approval,
        before_tool=agent.before_tool,
    )
    _mark_work(on_work)
    batch = _recovery_batch(ctx, run_id)
    ready, gates = await _prepare_tool_calls(
        [tc],
        batch=batch,
        call_counts=counts,
        lookups=lookups,
        deny=agent.deny,
        tenant_id=tenant_id,
        guardrail_engine=recorder.guardrail_engine,
        image_runtime=image_runtime,
        anticipated_skills=anticipated_skills,
        recorded_gates=ctx.pause_state.approval_gates
        if kind == "approval" and phase != "approved"
        else None,
        fallback_approval=frozenset({tc.name})
        if kind == "approval" and phase != "approved"
        else None,
        **policy,
    )
    # Direct policy-characterization callers still execute one prepared call;
    # batch recovery always prepares the full batch before reaching dispatch.
    if gates or prepare_only:
        return _ToolCallOutcome(batch.all_results, ready, bool(gates), gates or None)
    return await _execute_recovery_calls(agent, ready, lookups, counts, batch, image_runtime)


def _recovery_batch(ctx: _ResumeContext, run_id: str) -> Any:
    from dendrux.loops.react import _ToolBatch

    return _ToolBatch(
        strategy=ctx.resolved_strategy,
        recorder=ctx.recorder,
        notifier=ctx.notifier,
        run_id=run_id,
        iteration=ctx.pause_state.iteration,
        history=ctx.history,
        warnings=[],
    )


async def _execute_recovery_calls(
    agent: Agent,
    calls: list[ToolCall],
    lookups: Any,
    counts: dict[str, int],
    batch: Any,
    image_runtime: Any,
) -> Any:
    from dendrux.loops.react import (
        _dispatch_server_calls,
        _record_executed,
        _split_targets,
        _ToolCallOutcome,
    )

    server, client = _split_targets(calls, lookups)
    executed = await _dispatch_server_calls(
        server, lookups, batch, agent.tool_result_transform, image_runtime
    )
    await _record_executed(executed, counts, batch)
    return _ToolCallOutcome(batch.all_results, client)


def _hold_started_outcome(row: Any, outcome: Any, gates: dict[str, Any]) -> None:
    if outcome.pending_approval and row is not None and row.outcome == "started":
        for call in outcome.pending_calls:
            gates[call.id] = recovery_gate()


async def _recovery_pause_result(
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    state: dict[str, Any],
    descriptor: Any,
    held: list[ToolCall],
    gates: dict[str, Any],
) -> RunResult | None:
    if not held and not (state.get("phase") == "awaiting_input" and not descriptor):
        return None
    pause = replace(
        ctx.pause_state,
        history=ctx.history,
        pending_tool_calls=held,
        approval_gates=gates or None,
        pause_id=None,
        pending_decisions=None,
    )
    status = (
        RunStatus.WAITING_APPROVAL
        if gates
        else RunStatus.WAITING_CLIENT_TOOL
        if held
        else RunStatus.WAITING_HUMAN_INPUT
    )
    answer = None
    if status == RunStatus.WAITING_HUMAN_INPUT:
        trace = await store.get_trace(run_id, state["clarification_ref"])
        answer = json.loads(trace.content)["question"] if trace else ""
    return RunResult(
        run_id=run_id,
        status=status,
        answer=answer,
        steps=pause.steps,
        iteration_count=pause.iteration,
        usage=pause.usage,
        meta={"pause_state": pause},
    )


async def _dispatch_prepared_batch(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    actions: Any,
    held: list[ToolCall],
    gates: dict[str, Any],
    rows: dict[str, Any],
    lookups: Any,
    counts: dict[str, int],
    image_runtime: Any,
) -> tuple[list[ToolCall], RunResult | None]:
    if gates:
        return held, None
    remaining = {tc.id: tc for tc in held}
    dispatches = [a for a in actions if a.call_id in remaining]
    held = []
    for index, action in enumerate(dispatches):
        if action.call_id is None:
            raise PersistenceError(code="recovery_submission_missing")
        if action.kind == "wait_client":
            held.append(remaining[action.call_id])
            continue
        barrier = await _recovery_barrier(
            store,
            run_id,
            ctx.pause_state,
            ctx.history,
            dispatches,
            index,
            remaining,
            held,
            rows,
            gates,
        )
        if barrier is not None:
            return held, barrier
        outcome = await _execute_recovery_calls(
            agent,
            [remaining[action.call_id]],
            lookups,
            counts,
            _recovery_batch(ctx, run_id),
            image_runtime,
        )
        held.extend(outcome.pending_calls)
    return held, None


async def _resolve_saved_call(
    ctx: _ResumeContext,
    run_id: str,
    tc: ToolCall,
    action: Any,
    row: Any,
    held: list[ToolCall],
    gates: dict[str, Any],
) -> bool:
    if await _hold_recovery_call(ctx, run_id, tc, action, held, gates):
        return True
    if row is not None and row.outcome == "started":
        cast("PersistenceRecorder", ctx.recorder)._started_calls.add(tc.id)
    return await _observe_recovery_call(ctx, run_id, tc, row, action.kind)


async def resolve(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    *,
    image_runtime: Any,
    tenant_id: str | None,
    on_work: Callable[[], None] | None = None,
) -> RunResult | None:
    recorder = cast("PersistenceRecorder", ctx.recorder)
    pause = ctx.pause_state
    lookups = await agent.get_tool_lookups(loop=ctx.resolved_loop)
    state, descriptor = await _consume_recovery_submission(store, run_id, ctx, lookups)
    pending = state.get("pending", [])
    rows = (
        await store.get_tool_calls(run_id, iteration_index=pause.iteration, limit=MAX_PAGE)
        if pending
        else []
    )
    by_id = {row.tool_call_id: row for row in rows}
    calls = {tc.id: tc for tc in pause.pending_tool_calls}
    actions = resolve_pending(
        state, rows, current_idempotent={name: td.idempotent for name, td in lookups.defs.items()}
    )
    recorder.configure_tools(list(lookups.defs.values()))
    counts = recorder.execution_counts() or {}
    held: list[ToolCall] = []
    gates: dict[str, Any] = {}
    phases = {p["call_id"]: p["phase"] for p in pending}
    reserved_counts = counts.copy()
    anticipated_skills: set[str] = set()
    for index, action in enumerate(actions):
        barrier = await _recovery_barrier(
            store, run_id, pause, ctx.history, actions, index, calls, held, by_id, gates
        )
        if barrier is not None:
            return barrier
        if action.call_id is None:
            raise PersistenceError(code="recovery_submission_missing")
        tc = calls[action.call_id]
        row = by_id.get(tc.id)
        if await _resolve_saved_call(ctx, run_id, tc, action, row, held, gates):
            continue
        outcome = await _dispatch_recovery_call(
            agent,
            ctx,
            run_id,
            tc,
            action.kind,
            phases[tc.id],
            lookups,
            reserved_counts,
            image_runtime,
            tenant_id,
            on_work,
            prepare_only=True,
            anticipated_skills=anticipated_skills,
        )
        held.extend(outcome.pending_calls)
        gates.update(outcome.approval_gates or {})
        _hold_started_outcome(row, outcome, gates)
        for ready in outcome.pending_calls:
            reserved_counts[ready.name] = reserved_counts.get(ready.name, 0) + 1
    held, barrier = await _dispatch_prepared_batch(
        agent, store, run_id, ctx, actions, held, gates, by_id, lookups, counts, image_runtime
    )
    if barrier is not None:
        return barrier
    order_observations(recorder, ctx.history)
    return await _recovery_pause_result(store, run_id, ctx, state, descriptor, held, gates)
