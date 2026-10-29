"""Explicit recovery of a fenced ReAct checkpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from dendrux import _codes
from dendrux.errors import PersistenceError, RunNotFoundError, RunNotReclaimableError
from dendrux.loops._helpers import order_observations, project_message, record_message, record_tool
from dendrux.runtime.persistence import PersistenceRecorder, resolve_pending
from dendrux.runtime.state import claimed_store
from dendrux.types import (
    AgentStep,
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
    from dendrux.runtime.runner import _resume_core
    from dendrux.runtime.submit import governance_snapshot
    from dendrux.skills._session import SkillSession

    if not capabilities(store).recovery:
        _codes.log_code(logger, _codes.STORE_CAPABILITY, run_id=run_id)
        raise NotImplementedError(
            f"StateStore {type(store).__name__} does not declare the 'recovery' capability"
        )
    session = SkillSession.prepare(agent)
    session.run_id = run_id
    try:
        claim = await store.reclaim_run(
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
        try:
            pause = await rebuild_pause(
                agent, owned, run_id, claim.loop_state or {}, history_records=history_records
            )
        except (RunNotReclaimableError, PersistenceError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, PersistenceError) and exc.code not in (
                "recovery_call_missing",
                "recovery_state_missing",
            ):
                raise
            reason = (
                exc.reason
                if isinstance(exc, RunNotReclaimableError)
                else exc.code
                if isinstance(exc, PersistenceError)
                else "recovery_state_invalid"
            )
            await owned.finalize_with_event(
                run_id,
                status="error",
                allowed_current_statuses=["running"],
                event_type="run.error",
                event_data={"failure_reason": reason},
                failure_reason=reason,
                error=reason,
            )
            _codes.log_code(logger, _codes.REGISTRY[reason], run_id=run_id)
            raise RunNotReclaimableError(run_id, reason) from exc
        result = await _resume_core(
            run_id,
            state_store=owned,
            agent=agent,
            provider=agent._require_provider(),
            expected_status="running",
            extra_notifier=notifier,
            _skip_claim=True,
            _reclaim_pause=pause,
            _reclaim_history=history_records,
        )
        if result.status in (
            RunStatus.SUCCESS,
            RunStatus.ERROR,
            RunStatus.CANCELLED,
            RunStatus.MAX_ITERATIONS,
        ):
            _codes.log_code(logger, _codes.RUN_FINALIZED, run_id=run_id)
        return result


async def rebuild_pause(
    agent: Agent,
    store: StateStore,
    run_id: str,
    state: dict[str, Any],
    *,
    history_records: list[Any] | None = None,
) -> PauseState:
    from dendrux.runtime.replay import rebuild_history
    from dendrux.runtime.state import TraceRecord
    from dendrux.skills._session import current
    from dendrux.types import ImageViewState

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
        raise PersistenceError(code="recovery_call_missing")
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
    pause.iteration = state.get("iteration", 0)
    pause.trace_order_offset = state.get("trace_order_offset", 0)
    pause.usage = _usage_from_dict(state.get("usage", {}))
    pause.provider_kwargs = dict(state.get("provider_kwargs", {}))
    for kind, ref in (
        ("image_view", state.get("image_view_ref")),
        ("skills_session", state.get("skills", {}).get("session_ref")),
        ("clarification", state.get("clarification_ref")),
    ):
        if ref is None:
            continue
        trace = await store.get_trace(run_id, ref)
        if trace is None:
            raise PersistenceError(code="recovery_state_missing")
        payload = json.loads(trace.content)
        if kind == "image_view":
            pause.image_view = ImageViewState.from_dict(payload)
        elif kind == "skills_session":
            session = current(agent)
            if session is not None:
                session.restore_snapshot(payload)
        else:
            pause.pause_message = payload.get("pause_message")
    return pause


async def needs_resolution(store: StateStore, run_id: str, pause: PauseState) -> bool:
    if pause.schema_version != 2:
        return False
    if any(g.get("source") == "recovery" for g in (pause.approval_gates or {}).values()):
        return True
    if not pause.pending_tool_calls:
        return False
    calls = {tc.id for tc in pause.pending_tool_calls}
    return any(
        row.tool_call_id in calls and row.outcome == "started"
        for row in await store.get_tool_calls(run_id, iteration_index=pause.iteration, limit=1000)
    )


def recovery_gate() -> dict[str, str | None]:
    return {"source": "recovery", "reason": "replay_after_crash"}


async def resolve(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _ResumeContext,
    *,
    image_runtime: Any,
) -> RunResult | None:
    from dendrux.loops.react import _process_tool_calls
    from dendrux.runtime.submit import _submission_payload, deserialize_tool_results
    from dendrux.types import tool_result_message

    recorder = cast("PersistenceRecorder", ctx.recorder)
    pause = ctx.pause_state
    state = recorder._loop_state
    descriptor = state.get("submission")
    lookups = await agent.get_tool_lookups(loop=ctx.resolved_loop)
    if descriptor:
        raw = await store.get_pause_state(run_id) or {}
        payload = _submission_payload(raw)
        injected = []
        if descriptor["kind"] == "tool_results":
            injected = [
                tool_result_message(r)
                for r in deserialize_tool_results(payload["submitted_tool_results"])
            ]
        elif descriptor["kind"] == "input":
            from dendrux.runtime.runner import _stored_parts

            parts = _stored_parts(payload.get("submitted_user_parts"))
            injected = [
                Message.with_parts(Role.USER, parts)
                if parts
                else Message(Role.USER, payload["submitted_user_input"])
            ]
        if descriptor["kind"] == "approval":
            from dendrux.runtime.runner import (
                _approval_outcomes,
                _decided_payload,
                _record_approval_decided,
            )
            from dendrux.runtime.submit import deserialize_decisions
            from dendrux.types import ApprovalDecision

            decisions = (
                dict(deserialize_decisions(payload["submitted_decisions"]))
                if "submitted_decisions" in payload
                else None
            )
            denials = deserialize_tool_results(payload.get("submitted_tool_results", []))
            if decisions is None and payload.get("approved") is False:
                decisions = {
                    tr.call_id: ApprovalDecision(approved=False, reason=tr.error) for tr in denials
                }
            outcomes = _approval_outcomes(pause, lookups.target, decisions)
            await _record_approval_decided(
                recorder,
                ctx.notifier,
                run_id,
                pause.iteration,
                _decided_payload(run_id, outcomes),
                event_id=descriptor["submission_id"],
            )
            # Non-started denials can be consumed with their observation. Started
            # effects must instead resolve unknown through atomic completion.
            rows = await store.get_tool_calls(run_id, iteration_index=pause.iteration, limit=1000)
            started = {r.tool_call_id for r in rows if r.outcome == "started"}
            injected = [
                tool_result_message(
                    ToolResult(
                        name=tc.name,
                        call_id=tc.id,
                        payload=json.dumps(reason),
                        success=False,
                        error=reason,
                    )
                )
                for tc, outcome, reason in outcomes
                if outcome == "denied" and tc.id not in started
            ]
        injected = [await project_message(recorder, m) for m in injected]
        await recorder.consume_resume(injected)
        ctx.history.extend(injected)
        state = recorder._loop_state
    pending = state.get("pending", [])
    rows = (
        await store.get_tool_calls(run_id, iteration_index=pause.iteration, limit=1000)
        if pending
        else []
    )
    by_id = {r.tool_call_id: r for r in rows}
    calls = {tc.id: tc for tc in pause.pending_tool_calls}
    actions = resolve_pending(
        state, rows, current_idempotent={name: td.idempotent for name, td in lookups.defs.items()}
    )
    recorder.configure_tools(list(lookups.defs.values()))
    counts = recorder.execution_counts() or {}
    held: list[ToolCall] = []
    gates: dict[str, dict[str, str | None]] = {}
    phases = {p["call_id"]: p["phase"] for p in pending}
    for index, action in enumerate(actions):
        from dendrux.runtime.runner import _resume_barrier

        barrier = await _resume_barrier(store, run_id)
        if barrier is not None:
            remaining = held + [calls[a.call_id] for a in actions[index:] if a.call_id is not None]
            for tc in remaining:
                row = by_id.get(tc.id)
                if row is not None and row.outcome == "started":
                    gates[tc.id] = recovery_gate()
            paused = replace(
                pause,
                history=ctx.history,
                pending_tool_calls=remaining,
                approval_gates={**(pause.approval_gates or {}), **gates} or None,
                pause_reason="requested",
                pause_id=None,
            )
            return RunResult(
                run_id=run_id,
                status=barrier,
                steps=pause.steps,
                iteration_count=pause.iteration,
                usage=pause.usage,
                meta={"pause_state": paused} if barrier == RunStatus.PAUSED else {},
            )
        if action.call_id is None:
            raise PersistenceError(code="recovery_submission_missing")
        tc = calls[action.call_id]
        row = by_id.get(tc.id)
        if action.kind in ("approval_replay", "wait_client"):
            held.append(tc)
            if action.kind != "wait_client":
                gates[tc.id] = recovery_gate()
                await recorder.on_governance_event(
                    run_id,
                    "approval.requested",
                    pause.iteration,
                    {"call_id": tc.id, "tool_name": tc.name, **gates[tc.id]},
                    correlation_id=tc.id,
                )
            continue
        if row is not None and row.outcome == "started":
            recorder._started_calls.add(tc.id)
        if action.kind == "observe" and any(
            m.role == Role.TOOL and m.call_id == tc.id for m in ctx.history
        ):
            await recorder.resolve_observation(tc.id, None)
            continue
        if action.kind in ("observe", "resolve"):
            unknown = row is not None and row.outcome == "started"
            if action.kind == "observe" and row is not None and row.observation is not None:
                from dendrux.types import _message_from_dict

                message = _message_from_dict(row.observation)
                await recorder.resolve_observation(tc.id, message)
                ctx.history.append(message)
                continue
            tr = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=(
                    row.result["raw"]
                    if isinstance(row.result, dict)
                    and set(row.result) == {"raw"}
                    and isinstance(row.result["raw"], str)
                    else json.dumps(row.result, ensure_ascii=False, separators=(",", ":"))
                )
                if action.kind == "observe" and row is not None
                else '{"code":"outcome_unknown"}'
                if unknown
                else '{"code":"rejected"}',
                success=row.success if action.kind == "observe" and row is not None else False,
                error=row.error_message
                if action.kind == "observe" and row is not None
                else "outcome_unknown"
                if unknown
                else "rejected",
                meta={"code": "outcome_unknown"} if unknown else {},
            )
            if action.kind == "observe":
                message = await project_message(recorder, tool_result_message(tr))
                await recorder.resolve_observation(tc.id, message)
            else:
                await record_tool(recorder, run_id, tc, tr, pause.iteration)
                message = await record_message(
                    recorder, run_id, tool_result_message(tr), pause.iteration
                )
            ctx.history.append(message)
            continue
        if action.kind == "replay":
            recorder._replay_calls.add(tc.id)
        # Approved decisions skip only the gate. Validation, deny and call limits
        # still run through the same dispatcher as a live assistant batch.
        outcome = await _process_tool_calls(
            step=AgentStep(reasoning=None, action=tc),
            call_counts=counts,
            lookups=lookups,
            strategy=ctx.resolved_strategy,
            recorder=recorder,
            notifier=ctx.notifier,
            run_id=run_id,
            iteration=pause.iteration,
            history=ctx.history,
            warnings=[],
            deny=agent.deny,
            require_approval=(
                frozenset()
                if phases[tc.id] == "approved"
                else agent.require_approval | {tc.name}
                if action.kind == "approval"
                else agent.require_approval
            ),
            before_tool=None if phases[tc.id] == "approved" else agent.before_tool,
            guardrail_engine=recorder.guardrail_engine,
            tool_result_transform=agent.tool_result_transform,
            approved_skills=phases[tc.id] == "approved",
            image_runtime=image_runtime,
        )
        held.extend(outcome.pending_calls)
        gates.update(outcome.approval_gates or {})
        if row is not None and row.outcome == "started":
            for held_call in outcome.pending_calls:
                gates[held_call.id] = recovery_gate()
    order_observations(recorder, ctx.history)
    if held or state.get("phase") == "awaiting_input" and not descriptor:
        pause = replace(
            pause,
            history=ctx.history,
            pending_tool_calls=held,
            approval_gates=gates or None,
            pause_id=None,
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
    return None
