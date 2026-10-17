"""ReAct loop — think, act, observe, repeat.

The core agent execution loop. Orchestrates the cycle of:
  1. Strategy builds messages
  2. Provider calls the LLM
  3. Strategy parses the response into an AgentStep
  4. If ToolCall → execute → format result → append → repeat
  5. If Finish → return RunResult
  6. If max_iterations → return RunResult with MAX_ITERATIONS

The loop never touches provider-specific APIs or prompt formatting.
It operates entirely on Dendrux's universal types.

Notifier hooks fire at three kinds of points:
  - After each history.append() → on_message_appended
  - After provider.complete() → on_llm_call_completed
  - After _execute_tool() → on_tool_completed
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._budget import BudgetBreach, BudgetCarry, BudgetTracker, breach_payload
from dendrux.loops._helpers import (
    StreamInterruptedError,
    build_cache_key_prefix,
    guardrail_meta,
    interruptible,
    notify_governance,
    notify_llm,
    notify_llm_failed,
    notify_llm_started,
    notify_message,
    notify_tool,
    notify_tool_started,
    price_response,
    record_governance,
    record_llm,
    record_llm_failed,
    record_llm_started,
    record_message,
    record_tool,
    record_tool_started,
)
from dendrux.loops.base import Loop
from dendrux.tool import DEFAULT_TOOL_TIMEOUT
from dendrux.tools._lookups import ToolLookups, build_tool_lookups
from dendrux.tools._validate import check_tool_call, check_tool_json
from dendrux.types import (
    REASONING_MODEL_META_KEY,
    AgentStep,
    Clarification,
    Finish,
    GovernanceEventType,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    StreamEventType,
    ToolCall,
    ToolCallContext,
    ToolDecision,
    ToolResult,
    ToolResultContext,
    ToolTarget,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import BeforeToolHook, ToolResultTransform

logger = logging.getLogger(__name__)


_record_message = record_message
_record_llm = record_llm
_record_llm_started = record_llm_started
_record_llm_failed = record_llm_failed
_record_tool = record_tool
_record_tool_started = record_tool_started
_record_governance = record_governance
_notify_message = notify_message
_notify_llm = notify_llm
_notify_llm_started = notify_llm_started
_notify_llm_failed = notify_llm_failed
_notify_tool = notify_tool
_notify_tool_started = notify_tool_started
_notify_governance = notify_governance


class _ToolCallOutcome(NamedTuple):
    """Result of processing tool calls in a single iteration."""

    # All (tool_call, tool_result) pairs — limits + executed, in order
    all_results: list[tuple[ToolCall, ToolResult]]
    # Non-server tools that need pause (empty if all were server tools)
    pending_calls: list[ToolCall]
    # True if pending_calls are awaiting approval (vs client tool pause)
    pending_approval: bool = False
    # Approval-gated call ids → {"source", "reason"} (approval pauses only)
    approval_gates: dict[str, dict[str, str | None]] | None = None


def _accumulate_usage(total: UsageStats, step_usage: UsageStats) -> None:
    """Add per-call usage to running total. Mutates total in place.

    Cache fields treat None as 0 in summation so providers that don't
    report them don't poison the rollup. Once any step reports a value,
    the running total carries it.

    Cost is the opposite: a run total only means something if every step
    was priced, so one unpriced step (``cost_usd is None``) makes the total
    ``None`` and keeps it there. ``cost_unknown`` preserves this even if
    the unpriced call reported no tokens. Token counts also identify
    unknown totals restored from older pause snapshots without that flag.
    """
    cost_unknown = total.cost_unknown or (
        total.cost_usd is None
        and any(
            (
                total.input_tokens,
                total.output_tokens,
                total.total_tokens,
                total.cache_read_input_tokens,
                total.cache_creation_input_tokens,
            )
        )
    )
    total.input_tokens += step_usage.input_tokens
    total.output_tokens += step_usage.output_tokens
    total.total_tokens += step_usage.total_tokens
    total.usage_reported = total.usage_reported and step_usage.usage_reported
    total.cost_unknown = cost_unknown or step_usage.cost_usd is None
    if cost_unknown or step_usage.cost_usd is None:
        total.cost_usd = None
        total.cost_source = None
    else:
        if total.cost_usd is None:
            total.cost_usd = 0.0
            total.cost_source = step_usage.cost_source
        elif total.cost_source != step_usage.cost_source:
            total.cost_source = "mixed"
        total.cost_usd += step_usage.cost_usd
    if step_usage.cache_read_input_tokens is not None:
        total.cache_read_input_tokens = (
            total.cache_read_input_tokens or 0
        ) + step_usage.cache_read_input_tokens
    if step_usage.cache_creation_input_tokens is not None:
        total.cache_creation_input_tokens = (
            total.cache_creation_input_tokens or 0
        ) + step_usage.cache_creation_input_tokens
    if step_usage.reasoning_tokens is not None:
        total.reasoning_tokens = (total.reasoning_tokens or 0) + step_usage.reasoning_tokens


async def _append_assistant(
    response: LLMResponse,
    history: list[Message],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    warnings: list[str],
    *,
    phase: str,
) -> Message:
    """Create assistant message, append to history, record then notify.

    ``phase`` lands in ``Message.meta`` and from there in the persisted trace.
    """
    # Tag replay blocks with their producing model so providers never send
    # them to a different model (per-turn ``model=`` / retry on another model).
    meta: dict[str, Any] = {"phase": phase}
    if response.reasoning_blocks and response.model:
        meta[REASONING_MODEL_META_KEY] = response.model
    msg = Message(
        role=Role.ASSISTANT,
        content=response.text or "",
        tool_calls=response.tool_calls,
        meta=meta,
        reasoning=response.reasoning,
        reasoning_blocks=response.reasoning_blocks,
    )
    history.append(msg)
    await _record_message(recorder, run_id, msg, iteration)
    await _notify_message(notifier, run_id, msg, iteration, warnings)
    return msg


def _commentary(steps: list[AgentStep]) -> list[str]:
    """Non-empty assistant text from steps that ended in tool calls, in order."""
    return [
        s.raw_response
        for s in steps
        if isinstance(s.action, ToolCall) and s.raw_response and s.raw_response.strip()
    ]


def _snapshot_usage(usage: UsageStats) -> UsageStats:
    """Create an independent copy of usage stats for pause state."""
    return UsageStats(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cost_source=usage.cost_source,
        usage_reported=usage.usage_reported,
        cost_unknown=usage.cost_unknown,
    )


def _build_pause(
    *,
    agent_name: str,
    pending_calls: list[ToolCall],
    target_lookup: dict[str, ToolTarget],
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    approval_gates: dict[str, dict[str, str | None]] | None = None,
    budget: BudgetTracker | None = None,
) -> PauseState:
    """Build PauseState for client tool, approval or clarification pause."""
    pending_targets = (
        {tc.id: target_lookup.get(tc.name, ToolTarget.SERVER).value for tc in pending_calls}
        if pending_calls
        else {}
    )

    return PauseState(
        agent_name=agent_name,
        pending_tool_calls=pending_calls,
        pending_targets=pending_targets,
        history=list(history),
        steps=list(steps),
        iteration=iteration,
        trace_order_offset=len(history),
        usage=_snapshot_usage(usage),
        approval_gates=approval_gates,
        **(budget.pause_fields() if budget is not None else {}),
    )


INBOX_NOTE_PREFIX = "NOTE FROM USER:\n"


class _LegacyControl(NamedTuple):
    """Checkpoint flags of a pre-0.3 store: cancel only, no pause, no inbox."""

    cancel_requested: bool
    pause_requested: bool = False
    has_pending_notes: bool = False


async def _read_control(state_store: StateStore, run_id: str) -> Any:
    """``get_run_control`` when the store has it, else ``is_cancel_requested``."""
    getter = getattr(state_store, "get_run_control", None)
    if getter is not None:
        return await getter(run_id)
    return _LegacyControl(cancel_requested=bool(await state_store.is_cancel_requested(run_id)))


def _checkpoint_pause_result(
    *,
    agent_name: str,
    run_id: str,
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    meta: dict[str, Any],
    budget: BudgetTracker | None = None,
    breaches: list[BudgetBreach] | None = None,
) -> RunResult:
    """PAUSED result for a pause observed at the top of ``iteration``.

    Nothing of ``iteration`` has run, so the pause records ``iteration - 1``
    and resume continues at ``iteration`` with the full history. With
    ``breaches`` it is a budget pause (``pause_reason="budget"``), else a
    requested one.
    """
    exceeded = breach_payload(breaches) if breaches else None
    pause = PauseState(
        agent_name=agent_name,
        pending_tool_calls=[],
        history=list(history),
        steps=list(steps),
        iteration=iteration - 1,
        trace_order_offset=len(history),
        usage=_snapshot_usage(usage),
        pause_reason="budget" if exceeded else "requested",
        budget_exceeded=exceeded,
        **(budget.pause_fields() if budget is not None else {}),
    )
    extra = {"budget_exceeded": exceeded} if exceeded else {}
    return RunResult(
        run_id=run_id,
        status=RunStatus.PAUSED,
        steps=steps,
        commentary=_commentary(steps),
        iteration_count=iteration - 1,
        usage=usage,
        meta={"pause_state": pause, **meta, **extra},
    )


def _budget_cancel_result(
    *,
    run_id: str,
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    meta: dict[str, Any],
    breaches: list[BudgetBreach],
) -> RunResult:
    """CANCELLED result for a budget cancel at the top of ``iteration``.

    ``meta["budget_exceeded"]`` tells the runner to persist
    ``failure_reason="budget_exceeded"`` and a distinct event reason.
    """
    return RunResult(
        run_id=run_id,
        status=RunStatus.CANCELLED,
        steps=steps,
        commentary=_commentary(steps),
        iteration_count=iteration - 1,
        usage=usage,
        meta={**meta, "budget_exceeded": breach_payload(breaches)},
    )


async def _budget_stop(
    budget: BudgetTracker,
    *,
    agent_name: str,
    run_id: str,
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    meta: dict[str, Any],
) -> RunResult | None:
    """Checkpoint budget enforcement: a PAUSED/CANCELLED result, or None to go on."""
    breaches = await budget.checkpoint(iteration)
    if not breaches:
        return None
    if budget.action == "cancel":
        return _budget_cancel_result(
            run_id=run_id,
            steps=steps,
            iteration=iteration,
            usage=usage,
            meta=meta,
            breaches=breaches,
        )
    return _checkpoint_pause_result(
        agent_name=agent_name,
        run_id=run_id,
        history=history,
        steps=steps,
        iteration=iteration,
        usage=usage,
        meta=meta,
        budget=budget,
        breaches=breaches,
    )


async def _deliver_inbox(
    state_store: StateStore,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    history: list[Message],
    warnings: list[str],
) -> None:
    """Append undelivered inbox notes to history as USER messages, then mark them.

    At-least-once: each message is appended and recorded before the notes
    are marked delivered, so a crash in between redelivers rather than
    loses. Only the runner that owns the run calls this (claims guarantee a
    single owner). Note text never enters the ``inbox.delivered`` payload.
    """
    notes = await state_store.get_notes(run_id, undelivered_only=True)
    if not notes:
        return
    for note in notes:
        msg = Message(
            role=Role.USER,
            content=f"{INBOX_NOTE_PREFIX}{note.text}",
            kind="inbox_note",
            source="run_inbox",
            meta={"note_id": note.id},
        )
        history.append(msg)
        await _record_message(recorder, run_id, msg, iteration)
        await _notify_message(notifier, run_id, msg, iteration, warnings)
    await state_store.mark_notes_delivered(run_id, [n.id for n in notes], iteration=iteration)
    for note in notes:
        data = {"note_id": note.id}
        await _record_governance(
            recorder,
            run_id,
            GovernanceEventType.INBOX_DELIVERED,
            iteration,
            data,
            correlation_id=note.id,
        )
        await _notify_governance(
            notifier,
            run_id,
            GovernanceEventType.INBOX_DELIVERED,
            iteration,
            data,
            correlation_id=note.id,
            warnings=warnings,
        )


class _LoopState(NamedTuple):
    """Mutable iteration state — initialized once, shared across iterations."""

    history: list[Message]
    call_counts: dict[str, int]
    steps: list[AgentStep]
    usage: UsageStats
    warnings: list[str]


async def _init_loop_state(
    *,
    user_input: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    initial_history: list[Message] | None,
    initial_steps: list[AgentStep] | None,
    initial_usage: UsageStats | None,
) -> _LoopState:
    """Initialize mutable loop state — shared between run() and run_stream()."""
    if initial_history is not None:
        history = list(initial_history)
    else:
        from dendrux.context_blocks import wire_user_input

        user_msg = Message(role=Role.USER, content=wire_user_input(user_input, [], None))
        history = [user_msg]
        await _record_message(recorder, run_id, user_msg, 0)
        await _notify_message(notifier, run_id, user_msg, 0)

    call_counts: dict[str, int] = {}
    for msg in history:
        if msg.role == Role.TOOL and msg.name is not None:
            call_counts[msg.name] = call_counts.get(msg.name, 0) + 1

    steps: list[AgentStep] = list(initial_steps) if initial_steps else []
    usage = UsageStats(
        input_tokens=initial_usage.input_tokens if initial_usage else 0,
        output_tokens=initial_usage.output_tokens if initial_usage else 0,
        total_tokens=initial_usage.total_tokens if initial_usage else 0,
        cost_usd=initial_usage.cost_usd if initial_usage else None,
        cache_read_input_tokens=(initial_usage.cache_read_input_tokens if initial_usage else None),
        cache_creation_input_tokens=(
            initial_usage.cache_creation_input_tokens if initial_usage else None
        ),
        reasoning_tokens=(initial_usage.reasoning_tokens if initial_usage else None),
        cost_source=(initial_usage.cost_source if initial_usage else None),
        usage_reported=(initial_usage.usage_reported if initial_usage else True),
        cost_unknown=(
            initial_usage.cost_unknown or initial_usage.cost_usd is None if initial_usage else False
        ),
    )

    return _LoopState(
        history=history,
        call_counts=call_counts,
        steps=steps,
        usage=usage,
        warnings=[],
    )


async def _process_tool_calls(
    *,
    step: AgentStep,
    call_counts: dict[str, int],
    lookups: ToolLookups,
    strategy: Strategy,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    history: list[Message],
    warnings: list[str],
    deny: frozenset[str] | None = None,
    require_approval: frozenset[str] | None = None,
    guardrail_engine: GuardrailEngine | None = None,
    tool_result_transform: ToolResultTransform | None = None,
    approved_skills: bool = False,
    before_tool: BeforeToolHook | None = None,
    tenant_id: str | None = None,
) -> _ToolCallOutcome:
    """Execute tool calls: enforce limits, run server tools, update history.

    Returns a _ToolCallOutcome with all results and any pending (non-server)
    calls. Both run() and run_stream() call this — the caller decides
    whether to yield events from the results.

    Side effects: mutates call_counts, appends to history, records then notifies.
    """
    from dendrux.skills._session import SkillTool

    session = next((fn.session for fn in lookups.fn.values() if isinstance(fn, SkillTool)), None)
    if session is not None:
        session.run_id = run_id

        async def skill_sink(event: str, data: dict[str, Any], call_id: str | None) -> None:
            await _record_governance(
                recorder, run_id, event, iteration, data, correlation_id=call_id
            )
            await _notify_governance(
                notifier, run_id, event, iteration, data, correlation_id=call_id, warnings=warnings
            )

        session.sink = skill_sink
    all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
    all_results: list[tuple[ToolCall, ToolResult]] = []
    deny_set = deny or frozenset()

    # --- Phase -1: Reject calls whose arguments were not valid JSON ---
    # Runs before deny/approval: such a call has no usable arguments, so
    # pausing it for approval would ask a human to approve "{}", and the
    # model needs the parse error (not a policy message) to repair the call.
    parsed_calls: list[ToolCall] = []
    for tc in all_calls:
        json_result = check_tool_json(tc)
        if (
            json_result is None
            and session is not None
            and isinstance(lookups.fn.get(tc.name), SkillTool)
        ):
            json_result = check_tool_call(tc, lookups.defs.get(tc.name))
            if json_result is not None and tc.name == "use_skill":
                await session.preflight(tc)
        if json_result is None:
            parsed_calls.append(tc)
            continue
        await _record_tool(recorder, run_id, tc, json_result, iteration)
        await _notify_tool(notifier, run_id, tc, json_result, iteration, warnings)
        result_msg = strategy.format_tool_result(json_result)
        history.append(result_msg)
        await _record_message(recorder, run_id, result_msg, iteration)
        await _notify_message(notifier, run_id, result_msg, iteration, warnings)
        all_results.append((tc, json_result))

    # --- Phase 0: Deny check (governance) ---
    non_denied: list[ToolCall] = []
    for tc in parsed_calls:
        if tc.name in deny_set:
            deny_msg = f"Tool '{tc.name}' is denied by policy."
            deny_result = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=json.dumps({"denied": deny_msg}),
                success=False,
                error=deny_msg,
            )
            # No _record_tool / _notify_tool — denied tools are not executions.
            # Only policy.denied event + synthetic message for the model.
            event_data = {
                "tool_name": tc.name,
                "call_id": tc.id,
                "reason": "denied_by_policy",
            }
            await _record_governance(
                recorder, run_id, "policy.denied", iteration, event_data, correlation_id=tc.id
            )
            await _notify_governance(
                notifier,
                run_id,
                "policy.denied",
                iteration,
                event_data,
                correlation_id=tc.id,
                warnings=warnings,
            )
            result_msg = strategy.format_tool_result(deny_result)
            history.append(result_msg)
            await _record_message(recorder, run_id, result_msg, iteration)
            await _notify_message(notifier, run_id, result_msg, iteration, warnings)
            all_results.append((tc, deny_result))
        else:
            non_denied.append(tc)

    # --- Phase 0.5: Deanonymize tool call params ---
    if guardrail_engine is not None:
        for i, tc in enumerate(non_denied):
            if tc.params:
                deaned = guardrail_engine.deanonymize(tc.params)
                if deaned != tc.params:
                    non_denied[i] = ToolCall(
                        name=tc.name,
                        params=deaned,
                        id=tc.id,
                        provider_tool_call_id=tc.provider_tool_call_id,
                    )

    # Preflight before approval. Earlier valid activation barriers in this batch
    # may satisfy reads; execution rechecks against actual successful activations.
    if session is not None:
        ready: list[ToolCall] = []
        anticipated: set[str] = set()
        for tc in non_denied:
            failure = (
                await session.preflight(
                    tc,
                    requested=not approved_skills,
                    anticipated=anticipated,
                    approved=approved_skills,
                )
                if isinstance(lookups.fn.get(tc.name), SkillTool)
                else None
            )
            if failure is None:
                ready.append(tc)
                if tc.name == "use_skill" and isinstance(lookups.fn.get(tc.name), SkillTool):
                    anticipated.add(tc.params["name"])
                continue
            result_msg = strategy.format_tool_result(failure)
            history.append(result_msg)
            await _record_tool(recorder, run_id, tc, failure, iteration)
            await _notify_tool(notifier, run_id, tc, failure, iteration, warnings)
            await _record_message(recorder, run_id, result_msg, iteration)
            await _notify_message(notifier, run_id, result_msg, iteration, warnings)
            all_results.append((tc, failure))
        non_denied = ready

    # --- Phase 0.75: before_tool policy hook ---
    approval_set = require_approval or frozenset()
    gates: dict[str, dict[str, str | None]] = {
        tc.id: {"source": "static", "reason": "requires_approval"}
        for tc in non_denied
        if tc.name in approval_set
    }
    if before_tool is not None:
        non_denied = await _apply_before_tool(
            before_tool,
            non_denied,
            gates,
            lookups=lookups,
            strategy=strategy,
            recorder=recorder,
            notifier=notifier,
            run_id=run_id,
            iteration=iteration,
            history=history,
            warnings=warnings,
            all_results=all_results,
            tenant_id=tenant_id,
        )

    # --- Phase 1: Approval check (governance) ---
    if gates:
        # Any tool in the batch needs approval → entire remaining batch pauses.
        for tc in non_denied:
            gate = gates.get(tc.id)
            if gate is not None:
                if gate["source"] == "hook":
                    event_data = {
                        "tool_name": tc.name,
                        "call_id": tc.id,
                        "reason": gate["reason"] or "requires_approval",
                        "source": "hook",
                    }
                else:
                    event_data = {
                        "tool_name": tc.name,
                        "call_id": tc.id,
                        "reason": "requires_approval",
                    }
                await _record_governance(
                    recorder,
                    run_id,
                    "approval.requested",
                    iteration,
                    event_data,
                    correlation_id=tc.id,
                )
                await _notify_governance(
                    notifier,
                    run_id,
                    "approval.requested",
                    iteration,
                    event_data,
                    correlation_id=tc.id,
                    warnings=warnings,
                )
        return _ToolCallOutcome(
            all_results=all_results,
            pending_calls=non_denied,
            pending_approval=True,
            approval_gates=gates,
        )

    # --- Phase 1.5: Validate params against each tool's schema ---
    valid_calls: list[ToolCall] = []
    for tc in non_denied:
        invalid_result = check_tool_call(tc, lookups.defs.get(tc.name))
        if invalid_result is None:
            valid_calls.append(tc)
            continue
        await _record_tool(recorder, run_id, tc, invalid_result, iteration)
        await _notify_tool(notifier, run_id, tc, invalid_result, iteration, warnings)
        result_msg = strategy.format_tool_result(invalid_result)
        history.append(result_msg)
        await _record_message(recorder, run_id, result_msg, iteration)
        await _notify_message(notifier, run_id, result_msg, iteration, warnings)
        all_results.append((tc, invalid_result))

    # --- Phase 1: Enforce max_calls_per_run ---
    allowed_calls: list[ToolCall] = []
    batch_counts: dict[str, int] = {}
    for tc in valid_calls:
        max_calls = lookups.max_calls.get(tc.name)
        current = call_counts.get(tc.name, 0) + batch_counts.get(tc.name, 0)
        if max_calls is not None and current >= max_calls:
            limit_msg = (
                f"Tool '{tc.name}' has reached its maximum of {max_calls} calls for this run."
            )
            limit_result = ToolResult(
                name=tc.name,
                call_id=tc.id,
                payload=json.dumps({"limit": limit_msg}),
                success=False,
                error=limit_msg,
            )
            await _record_tool(recorder, run_id, tc, limit_result, iteration)
            await _notify_tool(notifier, run_id, tc, limit_result, iteration, warnings)
            result_msg = strategy.format_tool_result(limit_result)
            history.append(result_msg)
            await _record_message(recorder, run_id, result_msg, iteration)
            await _notify_message(notifier, run_id, result_msg, iteration, warnings)
            all_results.append((tc, limit_result))
        else:
            batch_counts[tc.name] = batch_counts.get(tc.name, 0) + 1
            allowed_calls.append(tc)

    # --- Phase 2: Split into server (execute now) vs non-server (pause) ---
    server_calls: list[ToolCall] = []
    pending_calls: list[ToolCall] = []
    for tc in allowed_calls:
        target = lookups.target.get(tc.name, ToolTarget.SERVER)
        if target == ToolTarget.SERVER:
            server_calls.append(tc)
        else:
            pending_calls.append(tc)

    # --- Phase 3: Execute server tools ---
    exec_groups = _build_execution_groups(server_calls, lookups.parallel)
    executed: list[tuple[ToolCall, ToolResult]] = []
    for group, is_parallel in exec_groups:
        if is_parallel and len(group) > 1:
            pairs = await asyncio.gather(
                *[
                    _execute_record_notify(
                        tc,
                        lookups,
                        recorder,
                        notifier,
                        run_id,
                        iteration,
                        warnings,
                        tool_result_transform,
                    )
                    for tc in group
                ]
            )
            executed.extend(pairs)
        else:
            for tc in group:
                await _record_tool_started(recorder, run_id, tc, iteration)
                await _notify_tool_started(notifier, run_id, tc, iteration, warnings)
                tool_result = await _execute_tool(tc, lookups, tool_result_transform)
                await _record_tool(recorder, run_id, tc, tool_result, iteration)
                await _notify_tool(notifier, run_id, tc, tool_result, iteration, warnings)
                executed.append((tc, tool_result))

    # --- Phase 4: Append history in original order (deterministic for LLM) ---
    for tc, tool_result in executed:
        call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
        result_msg = strategy.format_tool_result(tool_result)
        history.append(result_msg)
        await _record_message(recorder, run_id, result_msg, iteration)
        await _notify_message(notifier, run_id, result_msg, iteration, warnings)
        all_results.append((tc, tool_result))

    return _ToolCallOutcome(all_results=all_results, pending_calls=pending_calls)


async def _hook_tenant_id(
    hook: BeforeToolHook | None, state_store: StateStore | None, run_id: str
) -> str | None:
    """The run's tenant for ``ToolCallContext`` (read once, only with a hook)."""
    if hook is None or state_store is None:
        return None
    record = await state_store.get_run(run_id)
    return record.tenant_id if record is not None else None


_HOOK_ERROR_REASON = "policy hook error"
_CLIENT_APPROVAL_REASON = "approval is not supported for client tools"


def _tool_source(name: str, lookups: ToolLookups) -> Literal["local", "mcp", "skill"]:
    from dendrux.skills._session import SkillTool

    if isinstance(lookups.fn.get(name), SkillTool):
        return "skill"
    if "mcp_tool_name" in lookups.meta.get(name, {}):
        return "mcp"
    return "local"


async def _call_before_tool(
    hook: BeforeToolHook,
    tc: ToolCall,
    lookups: ToolLookups,
    *,
    run_id: str,
    iteration: int,
    tenant_id: str | None,
) -> tuple[ToolDecision | None, str | None]:
    """Run the hook on deep-copied inputs. Returns (decision, error_type).

    Only ``Exception`` is caught: cancellation and other ``BaseException``
    subclasses propagate. A non-``ToolDecision`` return is an error.
    """
    try:
        ctx = ToolCallContext(
            run_id=run_id,
            call_id=tc.id,
            tool_name=tc.name,
            params=copy.deepcopy(tc.params),
            target=lookups.target.get(tc.name, ToolTarget.SERVER),
            source=_tool_source(tc.name, lookups),
            meta=copy.deepcopy(lookups.meta.get(tc.name, {})),
            iteration=iteration,
            tenant_id=tenant_id,
        )
        decision = hook(ctx)
        if inspect.isawaitable(decision):
            decision = await decision
    except Exception as exc:
        logger.warning(
            "before_tool hook raised %s for tool '%s'; denying the call",
            type(exc).__name__,
            tc.name,
        )
        return None, type(exc).__name__
    if decision is not None and not isinstance(decision, ToolDecision):
        logger.warning(
            "before_tool hook returned %s for tool '%s' (expected ToolDecision or None); "
            "denying the call",
            type(decision).__name__,
            tc.name,
        )
        return None, "TypeError"
    return decision, None


async def _apply_before_tool(
    hook: BeforeToolHook,
    calls: list[ToolCall],
    gates: dict[str, dict[str, str | None]],
    *,
    lookups: ToolLookups,
    strategy: Strategy,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    history: list[Message],
    warnings: list[str],
    all_results: list[tuple[ToolCall, ToolResult]],
    tenant_id: str | None,
) -> list[ToolCall]:
    """Evaluate ``before_tool`` per call; returns the calls that continue.

    Denied and substituted calls get their result appended here. Hook
    ``require_approval`` adds a gate to ``gates`` (mutated in place).
    A hook deny overrides a static approval gate; nothing else does.
    """

    async def governance(event: str, data: dict[str, Any], call_id: str) -> None:
        await _record_governance(recorder, run_id, event, iteration, data, correlation_id=call_id)
        await _notify_governance(
            notifier, run_id, event, iteration, data, correlation_id=call_id, warnings=warnings
        )

    async def append(tc: ToolCall, result: ToolResult, *, record: bool) -> None:
        if record:
            await _record_tool(recorder, run_id, tc, result, iteration)
            await _notify_tool(notifier, run_id, tc, result, iteration, warnings)
        result_msg = strategy.format_tool_result(result)
        history.append(result_msg)
        await _record_message(recorder, run_id, result_msg, iteration)
        await _notify_message(notifier, run_id, result_msg, iteration, warnings)
        all_results.append((tc, result))

    async def deny(tc: ToolCall, reason: str) -> None:
        gates.pop(tc.id, None)
        await governance(
            "policy.denied",
            {"tool_name": tc.name, "call_id": tc.id, "reason": reason, "source": "hook"},
            tc.id,
        )
        result = ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=json.dumps({"denied": reason}),
            success=False,
            error=reason,
        )
        # Like static deny: not an execution, no tool_calls row.
        await append(tc, result, record=False)

    async def hook_error(tc: ToolCall, error_type: str) -> None:
        await governance(
            "policy.hook_error",
            {"tool_name": tc.name, "call_id": tc.id, "error_type": error_type},
            tc.id,
        )
        await deny(tc, _HOOK_ERROR_REASON)

    kept: list[ToolCall] = []
    for tc in calls:
        decision, error_type = await _call_before_tool(
            hook, tc, lookups, run_id=run_id, iteration=iteration, tenant_id=tenant_id
        )
        if error_type is not None:
            await hook_error(tc, error_type)
            continue
        if decision is None or decision.kind == "allow":
            kept.append(tc)
            continue
        if decision.kind == "deny":
            await deny(tc, decision.reason or _HOOK_ERROR_REASON)
            continue
        statically_gated = tc.id in gates
        if decision.kind == "require_approval":
            if lookups.target.get(tc.name, ToolTarget.SERVER) != ToolTarget.SERVER:
                await governance(
                    "policy.hook_warning",
                    {
                        "tool_name": tc.name,
                        "call_id": tc.id,
                        "warning": _CLIENT_APPROVAL_REASON,
                    },
                    tc.id,
                )
                await deny(tc, _CLIENT_APPROVAL_REASON)
                continue
            if not statically_gated:
                gates[tc.id] = {"source": "hook", "reason": decision.reason}
            kept.append(tc)
            continue
        # substitute
        if statically_gated:
            # Static approval gate stays in place; the call goes to a human.
            kept.append(tc)
            continue
        try:
            payload = json.dumps(decision.result)
        except (TypeError, ValueError) as exc:
            await hook_error(tc, type(exc).__name__)
            continue
        substituted = ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=payload,
            success=True,
            meta={"substituted": True},
        )
        await governance("policy.substituted", {"tool_name": tc.name, "call_id": tc.id}, tc.id)
        await append(tc, substituted, record=True)
    return kept


class ReActLoop(Loop):
    """Think → act → observe → repeat.

    The default loop for most agent tasks. Runs until the strategy returns
    a Finish action or max_iterations is reached.

    Usage:
        loop = ReActLoop()
        result = await loop.run(
            agent=agent,
            provider=provider,
            strategy=strategy,
            user_input="What is 15 + 27?",
        )
    """

    async def run(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
        run_id: str | None = None,
        recorder: LoopRecorder | None = None,
        notifier: LoopNotifier | None = None,
        initial_history: list[Message] | None = None,
        initial_steps: list[AgentStep] | None = None,
        iteration_offset: int = 0,
        initial_usage: UsageStats | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
        initial_pii_mapping: dict[str, str] | None = None,
        state_store: StateStore | None = None,
        budget_carry: BudgetCarry | None = None,
    ) -> RunResult:
        """Execute the ReAct loop, optionally resuming from a pause.

        ``budget_carry`` restores budget bookkeeping (fired keys) from the
        pause being resumed.
        """
        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        cache_key_prefix = build_cache_key_prefix(agent)
        lookups = await agent.get_tool_lookups(loop=self)
        tool_defs = agent.get_all_tool_defs(loop=self)
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            run_id=resolved_run_id,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings
        budget = BudgetTracker(
            agent.budget,
            run_id=resolved_run_id,
            usage=total_usage,
            recorder=recorder,
            notifier=notifier,
            warnings=notifier_warnings,
            state_store=state_store,
            carry=budget_carry,
        )
        before_tool: BeforeToolHook | None = getattr(agent, "before_tool", None)
        hook_tenant_id = await _hook_tenant_id(before_tool, state_store, resolved_run_id)
        g_engine = (
            GuardrailEngine(agent.guardrails, pii_mapping=initial_pii_mapping)
            if agent.guardrails
            else None
        )

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # Cooperative checkpoint before starting the next iteration:
            # cancel > requested pause > budget pause/cancel; then inbox notes
            # are delivered. The current iteration's LLM/tools are not
            # preempted — flags are seen here or at the runner's pre-pause
            # checkpoint.
            control = (
                await _read_control(state_store, resolved_run_id)
                if state_store is not None
                else None
            )
            if control is not None and control.cancel_requested:
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    steps=steps,
                    commentary=_commentary(steps),
                    iteration_count=iteration - 1,
                    usage=total_usage,
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )
            if control is not None and control.pause_requested:
                return _checkpoint_pause_result(
                    agent_name=agent.name,
                    run_id=resolved_run_id,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
                    meta=guardrail_meta(g_engine, notifier_warnings),
                    budget=budget,
                )
            budget_result = await _budget_stop(
                budget,
                agent_name=agent.name,
                run_id=resolved_run_id,
                history=history,
                steps=steps,
                iteration=iteration,
                usage=total_usage,
                meta=guardrail_meta(g_engine, notifier_warnings),
            )
            if budget_result is not None:
                return budget_result
            if control is not None and control.has_pending_notes:
                assert state_store is not None
                await _deliver_inbox(
                    state_store,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    history,
                    notifier_warnings,
                )

            messages, tools = strategy.build_messages(
                system_prompt=agent.get_system_prompt(loop=self),
                history=history,
                tool_defs=tool_defs,
            )

            # Incoming guardrail — scan ALL messages before LLM call
            _all_guardrail_findings: dict[str, Any] = {}
            if g_engine is not None:
                all_in_findings: list[Any] = []
                _in_was_redacted = False
                for msg_idx, msg in enumerate(messages):
                    if not msg.content:
                        continue
                    cleaned, findings, block_err = await g_engine.scan_incoming(msg.content)
                    if findings:
                        all_in_findings.extend(findings)
                    if block_err is not None:
                        await _record_governance(
                            recorder,
                            resolved_run_id,
                            "guardrail.blocked",
                            iteration,
                            {"direction": "incoming", "error": block_err},
                        )
                        await _notify_governance(
                            notifier,
                            resolved_run_id,
                            "guardrail.blocked",
                            iteration,
                            {"direction": "incoming", "error": block_err},
                            warnings=notifier_warnings,
                        )
                        return RunResult(
                            run_id=resolved_run_id,
                            status=RunStatus.ERROR,
                            error=block_err,
                            steps=steps,
                            commentary=_commentary(steps),
                            iteration_count=iteration,
                            usage=total_usage,
                            meta=guardrail_meta(g_engine, notifier_warnings),
                        )
                    if cleaned != msg.content:
                        _in_was_redacted = True
                        # replace() preserves the origin envelope
                        # (kind/placement/source) through redaction.
                        messages[msg_idx] = replace(msg, content=cleaned)
                if all_in_findings:
                    _in_entities = list({f.entity_type for f in all_in_findings})
                    await _record_governance(
                        recorder,
                        resolved_run_id,
                        "guardrail.detected",
                        iteration,
                        {
                            "direction": "incoming",
                            "findings_count": len(all_in_findings),
                            "entities": _in_entities,
                        },
                    )
                    await _notify_governance(
                        notifier,
                        resolved_run_id,
                        "guardrail.detected",
                        iteration,
                        {
                            "direction": "incoming",
                            "findings_count": len(all_in_findings),
                            "entities": _in_entities,
                        },
                        warnings=notifier_warnings,
                    )
                    _all_guardrail_findings["incoming"] = [
                        {"entity_type": f.entity_type, "score": f.score} for f in all_in_findings
                    ]
                if _in_was_redacted:
                    _red_entities = list({f.entity_type for f in all_in_findings})
                    await _record_governance(
                        recorder,
                        resolved_run_id,
                        "guardrail.redacted",
                        iteration,
                        {"direction": "incoming", "entities": _red_entities},
                    )
                    await _notify_governance(
                        notifier,
                        resolved_run_id,
                        "guardrail.redacted",
                        iteration,
                        {"direction": "incoming", "entities": _red_entities},
                        warnings=notifier_warnings,
                    )

            # LLM call — batch (lifecycle: started → completed/failed)
            await _record_llm_started(
                recorder,
                resolved_run_id,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
            )
            await _notify_llm_started(
                notifier,
                resolved_run_id,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
            )
            t0 = time.monotonic()
            try:
                with telemetry_context(
                    run_id=resolved_run_id,
                    iteration=iteration,
                    recorder=recorder,
                    notifier=notifier,
                ):
                    response = await provider.complete(
                        messages,
                        tools=tools,
                        run_id=resolved_run_id,
                        cache_key_prefix=cache_key_prefix,
                        **_pkw,
                    )
            except Exception as _llm_exc:
                _llm_fail_ms = int((time.monotonic() - t0) * 1000)
                await _record_llm_failed(
                    recorder,
                    resolved_run_id,
                    iteration,
                    _llm_exc,
                    duration_ms=_llm_fail_ms,
                )
                await _notify_llm_failed(
                    notifier,
                    resolved_run_id,
                    iteration,
                    _llm_exc,
                    notifier_warnings,
                    duration_ms=_llm_fail_ms,
                )
                raise
            llm_duration_ms = int((time.monotonic() - t0) * 1000)
            response = price_response(response, pricing=agent.pricing, default_model=provider.model)

            # Output guardrail — detection-only. Persistence stores raw
            # (DB is ground truth); the next scan_incoming redacts for
            # the LLM. Outgoing enforces block/warn policies.
            if g_engine is not None and (response.text or response.tool_calls):
                tc_params = (
                    [tc.params for tc in response.tool_calls] if response.tool_calls else None
                )
                out_findings, out_block = await g_engine.scan_outgoing(
                    response.text or "", tc_params
                )
                if out_findings:
                    _all_guardrail_findings["outgoing"] = [
                        {"entity_type": f.entity_type, "score": f.score} for f in out_findings
                    ]
                    _out_entities = list({f.entity_type for f in out_findings})
                    await _record_governance(
                        recorder,
                        resolved_run_id,
                        "guardrail.detected",
                        iteration,
                        {
                            "direction": "outgoing",
                            "findings_count": len(out_findings),
                            "entities": _out_entities,
                        },
                    )
                    await _notify_governance(
                        notifier,
                        resolved_run_id,
                        "guardrail.detected",
                        iteration,
                        {
                            "direction": "outgoing",
                            "findings_count": len(out_findings),
                            "entities": _out_entities,
                        },
                        warnings=notifier_warnings,
                    )
                if out_block is not None:
                    # Scrub response before recording — block means the
                    # original text/params contain the PII that triggered it.
                    _blocked_response = LLMResponse(
                        text="[blocked by guardrail]",
                        tool_calls=None,
                        raw=response.raw,
                        usage=response.usage,
                        provider_request=response.provider_request,
                        provider_response=None,
                    )
                    await _record_llm(
                        recorder,
                        resolved_run_id,
                        _blocked_response,
                        iteration,
                        semantic_messages=messages,
                        semantic_tools=tools,
                        duration_ms=llm_duration_ms,
                        guardrail_findings=_all_guardrail_findings or None,
                    )
                    await _notify_llm(
                        notifier,
                        resolved_run_id,
                        _blocked_response,
                        iteration,
                        notifier_warnings,
                        semantic_messages=messages,
                        semantic_tools=tools,
                        duration_ms=llm_duration_ms,
                        guardrail_findings=_all_guardrail_findings or None,
                    )
                    _accumulate_usage(total_usage, response.usage)
                    budget.add_response(response.usage)
                    await budget.evaluate(iteration)
                    await _record_governance(
                        recorder,
                        resolved_run_id,
                        "guardrail.blocked",
                        iteration,
                        {"direction": "outgoing", "error": out_block},
                    )
                    await _notify_governance(
                        notifier,
                        resolved_run_id,
                        "guardrail.blocked",
                        iteration,
                        {"direction": "outgoing", "error": out_block},
                        warnings=notifier_warnings,
                    )
                    return RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.ERROR,
                        error=out_block,
                        steps=steps,
                        commentary=_commentary(steps),
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=guardrail_meta(g_engine, notifier_warnings),
                    )

            # Record the LLM response (raw — DB is ground truth)
            # and accumulate usage.
            await _record_llm(
                recorder,
                resolved_run_id,
                response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
                guardrail_findings=_all_guardrail_findings or None,
            )
            await _notify_llm(
                notifier,
                resolved_run_id,
                response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
                guardrail_findings=_all_guardrail_findings or None,
            )
            _accumulate_usage(total_usage, response.usage)
            budget.add_response(response.usage)
            await budget.evaluate(iteration)

            step = strategy.parse_response(response)
            if response.reasoning is not None:
                step = replace(step, meta={**step.meta, "reasoning": response.reasoning})
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="final_answer",
                )
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.SUCCESS,
                    answer=step.action.answer,
                    steps=steps,
                    commentary=_commentary(steps),
                    iteration_count=iteration,
                    usage=total_usage,
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )

            if isinstance(step.action, Clarification):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="final_answer",
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
                    budget=budget,
                )
                meta: dict[str, Any] = {
                    "pause_state": pause,
                    **guardrail_meta(g_engine, notifier_warnings),
                }
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.WAITING_HUMAN_INPUT,
                    answer=step.action.question,
                    steps=steps,
                    commentary=_commentary(steps),
                    iteration_count=iteration,
                    usage=total_usage,
                    meta=meta,
                )

            if isinstance(step.action, ToolCall):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="commentary",
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    run_id=resolved_run_id,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                    deny=agent.deny,
                    require_approval=agent.require_approval,
                    tool_result_transform=agent.tool_result_transform,
                    guardrail_engine=g_engine,
                    before_tool=before_tool,
                    tenant_id=hook_tenant_id,
                )
                if outcome.pending_calls:
                    pause_status = (
                        RunStatus.WAITING_APPROVAL
                        if outcome.pending_approval
                        else RunStatus.WAITING_CLIENT_TOOL
                    )
                    pause = _build_pause(
                        agent_name=agent.name,
                        pending_calls=outcome.pending_calls,
                        target_lookup=lookups.target,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                        approval_gates=outcome.approval_gates,
                        budget=budget,
                    )
                    meta = {"pause_state": pause, **guardrail_meta(g_engine, notifier_warnings)}
                    return RunResult(
                        run_id=resolved_run_id,
                        status=pause_status,
                        steps=steps,
                        commentary=_commentary(steps),
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    )

        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.MAX_ITERATIONS,
            steps=steps,
            commentary=_commentary(steps),
            iteration_count=agent.max_iterations,
            usage=total_usage,
            meta=guardrail_meta(g_engine, notifier_warnings),
        )

    async def run_stream(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
        run_id: str | None = None,
        recorder: LoopRecorder | None = None,
        notifier: LoopNotifier | None = None,
        initial_history: list[Message] | None = None,
        initial_steps: list[AgentStep] | None = None,
        iteration_offset: int = 0,
        initial_usage: UsageStats | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
        state_store: StateStore | None = None,
        interrupt: asyncio.Event | None = None,
        budget_carry: BudgetCarry | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream the ReAct loop as RunEvents.

        Same iteration logic as run() — shared helpers ensure consistency.
        Swaps provider.complete() for complete_stream(), forwarding text
        deltas live and yielding TOOL_RESULT events after execution.
        Terminal event is RUN_COMPLETED or RUN_PAUSED.
        """
        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        cache_key_prefix = build_cache_key_prefix(agent)
        lookups = await agent.get_tool_lookups(loop=self)
        tool_defs = agent.get_all_tool_defs(loop=self)
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            run_id=resolved_run_id,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings
        budget = BudgetTracker(
            agent.budget,
            run_id=resolved_run_id,
            usage=total_usage,
            recorder=recorder,
            notifier=notifier,
            warnings=notifier_warnings,
            state_store=state_store,
            carry=budget_carry,
        )
        before_tool: BeforeToolHook | None = getattr(agent, "before_tool", None)
        hook_tenant_id = await _hook_tenant_id(before_tool, state_store, resolved_run_id)

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # Cooperative checkpoint (see run() for rationale). The in-process
            # interrupt is checked first: it is free, and it lets a cancel
            # that landed during tool execution stop the run before the next
            # provider call is even started.
            interrupted_here = interrupt is not None and interrupt.is_set()
            control = (
                await _read_control(state_store, resolved_run_id)
                if state_store is not None and not interrupted_here
                else None
            )
            if interrupted_here or (control is not None and control.cancel_requested):
                cancel_meta: dict[str, Any] = (
                    {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
                )
                yield RunEvent(
                    type=RunEventType.RUN_CANCELLED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.CANCELLED,
                        steps=steps,
                        commentary=_commentary(steps),
                        iteration_count=iteration - 1,
                        usage=total_usage,
                        meta=cancel_meta,
                    ),
                )
                return
            if control is not None and control.pause_requested:
                yield RunEvent(
                    type=RunEventType.RUN_PAUSED,
                    run_result=_checkpoint_pause_result(
                        agent_name=agent.name,
                        run_id=resolved_run_id,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                        meta={"notifier_warnings": notifier_warnings} if notifier_warnings else {},
                        budget=budget,
                    ),
                )
                return
            budget_result = await _budget_stop(
                budget,
                agent_name=agent.name,
                run_id=resolved_run_id,
                history=history,
                steps=steps,
                iteration=iteration,
                usage=total_usage,
                meta={"notifier_warnings": notifier_warnings} if notifier_warnings else {},
            )
            if budget_result is not None:
                yield RunEvent(
                    type=(
                        RunEventType.RUN_CANCELLED
                        if budget_result.status == RunStatus.CANCELLED
                        else RunEventType.RUN_PAUSED
                    ),
                    run_result=budget_result,
                )
                return
            if control is not None and control.has_pending_notes:
                assert state_store is not None
                await _deliver_inbox(
                    state_store,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    history,
                    notifier_warnings,
                )

            messages, tools = strategy.build_messages(
                system_prompt=agent.get_system_prompt(loop=self),
                history=history,
                tool_defs=tool_defs,
            )

            # LLM call — streaming (lifecycle: started → completed/failed)
            await _record_llm_started(
                recorder,
                resolved_run_id,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
            )
            await _notify_llm_started(
                notifier,
                resolved_run_id,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
            )
            t0 = time.monotonic()
            llm_response: LLMResponse | None = None
            _stream_telemetry = telemetry_context(
                run_id=resolved_run_id,
                iteration=iteration,
                recorder=recorder,
                notifier=notifier,
            )
            _stream_telemetry.__enter__()
            provider_stream = provider.complete_stream(
                messages,
                tools=tools,
                run_id=resolved_run_id,
                cache_key_prefix=cache_key_prefix,
                **_pkw,
            )
            partial_text: list[str] = []
            # Text after a tool block in the same call is known commentary.
            saw_tool_use = False
            interrupted = False
            try:
                try:
                    async for event in interruptible(provider_stream, interrupt):
                        if event.type == StreamEventType.TEXT_DELTA:
                            partial_text.append(event.text or "")
                            yield RunEvent(
                                type=RunEventType.TEXT_DELTA,
                                text=event.text,
                                iteration=iteration,
                                phase=event.phase or ("commentary" if saw_tool_use else None),
                            )
                        elif event.type == StreamEventType.REASONING_DELTA:
                            yield RunEvent(
                                type=RunEventType.REASONING_DELTA,
                                text=event.text,
                                iteration=iteration,
                            )
                        elif event.type == StreamEventType.TOOL_USE_START:
                            saw_tool_use = True
                            yield RunEvent(
                                type=RunEventType.TOOL_USE_START,
                                tool_name=event.tool_name,
                                tool_call_id=event.tool_call_id,
                                iteration=iteration,
                                phase="commentary",
                            )
                        elif event.type == StreamEventType.TOOL_USE_END:
                            saw_tool_use = True
                            yield RunEvent(
                                type=RunEventType.TOOL_USE_END,
                                tool_call=event.tool_call,
                                tool_name=event.tool_name,
                                tool_call_id=event.tool_call_id,
                                iteration=iteration,
                                phase="commentary",
                            )
                        elif event.type == StreamEventType.DONE:
                            llm_response = event.raw
                    # Validate inside the same try so a contract-violating
                    # provider stream still pairs with on_llm_call_failed.
                    if llm_response is None:
                        raise RuntimeError(
                            f"Provider stream ended without DONE event at iteration "
                            f"{iteration}. complete_stream() must yield "
                            f"StreamEvent(type=DONE, raw=LLMResponse)."
                        )
                except Exception as _stream_exc:
                    _stream_fail_ms = int((time.monotonic() - t0) * 1000)
                    await _record_llm_failed(
                        recorder,
                        resolved_run_id,
                        iteration,
                        _stream_exc,
                        duration_ms=_stream_fail_ms,
                    )
                    await _notify_llm_failed(
                        notifier,
                        resolved_run_id,
                        iteration,
                        _stream_exc,
                        notifier_warnings,
                        duration_ms=_stream_fail_ms,
                    )
                    if isinstance(_stream_exc, StreamInterruptedError):
                        interrupted = True
                    else:
                        raise
            finally:
                await provider_stream.aclose()
                _stream_telemetry.__exit__(None, None, None)

            if interrupted:
                # Mid-stream interrupt (cancel_run reached this process). The
                # interrupted call reported no usage, so total_usage covers
                # completed iterations only. Partial text becomes the answer.
                yield RunEvent(
                    type=RunEventType.RUN_CANCELLED,
                    run_result=_interrupted_result(
                        run_id=resolved_run_id,
                        partial_text=partial_text,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                        notifier_warnings=notifier_warnings,
                    ),
                )
                return

            # The try-block raised if no DONE arrived; only the interrupted path
            # (handled above) leaves this None.
            assert llm_response is not None
            llm_duration_ms = int((time.monotonic() - t0) * 1000)
            llm_response = price_response(
                llm_response, pricing=agent.pricing, default_model=provider.model
            )

            await _record_llm(
                recorder,
                resolved_run_id,
                llm_response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            await _notify_llm(
                notifier,
                resolved_run_id,
                llm_response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            _accumulate_usage(total_usage, llm_response.usage)
            budget.add_response(llm_response.usage)
            await budget.evaluate(iteration)

            step = strategy.parse_response(llm_response)
            if llm_response.reasoning is not None:
                step = replace(step, meta={**step.meta, "reasoning": llm_response.reasoning})
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="final_answer",
                )
                meta: dict[str, Any] = (
                    {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
                )
                yield RunEvent(
                    type=RunEventType.RUN_COMPLETED,
                    phase="final_answer",
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.SUCCESS,
                        answer=step.action.answer,
                        steps=steps,
                        commentary=_commentary(steps),
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    ),
                )
                return

            if isinstance(step.action, Clarification):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="final_answer",
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
                    budget=budget,
                )
                meta = {"pause_state": pause}
                if notifier_warnings:
                    meta["notifier_warnings"] = notifier_warnings
                yield RunEvent(
                    type=RunEventType.RUN_PAUSED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.WAITING_HUMAN_INPUT,
                        answer=step.action.question,
                        steps=steps,
                        commentary=_commentary(steps),
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    ),
                )
                return

            if isinstance(step.action, ToolCall):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    phase="commentary",
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    run_id=resolved_run_id,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                    deny=agent.deny,
                    require_approval=agent.require_approval,
                    tool_result_transform=agent.tool_result_transform,
                    guardrail_engine=None,  # streaming + guardrails banned at Agent level
                    before_tool=before_tool,
                    tenant_id=hook_tenant_id,
                )
                for tc, tool_result in outcome.all_results:
                    yield RunEvent(
                        type=RunEventType.TOOL_RESULT,
                        tool_call=tc,
                        tool_result=tool_result,
                        iteration=iteration,
                        phase="commentary",
                    )
                if outcome.pending_calls:
                    pause_status = (
                        RunStatus.WAITING_APPROVAL
                        if outcome.pending_approval
                        else RunStatus.WAITING_CLIENT_TOOL
                    )
                    pause = _build_pause(
                        agent_name=agent.name,
                        pending_calls=outcome.pending_calls,
                        target_lookup=lookups.target,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                        approval_gates=outcome.approval_gates,
                        budget=budget,
                    )
                    meta = {"pause_state": pause}
                    if notifier_warnings:
                        meta["notifier_warnings"] = notifier_warnings
                    yield RunEvent(
                        type=RunEventType.RUN_PAUSED,
                        run_result=RunResult(
                            run_id=resolved_run_id,
                            status=pause_status,
                            steps=steps,
                            commentary=_commentary(steps),
                            iteration_count=iteration,
                            usage=total_usage,
                            meta=meta,
                        ),
                    )
                    return

        meta = {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
        yield RunEvent(
            type=RunEventType.RUN_COMPLETED,
            run_result=RunResult(
                run_id=resolved_run_id,
                status=RunStatus.MAX_ITERATIONS,
                steps=steps,
                commentary=_commentary(steps),
                iteration_count=agent.max_iterations,
                usage=total_usage,
                meta=meta,
            ),
        )


def _interrupted_result(
    *,
    run_id: str,
    partial_text: list[str],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    notifier_warnings: list[str],
) -> RunResult:
    """Build the ``RunResult`` for a run whose provider stream was interrupted.

    ``answer`` is the text streamed so far (``None`` if nothing arrived).
    ``meta.interrupted`` is always True; ``meta.partial_output`` says whether
    ``answer`` carries anything.
    """
    answer = "".join(partial_text) or None
    meta: dict[str, Any] = {"interrupted": True, "partial_output": answer is not None}
    if notifier_warnings:
        meta["notifier_warnings"] = notifier_warnings
    return RunResult(
        run_id=run_id,
        status=RunStatus.CANCELLED,
        answer=answer,
        steps=steps,
        commentary=_commentary(steps),
        iteration_count=iteration,
        usage=usage,
        meta=meta,
    )


# Backwards compatibility alias for runner.py's existing import.
_build_tool_lookups = build_tool_lookups


def _build_execution_groups(
    tool_calls: list[ToolCall],
    parallel_lookup: dict[str, bool],
) -> list[tuple[list[ToolCall], bool]]:
    """Group contiguous tool calls by their parallel flag.

    Scans left to right, building contiguous groups of parallel=True tools.
    A parallel=False tool is a barrier — always forms its own singleton group.

    Returns list of (group, is_parallel) tuples in original order.
    Example: [A(par), B(par), C(nonpar), D(par)] →
             [([A,B], True), ([C], False), ([D], True)]
    Example: [A(seq), B(seq)] →
             [([A], False), ([B], False)]
    """
    if not tool_calls:
        return []

    groups: list[tuple[list[ToolCall], bool]] = []
    current_parallel_group: list[ToolCall] = []

    for tc in tool_calls:
        is_parallel = parallel_lookup.get(tc.name, True)
        if not is_parallel:
            # Flush any accumulated parallel group first
            if current_parallel_group:
                groups.append((current_parallel_group, True))
                current_parallel_group = []
            # Barrier: always a singleton group
            groups.append(([tc], False))
        else:
            current_parallel_group.append(tc)

    # Flush remaining parallel group
    if current_parallel_group:
        groups.append((current_parallel_group, True))

    return groups


async def _execute_record_notify(
    tool_call: ToolCall,
    lookups: ToolLookups,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    notifier_warnings: list[str],
    transform: ToolResultTransform | None = None,
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool, record, then notify — for parallel execution.

    Each tool records and streams its event as soon as it finishes,
    without waiting for siblings.
    """
    await _record_tool_started(recorder, run_id, tool_call, iteration)
    await _notify_tool_started(notifier, run_id, tool_call, iteration, notifier_warnings)
    result = await _execute_tool(tool_call, lookups, transform)
    await _record_tool(recorder, run_id, tool_call, result, iteration)
    await _notify_tool(notifier, run_id, tool_call, result, iteration, notifier_warnings)
    return tool_call, result


async def _execute_tool(
    tool_call: ToolCall,
    lookups: ToolLookups,
    transform: ToolResultTransform | None = None,
) -> ToolResult:
    """Execute a server tool function and return a ToolResult.

    Non-server tools are filtered out by the loop before reaching this
    function. They trigger a pause instead of execution.

    ``transform`` (``Agent.tool_result_transform``) rewrites the raw
    return value of a successful call before serialization, so the
    store, notifiers and the model all see the transformed result.
    A transform that raises fails the call.
    """
    from dendrux.skills._session import SkillTool

    fn = lookups.fn.get(tool_call.name)
    if fn is None:
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    if isinstance(fn, SkillTool):
        return await fn.session.execute(tool_call)

    timeout_s = lookups.timeout.get(tool_call.name, DEFAULT_TOOL_TIMEOUT)
    is_explicit = lookups.explicit_timeout.get(tool_call.name, False)

    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            coro = fn(**tool_call.params)
        else:
            coro = asyncio.to_thread(fn, **tool_call.params)
        result = await asyncio.wait_for(coro, timeout=timeout_s)

        duration_ms = int((time.monotonic() - start) * 1000)
        if transform is not None:
            try:
                result = await _apply_result_transform(transform, tool_call, lookups, result)
            except Exception as transform_err:
                error_msg = (
                    f"tool_result_transform failed for tool '{tool_call.name}': "
                    f"{type(transform_err).__name__}: {transform_err}"
                )
                logger.warning("%s", error_msg)
                return ToolResult(
                    name=tool_call.name,
                    call_id=tool_call.id,
                    payload=json.dumps({"error": error_msg}),
                    success=False,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
        try:
            payload = json.dumps(result)
        except (TypeError, ValueError) as ser_err:
            raise TypeError(
                f"Tool '{tool_call.name}' returned a non-JSON-serializable value: "
                f"{type(result).__name__}. Tool return values must be JSON-serializable "
                f"(str, int, float, bool, list, dict, or None)."
            ) from ser_err

        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=payload,
            success=True,
            duration_ms=duration_ms,
        )
    except TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        error_msg = f"Tool '{tool_call.name}' timed out after {timeout_s}s"
        if is_explicit:
            logger.warning("%s", error_msg)
        else:
            logger.warning(
                "%s (default). Set an explicit timeout: @tool(timeout_seconds=N)",
                error_msg,
            )
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": error_msg}),
            success=False,
            error=error_msg,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": str(e)}),
            success=False,
            error=str(e),
            duration_ms=duration_ms,
        )


async def _apply_result_transform(
    transform: ToolResultTransform,
    tool_call: ToolCall,
    lookups: ToolLookups,
    result: Any,
) -> Any:
    ctx = ToolResultContext(
        tool_name=tool_call.name,
        call_id=tool_call.id,
        params=tool_call.params,
        result=result,
        target=lookups.target.get(tool_call.name, ToolTarget.SERVER),
        meta=lookups.meta.get(tool_call.name, {}),
    )
    out = transform(ctx)
    if inspect.isawaitable(out):
        out = await out
    return out
