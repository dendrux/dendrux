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
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._helpers import (
    build_cache_key_prefix,
    guardrail_meta,
    notify_governance,
    notify_llm,
    notify_message,
    notify_tool,
    record_governance,
    record_llm,
    record_message,
    record_tool,
)
from dendrux.loops.base import Loop
from dendrux.tool import DEFAULT_TOOL_TIMEOUT
from dendrux.tools._lookups import ToolLookups, build_tool_lookups
from dendrux.types import (
    AgentStep,
    Budget,
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
    ToolResult,
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

logger = logging.getLogger(__name__)


_record_message = record_message
_record_llm = record_llm
_record_tool = record_tool
_record_governance = record_governance
_notify_message = notify_message
_notify_llm = notify_llm
_notify_tool = notify_tool
_notify_governance = notify_governance

# Sentinel value in fired_thresholds to track that budget.exceeded has fired.
_BUDGET_EXCEEDED_SENTINEL = -1.0


class _ToolCallOutcome(NamedTuple):
    """Result of processing tool calls in a single iteration."""

    # All (tool_call, tool_result) pairs — limits + executed, in order
    all_results: list[tuple[ToolCall, ToolResult]]
    # Non-server tools that need pause (empty if all were server tools)
    pending_calls: list[ToolCall]
    # True if pending_calls are awaiting approval (vs client tool pause)
    pending_approval: bool = False


def _accumulate_usage(total: UsageStats, step_usage: UsageStats) -> None:
    """Add per-call usage to running total. Mutates total in place.

    Cache fields treat None as 0 in summation so providers that don't
    report them don't poison the rollup. Once any step reports a value,
    the running total carries it.
    """
    total.input_tokens += step_usage.input_tokens
    total.output_tokens += step_usage.output_tokens
    total.total_tokens += step_usage.total_tokens
    if step_usage.cost_usd is not None:
        if total.cost_usd is None:
            total.cost_usd = 0.0
        total.cost_usd += step_usage.cost_usd
    if step_usage.cache_read_input_tokens is not None:
        total.cache_read_input_tokens = (
            total.cache_read_input_tokens or 0
        ) + step_usage.cache_read_input_tokens
    if step_usage.cache_creation_input_tokens is not None:
        total.cache_creation_input_tokens = (
            total.cache_creation_input_tokens or 0
        ) + step_usage.cache_creation_input_tokens


async def _check_budget(
    budget: Budget | None,
    total_usage: UsageStats,
    fired_thresholds: list[float],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    iteration: int,
    warnings: list[str],
) -> None:
    """Check budget thresholds and exceeded after usage accumulation.

    Advisory only — fires governance events but does not pause or stop.
    Each threshold and exceeded fires exactly once per run.
    """
    if budget is None:
        return

    used = total_usage.total_tokens
    max_t = budget.max_tokens
    fraction = used / max_t

    # Threshold events — fire once per fraction when first crossed
    for threshold in budget.warn_at:
        if fraction >= threshold and threshold not in fired_thresholds:
            fired_thresholds.append(threshold)
            event_data = {
                "fraction": threshold,
                "used": used,
                "max": max_t,
                "reason": "threshold_crossed",
            }
            await _record_governance(
                recorder,
                "budget.threshold",
                iteration,
                event_data,
            )
            await _notify_governance(
                notifier,
                "budget.threshold",
                iteration,
                event_data,
                warnings=warnings,
            )

    # Exceeded event — fire once when usage first reaches max
    if used >= max_t and _BUDGET_EXCEEDED_SENTINEL not in fired_thresholds:
        fired_thresholds.append(_BUDGET_EXCEEDED_SENTINEL)
        event_data = {
            "used": used,
            "max": max_t,
            "reason": "budget_exceeded",
        }
        await _record_governance(
            recorder,
            "budget.exceeded",
            iteration,
            event_data,
        )
        await _notify_governance(
            notifier,
            "budget.exceeded",
            iteration,
            event_data,
            warnings=warnings,
        )


async def _append_assistant(
    response: LLMResponse,
    history: list[Message],
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    iteration: int,
    warnings: list[str],
) -> Message:
    """Create assistant message, append to history, record then notify."""
    msg = Message(
        role=Role.ASSISTANT,
        content=response.text or "",
        tool_calls=response.tool_calls,
    )
    history.append(msg)
    await _record_message(recorder, msg, iteration)
    await _notify_message(notifier, msg, iteration, warnings)
    return msg


def _snapshot_usage(usage: UsageStats) -> UsageStats:
    """Create an independent copy of usage stats for pause state."""
    return UsageStats(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
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
) -> PauseState:
    """Build PauseState for client tool or clarification pause."""
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
    initial_history: list[Message] | None,
    initial_steps: list[AgentStep] | None,
    initial_usage: UsageStats | None,
) -> _LoopState:
    """Initialize mutable loop state — shared between run() and run_stream()."""
    if initial_history is not None:
        history = list(initial_history)
    else:
        user_msg = Message(role=Role.USER, content=user_input)
        history = [user_msg]
        await _record_message(recorder, user_msg, 0)
        await _notify_message(notifier, user_msg, 0)

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
    iteration: int,
    history: list[Message],
    warnings: list[str],
    deny: frozenset[str] | None = None,
    require_approval: frozenset[str] | None = None,
    guardrail_engine: GuardrailEngine | None = None,
) -> _ToolCallOutcome:
    """Execute tool calls: enforce limits, run server tools, update history.

    Returns a _ToolCallOutcome with all results and any pending (non-server)
    calls. Both run() and run_stream() call this — the caller decides
    whether to yield events from the results.

    Side effects: mutates call_counts, appends to history, records then notifies.
    """
    all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
    all_results: list[tuple[ToolCall, ToolResult]] = []
    deny_set = deny or frozenset()

    # --- Phase 0: Deny check (governance) ---
    non_denied: list[ToolCall] = []
    for tc in all_calls:
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
                recorder, "policy.denied", iteration, event_data, correlation_id=tc.id
            )
            await _notify_governance(
                notifier,
                "policy.denied",
                iteration,
                event_data,
                correlation_id=tc.id,
                warnings=warnings,
            )
            result_msg = strategy.format_tool_result(deny_result)
            history.append(result_msg)
            await _record_message(recorder, result_msg, iteration)
            await _notify_message(notifier, result_msg, iteration, warnings)
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

    # --- Phase 1: Approval check (governance) ---
    approval_set = require_approval or frozenset()
    if approval_set and any(tc.name in approval_set for tc in non_denied):
        # Any tool in the batch needs approval → entire remaining batch pauses.
        for tc in non_denied:
            if tc.name in approval_set:
                event_data = {
                    "tool_name": tc.name,
                    "call_id": tc.id,
                    "reason": "requires_approval",
                }
                await _record_governance(
                    recorder,
                    "approval.requested",
                    iteration,
                    event_data,
                    correlation_id=tc.id,
                )
                await _notify_governance(
                    notifier,
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
        )

    # --- Phase 1: Enforce max_calls_per_run ---
    allowed_calls: list[ToolCall] = []
    batch_counts: dict[str, int] = {}
    for tc in non_denied:
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
            await _record_tool(recorder, tc, limit_result, iteration)
            await _notify_tool(notifier, tc, limit_result, iteration, warnings)
            result_msg = strategy.format_tool_result(limit_result)
            history.append(result_msg)
            await _record_message(recorder, result_msg, iteration)
            await _notify_message(notifier, result_msg, iteration, warnings)
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
                        iteration,
                        warnings,
                    )
                    for tc in group
                ]
            )
            executed.extend(pairs)
        else:
            for tc in group:
                tool_result = await _execute_tool(tc, lookups)
                await _record_tool(recorder, tc, tool_result, iteration)
                await _notify_tool(notifier, tc, tool_result, iteration, warnings)
                executed.append((tc, tool_result))

    # --- Phase 4: Append history in original order (deterministic for LLM) ---
    for tc, tool_result in executed:
        call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
        result_msg = strategy.format_tool_result(tool_result)
        history.append(result_msg)
        await _record_message(recorder, result_msg, iteration)
        await _notify_message(notifier, result_msg, iteration, warnings)
        all_results.append((tc, tool_result))

        # Emit skill.invoked governance event when LLM activates a skill
        if tc.name == "use_skill" and tool_result.success:
            skill_name = (tc.params or {}).get("name", "")
            await _record_governance(
                recorder,
                GovernanceEventType.SKILL_INVOKED,
                iteration,
                {"skill_name": skill_name},
            )
            await _notify_governance(
                notifier,
                GovernanceEventType.SKILL_INVOKED,
                iteration,
                {"skill_name": skill_name},
                warnings=warnings,
            )

    return _ToolCallOutcome(all_results=all_results, pending_calls=pending_calls)


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
    ) -> RunResult:
        """Execute the ReAct loop, optionally resuming from a pause."""
        resolved_run_id = run_id or generate_ulid()
        _pkw = provider_kwargs or {}
        cache_key_prefix = build_cache_key_prefix(agent)
        lookups = await agent.get_tool_lookups()
        tool_defs = agent.get_all_tool_defs()
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings
        budget_fired: list[float] = []
        g_engine = (
            GuardrailEngine(agent.guardrails, pii_mapping=initial_pii_mapping)
            if agent.guardrails
            else None
        )

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # Cooperative cancellation: observe before starting next iteration.
            # Current iteration's LLM/tools are not preempted — cancel is seen
            # at this checkpoint or the runner's pre-pause checkpoint.
            if state_store is not None and await state_store.is_cancel_requested(resolved_run_id):
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.CANCELLED,
                    steps=steps,
                    iteration_count=iteration - 1,
                    usage=total_usage,
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )

            messages, tools = strategy.build_messages(
                system_prompt=agent.get_system_prompt(),
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
                            "guardrail.blocked",
                            iteration,
                            {"direction": "incoming", "error": block_err},
                        )
                        await _notify_governance(
                            notifier,
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
                            iteration_count=iteration,
                            usage=total_usage,
                            meta=guardrail_meta(g_engine, notifier_warnings),
                        )
                    if cleaned != msg.content:
                        _in_was_redacted = True
                        messages[msg_idx] = Message(
                            role=msg.role,
                            content=cleaned,
                            name=msg.name,
                            tool_calls=msg.tool_calls,
                            call_id=msg.call_id,
                            meta=msg.meta,
                        )
                if all_in_findings:
                    _in_entities = list({f.entity_type for f in all_in_findings})
                    await _record_governance(
                        recorder,
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
                        "guardrail.redacted",
                        iteration,
                        {"direction": "incoming", "entities": _red_entities},
                    )
                    await _notify_governance(
                        notifier,
                        "guardrail.redacted",
                        iteration,
                        {"direction": "incoming", "entities": _red_entities},
                        warnings=notifier_warnings,
                    )

            # LLM call — batch
            t0 = time.monotonic()
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
            llm_duration_ms = int((time.monotonic() - t0) * 1000)

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
                        _blocked_response,
                        iteration,
                        semantic_messages=messages,
                        semantic_tools=tools,
                        duration_ms=llm_duration_ms,
                        guardrail_findings=_all_guardrail_findings or None,
                    )
                    await _notify_llm(
                        notifier,
                        _blocked_response,
                        iteration,
                        notifier_warnings,
                        semantic_messages=messages,
                        semantic_tools=tools,
                        duration_ms=llm_duration_ms,
                    )
                    _accumulate_usage(total_usage, response.usage)
                    await _check_budget(
                        agent.budget,
                        total_usage,
                        budget_fired,
                        recorder,
                        notifier,
                        iteration,
                        notifier_warnings,
                    )
                    await _record_governance(
                        recorder,
                        "guardrail.blocked",
                        iteration,
                        {"direction": "outgoing", "error": out_block},
                    )
                    await _notify_governance(
                        notifier,
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
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=guardrail_meta(g_engine, notifier_warnings),
                    )

            # Record the LLM response (raw — DB is ground truth)
            # and accumulate usage.
            await _record_llm(
                recorder,
                response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
                guardrail_findings=_all_guardrail_findings or None,
            )
            await _notify_llm(
                notifier,
                response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            _accumulate_usage(total_usage, response.usage)
            await _check_budget(
                agent.budget,
                total_usage,
                budget_fired,
                recorder,
                notifier,
                iteration,
                notifier_warnings,
            )

            step = strategy.parse_response(response)
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                return RunResult(
                    run_id=resolved_run_id,
                    status=RunStatus.SUCCESS,
                    answer=step.action.answer,
                    steps=steps,
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
                    iteration,
                    notifier_warnings,
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
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
                    iteration,
                    notifier_warnings,
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                    deny=agent.deny,
                    require_approval=agent.require_approval,
                    guardrail_engine=g_engine,
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
                    )
                    meta = {"pause_state": pause, **guardrail_meta(g_engine, notifier_warnings)}
                    return RunResult(
                        run_id=resolved_run_id,
                        status=pause_status,
                        steps=steps,
                        iteration_count=iteration,
                        usage=total_usage,
                        meta=meta,
                    )

        return RunResult(
            run_id=resolved_run_id,
            status=RunStatus.MAX_ITERATIONS,
            steps=steps,
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
        lookups = await agent.get_tool_lookups()
        tool_defs = agent.get_all_tool_defs()
        state = await _init_loop_state(
            user_input=user_input,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            initial_usage=initial_usage,
        )
        history, call_counts, steps = state.history, state.call_counts, state.steps
        total_usage, notifier_warnings = state.usage, state.warnings
        budget_fired: list[float] = []

        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # Cooperative cancellation checkpoint (see run() for rationale).
            if state_store is not None and await state_store.is_cancel_requested(resolved_run_id):
                cancel_meta: dict[str, Any] = (
                    {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
                )
                yield RunEvent(
                    type=RunEventType.RUN_CANCELLED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.CANCELLED,
                        steps=steps,
                        iteration_count=iteration - 1,
                        usage=total_usage,
                        meta=cancel_meta,
                    ),
                )
                return

            messages, tools = strategy.build_messages(
                system_prompt=agent.get_system_prompt(),
                history=history,
                tool_defs=tool_defs,
            )

            # LLM call — streaming
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
            try:
                async for event in provider_stream:
                    if event.type == StreamEventType.TEXT_DELTA:
                        yield RunEvent(type=RunEventType.TEXT_DELTA, text=event.text)
                    elif event.type == StreamEventType.TOOL_USE_START:
                        yield RunEvent(
                            type=RunEventType.TOOL_USE_START,
                            tool_name=event.tool_name,
                            tool_call_id=event.tool_call_id,
                        )
                    elif event.type == StreamEventType.TOOL_USE_END:
                        yield RunEvent(
                            type=RunEventType.TOOL_USE_END,
                            tool_call=event.tool_call,
                            tool_name=event.tool_name,
                            tool_call_id=event.tool_call_id,
                        )
                    elif event.type == StreamEventType.DONE:
                        llm_response = event.raw
            finally:
                await provider_stream.aclose()
                _stream_telemetry.__exit__(None, None, None)

            llm_duration_ms = int((time.monotonic() - t0) * 1000)

            if llm_response is None:
                raise RuntimeError(
                    f"Provider stream ended without DONE event at iteration {iteration}. "
                    f"complete_stream() must yield StreamEvent(type=DONE, raw=LLMResponse)."
                )

            await _record_llm(
                recorder,
                llm_response,
                iteration,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            await _notify_llm(
                notifier,
                llm_response,
                iteration,
                notifier_warnings,
                semantic_messages=messages,
                semantic_tools=tools,
                duration_ms=llm_duration_ms,
            )
            _accumulate_usage(total_usage, llm_response.usage)
            await _check_budget(
                agent.budget,
                total_usage,
                budget_fired,
                recorder,
                notifier,
                iteration,
                notifier_warnings,
            )

            step = strategy.parse_response(llm_response)
            steps.append(step)

            if isinstance(step.action, Finish):
                await _append_assistant(
                    llm_response,
                    history,
                    recorder,
                    notifier,
                    iteration,
                    notifier_warnings,
                )
                meta: dict[str, Any] = (
                    {"notifier_warnings": notifier_warnings} if notifier_warnings else {}
                )
                yield RunEvent(
                    type=RunEventType.RUN_COMPLETED,
                    run_result=RunResult(
                        run_id=resolved_run_id,
                        status=RunStatus.SUCCESS,
                        answer=step.action.answer,
                        steps=steps,
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
                    iteration,
                    notifier_warnings,
                )
                pause = _build_pause(
                    agent_name=agent.name,
                    pending_calls=[],
                    target_lookup=lookups.target,
                    history=history,
                    steps=steps,
                    iteration=iteration,
                    usage=total_usage,
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
                    iteration,
                    notifier_warnings,
                )
                outcome = await _process_tool_calls(
                    step=step,
                    call_counts=call_counts,
                    lookups=lookups,
                    strategy=strategy,
                    recorder=recorder,
                    notifier=notifier,
                    iteration=iteration,
                    history=history,
                    warnings=notifier_warnings,
                    deny=agent.deny,
                    require_approval=agent.require_approval,
                    guardrail_engine=None,  # streaming + guardrails banned at Agent level
                )
                for tc, tool_result in outcome.all_results:
                    yield RunEvent(
                        type=RunEventType.TOOL_RESULT,
                        tool_call=tc,
                        tool_result=tool_result,
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
                iteration_count=agent.max_iterations,
                usage=total_usage,
                meta=meta,
            ),
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
    iteration: int,
    notifier_warnings: list[str],
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool, record, then notify — for parallel execution.

    Each tool records and streams its event as soon as it finishes,
    without waiting for siblings.
    """
    result = await _execute_tool(tool_call, lookups)
    await _record_tool(recorder, tool_call, result, iteration)
    await _notify_tool(notifier, tool_call, result, iteration, notifier_warnings)
    return tool_call, result


async def _execute_tool(
    tool_call: ToolCall,
    lookups: ToolLookups,
) -> ToolResult:
    """Execute a server tool function and return a ToolResult.

    Non-server tools are filtered out by the loop before reaching this
    function. They trigger a pause instead of execution.
    """
    fn = lookups.fn.get(tool_call.name)
    if fn is None:
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

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
