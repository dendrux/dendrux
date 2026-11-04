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
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    ApprovalReason,
    DenialReason,
    FailureReason,
    PauseReason,
)
from dendrux._meta_keys import (
    NOT_EXECUTED_META_KEY,
)
from dendrux.errors import ImageError, ImagePersistenceUnsupportedError
from dendrux.guardrails._engine import GuardrailEngine
from dendrux.llm._calls import complete_call
from dendrux.llm._evidence import (
    RejectedArguments,
    RejectedToolName,
    persistable_call,
)
from dendrux.llm._retry_telemetry import telemetry_context
from dendrux.loops._budget import BudgetBreach, BudgetCarry, BudgetTracker, breach_payload
from dendrux.loops._helpers import (
    build_cache_key_prefix,
    final_stop_reason,
    governance_sinks,
    guardrail_meta,
    lease_scope,
    notify_governance,
    notify_llm,
    notify_llm_started,
    notify_message,
    notify_tool,
    notify_tool_started,
    order_observations,
    project_message,
    record_governance,
    record_llm,
    record_llm_started,
    record_message,
    record_tool,
    record_tool_started,
)
from dendrux.loops._images import (
    ImageCallSite,
    no_control,
    pause_image_view,
    release_wire,
    tool_result_images_dropped,
)
from dendrux.loops._llm_call import EventRelay, ProviderCallSite, StreamCall, stream_call_events
from dendrux.loops._policy_context import initial_call_counts, static_denial, tool_call_context
from dendrux.loops.base import Loop
from dendrux.media._ids import IMAGE_ID_RE
from dendrux.media._image import Image, ToolOutput
from dendrux.media._ingest import ingest
from dendrux.media._policy import ImagePolicy
from dendrux.skills._session import SkillTool
from dendrux.tool import DEFAULT_TOOL_TIMEOUT
from dendrux.tools._coerce import ArgumentCoercionError, build_call, get_args_spec
from dendrux.tools._limits import (
    ToolResultTooLargeError,
    cap_value_payload,
    enforce_call_limits,
    result_cap,
    size_limit_payload,
)
from dendrux.tools._mask import approval_view, placeholder_mapping
from dendrux.tools._pool import (
    CallTracker,
    ToolCallLimiter,
    ToolPoolSaturatedError,
    run_sync_tool,
)
from dendrux.tools._validate import (
    check_tool_call,
    check_tool_json,
    coercion_failure,
    prepare_tool_call,
)
from dendrux.types import (
    REASONING_MODEL_META_KEY,
    AgentStep,
    Clarification,
    Finish,
    GovernanceEventType,
    ImagePart,
    ImageRef,
    ImageViewState,
    Limits,
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
    ToolDecision,
    ToolResult,
    ToolResultContext,
    ToolTarget,
    UsageStats,
    generate_ulid,
)

_IMAGE_REFERENCE_MAX_DEPTH = 50
_IMAGE_SEARCH_MAX_DEPTH = 20
_IMAGE_WIRE_OVERHEAD_BYTES = 4096

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from pydantic import BaseModel

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.media._prepared import PreparedRequest
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import StateStore
    from dendrux.skills._session import SkillSession
    from dendrux.strategies.base import Strategy
    from dendrux.tools._lookups import ToolLookups
    from dendrux.types import BeforeToolHook, StreamEvent, ToolDef, ToolResultTransform

logger = logging.getLogger(__name__)


_record_message = record_message
_record_llm = record_llm
_record_llm_started = record_llm_started
_record_tool = record_tool
_record_tool_started = record_tool_started
_record_governance = record_governance
_notify_message = notify_message
_notify_llm = notify_llm
_notify_llm_started = notify_llm_started
_notify_tool = notify_tool
_notify_tool_started = notify_tool_started
_notify_governance = notify_governance


def _blocked_data(direction: str, error: str, findings: list[Any], location: str) -> dict[str, Any]:
    from collections import Counter

    counts = dict(Counter(f.entity_type for f in findings))
    return {
        "direction": direction,
        "error": error,
        "code": FailureReason.GUARDRAIL_BLOCKED.value,
        "kinds": sorted(counts),
        "counts": counts,
        "location": location,
    }


def _image_block_data(error: str, state: ImageViewState) -> dict[str, Any]:
    from collections import Counter

    counts = dict(
        Counter(
            kind
            for verdict in state.guardrail_verdicts.values()
            if verdict.action == "block"
            for kind in verdict.entities
        )
    )
    return {
        **_blocked_data("incoming_image", error, [], "incoming_image"),
        "kinds": sorted(counts),
        "counts": counts,
    }


def _incoming_location(msg: Message, index: int) -> str:
    call = f", tool_call_id={msg.call_id}" if msg.call_id else ""
    return f"message {index} (role={msg.role.value}{call})"


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
    total.cost_unknown = cost_unknown or step_usage.cost_unknown or step_usage.cost_usd is None
    if total.cost_unknown:
        total.cost_usd = None
        total.cost_source = None
    else:
        if total.cost_usd is None:
            total.cost_usd = 0.0
            total.cost_source = step_usage.cost_source
        elif total.cost_source != step_usage.cost_source:
            total.cost_source = "mixed"
        if step_usage.cost_usd is not None:
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
    return await _commit_message(history, msg, recorder, notifier, run_id, iteration, warnings)


async def _commit_message(
    history: list[Message],
    msg: Message,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    warnings: list[str],
) -> Message:
    """Record then notify ``msg``, the last history entry; history keeps the recorded form."""
    msg = await _record_message(recorder, run_id, msg, iteration)
    history[-1] = msg
    await _notify_message(notifier, run_id, msg, iteration, warnings)
    return msg


def _format_result(strategy: Strategy, result: ToolResult) -> Message:
    """``strategy.format_tool_result`` plus the dropped-images guard."""
    msg = strategy.format_tool_result(result)
    tool_result_images_dropped(result.images, msg)
    return msg


def _commentary(steps: list[AgentStep]) -> list[str]:
    """Non-empty assistant text from steps that ended in tool calls, in order."""
    return [
        s.raw_response
        for s in steps
        if isinstance(s.action, ToolCall) and s.raw_response and s.raw_response.strip()
    ]


def _pending_limit_result(
    *,
    agent: Agent,
    state_store: StateStore | None,
    response: LLMResponse,
    run_id: str,
    iteration: int,
    steps: list[AgentStep],
    usage: UsageStats,
    meta: dict[str, Any],
) -> RunResult | None:
    if (
        state_store is None
        or getattr(state_store, "storage_format", 1) != STORAGE_FORMAT_CHECKPOINTED
    ):
        return None
    if len(response.tool_calls or ()) <= agent._limits.max_pending_calls:
        return None
    _codes.log_code(logger, _codes.PENDING_LIMIT, run_id=run_id)
    return RunResult(
        run_id=run_id,
        status=RunStatus.ERROR,
        error="Assistant batch exceeds the pending-call limit.",
        steps=steps,
        commentary=_commentary(steps),
        iteration_count=iteration,
        usage=usage,
        meta={**meta, "failure_reason": FailureReason.PENDING_LIMIT.value},
    )


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
    image_view: ImageViewState | None = None,
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
        image_view=image_view,
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
    image_view: ImageViewState | None = None,
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
        pause_reason=PauseReason.BUDGET.value if exceeded else PauseReason.REQUESTED.value,
        budget_exceeded=exceeded,
        image_view=image_view,
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


def _control_stop_result(
    control: Any,
    *,
    agent_name: str,
    run_id: str,
    history: list[Message],
    steps: list[AgentStep],
    iteration: int,
    usage: UsageStats,
    meta: dict[str, Any],
    budget: BudgetTracker | None,
    image_view: ImageViewState | None,
) -> RunResult:
    """CANCELLED / PAUSED result for a control read just before an image retry (N-15.1).

    Nothing of ``iteration`` was committed (the assistant turn never
    arrived), so it ends exactly like the top-of-iteration checkpoint.
    """
    if control.cancel_requested:
        return RunResult(
            run_id=run_id,
            status=RunStatus.CANCELLED,
            steps=steps,
            commentary=_commentary(steps),
            iteration_count=iteration - 1,
            usage=usage,
            meta=meta,
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
        image_view=image_view,
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
    image_view: Callable[[], ImageViewState | None] | None = None,
) -> RunResult | None:
    """Checkpoint budget enforcement: a PAUSED/CANCELLED result, or None to go on.

    ``image_view`` builds the pause's image view lazily (only when pausing).
    """
    action, breaches = await budget.checkpoint(iteration)
    if not breaches:
        return None
    if action == "cancel":
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
        image_view=image_view() if image_view is not None else None,
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
        await _commit_message(history, msg, recorder, notifier, run_id, iteration, warnings)
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
    image_view: ImageViewState


def _bind_checkpoint(
    recorder: LoopRecorder | None,
    *,
    state: _LoopState,
    budget: BudgetTracker,
    provider_kwargs: dict[str, Any],
    tool_defs: Any,
    engine: GuardrailEngine | None,
    agent: Agent,
    strategy: Strategy,
    loop_type: str = "react",
) -> None:
    bind = getattr(recorder, "bind_loop", None)
    if bind is not None:
        bind(
            lambda: {
                "loop": loop_type,
                "format_result": lambda result: _format_result(strategy, result),
                "idempotent": {t.name: t.idempotent for t in tool_defs},
                "steps": state.steps,
                "usage": state.usage,
                "budget": budget,
                "tool_counts": state.call_counts,
                "declared_tools": {t.name for t in tool_defs},
                "tool_targets": {t.name: t.target.value for t in tool_defs},
                "approval_tools": agent.require_approval,
                "denied_tools": agent._deny,
                "provider_kwargs": provider_kwargs,
                "image_view": state.image_view,
                "has_images": any(m.parts for m in state.history),
                "pii_mapping": engine.get_pii_mapping() if engine else None,
                "pii_counters": engine._counters if engine else {},
            },
            budget,
        )


async def _checkpoint_view(
    recorder: LoopRecorder | None,
    messages: list[Message],
    iteration: int,
) -> None:
    checkpoint = getattr(recorder, "checkpoint_view", None)
    if checkpoint is not None:
        await checkpoint(messages, iteration)


async def _init_loop_state(
    *,
    user_input: str,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    initial_history: list[Message] | None,
    initial_steps: list[AgentStep] | None,
    initial_usage: UsageStats | None,
    _initial_call_counts: dict[str, int] | None = None,
    image_view_state: ImageViewState | None = None,
) -> _LoopState:
    """Initialize mutable loop state — shared between run() and run_stream()."""
    if initial_history is not None:
        history = [await project_message(recorder, m) for m in initial_history]
    else:
        from dendrux.context_blocks import wire_user_input

        user_msg = Message(role=Role.USER, content=wire_user_input(user_input, [], None))
        history = [user_msg]
        user_msg = await _record_message(recorder, run_id, user_msg, 0)
        history[-1] = user_msg
        await _notify_message(notifier, run_id, user_msg, 0)

    call_counts = initial_call_counts(history, _initial_call_counts)

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
        image_view=image_view_state if image_view_state is not None else ImageViewState(),
    )


def _rejected_call_result(tc: ToolCall) -> ToolResult:
    from dendrux.tools._validate import invalid_params_result

    if isinstance(tc, RejectedToolName):
        return _unknown_tool_result(tc, None)
    return invalid_params_result(tc, "Tool arguments must contain persistable text and JSON.")


def _persistable_argument_validator(validator: Any) -> Any:
    from functools import wraps

    @wraps(validator)
    def validate(tc: ToolCall, *args: Any, **kwargs: Any) -> Any:
        if isinstance(tc, (RejectedToolName, RejectedArguments)):
            return _rejected_call_result(tc)
        return validator(tc, *args, **kwargs)

    return validate


prepare_tool_call = _persistable_argument_validator(prepare_tool_call)


class _Governance:
    """Record then notify a governance event of one iteration.

    Also the skill session's sink, so it holds only what the event needs.
    """

    __slots__ = ("notifier", "recorder", "run_id", "iteration", "warnings")

    def __init__(
        self,
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
        run_id: str,
        iteration: int,
        warnings: list[str],
    ) -> None:
        self.recorder = recorder
        self.notifier = notifier
        self.run_id = run_id
        self.iteration = iteration
        self.warnings = warnings

    async def __call__(self, event: str, data: dict[str, Any], call_id: str | None) -> None:
        await _record_governance(
            self.recorder, self.run_id, event, self.iteration, data, correlation_id=call_id
        )
        await _notify_governance(
            self.notifier,
            self.run_id,
            event,
            self.iteration,
            data,
            correlation_id=call_id,
            warnings=self.warnings,
        )


class _ToolBatch:
    """The shared state of one ``_process_tool_calls`` batch and its writes.

    ``all_results`` and ``history`` grow in phase order. Every write keeps
    the order the stages always used: a rejected call gets its tool row,
    then its message; a skill preflight failure joins history first.
    """

    __slots__ = ("strategy", "history", "all_results", "governance")

    def __init__(
        self,
        *,
        strategy: Strategy,
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
        run_id: str,
        iteration: int,
        history: list[Message],
        warnings: list[str],
    ) -> None:
        self.strategy = strategy
        self.history = history
        self.all_results: list[tuple[ToolCall, ToolResult]] = []
        self.governance = _Governance(recorder, notifier, run_id, iteration, warnings)

    @property
    def recorder(self) -> LoopRecorder | None:
        return self.governance.recorder

    @property
    def notifier(self) -> LoopNotifier | None:
        return self.governance.notifier

    @property
    def run_id(self) -> str:
        return self.governance.run_id

    @property
    def iteration(self) -> int:
        return self.governance.iteration

    @property
    def warnings(self) -> list[str]:
        return self.governance.warnings

    async def restore_skipped(self, data: dict[str, Any], call_id: str) -> None:
        await self.governance(GovernanceEventType.GUARDRAIL_RESTORE_SKIPPED.value, data, call_id)

    async def tool_row(self, tc: ToolCall, result: ToolResult) -> None:
        await _record_tool(self.recorder, self.run_id, tc, result, self.iteration)
        await _notify_tool(self.notifier, self.run_id, tc, result, self.iteration, self.warnings)

    async def append_message(self, tc: ToolCall, result: ToolResult) -> None:
        """Append the result message, record then notify it, and keep the pair."""
        msg = _format_result(self.strategy, result)
        self.history.append(msg)
        await self._commit(tc, result, msg)

    async def reject(self, tc: ToolCall, result: ToolResult) -> None:
        """A call answered before execution: tool row, then message."""
        await self.tool_row(tc, result)
        await self.append_message(tc, result)

    async def reject_listed_first(self, tc: ToolCall, result: ToolResult) -> None:
        """Skill preflight order: the message joins history before the tool row."""
        msg = _format_result(self.strategy, result)
        self.history.append(msg)
        await self.tool_row(tc, result)
        await self._commit(tc, result, msg)

    async def _commit(self, tc: ToolCall, result: ToolResult, msg: Message) -> None:
        await _commit_message(
            self.history,
            msg,
            self.recorder,
            self.notifier,
            self.run_id,
            self.iteration,
            self.warnings,
        )
        self.all_results.append((tc, result))


def _bind_skill_session(lookups: ToolLookups, governance: _Governance) -> SkillSession | None:
    """The batch's skill session (None without skill tools), bound to this run and iteration."""
    session = next((fn.session for fn in lookups.fn.values() if isinstance(fn, SkillTool)), None)
    if session is not None:
        session.run_id = governance.run_id
        session.sink = governance
    return session


async def _json_failure(
    tc: ToolCall, lookups: ToolLookups, session: SkillSession | None
) -> ToolResult | None:
    """Invalid JSON arguments, or an invalid call of a skill tool (activations preflight)."""
    result = check_tool_json(tc)
    if result is not None or session is None:
        return result
    if not isinstance(lookups.fn.get(tc.name), SkillTool):
        return None
    result = check_tool_call(tc, lookups.defs.get(tc.name))
    if result is not None and tc.name == "use_skill":
        await session.preflight(tc)
    return result


async def _reject_invalid_json(
    calls: list[ToolCall],
    lookups: ToolLookups,
    session: SkillSession | None,
    batch: _ToolBatch,
) -> list[ToolCall]:
    """Stage 1. Runs before deny and approval: a call with no usable arguments
    must never reach a human as "{}", and the model needs the parse error
    (not a policy message) to repair it."""
    parsed: list[ToolCall] = []
    for tc in calls:
        result = await _json_failure(tc, lookups, session)
        if result is None:
            parsed.append(tc)
        else:
            await batch.reject(tc, result)
    return parsed


async def _apply_static_deny(
    calls: list[ToolCall], deny: frozenset[str], batch: _ToolBatch
) -> list[ToolCall]:
    """Stage 2. A denied call is not an execution: governance event and message, no tool row."""
    kept: list[ToolCall] = []
    for tc in calls:
        if tc.name not in deny:
            kept.append(tc)
            continue
        result, data = static_denial(tc.name, tc.id)
        await batch.governance(GovernanceEventType.POLICY_DENIED.value, data, tc.id)
        await _append_denial_message(tc, result, batch)
    return kept


async def _append_denial_message(tc: ToolCall, result: ToolResult, batch: _ToolBatch) -> None:
    msg = _format_result(batch.strategy, result)
    msg.meta[NOT_EXECUTED_META_KEY] = True
    batch.history.append(msg)
    await batch._commit(tc, result, msg)


async def _reject_unknown(
    calls: list[ToolCall], lookups: ToolLookups, batch: _ToolBatch
) -> list[ToolCall]:
    """Stage 3. Unknown tools are answered now, never hooked or approved."""
    known: list[ToolCall] = []
    for tc in calls:
        if tc.name in lookups.fn:
            known.append(tc)
        else:
            await batch.reject(tc, _unknown_tool_result(tc, lookups))
    return known


async def _restore_calls(
    engine: GuardrailEngine | None,
    calls: list[ToolCall],
    lookups: ToolLookups,
    batch: _ToolBatch,
) -> list[ToolCall]:
    """Stage 4. Real values for the PII placeholders, as the guardrail decides."""
    if engine is None:
        return calls
    for i, tc in enumerate(calls):
        if tc.params:
            params = await _restore_params(engine, tc, lookups, batch.restore_skipped)
            if params != tc.params:
                calls[i] = replace(tc, params=params)
    return calls


async def _validate_calls(
    calls: list[ToolCall],
    lookups: ToolLookups,
    placeholders: frozenset[str],
    batch: _ToolBatch,
) -> list[ToolCall]:
    """Stage 5. Validate, then coerce local tools to normalised arguments.

    Before hooks and approval: a human never approves a malformed call, and
    everything downstream sees exactly the arguments the function receives.
    """
    valid: list[ToolCall] = []
    for tc in calls:
        prepared = prepare_tool_call(
            tc, lookups.defs.get(tc.name), lookups.fn.get(tc.name), placeholders=placeholders
        )
        if isinstance(prepared, ToolResult):
            await batch.reject(tc, prepared)
        else:
            valid.append(prepared)
    return valid


async def _enforce_limits(
    calls: list[ToolCall],
    lookups: ToolLookups,
    call_counts: dict[str, int],
    batch: _ToolBatch,
) -> list[ToolCall]:
    """Stage 6. ``max_calls_per_run``, before hooks and approval (counts are read only)."""
    kept, over_limit = enforce_call_limits(
        calls, max_calls=lookups.max_calls, call_counts=call_counts
    )
    for tc, result in over_limit:
        await batch.reject(tc, result)
    return kept


async def _preflight_skills(
    calls: list[ToolCall],
    lookups: ToolLookups,
    session: SkillSession,
    approved: bool,
    batch: _ToolBatch,
) -> list[ToolCall]:
    """Stage 6b. Skill preflight before approval.

    Earlier valid activations in this batch may satisfy reads; execution
    rechecks against the activations that actually succeed.
    """
    ready: list[ToolCall] = []
    anticipated: set[str] = set()
    for tc in calls:
        is_skill = isinstance(lookups.fn.get(tc.name), SkillTool)
        failure = (
            await session.preflight(
                tc, requested=not approved, anticipated=anticipated, approved=approved
            )
            if is_skill
            else None
        )
        if failure is not None:
            await batch.reject_listed_first(tc, failure)
            continue
        ready.append(tc)
        if is_skill and tc.name == "use_skill":
            anticipated.add(tc.params["name"])
    return ready


def _static_gates(
    calls: list[ToolCall], require_approval: frozenset[str] | None
) -> dict[str, dict[str, str | None]]:
    approval_set = require_approval or frozenset()
    return {
        tc.id: {"source": "static", "reason": ApprovalReason.REQUIRES_APPROVAL.value}
        for tc in calls
        if tc.name in approval_set
    }


def _approval_data(
    tc: ToolCall,
    gate: dict[str, str | None],
    sent_params: dict[str, dict[str, Any]],
    pii_mapping: dict[str, str],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tool_name": tc.name,
        "call_id": tc.id,
        "reason": ApprovalReason.REQUIRES_APPROVAL.value,
    }
    if gate["source"] == "hook":
        data["reason"] = gate["reason"] or ApprovalReason.REQUIRES_APPROVAL.value
        data["source"] = "hook"
    # What is being approved: the call as the model sent it (placeholder
    # form, pre-coercion). Coercion is deterministic from these arguments,
    # and no restored value can appear.
    data["params"] = approval_view(sent_params.get(tc.id, _NO_SENT_PARAMS), pii_mapping)
    return data


async def _request_approvals(
    calls: list[ToolCall],
    gates: dict[str, dict[str, str | None]],
    sent_params: dict[str, dict[str, Any]],
    pii_mapping: dict[str, str],
    batch: _ToolBatch,
) -> None:
    """Stage 8. One ``approval.requested`` per gated call; the whole batch pauses."""
    for tc in calls:
        gate = gates.get(tc.id)
        if gate is not None:
            await batch.governance(
                GovernanceEventType.APPROVAL_REQUESTED.value,
                _approval_data(tc, gate, sent_params, pii_mapping),
                tc.id,
            )


def _split_targets(
    calls: list[ToolCall], lookups: ToolLookups
) -> tuple[list[ToolCall], list[ToolCall]]:
    """Stage 9. Server calls run now; the others pause for the client."""
    server_calls: list[ToolCall] = []
    pending_calls: list[ToolCall] = []
    for tc in calls:
        if lookups.target.get(tc.name, ToolTarget.SERVER) == ToolTarget.SERVER:
            server_calls.append(tc)
        else:
            pending_calls.append(tc)
    return server_calls, pending_calls


async def _dispatch_server_calls(
    calls: list[ToolCall],
    lookups: ToolLookups,
    batch: _ToolBatch,
    transform: ToolResultTransform | None,
    image_runtime: ImageRuntime | None,
) -> list[tuple[ToolCall, ToolResult]]:
    """Stage 10. Execute in groups; each call writes its tool row as it finishes."""
    executed: list[tuple[ToolCall, ToolResult]] = []
    for group, is_parallel in _build_execution_groups(calls, lookups.parallel):
        if is_parallel and len(group) > 1:
            executed.extend(
                await _execute_parallel_group(
                    group,
                    lookups,
                    batch.recorder,
                    batch.notifier,
                    batch.run_id,
                    batch.iteration,
                    batch.warnings,
                    transform,
                    image_runtime=image_runtime,
                )
            )
            continue
        for tc in group:
            executed.append(
                await _execute_record_notify(
                    tc,
                    lookups,
                    batch.recorder,
                    batch.notifier,
                    batch.run_id,
                    batch.iteration,
                    batch.warnings,
                    transform,
                    image_runtime=image_runtime,
                )
            )
    return executed


async def _record_executed(
    executed: list[tuple[ToolCall, ToolResult]],
    call_counts: dict[str, int],
    batch: _ToolBatch,
) -> None:
    """Stage 11. Messages in dispatch order (deterministic for the model); count executions."""
    for tc, result in executed:
        call_counts[tc.name] = call_counts.get(tc.name, 0) + 1
        await batch.append_message(tc, result)


def _pii_placeholders(engine: GuardrailEngine | None) -> dict[str, str]:
    return placeholder_mapping(engine.get_pii_mapping()) if engine is not None else {}


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
    image_runtime: ImageRuntime | None = None,
) -> _ToolCallOutcome:
    """Execute tool calls: enforce limits, run server tools, update history.

    Returns a _ToolCallOutcome with all results and any pending (non-server)
    calls. Both run() and run_stream() call this — the caller decides
    whether to yield events from the results.

    Side effects: mutates call_counts, appends to history, records then notifies.
    The stages run in this order: invalid JSON, static deny, unknown tool,
    restore, validate, call limits, skill preflight, hook, approval (the only
    early return), target split, dispatch, record.
    """
    batch = _ToolBatch(
        strategy=strategy,
        recorder=recorder,
        notifier=notifier,
        run_id=run_id,
        iteration=iteration,
        history=history,
        warnings=warnings,
    )
    session = _bind_skill_session(lookups, batch.governance)
    all_calls: list[ToolCall] = step.meta.get("all_tool_calls", [step.action])
    calls = await _reject_invalid_json(all_calls, lookups, session, batch)
    calls = await _apply_static_deny(calls, deny or frozenset(), batch)
    calls = await _reject_unknown(calls, lookups, batch)
    pii_mapping = _pii_placeholders(guardrail_engine)
    # The arguments as the model sent them (placeholder form, pre-restore and
    # pre-coercion), per call id: the only form approval views ever show.
    sent_params: dict[str, dict[str, Any]] = {tc.id: tc.params for tc in calls}
    calls = await _restore_calls(guardrail_engine, calls, lookups, batch)
    calls = await _validate_calls(calls, lookups, frozenset(pii_mapping), batch)
    calls = await _enforce_limits(calls, lookups, call_counts, batch)
    if session is not None:
        calls = await _preflight_skills(calls, lookups, session, approved_skills, batch)
    gates = _static_gates(calls, require_approval)
    if before_tool is not None:
        calls = await _apply_before_tool(
            before_tool,
            calls,
            gates,
            batch,
            lookups=lookups,
            tenant_id=tenant_id,
            image_runtime=image_runtime,
        )
    if gates:
        await _request_approvals(calls, gates, sent_params, pii_mapping, batch)
        return _ToolCallOutcome(
            all_results=batch.all_results,
            pending_calls=calls,
            pending_approval=True,
            approval_gates=gates,
        )
    server_calls, pending_calls = _split_targets(calls, lookups)
    executed = await _dispatch_server_calls(
        server_calls, lookups, batch, tool_result_transform, image_runtime
    )
    await _record_executed(executed, call_counts, batch)
    return _ToolCallOutcome(all_results=batch.all_results, pending_calls=pending_calls)


async def _hook_tenant_id(
    hook: BeforeToolHook | None, state_store: StateStore | None, run_id: str
) -> str | None:
    """The run's tenant for ``ToolCallContext`` (read once, only with a hook)."""
    if hook is None or state_store is None:
        return None
    record = await state_store.get_run(run_id)
    return record.tenant_id if record is not None else None


_HOOK_ERROR_REASON = DenialReason.HOOK_ERROR.value
_CLIENT_APPROVAL_REASON = DenialReason.CLIENT_APPROVAL_UNSUPPORTED.value


def _tool_source(name: str, lookups: ToolLookups) -> Literal["local", "mcp", "skill"]:
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
    images: tuple[ImageRef, ...] = (),
) -> tuple[ToolDecision | None, str | None]:
    """Run the hook on deep-copied inputs. Returns (decision, error_type).

    Only ``Exception`` is caught: cancellation and other ``BaseException``
    subclasses propagate. A non-``ToolDecision`` return is an error.
    """
    try:
        ctx = tool_call_context(
            run_id=run_id,
            call_id=tc.id,
            tool_name=tc.name,
            params=tc.params,
            target=lookups.target.get(tc.name, ToolTarget.SERVER),
            source=_tool_source(tc.name, lookups),
            meta=lookups.meta.get(tc.name, {}),
            iteration=iteration,
            tenant_id=tenant_id,
            images=images,
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
    batch: _ToolBatch,
    *,
    lookups: ToolLookups,
    tenant_id: str | None,
    image_runtime: ImageRuntime | None = None,
) -> list[ToolCall]:
    """Evaluate ``before_tool`` per call; returns the calls that continue.

    Denied and substituted calls get their result appended here. Hook
    ``require_approval`` adds a gate to ``gates`` (mutated in place).
    A hook deny overrides a static approval gate; nothing else does.
    """
    kept: list[ToolCall] = []
    run_images = _run_image_index(batch.history)
    for tc in calls:
        decision, error_type = await _call_before_tool(
            hook,
            tc,
            lookups,
            run_id=batch.run_id,
            iteration=batch.iteration,
            tenant_id=tenant_id,
            images=_referenced_images(tc.params, run_images),
        )
        if error_type is not None:
            await _hook_error(tc, error_type, gates, batch)
        elif await _apply_decision(decision, tc, gates, batch, lookups, image_runtime):
            kept.append(tc)
    return kept


async def _apply_decision(
    decision: ToolDecision | None,
    tc: ToolCall,
    gates: dict[str, dict[str, str | None]],
    batch: _ToolBatch,
    lookups: ToolLookups,
    image_runtime: ImageRuntime | None,
) -> bool:
    """Apply one hook decision; True when the call continues."""
    if decision is None or decision.kind == "allow":
        return True
    if decision.kind == "deny":
        await _hook_deny(tc, decision.reason or _HOOK_ERROR_REASON, gates, batch)
        return False
    if decision.kind == "require_approval":
        return await _hook_gate(decision, tc, gates, batch, lookups)
    if tc.id in gates:
        # Substitute on a statically gated call: the gate stays, a human decides.
        return True
    await _substitute(decision, tc, gates, batch, image_runtime)
    return False


async def _hook_gate(
    decision: ToolDecision,
    tc: ToolCall,
    gates: dict[str, dict[str, str | None]],
    batch: _ToolBatch,
    lookups: ToolLookups,
) -> bool:
    """Hook ``require_approval``: a server call gets a gate; a client call is denied."""
    if lookups.target.get(tc.name, ToolTarget.SERVER) != ToolTarget.SERVER:
        await batch.governance(
            GovernanceEventType.POLICY_HOOK_WARNING.value,
            {"tool_name": tc.name, "call_id": tc.id, "warning": _CLIENT_APPROVAL_REASON},
            tc.id,
        )
        await _hook_deny(tc, _CLIENT_APPROVAL_REASON, gates, batch)
        return False
    if tc.id not in gates:
        gates[tc.id] = {"source": "hook", "reason": decision.reason}
    return True


async def _hook_deny(
    tc: ToolCall, reason: str, gates: dict[str, dict[str, str | None]], batch: _ToolBatch
) -> None:
    gates.pop(tc.id, None)
    await batch.governance(
        GovernanceEventType.POLICY_DENIED.value,
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
    await _append_denial_message(tc, result, batch)


async def _hook_error(
    tc: ToolCall, error_type: str, gates: dict[str, dict[str, str | None]], batch: _ToolBatch
) -> None:
    await batch.governance(
        GovernanceEventType.POLICY_HOOK_ERROR.value,
        {"tool_name": tc.name, "call_id": tc.id, "error_type": error_type},
        tc.id,
    )
    await _hook_deny(tc, _HOOK_ERROR_REASON, gates, batch)


async def _substitute(
    decision: ToolDecision,
    tc: ToolCall,
    gates: dict[str, dict[str, str | None]],
    batch: _ToolBatch,
    image_runtime: ImageRuntime | None,
) -> None:
    """Hook ``substitute``: governance, tool row, message; a bad result denies the call."""
    value = decision.result
    try:
        payload = json.dumps(value.result if isinstance(value, ToolOutput) else value)
    except (TypeError, ValueError) as exc:
        await _hook_error(tc, type(exc).__name__, gates, batch)
        return
    sub_images: tuple[ImagePart, ...] = ()
    if isinstance(value, ToolOutput) and value.images:
        try:
            sub_images = await _ingest_tool_images(value.images, tc.name, image_runtime)
        except Exception as exc:
            # Fails closed like a serialisation error: the call is denied.
            await _hook_error(tc, type(exc).__name__, gates, batch)
            return
    substituted = ToolResult(
        name=tc.name,
        call_id=tc.id,
        payload=payload,
        success=True,
        meta={"substituted": True},
        images=sub_images,
    )
    await batch.governance(
        GovernanceEventType.POLICY_SUBSTITUTED.value,
        {"tool_name": tc.name, "call_id": tc.id},
        tc.id,
    )
    await batch.tool_row(tc, substituted)
    await batch.append_message(tc, substituted)


def _run_image_index(history: list[Message]) -> dict[str, list[ImageRef]]:
    """``image_id`` -> distinct refs of this run's image slots, in slot order."""
    index: dict[str, list[ImageRef]] = {}
    for msg in history:
        for part in msg.images:
            refs = index.setdefault(part.ref.image_id, [])
            if part.ref not in refs:
                refs.append(part.ref)
    return index


def _referenced_images(params: Any, run_images: dict[str, list[ImageRef]]) -> tuple[ImageRef, ...]:
    """Refs whose ``img_...`` id is a whole string value anywhere in ``params`` (OQ-9)."""
    if not run_images:
        return ()
    found: set[str] = set()

    def walk(value: Any, depth: int) -> None:
        if depth > _IMAGE_REFERENCE_MAX_DEPTH:
            return
        if isinstance(value, str):
            if value in run_images and IMAGE_ID_RE.fullmatch(value):
                found.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)

    walk(params, 0)
    out: list[ImageRef] = []
    for image_id, refs in run_images.items():
        if image_id in found:
            out.extend(ref for ref in refs if ref not in out)
    return tuple(out)


async def _ingest_tool_images(
    images: Sequence[Any],
    tool_name: str,
    image_runtime: ImageRuntime | None,
) -> tuple[ImagePart, ...]:
    """Strict ingest of a local tool's / substitute's ``ToolOutput`` images."""
    if image_runtime is None:
        raise ImagePersistenceUnsupportedError("loop")
    return await ingest(
        tuple(images),
        policy=image_runtime.policy,
        runtime=image_runtime,
        origin="tool",
        tool_name=tool_name,
    )


async def _ingest_mcp_images(
    images: Sequence[Any],
    tool_name: str,
    image_runtime: ImageRuntime | None,
) -> tuple[tuple[ImagePart, ...], tuple[str, ...]]:
    """Lenient ingest of MCP images: failures become notes, the call still succeeds (N-11)."""
    if image_runtime is None:
        raise ImagePersistenceUnsupportedError("loop")
    outcomes = await ingest(
        tuple(images),
        policy=image_runtime.policy,
        runtime=image_runtime,
        origin="mcp",
        tool_name=tool_name,
        lenient=True,
        collect_errors=True,
    )
    parts: list[ImagePart] = []
    notes: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, ImageError):
            reason = getattr(outcome, "reason", None) or getattr(outcome, "scope", None)
            notes.append(f"[image from MCP could not be used: {reason or 'invalid'}]")
        else:
            parts.append(outcome)
    return tuple(parts), tuple(notes)


def _contains_image(value: Any, depth: int = 0) -> bool:
    if isinstance(value, (Image, ImageRef, ImagePart)):
        return True
    if depth > _IMAGE_SEARCH_MAX_DEPTH:
        return False
    if isinstance(value, dict):
        return any(_contains_image(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_image(v, depth + 1) for v in value)
    return False


def _image_policy(agent: Any, image_runtime: ImageRuntime | None) -> ImagePolicy:
    if image_runtime is not None:
        return image_runtime.policy
    policy = getattr(agent, "image_policy", None)
    return policy if isinstance(policy, ImagePolicy) else ImagePolicy()


def _mark_call_rejections(
    history: list[Message], start: int, results: list[tuple[ToolCall, ToolResult]]
) -> None:
    """Keep generated cap rejections out of execution counts after a resume."""
    rejected = {tc.id for tc, result in results if result.meta.get("code") == "max_calls"}
    if rejected:
        for message in history[start:]:
            if message.role == Role.TOOL and message.call_id in rejected:
                message.meta[NOT_EXECUTED_META_KEY] = True


@dataclass(slots=True)
class _RunContext:
    """Per-run state shared by ``ReActLoop.run`` and ``run_stream`` and their helpers."""

    loop: Loop
    agent: Agent
    provider: LLMProvider
    strategy: Strategy
    run_id: str
    recorder: LoopRecorder | None
    notifier: LoopNotifier | None
    state_store: StateStore | None
    interrupt: asyncio.Event | None
    image_runtime: ImageRuntime | None
    pkw: dict[str, Any]
    cache_key_prefix: str | None
    lookups: ToolLookups
    tool_defs: list[ToolDef]
    state: _LoopState
    images: ImageCallSite
    budget: BudgetTracker
    before_tool: BeforeToolHook | None
    hook_tenant_id: str | None
    calls: ProviderCallSite
    g_engine: GuardrailEngine | None = None

    def meta(self) -> dict[str, Any]:
        """Terminal result meta: the PII mapping (batch only) and notifier warnings."""
        return guardrail_meta(self.g_engine, self.state.warnings)

    def pause_view(self) -> ImageViewState | None:
        return pause_image_view(self.state.image_view, self.state.history)

    async def control_reader(self) -> Any:
        """Run control for the image-retry checkpoint.

        An in-process cancel (``cancel_run`` in this process) wins, as at the
        top-of-iteration checkpoint.
        """
        if self.interrupt is not None and self.interrupt.is_set():
            return _LegacyControl(cancel_requested=True)
        if self.state_store is None:
            return await no_control()
        return await _read_control(self.state_store, self.run_id)


async def _start_run(
    loop: Loop,
    agent: Agent,
    provider: LLMProvider,
    strategy: Strategy,
    user_input: str,
    *,
    run_id: str | None,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    initial_history: list[Message] | None,
    initial_steps: list[AgentStep] | None,
    iteration_offset: int,
    initial_usage: UsageStats | None,
    _initial_call_counts: dict[str, int] | None,
    provider_kwargs: dict[str, Any] | None,
    state_store: StateStore | None,
    budget_carry: BudgetCarry | None,
    image_runtime: ImageRuntime | None,
    image_view_state: ImageViewState | None,
    interrupt: asyncio.Event | None,
) -> _RunContext:
    """Shared setup of ``run`` and ``run_stream``: ids, tools, loop state, images, budget, hook."""
    resolved_run_id = run_id or generate_ulid()
    pkw = provider_kwargs or {}
    cache_key_prefix = build_cache_key_prefix(agent)
    lookups = await agent.get_tool_lookups(loop=loop)
    tool_defs = agent.get_all_tool_defs(loop=loop)
    state = await _init_loop_state(
        user_input=user_input,
        recorder=recorder,
        notifier=notifier,
        run_id=resolved_run_id,
        initial_history=initial_history,
        initial_steps=initial_steps,
        initial_usage=initial_usage,
        _initial_call_counts=_initial_call_counts,
        image_view_state=image_view_state,
    )
    images = ImageCallSite(
        state=state.image_view,
        policy=_image_policy(agent, image_runtime),
        runtime=image_runtime,
        provider=provider,
        model=str(pkw.get("model") or provider.model),
        recorder=recorder,
        notifier=notifier,
        run_id=resolved_run_id,
        warnings=state.warnings,
    )
    budget = BudgetTracker(
        agent.budget,
        run_id=resolved_run_id,
        usage=state.usage,
        recorder=recorder,
        notifier=notifier,
        warnings=state.warnings,
        state_store=state_store,
        carry=budget_carry,
        fresh=iteration_offset == 0,
    )
    before_tool: BeforeToolHook | None = getattr(agent, "before_tool", None)
    hook_tenant_id = await _hook_tenant_id(before_tool, state_store, resolved_run_id)
    return _RunContext(
        loop=loop,
        agent=agent,
        provider=provider,
        strategy=strategy,
        run_id=resolved_run_id,
        recorder=recorder,
        notifier=notifier,
        state_store=state_store,
        interrupt=interrupt,
        image_runtime=image_runtime,
        pkw=pkw,
        cache_key_prefix=cache_key_prefix,
        lookups=lookups,
        tool_defs=tool_defs,
        state=state,
        images=images,
        budget=budget,
        before_tool=before_tool,
        hook_tenant_id=hook_tenant_id,
        calls=ProviderCallSite(
            images,
            agent=agent,
            provider=provider,
            pkw=pkw,
            recorder=recorder,
            notifier=notifier,
            run_id=resolved_run_id,
            warnings=state.warnings,
        ),
    )


def _bind_run_checkpoint(ctx: _RunContext) -> None:
    _bind_checkpoint(
        ctx.recorder,
        state=ctx.state,
        budget=ctx.budget,
        provider_kwargs=ctx.pkw,
        tool_defs=ctx.tool_defs,
        strategy=ctx.strategy,
        engine=ctx.g_engine,
        agent=ctx.agent,
    )


async def _gate_control(ctx: _RunContext) -> Any:
    """The checkpoint's run control. An in-process interrupt is free and is
    checked first: it reads as a cancel without a store read."""
    if ctx.interrupt is not None and ctx.interrupt.is_set():
        return _LegacyControl(cancel_requested=True)
    if ctx.state_store is None:
        return None
    return await _read_control(ctx.state_store, ctx.run_id)


def _control_gate_result(ctx: _RunContext, control: Any, iteration: int) -> RunResult | None:
    """CANCELLED or PAUSED for a cancel or a requested pause, else None (cancel wins)."""
    if control is None:
        return None
    if control.cancel_requested:
        return RunResult(
            run_id=ctx.run_id,
            status=RunStatus.CANCELLED,
            steps=ctx.state.steps,
            commentary=_commentary(ctx.state.steps),
            iteration_count=iteration - 1,
            usage=ctx.state.usage,
            meta=ctx.meta(),
        )
    if control.pause_requested:
        return _checkpoint_pause_result(
            agent_name=ctx.agent.name,
            run_id=ctx.run_id,
            history=ctx.state.history,
            steps=ctx.state.steps,
            iteration=iteration,
            usage=ctx.state.usage,
            meta=ctx.meta(),
            budget=ctx.budget,
            image_view=ctx.pause_view(),
        )
    return None


async def _iteration_gate(ctx: _RunContext, iteration: int) -> RunResult | None:
    """Cooperative checkpoint before an iteration starts.

    Cancel > requested pause > budget pause/cancel; then inbox notes are
    delivered. The current iteration's LLM call and tools are not preempted:
    flags are seen here or at the runner's pre-pause checkpoint.
    """
    control = await _gate_control(ctx)
    stopped = _control_gate_result(ctx, control, iteration)
    if stopped is not None:
        return stopped
    budget_result = await _budget_stop(
        ctx.budget,
        agent_name=ctx.agent.name,
        run_id=ctx.run_id,
        history=ctx.state.history,
        steps=ctx.state.steps,
        iteration=iteration,
        usage=ctx.state.usage,
        meta=ctx.meta(),
        image_view=ctx.pause_view,
    )
    if budget_result is not None:
        return budget_result
    if control is not None and control.has_pending_notes:
        assert ctx.state_store is not None
        await _deliver_inbox(
            ctx.state_store,
            ctx.recorder,
            ctx.notifier,
            ctx.run_id,
            iteration,
            ctx.state.history,
            ctx.state.warnings,
        )
    return None


def _stop_event(result: RunResult) -> RunEvent:
    """The stream event for a checkpoint stop: cancelled or paused."""
    return RunEvent(
        type=(
            RunEventType.RUN_CANCELLED
            if result.status == RunStatus.CANCELLED
            else RunEventType.RUN_PAUSED
        ),
        run_result=result,
    )


class _StreamDeltas:
    """ReAct's events for a streamed call; text after a tool block is known commentary."""

    __slots__ = ("iteration", "saw_tool_use")

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration
        self.saw_tool_use = False

    def __call__(self, event: StreamEvent) -> RunEvent | None:
        if event.type == StreamEventType.TEXT_DELTA:
            return RunEvent(
                type=RunEventType.TEXT_DELTA,
                text=event.text,
                iteration=self.iteration,
                phase=event.phase or ("commentary" if self.saw_tool_use else None),
            )
        if event.type == StreamEventType.REASONING_DELTA:
            return RunEvent(
                type=RunEventType.REASONING_DELTA, text=event.text, iteration=self.iteration
            )
        if event.type == StreamEventType.TOOL_USE_START:
            self.saw_tool_use = True
            return RunEvent(
                type=RunEventType.TOOL_USE_START,
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                iteration=self.iteration,
                phase="commentary",
            )
        if event.type == StreamEventType.TOOL_USE_END:
            self.saw_tool_use = True
            return RunEvent(
                type=RunEventType.TOOL_USE_END,
                tool_call=persistable_call(event.tool_call),
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                iteration=self.iteration,
                phase="commentary",
            )
        return None


async def _stream_call(
    ctx: _RunContext, messages: list[Message], tools: Any, iteration: int
) -> StreamCall:
    """The iteration's streamed provider call over its image request view (§5.1).

    Image guardrails are batch-only: stream() rejects guardrails at the Agent level.
    """
    return StreamCall(
        ctx.calls,
        await ctx.images.prepare(messages, iteration=iteration),
        messages,
        tools=tools,
        iteration=iteration,
        interrupt=ctx.interrupt,
        control_reader=ctx.control_reader,
        cache_key_prefix=ctx.cache_key_prefix,
        translate=_StreamDeltas(iteration),
        check=_require_done,
    )


def _call_end_event(ctx: _RunContext, call: StreamCall, iteration: int) -> RunEvent | None:
    """The terminal event of a streamed call that ended without a response, else None."""
    state = ctx.state
    if call.stop is not None:
        return _stop_event(
            _control_stop_result(
                call.stop,
                agent_name=ctx.agent.name,
                run_id=ctx.run_id,
                history=state.history,
                steps=state.steps,
                iteration=iteration,
                usage=state.usage,
                meta=guardrail_meta(None, state.warnings),
                budget=ctx.budget,
                image_view=pause_image_view(state.image_view, state.history),
            )
        )
    if call.interrupted:
        # Mid-stream interrupt (cancel_run reached this process). The
        # interrupted call reported no usage, so the usage covers completed
        # iterations only. Partial text becomes the answer.
        return RunEvent(
            type=RunEventType.RUN_CANCELLED,
            run_result=_interrupted_result(
                run_id=ctx.run_id,
                partial_text=call.partial_text,
                steps=state.steps,
                iteration=iteration,
                usage=state.usage,
                notifier_warnings=state.warnings,
            ),
        )
    return None


async def _answer_event(
    ctx: _RunContext, response: LLMResponse, action: Any, iteration: int
) -> RunEvent | None:
    """The stream's terminal event for a Finish or a Clarification, else None."""
    if isinstance(action, Finish):
        return RunEvent(
            type=RunEventType.RUN_COMPLETED,
            phase="final_answer",
            run_result=await _finish_result(ctx, response, action, iteration),
        )
    if isinstance(action, Clarification):
        return RunEvent(
            type=RunEventType.RUN_PAUSED,
            run_result=await _clarification_result(ctx, response, action, iteration),
        )
    return None


def _require_done(response: LLMResponse | None, iteration: int) -> None:
    if response is None:
        raise RuntimeError(
            f"Provider stream ended without DONE event at iteration "
            f"{iteration}. complete_stream() must yield "
            f"StreamEvent(type=DONE, raw=LLMResponse)."
        )


def _build_messages(ctx: _RunContext) -> tuple[list[Message], Any]:
    return ctx.strategy.build_messages(
        system_prompt=ctx.agent.get_system_prompt(loop=ctx.loop),
        history=ctx.state.history,
        tool_defs=ctx.tool_defs,
    )


def _llm_findings(
    ctx: _RunContext, prepared: PreparedRequest, base: dict[str, Any] | None
) -> dict[str, Any] | None:
    return None if base is None else ctx.images.findings(prepared, base)


async def _after_response(
    ctx: _RunContext,
    response: LLMResponse,
    *,
    prepared: PreparedRequest,
    tools: Any,
    iteration: int,
    duration_ms: int,
    findings: dict[str, Any] | None,
) -> tuple[AgentStep, RunResult | None]:
    """Record the response (raw: the DB is ground truth), count usage and budget, parse it.

    Returns the step and, past the pending-call limit, the ERROR result.
    ``findings`` is the batch's guardrail findings (``None`` on stream).
    """
    await _record_llm(
        ctx.recorder,
        ctx.run_id,
        response,
        iteration,
        semantic_messages=prepared.semantic,
        semantic_tools=tools,
        duration_ms=duration_ms,
        guardrail_findings=_llm_findings(ctx, prepared, findings),
    )
    await _notify_llm(
        ctx.notifier,
        ctx.run_id,
        response,
        iteration,
        ctx.state.warnings,
        semantic_messages=prepared.semantic,
        semantic_tools=tools,
        duration_ms=duration_ms,
        guardrail_findings=_llm_findings(ctx, prepared, findings),
    )
    _accumulate_usage(ctx.state.usage, response.usage)
    ctx.budget.add_response(response.usage)
    await ctx.budget.evaluate(iteration)
    step = ctx.strategy.parse_response(response)
    if response.reasoning is not None:
        step = replace(step, meta={**step.meta, "reasoning": response.reasoning})
    ctx.state.steps.append(step)
    limited = _pending_limit_result(
        agent=ctx.agent,
        state_store=ctx.state_store,
        response=response,
        run_id=ctx.run_id,
        iteration=iteration,
        steps=ctx.state.steps,
        usage=ctx.state.usage,
        meta=ctx.meta(),
    )
    return step, limited


async def _append_turn(
    ctx: _RunContext, response: LLMResponse, iteration: int, *, phase: str
) -> None:
    await _append_assistant(
        response,
        ctx.state.history,
        ctx.recorder,
        ctx.notifier,
        ctx.run_id,
        iteration,
        ctx.state.warnings,
        phase=phase,
    )


async def _finish_result(
    ctx: _RunContext, response: LLMResponse, action: Finish, iteration: int
) -> RunResult:
    await _append_turn(ctx, response, iteration, phase="final_answer")
    return RunResult(
        run_id=ctx.run_id,
        status=RunStatus.SUCCESS,
        answer=action.answer,
        steps=ctx.state.steps,
        commentary=_commentary(ctx.state.steps),
        iteration_count=iteration,
        usage=ctx.state.usage,
        meta=ctx.meta(),
        stop_reason=final_stop_reason(
            response, run_id=ctx.run_id, default_model=ctx.provider.model
        ),
    )


def _pause_result(
    ctx: _RunContext,
    status: RunStatus,
    iteration: int,
    *,
    pending_calls: list[ToolCall],
    approval_gates: dict[str, dict[str, str | None]] | None = None,
    answer: str | None = None,
) -> RunResult:
    pause = _build_pause(
        agent_name=ctx.agent.name,
        pending_calls=pending_calls,
        target_lookup=ctx.lookups.target,
        history=ctx.state.history,
        steps=ctx.state.steps,
        iteration=iteration,
        usage=ctx.state.usage,
        approval_gates=approval_gates,
        budget=ctx.budget,
        image_view=ctx.pause_view(),
    )
    return RunResult(
        run_id=ctx.run_id,
        status=status,
        answer=answer,
        steps=ctx.state.steps,
        commentary=_commentary(ctx.state.steps),
        iteration_count=iteration,
        usage=ctx.state.usage,
        meta={"pause_state": pause, **ctx.meta()},
    )


async def _clarification_result(
    ctx: _RunContext, response: LLMResponse, action: Clarification, iteration: int
) -> RunResult:
    await _append_turn(ctx, response, iteration, phase="final_answer")
    return _pause_result(
        ctx, RunStatus.WAITING_HUMAN_INPUT, iteration, pending_calls=[], answer=action.question
    )


async def _run_tool_step(
    ctx: _RunContext, response: LLMResponse, step: AgentStep, iteration: int
) -> tuple[_ToolCallOutcome, int]:
    """Append the assistant turn and run its tool calls; returns the outcome and
    where this iteration's tool messages start in history."""
    await _append_turn(ctx, response, iteration, phase="commentary")
    if ctx.state_store is not None:
        # The provider may have returned after sweep revoked this worker.
        await _read_control(ctx.state_store, ctx.run_id)
    tool_history_start = len(ctx.state.history)
    outcome = await _process_tool_calls(
        step=step,
        call_counts=ctx.state.call_counts,
        lookups=ctx.lookups,
        strategy=ctx.strategy,
        recorder=ctx.recorder,
        notifier=ctx.notifier,
        run_id=ctx.run_id,
        iteration=iteration,
        history=ctx.state.history,
        warnings=ctx.state.warnings,
        deny=ctx.agent.deny,
        require_approval=ctx.agent.require_approval,
        tool_result_transform=ctx.agent.tool_result_transform,
        # Always None on stream: streaming with guardrails is refused by the Agent.
        guardrail_engine=ctx.g_engine,
        before_tool=ctx.before_tool,
        tenant_id=ctx.hook_tenant_id,
        image_runtime=ctx.image_runtime,
    )
    return outcome, tool_history_start


def _after_tools(
    ctx: _RunContext, outcome: _ToolCallOutcome, tool_history_start: int, iteration: int
) -> RunResult | None:
    """Mark generated rejections, order observations; a pause result if calls are pending."""
    _mark_call_rejections(ctx.state.history, tool_history_start, outcome.all_results)
    order_observations(ctx.recorder, ctx.state.history)
    if not outcome.pending_calls:
        return None
    status = (
        RunStatus.WAITING_APPROVAL if outcome.pending_approval else RunStatus.WAITING_CLIENT_TOOL
    )
    return _pause_result(
        ctx,
        status,
        iteration,
        pending_calls=outcome.pending_calls,
        approval_gates=outcome.approval_gates,
    )


def _max_iterations_result(ctx: _RunContext) -> RunResult:
    return RunResult(
        run_id=ctx.run_id,
        status=RunStatus.MAX_ITERATIONS,
        steps=ctx.state.steps,
        commentary=_commentary(ctx.state.steps),
        iteration_count=ctx.agent.max_iterations,
        usage=ctx.state.usage,
        meta=ctx.meta(),
    )


def _run_guardrail_engine(
    agent: Agent, recorder: LoopRecorder | None, pii_mapping: dict[str, str] | None, run_id: str
) -> GuardrailEngine | None:
    """The recorder's engine (shared with persistence), else one for the agent's guardrails."""
    shared: GuardrailEngine | None = getattr(recorder, "guardrail_engine", None)
    if shared:
        return shared
    if not agent.guardrails:
        return None
    return GuardrailEngine(agent.guardrails, pii_mapping=pii_mapping, run_id=run_id)


async def _incoming_project(
    ctx: _RunContext, engine: GuardrailEngine, msg: Message, index: int
) -> tuple[Message, list[Any], str | None]:
    shared = getattr(ctx.recorder, "guardrail_engine", None) is engine
    project = engine.project_incoming_message if shared else engine.scan_incoming_message
    return await project(msg, index=index)


async def _incoming_blocked(
    ctx: _RunContext, error: str, findings: list[Any], msg: Message, index: int, iteration: int
) -> RunResult:
    await _record_governance(
        ctx.recorder,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_BLOCKED.value,
        iteration,
        _blocked_data("incoming", error, findings, _incoming_location(msg, index)),
    )
    await _notify_governance(
        ctx.notifier,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_BLOCKED.value,
        iteration,
        _blocked_data("incoming", error, findings, _incoming_location(msg, index)),
        warnings=ctx.state.warnings,
    )
    return _guardrail_error(ctx, error, iteration)


def _guardrail_error(ctx: _RunContext, error: str, iteration: int) -> RunResult:
    return RunResult(
        run_id=ctx.run_id,
        status=RunStatus.ERROR,
        error=error,
        steps=ctx.state.steps,
        commentary=_commentary(ctx.state.steps),
        iteration_count=iteration,
        usage=ctx.state.usage,
        meta={**ctx.meta(), "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value},
    )


async def _incoming_events(
    ctx: _RunContext,
    findings: list[Any],
    redacted: bool,
    iteration: int,
    all_findings: dict[str, Any],
) -> None:
    """``guardrail.detected`` and ``guardrail.redacted`` for the incoming scan."""
    if findings:
        entities = list({f.entity_type for f in findings})
        await _record_governance(
            ctx.recorder,
            ctx.run_id,
            GovernanceEventType.GUARDRAIL_DETECTED.value,
            iteration,
            {"direction": "incoming", "findings_count": len(findings), "entities": entities},
        )
        await _notify_governance(
            ctx.notifier,
            ctx.run_id,
            GovernanceEventType.GUARDRAIL_DETECTED.value,
            iteration,
            {"direction": "incoming", "findings_count": len(findings), "entities": entities},
            warnings=ctx.state.warnings,
        )
        all_findings["incoming"] = [
            {"entity_type": f.entity_type, "score": f.score} for f in findings
        ]
    if redacted:
        red_entities = list({f.entity_type for f in findings})
        await _record_governance(
            ctx.recorder,
            ctx.run_id,
            GovernanceEventType.GUARDRAIL_REDACTED.value,
            iteration,
            {"direction": "incoming", "entities": red_entities},
        )
        await _notify_governance(
            ctx.notifier,
            ctx.run_id,
            GovernanceEventType.GUARDRAIL_REDACTED.value,
            iteration,
            {"direction": "incoming", "entities": red_entities},
            warnings=ctx.state.warnings,
        )


async def _scan_incoming(
    ctx: _RunContext, messages: list[Message], iteration: int, all_findings: dict[str, Any]
) -> RunResult | None:
    """Incoming guardrail (batch only): scan every message before the LLM call.

    Redacted messages replace their originals in ``messages``; a block ends
    the run with an ERROR result.
    """
    engine = ctx.g_engine
    if engine is None:
        return None
    all_in_findings: list[Any] = []
    redacted = False
    for index, msg in enumerate(messages):
        if not msg.content:
            continue
        cleaned, findings, block_error = await _incoming_project(ctx, engine, msg, index)
        if findings:
            all_in_findings.extend(findings)
        if block_error is not None:
            return await _incoming_blocked(ctx, block_error, findings, msg, index, iteration)
        if cleaned is not msg:
            redacted = True
            # The engine's replace() preserves the origin envelope
            # (kind/placement/source) through redaction.
            messages[index] = cleaned
    await _incoming_events(ctx, all_in_findings, redacted, iteration, all_findings)
    return None


async def _outgoing_detected(
    ctx: _RunContext, findings: list[Any], iteration: int, all_findings: dict[str, Any]
) -> None:
    all_findings["outgoing"] = [{"entity_type": f.entity_type, "score": f.score} for f in findings]
    entities = list({f.entity_type for f in findings})
    await _record_governance(
        ctx.recorder,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_DETECTED.value,
        iteration,
        {"direction": "outgoing", "findings_count": len(findings), "entities": entities},
    )
    await _notify_governance(
        ctx.notifier,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_DETECTED.value,
        iteration,
        {"direction": "outgoing", "findings_count": len(findings), "entities": entities},
        warnings=ctx.state.warnings,
    )


async def _outgoing_blocked(
    ctx: _RunContext,
    response: LLMResponse,
    *,
    prepared: PreparedRequest,
    tools: Any,
    iteration: int,
    duration_ms: int,
    all_findings: dict[str, Any],
    error: str,
    findings: list[Any],
) -> RunResult:
    """Record the scrubbed response, count it, then ``guardrail.blocked`` and ERROR."""
    # Scrub before recording: a block means the original text/params
    # contain the PII that triggered it.
    blocked_response = LLMResponse(
        text="[blocked by guardrail]",
        tool_calls=None,
        raw=response.raw,
        usage=response.usage,
        provider_request=response.provider_request,
        provider_response=None,
        stop_reason=response.stop_reason,
        raw_stop_reason=response.raw_stop_reason,
    )
    await _record_llm(
        ctx.recorder,
        ctx.run_id,
        blocked_response,
        iteration,
        semantic_messages=prepared.semantic,
        semantic_tools=tools,
        duration_ms=duration_ms,
        guardrail_findings=ctx.images.findings(prepared, all_findings),
    )
    await _notify_llm(
        ctx.notifier,
        ctx.run_id,
        blocked_response,
        iteration,
        ctx.state.warnings,
        semantic_messages=prepared.semantic,
        semantic_tools=tools,
        duration_ms=duration_ms,
        guardrail_findings=ctx.images.findings(prepared, all_findings),
    )
    _accumulate_usage(ctx.state.usage, response.usage)
    ctx.budget.add_response(response.usage)
    await ctx.budget.evaluate(iteration)
    await _record_governance(
        ctx.recorder,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_BLOCKED.value,
        iteration,
        _blocked_data("outgoing", error, findings, "output"),
    )
    await _notify_governance(
        ctx.notifier,
        ctx.run_id,
        GovernanceEventType.GUARDRAIL_BLOCKED.value,
        iteration,
        _blocked_data("outgoing", error, findings, "output"),
        warnings=ctx.state.warnings,
    )
    return _guardrail_error(ctx, error, iteration)


async def _scan_outgoing(
    ctx: _RunContext,
    response: LLMResponse,
    *,
    prepared: PreparedRequest,
    tools: Any,
    iteration: int,
    duration_ms: int,
    all_findings: dict[str, Any],
) -> RunResult | None:
    """Output guardrail (batch only): detection, and block/warn policies.

    Persistence stores the raw response (the DB is ground truth); the next
    incoming scan redacts for the LLM.
    """
    engine = ctx.g_engine
    if engine is None or not (response.text or response.tool_calls):
        return None
    tc_params = [tc.params for tc in response.tool_calls] if response.tool_calls else None
    out_findings, out_block = await engine.scan_outgoing(response.text or "", tc_params)
    if out_findings:
        await _outgoing_detected(ctx, out_findings, iteration, all_findings)
    if out_block is None:
        return None
    return await _outgoing_blocked(
        ctx,
        response,
        prepared=prepared,
        tools=tools,
        iteration=iteration,
        duration_ms=duration_ms,
        all_findings=all_findings,
        error=out_block,
        findings=out_findings,
    )


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
        _initial_call_counts: dict[str, int] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
        initial_pii_mapping: dict[str, str] | None = None,
        state_store: StateStore | None = None,
        budget_carry: BudgetCarry | None = None,
        image_runtime: ImageRuntime | None = None,
        image_view_state: ImageViewState | None = None,
    ) -> RunResult:
        """Execute the ReAct loop, optionally resuming from a pause.

        ``budget_carry`` restores budget bookkeeping (fired keys) from the
        pause being resumed. ``image_runtime`` is the run's image runtime
        (tenant scope, catalog, stores); ``image_view_state`` restores the
        request-view state from a pause or seeds it for a new run.
        """
        ctx = await _start_run(
            self,
            agent,
            provider,
            strategy,
            user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            iteration_offset=iteration_offset,
            initial_usage=initial_usage,
            _initial_call_counts=_initial_call_counts,
            provider_kwargs=provider_kwargs,
            state_store=state_store,
            budget_carry=budget_carry,
            image_runtime=image_runtime,
            image_view_state=image_view_state,
            interrupt=None,
        )
        resolved_run_id, _pkw, cache_key_prefix = ctx.run_id, ctx.pkw, ctx.cache_key_prefix
        history, steps, total_usage = ctx.state.history, ctx.state.steps, ctx.state.usage
        notifier_warnings, image_state = ctx.state.warnings, ctx.state.image_view
        images, budget, _control_reader = ctx.images, ctx.budget, ctx.control_reader
        calls = ctx.calls
        g_engine = _run_guardrail_engine(agent, recorder, initial_pii_mapping, resolved_run_id)
        ctx.g_engine = g_engine
        _bind_run_checkpoint(ctx)
        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            stopped = await _iteration_gate(ctx, iteration)
            if stopped is not None:
                return stopped

            messages, tools = _build_messages(ctx)
            _all_guardrail_findings: dict[str, Any] = {}
            blocked = await _scan_incoming(ctx, messages, iteration, _all_guardrail_findings)
            if blocked is not None:
                return blocked

            await _checkpoint_view(recorder, messages, iteration)

            # Image request view (§5.1): placeholders, caps, verified bytes.
            prepared = await images.prepare(
                messages, iteration=iteration, guardrails=agent.guardrails or ()
            )
            if prepared.block_error is not None:
                return await self._image_blocked(
                    prepared.block_error,
                    images=images,
                    run_id=resolved_run_id,
                    iteration=iteration,
                    steps=steps,
                    usage=total_usage,
                    meta=guardrail_meta(g_engine, notifier_warnings),
                )

            # LLM call — batch (lifecycle: started → completed/failed).
            # An image rejection may retry the call once (§5.7).
            retried = False
            while True:
                await _record_llm_started(
                    recorder,
                    resolved_run_id,
                    iteration,
                    semantic_messages=prepared.semantic,
                    semantic_tools=tools,
                )
                await _notify_llm_started(
                    notifier,
                    resolved_run_id,
                    iteration,
                    notifier_warnings,
                    semantic_messages=prepared.semantic,
                    semantic_tools=tools,
                )
                t0 = time.monotonic()
                failure: Exception | None = None
                try:
                    with telemetry_context(
                        run_id=resolved_run_id,
                        iteration=iteration,
                        sinks=governance_sinks(recorder, notifier, resolved_run_id),
                    ):
                        async with lease_scope(recorder):
                            response = await complete_call(
                                provider,
                                prepared.wire,
                                tools=tools,
                                run_id=resolved_run_id,
                                cache_key_prefix=cache_key_prefix,
                                **_pkw,
                            )
                except Exception as _llm_exc:
                    await calls.record_failure(_llm_exc, t0, iteration)
                    if not prepared.slots:
                        raise
                    failure = _llm_exc
                    prepared = release_wire(prepared)
                if failure is None:
                    break
                stop, error, decision = await calls.decide(
                    failure,
                    prepared,
                    iteration,
                    retried=retried,
                    visible_deltas=0,
                    control_reader=_control_reader,
                )
                del failure
                if stop is not None:
                    return _control_stop_result(
                        stop,
                        agent_name=agent.name,
                        run_id=resolved_run_id,
                        history=history,
                        steps=steps,
                        iteration=iteration,
                        usage=total_usage,
                        meta=guardrail_meta(g_engine, notifier_warnings),
                        budget=budget,
                        image_view=pause_image_view(image_state, history),
                    )
                if error is not None:
                    raise error
                retried = True
                prepared = await images.prepare(
                    messages,
                    iteration=iteration,
                    guardrails=agent.guardrails or (),
                    force_tombstones=decision.tombstones,
                )
                if prepared.block_error is not None:
                    return await self._image_blocked(
                        prepared.block_error,
                        images=images,
                        run_id=resolved_run_id,
                        iteration=iteration,
                        steps=steps,
                        usage=total_usage,
                        meta=guardrail_meta(g_engine, notifier_warnings),
                    )
            prepared, llm_duration_ms, response = calls.settle(prepared, response, t0)
            blocked = await _scan_outgoing(
                ctx,
                response,
                prepared=prepared,
                tools=tools,
                iteration=iteration,
                duration_ms=llm_duration_ms,
                all_findings=_all_guardrail_findings,
            )
            if blocked is not None:
                return blocked

            step, limited = await _after_response(
                ctx,
                response,
                prepared=prepared,
                tools=tools,
                iteration=iteration,
                duration_ms=llm_duration_ms,
                findings=_all_guardrail_findings,
            )
            if limited is not None:
                return limited
            if isinstance(step.action, Finish):
                return await _finish_result(ctx, response, step.action, iteration)
            if isinstance(step.action, Clarification):
                return await _clarification_result(ctx, response, step.action, iteration)
            if isinstance(step.action, ToolCall):
                outcome, tool_history_start = await _run_tool_step(ctx, response, step, iteration)
                paused = _after_tools(ctx, outcome, tool_history_start, iteration)
                if paused is not None:
                    return paused

        return _max_iterations_result(ctx)

    @staticmethod
    async def _image_blocked(
        error: str,
        *,
        images: ImageCallSite,
        run_id: str,
        iteration: int,
        steps: list[AgentStep],
        usage: UsageStats,
        meta: dict[str, Any],
    ) -> RunResult:
        """ERROR result for an image guardrail ``block`` (commit step 4)."""
        await images.emit(
            GovernanceEventType.GUARDRAIL_BLOCKED.value,
            _image_block_data(error, images.state),
            iteration,
        )
        return RunResult(
            run_id=run_id,
            status=RunStatus.ERROR,
            error=error,
            steps=steps,
            commentary=_commentary(steps),
            iteration_count=iteration,
            usage=usage,
            meta={**meta, "failure_reason": FailureReason.GUARDRAIL_BLOCKED.value},
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
        _initial_call_counts: dict[str, int] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
        state_store: StateStore | None = None,
        interrupt: asyncio.Event | None = None,
        budget_carry: BudgetCarry | None = None,
        image_runtime: ImageRuntime | None = None,
        image_view_state: ImageViewState | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream the ReAct loop as RunEvents.

        Same iteration logic as run() — shared helpers ensure consistency.
        Swaps provider.complete() for complete_stream(), forwarding text
        deltas live and yielding TOOL_RESULT events after execution.
        Terminal event is RUN_COMPLETED or RUN_PAUSED.
        """
        ctx = await _start_run(
            self,
            agent,
            provider,
            strategy,
            user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            iteration_offset=iteration_offset,
            initial_usage=initial_usage,
            _initial_call_counts=_initial_call_counts,
            provider_kwargs=provider_kwargs,
            state_store=state_store,
            budget_carry=budget_carry,
            image_runtime=image_runtime,
            image_view_state=image_view_state,
            interrupt=interrupt,
        )
        _bind_run_checkpoint(ctx)
        start_iteration = iteration_offset + 1
        end_iteration = agent.max_iterations + 1
        for iteration in range(start_iteration, end_iteration):
            # Cooperative checkpoint (see run()). The in-process interrupt is
            # checked first: it is free, and it lets a cancel that landed
            # during tool execution stop the run before the next provider
            # call is even started.
            stopped = await _iteration_gate(ctx, iteration)
            if stopped is not None:
                yield _stop_event(stopped)
                return

            messages, tools = _build_messages(ctx)
            await _checkpoint_view(recorder, messages, iteration)

            # LLM call — streaming (lifecycle: started → completed/failed). The
            # call is driven here, in the consumer's task, and closed with it.
            call = await _stream_call(ctx, messages, tools, iteration)
            relay = EventRelay(stream_call_events(call))
            async with aclosing(relay.events):
                while (event := await relay.next()) is not None:
                    try:
                        yield event
                    except Exception as thrown:
                        relay.throw(thrown)
            ended = _call_end_event(ctx, call, iteration)
            if ended is not None:
                yield ended
                return
            # The call raised if no DONE arrived; only a stop or an interrupt
            # (handled above) leaves this None.
            assert call.response is not None
            prepared, llm_duration_ms, llm_response = ctx.calls.settle(
                call.prepared, call.response, call.t0
            )
            # The call still holds the byte-bearing request; only the released one goes on.
            del call
            step, limited = await _after_response(
                ctx,
                llm_response,
                prepared=prepared,
                tools=tools,
                iteration=iteration,
                duration_ms=llm_duration_ms,
                findings=None,
            )
            if limited is not None:
                yield RunEvent(type=RunEventType.RUN_ERROR, run_result=limited)
                return

            answered = await _answer_event(ctx, llm_response, step.action, iteration)
            if answered is not None:
                yield answered
                return

            if isinstance(step.action, ToolCall):
                outcome, tool_history_start = await _run_tool_step(
                    ctx, llm_response, step, iteration
                )
                for tc, tool_result in outcome.all_results:
                    yield RunEvent(
                        type=RunEventType.TOOL_RESULT,
                        tool_call=tc,
                        tool_result=tool_result,
                        iteration=iteration,
                        phase="commentary",
                    )
                paused = _after_tools(ctx, outcome, tool_history_start, iteration)
                if paused is not None:
                    yield RunEvent(type=RunEventType.RUN_PAUSED, run_result=paused)
                    return

        yield RunEvent(type=RunEventType.RUN_COMPLETED, run_result=_max_iterations_result(ctx))


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
    *,
    image_runtime: ImageRuntime | None = None,
    tracker: CallTracker | None = None,
) -> tuple[ToolCall, ToolResult]:
    """Execute a tool, record, then notify.

    In a parallel batch each tool records and streams its event as soon as
    it finishes, without waiting for siblings.
    """
    await _record_tool_started(recorder, run_id, tool_call, iteration)
    await _notify_tool_started(notifier, run_id, tool_call, iteration, notifier_warnings)
    async with lease_scope(recorder):
        result = await _execute_tool(
            tool_call,
            lookups,
            transform,
            image_runtime=image_runtime,
            tracker=tracker,
            run_id=run_id,
        )
    if tracker is not None:
        tracker.result = result
    await _record_tool(recorder, run_id, tool_call, result, iteration)
    if tracker is not None:
        tracker.recorded = True
    await _notify_tool(notifier, run_id, tool_call, result, iteration, notifier_warnings)
    return tool_call, result


_SIBLING_CANCELLED = "cancelled: another tool call in the same batch failed"
_SIBLING_UNKNOWN = (
    "outcome unknown: another tool call in the same batch failed while this one was "
    "running; it may still finish and apply its effects"
)


async def _execute_parallel_group(
    group: list[ToolCall],
    lookups: ToolLookups,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
    run_id: str,
    iteration: int,
    notifier_warnings: list[str],
    transform: ToolResultTransform | None = None,
    *,
    image_runtime: ImageRuntime | None = None,
) -> list[tuple[ToolCall, ToolResult]]:
    """Run a parallel batch; a run-ending failure stops its siblings.

    When one call fails in a way that ends the run (a persistence failure,
    say), the other calls are cancelled and awaited before the failure
    propagates, so none keeps running or writes rows after the run ended.
    Each stopped sibling is recorded best effort: ``cancelled`` for async
    calls and calls that never started, ``outcome_unknown`` for sync calls
    whose thread had already started (a thread cannot be stopped). Outer
    cancellation stops the siblings the same way.
    """
    trackers = [CallTracker() for _ in group]
    tasks = [
        asyncio.ensure_future(
            _execute_record_notify(
                tc,
                lookups,
                recorder,
                notifier,
                run_id,
                iteration,
                notifier_warnings,
                transform,
                image_runtime=image_runtime,
                tracker=tracker,
            )
        )
        for tc, tracker in zip(group, trackers, strict=True)
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        await _stop_siblings(tasks)
        _log_stopped(run_id, group, tasks, trackers, reason="cancelled")
        raise
    failed = next(
        (t for t in tasks if t.done() and not t.cancelled() and t.exception() is not None),
        None,
    )
    if failed is None:
        return [t.result() for t in tasks]
    error = failed.exception()
    assert error is not None
    await _stop_siblings(tasks)
    stopped = _log_stopped(run_id, group, tasks, trackers, reason="sibling_failed")
    for tc, tracker in stopped:
        if tracker.result is not None:
            # The tool returned and its own row write had started (it may or
            # may not have landed). Never write a second row for the call.
            if not tracker.recorded:
                _codes.log_code(logger, _codes.TOOL_ROW_MAY_BE_MISSING, run_id=run_id)
            continue
        message = _SIBLING_UNKNOWN if tracker.abandoned else _SIBLING_CANCELLED
        code = "outcome_unknown" if tracker.abandoned else "cancelled"
        result = ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=json.dumps({"error": message, "code": code}),
            success=False,
            error=message,
            meta={"code": code},
        )
        try:
            await _record_tool(recorder, run_id, tc, result, iteration)
            await _notify_tool(notifier, run_id, tc, result, iteration, notifier_warnings)
        except Exception:
            _codes.log_code(logger, _codes.REGISTRY[code], run_id=run_id)
    raise error


async def _stop_siblings(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancel unfinished tasks and wait until each has unwound."""
    pending = [t for t in tasks if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        # Shielded so a second cancellation cannot leave siblings running.
        await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))


def _log_stopped(
    run_id: str,
    group: list[ToolCall],
    tasks: Sequence[asyncio.Future[Any]],
    trackers: list[CallTracker],
    *,
    reason: str,
) -> list[tuple[ToolCall, CallTracker]]:
    """Log each sibling that was stopped; return them."""
    stopped = [
        (tc, tracker)
        for tc, task, tracker in zip(group, tasks, trackers, strict=True)
        if task.cancelled()
    ]
    for _tc, tracker in stopped:
        if tracker.result is not None:
            continue  # it finished; only its row or event was cut short
        _codes.log_code(
            logger,
            _codes.REGISTRY["outcome_unknown" if tracker.abandoned else "cancelled"],
            run_id=run_id,
            reason=reason,
        )
    return stopped


_UNKNOWN_TOOL_LIST_LIMIT = 50


def _unknown_tool_message(name: str, lookups: ToolLookups) -> str:
    """``Unknown tool: <name>`` plus the tools the model was offered.

    ``lookups.defs`` holds the same tools the request's tool defs carry
    (local, MCP and skill tools). Sorted, at most 50 names.
    """
    offered = sorted(lookups.defs)
    if not offered:
        return f"Unknown tool: {name}. No tools are available."
    listed = ", ".join(offered[:_UNKNOWN_TOOL_LIST_LIMIT])
    extra = len(offered) - _UNKNOWN_TOOL_LIST_LIMIT
    if extra > 0:
        listed += f", … and {extra} more"
    return f"Unknown tool: {name}. Available tools: {listed}"


def _tool_failure_message(name: str, lookups: ToolLookups | None) -> str:
    if lookups is None:
        return "Unknown tool: name exceeds the supported length."
    return _unknown_tool_message(name, lookups)


def _unknown_tool_result(tool_call: ToolCall, lookups: ToolLookups | None) -> ToolResult:
    message = _tool_failure_message(tool_call.name, lookups)
    return ToolResult(
        name=tool_call.name,
        call_id=tool_call.id,
        payload=json.dumps({"error": message}, ensure_ascii=False),
        success=False,
        error=message,
        meta={"code": "unknown_tool"},
    )


_NO_SENT_PARAMS: dict[str, Any] = {}


async def _restore_params(
    engine: GuardrailEngine,
    tool_call: ToolCall,
    lookups: ToolLookups,
    on_skipped: Callable[[dict[str, Any], str], Awaitable[None]],
) -> dict[str, Any]:
    """Real values for the call's PII placeholders, as the guardrail decides.

    The PII guardrail owns the restore decision (``PII(restore=...)``,
    ``@tool(receives_pii=...)``); this is its single call site in the tool
    pipeline. Placeholders it deliberately leaves in place are reported as
    ``guardrail.restore_skipped`` (counts and kinds only).
    """
    restore = getattr(engine, "restore_params", None)
    if restore is None:  # guardrail engines without the restore allowlist
        return engine.deanonymize(tool_call.params)
    outcome = restore(tool_call.params, tool_def=lookups.defs.get(tool_call.name))
    if outcome.skipped:
        await on_skipped(outcome.event_data(tool_call.name, tool_call.id), tool_call.id)
    params: dict[str, Any] = outcome.params
    return params


def _image_wire_bytes(image_runtime: ImageRuntime | None) -> int:
    """Read-cap room for one image block (base64 of the policy's largest source)."""
    if image_runtime is None:
        return 0
    return -(-image_runtime.policy.max_source_bytes * 4 // 3) + _IMAGE_WIRE_OVERHEAD_BYTES


class _ToolTimedOutError(Exception):
    """The tool's own deadline (not a ``TimeoutError`` the tool raised)."""


async def _invoke_tool(
    fn: Callable[..., Any],
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    timeout_s: float,
    limiter: ToolCallLimiter | None,
    result_limit: int,
    image_wire_bytes: int,
    tracker: CallTracker,
) -> Any:
    """Run the tool under its deadline; async on the loop, sync on the tool pool."""
    deadline = asyncio.timeout(timeout_s)
    try:
        async with deadline:
            with result_cap(result_limit, image_bytes=image_wire_bytes):
                if inspect.iscoroutinefunction(fn):
                    tracker.started = True
                    return await fn(*args, **kwargs)
                return await run_sync_tool(fn, args, kwargs, limiter=limiter, tracker=tracker)
    except TimeoutError:
        if not deadline.expired():
            raise  # the tool raised its own TimeoutError: an ordinary failure
        if tracker.completed is not None:
            # The thread finished right at the deadline: the outcome is known.
            return tracker.completed.result()
        raise _ToolTimedOutError from None


def _timeout_result(
    tool_call: ToolCall,
    timeout_s: float,
    tracker: CallTracker,
    *,
    is_explicit: bool,
    duration_ms: int,
    run_id: str | None,
) -> ToolResult:
    name = tool_call.name
    meta: dict[str, Any]
    if tracker.started:
        code = "timeout_outcome_unknown"
        error_msg = (
            f"Tool '{name}' timed out after {timeout_s}s while running. It may still finish "
            "and apply its effects: check before retrying, and do not retry blindly if the "
            "tool changes anything."
        )
        meta = {"code": code}
        if tracker.abandoned:
            meta["zombie_threads"] = tracker.zombies
    else:
        code = "timeout_not_started"
        error_msg = (
            f"Tool '{name}' did not start within {timeout_s}s because every tool thread was "
            "busy. It was not run and is safe to retry."
        )
        meta = {"code": code}
    _codes.log_code(
        logger,
        _codes.REGISTRY[code],
        run_id=run_id,
        timeout_s=timeout_s,
        zombie_threads=meta.get("zombie_threads", 0),
    )
    return ToolResult(
        name=name,
        call_id=tool_call.id,
        payload=json.dumps({"error": error_msg, "code": code}),
        success=False,
        error=error_msg,
        duration_ms=duration_ms,
        meta=meta,
    )


async def _execute_tool(
    tool_call: ToolCall,
    lookups: ToolLookups,
    transform: ToolResultTransform | None = None,
    *,
    image_runtime: ImageRuntime | None = None,
    tracker: CallTracker | None = None,
    run_id: str | None = None,
) -> ToolResult:
    """Execute a server tool function and return a ToolResult.

    Non-server tools are filtered out by the loop before reaching this
    function. They trigger a pause instead of execution.

    A local ``@tool`` receives arguments rebuilt from the call's normalised
    JSON (:mod:`dendrux.tools._coerce`); a call that no longer validates
    fails as ``invalid_params`` without running. Async tools run on the
    loop; sync tools run on the bounded tool pool under the Agent's
    ``tool_threads`` limiter. A timeout reports ``timeout_not_started`` (the
    call never ran) or ``timeout_outcome_unknown`` (it started and may still
    finish); a ``TimeoutError`` the tool raises itself is an ordinary error.

    ``transform`` (``Agent.tool_result_transform``) rewrites the raw
    return value of a successful call before serialization, so the
    store, notifiers and the model all see the transformed result.
    A transform that raises fails the call.

    A ``ToolOutput`` result serialises its ``result`` like a plain return and
    ingests its images through ``image_runtime`` (strict for local tools;
    lenient with per-image notes for MCP tools). An image ingest failure
    fails the call like any tool exception. The serialised payload is capped
    at ``Limits.max_tool_result_bytes`` (images excluded).
    """
    fn = lookups.fn.get(tool_call.name)
    if fn is None:
        return _unknown_tool_result(tool_call, lookups)

    if isinstance(fn, SkillTool):
        return await fn.session.execute(tool_call)

    tracker = tracker if tracker is not None else CallTracker()
    limiter = lookups.call_limiter if isinstance(lookups.call_limiter, ToolCallLimiter) else None
    limits = limiter.limits if limiter is not None else Limits()
    timeout_s = lookups.timeout.get(tool_call.name, DEFAULT_TOOL_TIMEOUT)
    is_explicit = lookups.explicit_timeout.get(tool_call.name, False)

    spec = get_args_spec(fn)
    tool_def = lookups.defs.get(tool_call.name)
    args: list[Any] = []
    kwargs: dict[str, Any] = dict(tool_call.params or {})
    if spec is not None and tool_def is not None:
        try:
            args, kwargs = build_call(spec, tool_call.params or {})
        except ArgumentCoercionError as exc:
            return coercion_failure(tool_call, tool_def, exc)

    start = time.monotonic()
    try:
        result = await _invoke_tool(
            fn,
            args,
            kwargs,
            timeout_s=timeout_s,
            limiter=limiter,
            result_limit=limits.max_tool_result_bytes,
            image_wire_bytes=_image_wire_bytes(image_runtime),
            tracker=tracker,
        )

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
        images: tuple[ImagePart, ...] = ()
        notes: tuple[str, ...] = ()
        if isinstance(result, ToolOutput):
            output = result
            result = output.result
        else:
            output = None
        try:
            payload = json.dumps(result)
        except (TypeError, ValueError) as ser_err:
            hint = (
                " To return images, return ToolOutput(result, images=[...])."
                if _contains_image(result)
                else ""
            )
            raise TypeError(
                f"Tool '{tool_call.name}' returned a non-JSON-serializable value: "
                f"{type(result).__name__}. Tool return values must be JSON-serializable "
                f"(str, int, float, bool, list, dict, or None).{hint}"
            ) from ser_err
        payload, cap_meta = cap_value_payload(result, payload, limits.max_tool_result_bytes)
        if cap_meta is not None:
            _codes.log_code(
                logger,
                _codes.SIZE_LIMIT,
                run_id=run_id,
                bytes=cap_meta["bytes"],
                limit=limits.max_tool_result_bytes,
                truncated=bool(cap_meta.get("truncated")),
            )
            if not cap_meta.get("truncated"):
                return ToolResult(
                    name=tool_call.name,
                    call_id=tool_call.id,
                    payload=payload,
                    success=False,
                    error="result too large",
                    duration_ms=duration_ms,
                    meta=cap_meta,
                )
        if output is not None and output.images:
            try:
                if _tool_source(tool_call.name, lookups) == "mcp":
                    images, notes = await _ingest_mcp_images(
                        output.images, tool_call.name, image_runtime
                    )
                else:
                    images = await _ingest_tool_images(output.images, tool_call.name, image_runtime)
            except TimeoutError:
                # Not the tool's timeout: an image source (e.g. a URL) timed out.
                error_msg = f"Image ingest for tool '{tool_call.name}' timed out"
                return ToolResult(
                    name=tool_call.name,
                    call_id=tool_call.id,
                    payload=json.dumps({"error": error_msg}),
                    success=False,
                    error=error_msg,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=payload,
            success=True,
            duration_ms=duration_ms,
            images=images,
            notes=notes,
            meta=cap_meta or {},
        )
    except _ToolTimedOutError:
        return _timeout_result(
            tool_call,
            timeout_s,
            tracker,
            is_explicit=is_explicit,
            duration_ms=int((time.monotonic() - start) * 1000),
            run_id=run_id,
        )
    except ToolResultTooLargeError as too_large:
        # The source stopped reading an oversized result (MCP read cap).
        payload, cap_meta = size_limit_payload(too_large.size)
        _codes.log_code(
            logger,
            _codes.SIZE_LIMIT,
            run_id=run_id,
            bytes=too_large.size,
            limit=limits.max_tool_result_bytes,
        )
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=payload,
            success=False,
            error="result too large",
            duration_ms=int((time.monotonic() - start) * 1000),
            meta=cap_meta,
        )
    except ToolPoolSaturatedError:
        error_msg = (
            f"Tool '{tool_call.name}' could not be queued because the tool pool is full. "
            "It was not run and is safe to retry."
        )
        _codes.log_code(logger, _codes.TOOL_POOL_SATURATED, run_id=run_id)
        return ToolResult(
            name=tool_call.name,
            call_id=tool_call.id,
            payload=json.dumps({"error": error_msg, "code": "tool_pool_saturated"}),
            success=False,
            error=error_msg,
            meta={"code": "tool_pool_saturated"},
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
