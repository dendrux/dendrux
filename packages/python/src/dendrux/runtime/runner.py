"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with an explicit
provider and strategy. This is the top-level API developers interact with.

Sprint 1: caller provides the LLM provider instance, defaults to
NativeToolCalling strategy and ReActLoop. Future sprints add provider
registry (model string → provider resolution), strategy selection from
agent config, and more loop types.

Sprint 2 adds optional state_store for persistence. When provided:
  - Runner owns the run_id (generates it, passes to loop)
  - PersistenceRecorder records traces, tool calls, and usage (fail-closed)
  - finalize_run() is called in try/finally to guarantee persistence
  - Developer's extra_notifier is passed separately (best-effort)
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from contextlib import nullcontext as nullcontext
from copy import deepcopy
from dataclasses import dataclass as dataclass
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, overload
from typing import Literal as Literal
from typing import get_args as get_args

import dendrux.chat as _chat
import dendrux.loops._helpers as _hooks
import dendrux.runtime.context as _context
import dendrux.runtime.replay as _replay
import dendrux.runtime.state as _state
from dendrux import _codes
from dendrux._lifecycle import CANCEL_FINALIZE_STATUS_VALUES as CANCEL_FINALIZE_STATUS_VALUES
from dendrux._lifecycle import (
    PAUSE_STATUSES as _PAUSE_STATUSES,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux._lifecycle import (
    RESUME_INPUT_STATUS_VALUES as RESUME_INPUT_STATUS_VALUES,
)
from dendrux._lifecycle import (
    RESUME_RESULTS_STATUS_VALUES as RESUME_RESULTS_STATUS_VALUES,
)
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    TERMINAL_STATUSES,
    CancelReason,
    DenialReason,
    FailureReason,
    PauseReason,
    StoredEventType,
)
from dendrux._meta_keys import BLOCKED_META_KEY as BLOCKED_META_KEY
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY,
    LOOP_META_KEY,
    MAX_DELEGATION_DEPTH_META_KEY,
    NOT_EXECUTED_META_KEY,
    SEED_PREFIX_META_KEY,
    SKILLS_CONTEXT_META_KEY,
)
from dendrux._meta_keys import ERROR_INFO_META_KEY as ERROR_INFO_META_KEY
from dendrux._meta_keys import OUTPUT_TYPE_META_KEY as OUTPUT_TYPE_META_KEY
from dendrux._meta_keys import RETRY_OF_META_KEY as RETRY_OF_META_KEY
from dendrux._meta_keys import (
    STOP_REASON_META_KEY as STOP_REASON_META_KEY,
)
from dendrux._sentinel import _OMITTED, _OmittedType, _UnsetType
from dendrux.chat import (
    history_has_images,
    normalize_chat_history,
    validate_chat_history,
)
from dendrux.chat import split_user_content as split_user_content
from dendrux.context_blocks import (
    ContextBlock,
    fold_context,
    render_band,
    validate_context,
    wire_user_input,
    wire_user_parts,
)
from dendrux.errors import (
    BlobNotFoundError as BlobNotFoundError,
)
from dendrux.errors import (
    InvalidImageError as InvalidImageError,
)
from dendrux.errors import (
    RunFencedError,
    RunNotFoundError,
    safe_exception_info,
)
from dendrux.errors import StoreOutcomeUnknown as StoreOutcomeUnknown
from dendrux.loops._budget import BUDGET_EXCEEDED as BUDGET_EXCEEDED
from dendrux.loops._budget import BudgetCarry
from dendrux.loops._helpers import notifier_span_link as notifier_span_link
from dendrux.loops._helpers import notifier_trace_link as notifier_trace_link
from dendrux.loops._helpers import (
    notify_message,
    order_observations,
    project_message,
    record_message,
    record_run_started,
    record_tool,
)
from dendrux.loops._helpers import notify_run_failed as notify_run_failed
from dendrux.loops._helpers import (
    notify_run_finished as notify_run_finished,
)
from dendrux.loops._helpers import notify_run_started as notify_run_started
from dendrux.loops._helpers import record_run_failed as record_run_failed
from dendrux.loops._helpers import (
    record_run_finished as record_run_finished,
)
from dendrux.loops._policy_context import (
    dispatch_policy,
    history_call_counts,
    static_denial,
)
from dendrux.loops.react import MANAGED_HANDOFF_KEY as MANAGED_HANDOFF_KEY
from dendrux.loops.react import ReActLoop
from dendrux.runtime import (
    _managed,
    _managed_budget,
    _reclaim,
    _run_contracts,
    _run_entry,
    _run_errors,
    _run_events,
    _run_images,
    _run_observers,
    _run_outcome,
    _run_pending,
    _run_start,
    _run_tools,
)
from dendrux.runtime._entry_checks import resolve_runner_tenant, runner_entry_checks
from dendrux.runtime._managed import (
    DESCEND_KEY as MANAGED_DESCEND_KEY,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError,
)
from dendrux.runtime._managed_cancel import ManagedStopLifecycle as ManagedStopLifecycle
from dendrux.runtime._managed_cancel import (
    stopped_resume_result as _stopped_resume_result,
)
from dendrux.runtime._run_contracts import _UNSET_DEPTH as _UNSET_DEPTH
from dendrux.runtime._run_contracts import _accepts_kwarg as _accepts_kwarg
from dendrux.runtime._run_contracts import _ApprovalResume as _ApprovalResume
from dendrux.runtime._run_contracts import _PauseReads as _PauseReads
from dendrux.runtime._run_contracts import _ResumeContext as _ResumeContext
from dendrux.runtime._run_contracts import _ResumeProgress as _ResumeProgress
from dendrux.runtime._run_contracts import _signature_params as _signature_params
from dendrux.runtime._run_entry import _DEFAULT_RECOVERY_ITERATIONS as _DEFAULT_RECOVERY_ITERATIONS
from dendrux.runtime._run_entry import _apply_replay as _apply_replay
from dendrux.runtime._run_entry import _base_run_metadata as _base_run_metadata
from dendrux.runtime._run_entry import _bind_runtime_run as _bind_runtime_run
from dendrux.runtime._run_entry import _blocked_entry_result as _blocked_entry_result
from dendrux.runtime._run_entry import _create_projected_run as _create_projected_run
from dendrux.runtime._run_entry import _create_runtime_run as _create_runtime_run
from dendrux.runtime._run_entry import _entry_replay as _entry_replay
from dendrux.runtime._run_entry import _entry_seed_history as _entry_seed_history
from dendrux.runtime._run_entry import _load_new_run_replay as _load_new_run_replay
from dendrux.runtime._run_entry import _managed_creation as _managed_creation
from dendrux.runtime._run_entry import _new_run_history as _new_run_history
from dendrux.runtime._run_entry import _new_run_image_view as _new_run_image_view
from dendrux.runtime._run_entry import _new_run_input as _new_run_input
from dendrux.runtime._run_entry import _new_run_loop as _new_run_loop
from dendrux.runtime._run_entry import _new_run_metadata as _new_run_metadata
from dendrux.runtime._run_entry import _project_seed_messages as _project_seed_messages
from dendrux.runtime._run_entry import _record_seeded_user as _record_seeded_user
from dendrux.runtime._run_entry import _recovery_loop_name as _recovery_loop_name
from dendrux.runtime._run_entry import _require_idempotency_store as _require_idempotency_store
from dendrux.runtime._run_entry import (
    _resolve_max_delegation_depth as _resolve_max_delegation_depth,
)
from dendrux.runtime._run_entry import _run_projected_loop as _run_projected_loop
from dendrux.runtime._run_entry import _runtime_seeds as _runtime_seeds
from dendrux.runtime._run_entry import _text_history as _text_history
from dendrux.runtime._run_entry import _verify_projected_creation as _verify_projected_creation
from dendrux.runtime._run_errors import ERROR_TEXT_CHARS as ERROR_TEXT_CHARS
from dendrux.runtime._run_errors import _database_error_code as _database_error_code
from dendrux.runtime._run_errors import _database_error_info as _database_error_info
from dendrux.runtime._run_errors import _error_info_from_meta as _error_info_from_meta
from dendrux.runtime._run_errors import _error_payload as _error_payload
from dendrux.runtime._run_errors import _error_store_kwargs as _error_store_kwargs
from dendrux.runtime._run_errors import _nonprovider_error as _nonprovider_error
from dendrux.runtime._run_errors import _provider_error_info as _provider_error_info
from dendrux.runtime._run_errors import _public_meta as _public_meta
from dendrux.runtime._run_errors import _safe_error as _safe_error
from dendrux.runtime._run_errors import logger as logger
from dendrux.runtime._run_events import _attach_undelivered_notes as _attach_undelivered_notes
from dendrux.runtime._run_events import _check_claim as _check_claim
from dendrux.runtime._run_events import _drain_task_cancellation as _drain_task_cancellation
from dendrux.runtime._run_events import _emit_event as _emit_event
from dendrux.runtime._run_events import _emit_event_safe as _emit_event_safe
from dendrux.runtime._run_events import _fenced_stream_event as _fenced_stream_event
from dendrux.runtime._run_events import _finalize_running_event as _finalize_running_event
from dendrux.runtime._run_events import _finish_fenced_lifecycle as _finish_fenced_lifecycle
from dendrux.runtime._run_events import _may_swallow_cancellation as _may_swallow_cancellation
from dendrux.runtime._run_events import _pause_request_message as _pause_request_message
from dendrux.runtime._run_events import _pause_with_event as _pause_with_event
from dendrux.runtime._run_events import _persist_run_cancellation as _persist_run_cancellation
from dendrux.runtime._run_events import _persist_task_cancellation as _persist_task_cancellation
from dendrux.runtime._run_events import _read_cancel_winner as _read_cancel_winner
from dendrux.runtime._run_events import _shielded_cleanup as _shielded_cleanup
from dendrux.runtime._run_events import _take_task_interrupt as _take_task_interrupt
from dendrux.runtime._run_events import _transition_with_event as _transition_with_event
from dendrux.runtime._run_images import _carry_image_view as _carry_image_view
from dendrux.runtime._run_images import _entry_media as _entry_media
from dendrux.runtime._run_images import _EntryImages as _EntryImages
from dendrux.runtime._run_images import (
    _idempotent_hit_after_image_error as _idempotent_hit_after_image_error,
)
from dendrux.runtime._run_images import _image_loop_kwargs as _image_loop_kwargs
from dendrux.runtime._run_images import _image_retry_parts as _image_retry_parts
from dendrux.runtime._run_images import _image_runtime as _image_runtime
from dendrux.runtime._run_images import _ingest_entry_images as _ingest_entry_images
from dendrux.runtime._run_images import _ingest_resume_images as _ingest_resume_images
from dendrux.runtime._run_images import _input_data as _input_data
from dendrux.runtime._run_images import _input_edge_kwargs as _input_edge_kwargs
from dendrux.runtime._run_images import _live_edges as _live_edges
from dendrux.runtime._run_images import _memory_catalog as _memory_catalog
from dendrux.runtime._run_images import _record_user_trace as _record_user_trace
from dendrux.runtime._run_images import _recorder_image_kwargs as _recorder_image_kwargs
from dendrux.runtime._run_images import _register_memory_input as _register_memory_input
from dendrux.runtime._run_images import _release_memory_edges as _release_memory_edges
from dendrux.runtime._run_images import _replay_has_images as _replay_has_images
from dendrux.runtime._run_images import _replay_sources as _replay_sources
from dendrux.runtime._run_images import _require_image_loop as _require_image_loop
from dendrux.runtime._run_images import _resume_image_view as _resume_image_view
from dendrux.runtime._run_images import _same_pause as _same_pause
from dendrux.runtime._run_images import _seed_edges as _seed_edges
from dendrux.runtime._run_images import _single_full_source as _single_full_source
from dendrux.runtime._run_images import _start_image_view as _start_image_view
from dendrux.runtime._run_images import _stored_parts as _stored_parts
from dendrux.runtime._run_images import _submit_image_runtime as _submit_image_runtime
from dendrux.runtime._run_images import _tool_image_kwargs as _tool_image_kwargs
from dendrux.runtime._run_images import _write_edges as _write_edges
from dendrux.runtime._run_images import ingest_submitted_tool_images as ingest_submitted_tool_images
from dendrux.runtime._run_images import ingest_submitted_user_parts as ingest_submitted_user_parts
from dendrux.runtime._run_observers import _attach_mcp_meta as _attach_mcp_meta
from dendrux.runtime._run_observers import (
    _attach_run_error_meta as _attach_run_error_meta,
)
from dendrux.runtime._run_observers import _cause_type as _cause_type
from dendrux.runtime._run_observers import _start_trace as _start_trace
from dendrux.runtime._run_observers import _TraceNotifier as _TraceNotifier
from dendrux.runtime._run_outcome import _closing_loop_stream as _closing_loop_stream
from dendrux.runtime._run_outcome import _completion_evidence as _completion_evidence
from dendrux.runtime._run_outcome import _finalize_loop_cancelled as _finalize_loop_cancelled
from dendrux.runtime._run_outcome import _finish_stream_event as _finish_stream_event
from dendrux.runtime._run_outcome import _loop_cancel_request as _loop_cancel_request
from dendrux.runtime._run_outcome import _partial_answer as _partial_answer
from dendrux.runtime._run_outcome import _persist_loop_outcome as _persist_loop_outcome
from dendrux.runtime._run_outcome import _persist_managed_handoff as _persist_managed_handoff
from dendrux.runtime._run_outcome import _persist_pause as _persist_pause
from dendrux.runtime._run_outcome import _persisted_loop_result as _persisted_loop_result
from dendrux.runtime._run_outcome import _prepare_stream_result as _prepare_stream_result
from dendrux.runtime._run_outcome import _record_stop_reason as _record_stop_reason
from dendrux.runtime._run_outcome import _stream_error_meta as _stream_error_meta
from dendrux.runtime._run_outcome import _stream_loop_events as _stream_loop_events
from dendrux.runtime._run_outcome import _terminal_failure_kwargs as _terminal_failure_kwargs
from dendrux.runtime._run_outcome import _terminal_failure_reason as _terminal_failure_reason
from dendrux.runtime._run_outcome import _terminal_loop_event as _terminal_loop_event
from dendrux.runtime._run_outcome import _terminal_stream_result as _terminal_stream_result
from dendrux.runtime._run_outcome import _terminal_writer as _terminal_writer
from dendrux.runtime._run_outcome import _track_partial_text as _track_partial_text
from dendrux.runtime._run_outcome import _unstored_failure_reason as _unstored_failure_reason
from dendrux.runtime._run_outcome import _wrap_loop_result as _wrap_loop_result
from dendrux.runtime._run_pending import _CONTROL_EVENT_LIMIT as _CONTROL_EVENT_LIMIT
from dendrux.runtime._run_pending import _STOP_REASONS as _STOP_REASONS
from dendrux.runtime._run_pending import (
    _attach_persisted_pending as _attach_persisted_pending,
)
from dendrux.runtime._run_pending import _build_cached_result as _build_cached_result
from dendrux.runtime._run_pending import _cached_output as _cached_output
from dendrux.runtime._run_pending import (
    _committed_pause_result as _committed_pause_result,
)
from dendrux.runtime._run_pending import (
    _ensure_pause_generation as _ensure_pause_generation,
)
from dendrux.runtime._run_pending import _final_run_recovery as _final_run_recovery
from dendrux.runtime._run_pending import _in_memory_pending as _in_memory_pending
from dendrux.runtime._run_pending import (
    _load_persisted_pending as _load_persisted_pending,
)
from dendrux.runtime._run_pending import _pause_event_data as _pause_event_data
from dendrux.runtime._run_pending import _pending_moved as _pending_moved
from dendrux.runtime._run_pending import (
    _raise_resume_claim_failure as _raise_resume_claim_failure,
)
from dendrux.runtime._run_pending import _read_recovery_info as _read_recovery_info
from dendrux.runtime._run_pending import _read_status as _read_status
from dendrux.runtime._run_pending import _recovery_info as _recovery_info
from dendrux.runtime._run_pending import _replay_from_meta as _replay_from_meta
from dendrux.runtime._run_pending import _reread_moved as _reread_moved
from dendrux.runtime._run_pending import (
    _stop_reason_from_meta as _stop_reason_from_meta,
)
from dendrux.runtime._run_start import _creation_claim as _creation_claim
from dendrux.runtime._run_start import _failure_fields as _failure_fields
from dendrux.runtime._run_start import _failure_finalize_fields as _failure_finalize_fields
from dendrux.runtime._run_start import _finish_lifecycle as _finish_lifecycle
from dendrux.runtime._run_start import _persist_run_failure as _persist_run_failure
from dendrux.runtime._run_start import _record_finished as _record_finished
from dendrux.runtime._run_start import _recorder_tool_defs as _recorder_tool_defs
from dendrux.runtime._run_start import _RunLifecycle as _RunLifecycle
from dendrux.runtime._run_start import _runtime_recorder as _runtime_recorder
from dendrux.runtime._run_start import _start_new_run as _start_new_run
from dendrux.runtime._run_start import _stream_run_failure as _stream_run_failure
from dendrux.runtime._run_tools import _TOOL_NAME_CHARS as _TOOL_NAME_CHARS
from dendrux.runtime._run_tools import _TOOL_NAME_SCAN_LIMIT as _TOOL_NAME_SCAN_LIMIT
from dendrux.runtime._run_tools import _discover_runtime_tools as _discover_runtime_tools
from dendrux.runtime._run_tools import _emit_declared_tools as _emit_declared_tools
from dendrux.runtime._run_tools import _emit_init_events as _emit_init_events
from dendrux.runtime._run_tools import _emit_init_governance_event as _emit_init_governance_event
from dendrux.runtime._run_tools import _local_tool_names as _local_tool_names
from dendrux.runtime._run_tools import _mcp_error_payload as _mcp_error_payload
from dendrux.runtime._run_tools import _MCPDiscoveryError as _MCPDiscoveryError
from dendrux.runtime._run_tools import _snapshot_mcp_meta as _snapshot_mcp_meta
from dendrux.runtime._run_tools import _validate_loop_skill_compat as _validate_loop_skill_compat
from dendrux.runtime._tenant_checks import _tenant_scope_chosen
from dendrux.runtime._tracelink import validate_trace_link as validate_trace_link
from dendrux.runtime.context import (
    DelegationContext,
    StreamContext,
    get_store_identity,
    reset_delegation_context,
    resolve_child_tenant,
    resolve_parent_link,
    set_delegation_context,
    take_managed_invocation,
)
from dendrux.runtime.context import get_delegation_context as get_delegation_context
from dendrux.runtime.replay import (
    PARTS_KEY,
    REPLAY_META_KEY,
    ReplayLast,
    ReplayPlan,
    ReplayRequest,
    _pair_tool_calls,
    current_prefix,
    expand_replay,
    has_images,
    image_slot_count,
    rebuild_messages,
    user_trace_message,
    validate_replay_args,
)
from dendrux.runtime.replay import LoadedReplay as LoadedReplay
from dendrux.runtime.replay import _load_replay_runs as _load_replay_runs
from dendrux.runtime.replay import build_seed as build_seed
from dendrux.runtime.replay import (
    initial_image_view as initial_image_view,
)
from dendrux.runtime.replay import (
    load_image_sources as load_image_sources,
)
from dendrux.runtime.replay import load_replay_runs as load_replay_runs
from dendrux.runtime.replay import (
    parts_to_dicts as parts_to_dicts,
)
from dendrux.runtime.replay import record_run_prefix as record_run_prefix
from dendrux.runtime.state import ClaimResult as ClaimResult
from dendrux.runtime.state import _initial_loop_state as _initial_loop_state
from dendrux.runtime.state import claimed_store as claimed_store
from dendrux.runtime.state import unclaimed_store as unclaimed_store
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit
from dendrux.runtime.submit import GOVERNANCE_KEY as GOVERNANCE_KEY
from dendrux.runtime.submit import _claim_failure_mapping as _claim_failure_mapping
from dendrux.runtime.submit import (
    _claim_request,
    _claim_submission,
    _claimed_submission_view,
    _release_supported,
    legacy_approval_denial,
    prepare_resume_results,
    report_governance_change,
    require_pause_id,
    submission_validator,
    validate_result_names,
)
from dendrux.runtime.submit import governance_snapshot as governance_snapshot
from dendrux.skills._session import META_KEY, SkillSession, continuation, current
from dendrux.skills._session import (
    current as current_skill_session,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    AgentStep,
    ApprovalDecision,
    GovernanceEventType,
    IdempotencyConflictError,
    ImagePart,
    ImageViewState,
    Message,
    PauseState,
    Role,
    RunAlreadyActiveError,
    RunEventType,
    RunStatus,
    UsageStats,
    _usage_from_dict,
    generate_ulid,
    tool_result_message,
)
from dendrux.types import (
    ErrorInfo as ErrorInfo,
)
from dendrux.types import (
    ImageRef as ImageRef,
)
from dendrux.types import ReplaySummary as ReplaySummary
from dendrux.types import StopReason as StopReason
from dendrux.types import (
    TextPart as TextPart,
)
from dendrux.types import compute_idempotency_fingerprint as compute_idempotency_fingerprint
from dendrux.types import (
    content_part_from_dict as content_part_from_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Sequence,
    )

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.media._catalog import BlobEdge as BlobEdge
    from dendrux.media._image import UserContent
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime._continuation_guardrails import _SubmissionProjection
    from dendrux.runtime._entry_guardrails import _EntryProjection as _EntryProjection
    from dendrux.runtime.state import ClaimResult, RunRecord, StateStore
    from dendrux.strategies.base import Strategy
    from dendrux.types import (
        ContentPart,
        CreateRunResult,
        Limits,
        RunEvent,
        RunResult,
        RunStream,
        ToolCall,
        ToolResult,
    )


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


def _budget_carry_kwargs(loop_method: Any, pause_state: PauseState) -> dict[str, Any]:
    """``budget_carry=`` for a loop that accepts it (built-in loops); else nothing.

    Custom loops written against the pre-0.3 ``Loop`` signature keep working;
    they simply cannot restore fired budget keys.
    """
    params = _run_contracts._signature_params(loop_method)
    if params is None or "budget_carry" not in params:
        return {}
    return {"budget_carry": BudgetCarry.from_pause(pause_state)}


def _carry_budget_fields(new: PauseState, old: PauseState) -> PauseState:
    """Copy fired budget keys from the pause being resumed onto a runner-built pause."""
    return replace(
        new,
        budget_fired=old.budget_fired,
        budget_unpriced_reported=old.budget_unpriced_reported,
        budget_known_cost_usd=old.budget_known_cost_usd,
        budget_unpriced_calls=old.budget_unpriced_calls,
    )


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunResult: ...


@overload
async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunResult: ...


async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrux agent. It wires
    together the agent definition, LLM provider, strategy, and loop,
    then executes the loop until completion.

    Args:
        agent: Agent definition (model, tools, prompt, limits).
        provider: LLM provider to use for this run.
        user_input: The user's input to process.
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop. Defaults to ReActLoop.
        state_store: Optional persistence backend. If provided, the run
            is persisted to the database with full traces.
        tenant_id: Optional tenant ID for multi-tenant isolation.
        metadata: Optional developer linking data (thread_id, user_id, etc.).
            Stored in agent_runs.meta — Dendrux stores it, never reads it.
        extra_notifier: Optional additional LoopNotifier for best-effort
            notifications (e.g. ConsoleNotifier).
        max_delegation_depth: Maximum allowed delegation depth for the
            run tree. Default 10. None means unbounded. Propagated to
            all child runs via contextvar. Raises DelegationDepthExceededError
            if a child run would exceed this depth.
        **kwargs: Reserved for future use.

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrux import Agent, tool, run
        from dendrux.llm import AnthropicProvider

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a calculator.",
        )
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        result = await run(agent, provider=provider, user_input="What is 15 + 27?")
        print(result.answer)
    """
    result = await _run_once(
        agent,
        provider=provider,
        user_input=user_input,
        history=history,
        context=context,
        activate_skills=activate_skills,
        runs=runs,
        replay_last=replay_last,
        replay_reasoning=replay_reasoning,
        strategy=strategy,
        loop=loop,
        state_store=state_store,
        tenant_id=tenant_id,
        metadata=metadata,
        extra_notifier=extra_notifier,
        max_delegation_depth=max_delegation_depth,
        idempotency_key=idempotency_key,
        output_type=output_type,
        **kwargs,
    )
    # A managed parent exits at its handoff; the driver runs the child in this
    # same task and unwinds terminal outcomes back to the root (0.7).
    return await _managed.drive(result, notifier=extra_notifier)


async def _run_once(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _run_contracts._UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    **kwargs: Any,
) -> RunResult:
    """One run frame: a new run, or a managed child created by its atomic bind."""
    invocation = take_managed_invocation()
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "run", tenant_id, state_store=state_store)
    _run_entry._require_idempotency_store(state_store, idempotency_key)
    await ensure_agent_limits_fit(agent, state_store)
    skill_session = SkillSession.prepare(
        agent, loop=_run_entry._new_run_loop(agent, loop), explicit=activate_skills
    )
    with skill_session.bind():
        # Validate + resolve depth early — before any side effects.
        # _resolve_max_delegation_depth is a pure policy helper; side effect
        # is limited to dedup warning state on parent_ctx.warned_mismatches.
        # A managed child never inherits the caller's delegation context: its
        # parent link, level and depth come from its bound invocation.
        parent_ctx = _managed.parent_context(invocation, _context.get_delegation_context)
        effective_max_depth = _run_entry._resolve_max_delegation_depth(
            max_delegation_depth, parent_ctx
        )
        tenant_id = resolve_child_tenant(parent_ctx, tenant_id)
        runner_entry_checks(
            agent,
            "run",
            tenant_id=tenant_id,
            metadata=metadata,
            user_input=user_input,
            history=history,
            context=context,
            provider_kwargs=kwargs,
            idempotency_key=idempotency_key,
            provider=provider,
        )
        resolved_strategy = strategy or NativeToolCalling()
        resolved_loop = _run_entry._new_run_loop(agent, loop)
        _run_tools._validate_loop_skill_compat(agent, resolved_loop)
        root_limits = await _managed.enter(agent, invocation, state_store, resolved_loop)
        # Validate input + chat history before any side effects. Raises
        # TypeError / ValueError on bad input so no DB row is created.
        user_input, input_items, image_input, normalized_history = _run_entry._new_run_input(
            user_input, history, resolved_loop
        )
        replay_req, replay_runs, replay_omitted = await _run_entry._entry_replay(
            state_store,
            tenant_id,
            runs,
            replay_last,
            replay_reasoning,
            resolved_loop,
            context,
            agent._limits,
        )
        replay_plan: ReplayPlan | None = None
        provider_kwargs = dict(kwargs)
        # A managed child's id is its parent's bind candidate, minted at the handoff.
        run_id = _managed.new_run_id(invocation)
        from dendrux.runtime._entry_guardrails import _entry_projection

        entry_projection = _entry_projection(agent, run_id)
        _system_prompt = (
            await entry_projection.project(
                Message(Role.SYSTEM, agent.get_system_prompt(loop=resolved_loop)), 0
            )
        ).content
        # Images: ingest before the first durable write (§5.2). Typed errors
        # raise here, before any run row or claim exists.
        image_runtime, user_parts, normalized_history, cached = await _run_images._entry_media(
            agent,
            provider,
            resolved_loop,
            state_store,
            tenant_id,
            run_id,
            image_input,
            user_input,
            input_items,
            history,
            normalized_history,
            provider_kwargs,
            idempotency_key,
            output_type,
            context,
            replay_req,
            skill_session,
            entry_projection=entry_projection,
        )
        if cached is not None:
            return cached
        # When chat history is provided, the loop expects initial_history to
        # be the COMPLETE message list (loops use initial_history as-is and
        # do not separately append user_input — same contract as resume).
        # The runner will record the new user_input itself (after the recorder
        # is constructed) so seeded prior turns stay out of react_traces.
        # With runs=, the replayed prefix is folded in after expansion.
        (
            seeded_history,
            normalized_history,
            context,
            user_input,
            user_parts,
        ) = await entry_projection.prepare(
            normalized_history, context, user_input, user_parts, skill_session
        )
        seeded_history = _run_entry._entry_seed_history(replay_req, seeded_history)

        # --- Delegation context ---
        parent_run_id, delegation_level = _managed.parent_link(invocation, parent_ctx, state_store)

        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        result: RunResult | None = None
        lifecycle = _run_start._RunLifecycle(
            run_id, state_store, getattr(agent, "_task_manager", None)
        )

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = entry_projection.engine
        try:
            try:
                if state_store is not None:
                    # Create the run record before the loop starts
                    # Merge loop type + depth limit into developer metadata
                    run_meta, idem_fingerprint = _run_entry._new_run_metadata(
                        agent,
                        resolved_loop,
                        skill_session,
                        metadata,
                        effective_max_depth,
                        user_input=user_input,
                        idempotency_key=idempotency_key,
                        output_type=output_type,
                        history=normalized_history,
                        context=context,
                        replay=replay_req,
                        parts=user_parts,
                    )

                    seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
                        state_store,
                        agent,
                        provider,
                        resolved_loop,
                        resolved_strategy,
                        provider_kwargs,
                        seeded_history,
                        user_input=user_input,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        replay_req=replay_req,
                        replay_runs=replay_runs,
                        replay_omitted=replay_omitted,
                        normalized_history=normalized_history,
                        context=context,
                        user_parts=user_parts,
                        engine=run_guardrails,
                        system_prompt=_system_prompt,
                    )
                    created = await _run_entry._create_projected_run(
                        entry_projection.error is not None,
                        state_store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_run_images._input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=tenant_id,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=idem_fingerprint,
                        **_run_images._input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                        invocation=invocation,
                    )
                    # A child bound by another caller reads as existing_active below.
                    create_result, child_claim = _run_entry._managed_creation(created)

                    # Handle idempotency outcomes
                    if create_result.outcome == "existing_terminal":
                        return await _run_pending._build_cached_result(
                            state_store, create_result.run_id, output_type
                        )
                    if create_result.outcome == "existing_active":
                        raise RunAlreadyActiveError(create_result.run_id, create_result.status)

                    run_id = create_result.run_id
                    state_store = lifecycle._claim(
                        state_store, resolved_loop, create_result, child_claim, root_limits
                    )
                    await _run_entry._verify_projected_creation(
                        entry_projection.error is not None, state_store, run_id
                    )

                trace_link, extra_notifier = await _run_start._start_new_run(
                    lifecycle,
                    agent,
                    provider,
                    tenant_id,
                    seed_kwargs,
                    run_guardrails,
                    extra_notifier,
                    parent_ctx,
                    {"agent_name": agent.name, "system_prompt": _system_prompt},
                )

                # Set delegation context for the duration of this run so nested
                # agent.run() calls inside tools inherit the parent link.
                this_ctx = DelegationContext(
                    run_id=run_id,
                    trace_link=trace_link,
                    delegation_level=delegation_level,
                    persisted=state_store is not None,
                    store_identity=get_store_identity(state_store),
                    max_delegation_depth=effective_max_depth,
                    tenant_id=tenant_id,
                    tenant_explicit=tenant_explicit,
                    state_store=state_store,
                    lease_interval_s=agent.lease_interval.total_seconds(),
                    loop=asyncio.get_running_loop(),
                )
                ctx_token = set_delegation_context(this_ctx)

                # Emit init governance events (skills + MCP) inside the try block
                # so discovery failures flow through the error cleanup path.
                # _MCPDiscoveryError is caught here, mcp.error emitted, then the
                # original cause is re-raised for the outer except Exception.
                await _run_tools._discover_runtime_tools(
                    agent, lifecycle.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
                )

                replay_plan, seeded_history = await _run_entry._apply_replay(
                    agent,
                    provider,
                    resolved_loop,
                    provider_kwargs,
                    state_store=state_store,
                    run_id=run_id,
                    replay_req=replay_req,
                    replay_runs=replay_runs,
                    replay_omitted=replay_omitted,
                    normalized_history=normalized_history,
                    context=context,
                    user_input=user_input,
                    seeded_history=seeded_history,
                    user_parts=user_parts,
                    system_prompt=_system_prompt,
                )
                await _run_images._register_memory_input(
                    image_runtime, state_store, run_id, user_parts
                )
                image_view = await _run_entry._new_run_image_view(
                    image_runtime,
                    state_store,
                    run_id,
                    seeded_history,
                    replay_runs,
                    replay_plan,
                    user_parts,
                )

                # When chat history was provided, the loop receives a complete
                # message list as initial_history and skips its own user-message
                # recording. The runner records the new user_input itself so it
                # still lands in react_traces (seeded prior turns are NOT recorded
                # — they live in the dev's chat DB).
                await _run_entry._record_seeded_user(
                    lifecycle.recorder,
                    extra_notifier,
                    run_id,
                    user_input,
                    seeded_history,
                    context,
                    skill_session,
                )

                result = await _run_entry._run_projected_loop(
                    entry_projection,
                    resolved_loop,
                    user_parts,
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=user_input,
                    run_id=run_id,
                    recorder=lifecycle.recorder,
                    notifier=extra_notifier,
                    initial_history=seeded_history,
                    provider_kwargs=provider_kwargs or None,
                    output_type=output_type,
                    state_store=state_store,
                    **_run_images._image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
                )

                if replay_plan is not None:
                    result.replay = replay_plan.summary
                _run_observers._attach_mcp_meta(result, mcp_meta)
                _run_pending._ensure_pause_generation(result, provider_kwargs)
                if state_store is not None:
                    result = await _run_outcome._persist_loop_outcome(
                        recorder=lifecycle.recorder,
                        provider_kwargs=provider_kwargs,
                        state_store=state_store,
                        run_id=run_id,
                        result=result,
                    )
                # A parent at a managed handoff exits here without finishing.
                await lifecycle._finish(lifecycle.recorder, extra_notifier, result)

                result.meta = _run_errors._public_meta(result.meta)

                return result

            except (ManagedRootStoppedError, RunFencedError) as stop:
                return await lifecycle._stopped(stop, lifecycle.recorder, extra_notifier)

            except Exception as exc:
                await lifecycle._failed(exc, mcp_meta, lifecycle.recorder, extra_notifier)
                raise

        except asyncio.CancelledError:
            cancelled_result, swallow = await lifecycle._cancelled(
                result, mcp_meta, lifecycle.recorder, extra_notifier
            )
            if swallow:
                return cancelled_result
            raise

        finally:
            _run_images._release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


async def _check_retry_source(
    state_store: StateStore, source_run: RunRecord, original_run_id: str
) -> None:

    if source_run.status not in TERMINAL_STATUSES:
        raise ValueError(
            f"Run '{original_run_id}' is in status '{source_run.status}' — "
            f"only terminal runs (error, cancelled, max_iterations, success) can be retried."
        )

    # 2. Validate source run was not SingleCall (no iterative history to seed)
    from dendrux.runtime._capabilities import capabilities

    source_loop = (source_run.meta or {}).get(LOOP_META_KEY, "")
    if (
        source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED
        and capabilities(state_store).checkpointing
    ):
        source_state = await state_store.get_loop_state(original_run_id)
        source_loop = (source_state or {}).get("loop")
    if source_loop in ("SingleCall", "single_call"):
        raise ValueError(
            "SingleCall runs cannot be retried with context — "
            "they have no iterative history to seed. Call agent.run() instead."
        )


async def _retry_history(
    state_store: StateStore,
    source_run: RunRecord,
    agent: Agent,
    provider: LLMProvider,
    kwargs: dict[str, Any],
) -> tuple[Any, list[Message]]:
    from dendrux.runtime._capabilities import capabilities
    from dendrux.runtime.replay import conversation_records

    traces = (
        await conversation_records(state_store, source_run.id, limits=agent._limits)
        if source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED
        and capabilities(state_store).checkpointing
        else await state_store.get_traces(source_run.id)
    )
    retry_model = (kwargs or {}).get("model") or provider.model
    initial_history = rebuild_messages(traces, reasoning_for=retry_model)
    _trim_retry_answer(source_run, initial_history)
    initial_history = _pair_tool_calls(initial_history)
    if source_run.storage_format == STORAGE_FORMAT_CHECKPOINTED:
        initial_history = [m for m in initial_history if m.role != Role.SYSTEM]

    if not initial_history:
        raise ValueError(
            f"Run '{source_run.id}' has no persisted traces. Cannot retry without prior context."
        )

    return traces, initial_history


def _trim_retry_answer(source_run: RunRecord, history: list[Message]) -> None:
    if source_run.status == RunStatus.SUCCESS.value and history:
        final = history[-1]
        if final.role == Role.ASSISTANT and not final.tool_calls:
            history.pop()


def _retry_source_input(source_run: RunRecord, traces: Any) -> tuple[Any, Any, Any, Any]:
    users = [t for t in traces if t.role == "user"]
    raw_user = next((t for t in users if not (t.meta or {}).get(SEED_PREFIX_META_KEY)), users[0])
    user_input = (source_run.input_data or {}).get("input", raw_user.content)
    source_parts = _run_images._stored_parts((source_run.input_data or {}).get(PARTS_KEY))

    return user_input, source_parts, raw_user, users


def _refresh_retry_user(
    initial_history: list[Message],
    users: Any,
    raw_user: Any,
    user_input: str,
    source_parts: tuple[ContentPart, ...] | None,
    skill_session: SkillSession,
) -> tuple[list[ContextBlock], Message]:
    own_ordinal = users.index(raw_user)
    first_user = [i for i, m in enumerate(initial_history) if m.role == Role.USER][own_ordinal]
    retry_context = [
        ContextBlock(**b) for b in (raw_user.meta or {}).get(SKILLS_CONTEXT_META_KEY, [])
    ]
    first_msg = initial_history[first_user]
    if first_msg.parts is not None and source_parts is not None:
        retry_user = initial_history[first_user] = Message.with_parts(
            Role.USER,
            wire_user_parts(source_parts, skill_session.envelopes(), retry_context),
            placement=first_msg.placement,
        )
    else:
        retry_user = initial_history[first_user] = replace(
            first_msg,
            content=wire_user_input(user_input, skill_session.envelopes(), retry_context),
        )

    return retry_context, retry_user


async def _retry_replay(
    source_run: RunRecord,
    resolved_loop: Loop,
    state_store: StateStore,
    tenant_id: str | None,
    limits: Limits,
    original_run_id: str,
) -> tuple[ReplayRequest | None, list[Any]]:
    """The retried run's ``runs=`` sources, after the blocked-content admission.

    Refuses (R1.8) when the retried run or any of its recorded sources is
    identified as blocked, after their scope/status checks and before any history
    is rebuilt or a run is created. One scoped verdict lookup covers all of them.
    """
    replay_req = _retry_replay_request(source_run, resolved_loop)
    # The verdict and any refusal name the run the caller authorized and asked for.
    retried = replace(source_run, id=original_run_id)
    if replay_req is None:
        from dendrux.runtime._blocked_content import _blocked_sources

        if await _blocked_sources(state_store, [retried], tenant_id=tenant_id):
            _refuse_blocked_retry(original_run_id)
        return None, []
    loaded = await _replay._load_replay_runs(
        state_store, replay_req, tenant_id=tenant_id, limits=limits, extra=(retried,)
    )
    identified = (*loaded.extra_blocked, *loaded.omitted)
    if identified:
        _refuse_blocked_retry(identified[0])
    return replay_req, loaded.runs


def _retry_replay_request(source_run: RunRecord, resolved_loop: Loop) -> ReplayRequest | None:
    replay_req: ReplayRequest | None = None
    source_replay = (source_run.meta or {}).get(REPLAY_META_KEY)
    if isinstance(source_replay, dict) and source_replay.get("runs"):
        replay_req = validate_replay_args(
            list(source_replay["runs"]),
            source_replay.get("replay_last", 0),
            bool(source_replay.get("replay_reasoning", True)),
            loop=resolved_loop,
            context=None,
        )
    return replay_req


def _refuse_blocked_retry(run_id: str) -> NoReturn:
    from dendrux.errors import ReplayRunError

    _codes.log_code(logger, _codes.BLOCKED_CONTENT, run_id=run_id)
    raise ReplayRunError(
        run_id,
        "blocked_content",
        "Source history contains blocked content; start a new run with clean input.",
    )


def _retry_media(
    agent: Agent,
    store: StateStore,
    loop: Loop,
    tenant: str | None,
    run_id: str,
    parts: tuple[ContentPart, ...] | None,
    history: list[Message],
    replay: list[Any],
) -> ImageRuntime | None:
    if parts is not None or has_images(history) or _run_images._replay_has_images(replay):
        _run_images._require_image_loop(loop)
    elif not _run_contracts._accepts_kwarg(loop.run, "image_runtime"):
        return None
    return _run_images._image_runtime(agent, store, tenant, run_id)


async def _retry_edges(
    runtime: ImageRuntime | None, parts: tuple[ContentPart, ...] | None
) -> dict[str, Any]:
    if parts is None or runtime is None:
        return {}
    live = await _run_images._live_edges(
        runtime, _run_images._input_edge_kwargs(runtime, parts).get("blob_refs", [])
    )
    return {"blob_refs": live, "blob_refs_scope": runtime.tenant_scope} if live else {}


async def _retry_source(
    store: StateStore, run_id: str, tenant_id: str | None
) -> tuple[RunRecord, Any]:
    source = await store.get_run(run_id)
    if source is None:
        raise RunNotFoundError(run_id)
    if source.tenant_id != tenant_id:
        raise RunNotFoundError(run_id)
    return source, (source.meta or {}).get(META_KEY, {}).get("explicit", [])


async def _start_retry_history(
    agent: Agent,
    provider: LLMProvider,
    loop: Loop,
    provider_kwargs: dict[str, Any],
    store: StateStore,
    run_id: str,
    original_run_id: str,
    source: RunRecord,
    replay_req: ReplayRequest | None,
    replay_runs: list[Any],
    history: list[Message],
    image_runtime: ImageRuntime | None,
    explicit: Any,
) -> tuple[ReplayPlan | None, list[Message], ImageViewState | None]:
    from dendrux.runtime._capabilities import capabilities

    current = current_prefix(agent, provider, loop, provider_kwargs)
    replay_plan: ReplayPlan | None = None
    if replay_req is not None:
        replay_plan = expand_replay(
            replay_runs, replay_req, current=current, guardrails=bool(agent.guardrails)
        )
        if (
            source.storage_format != STORAGE_FORMAT_CHECKPOINTED
            or not capabilities(store).checkpointing
        ):
            history = [*replay_plan.messages, *history]
    seeded_slots = image_slot_count(history)
    await _replay.record_run_prefix(
        store,
        run_id,
        current=current,
        seed={"retry_of": original_run_id},
        plan=replay_plan,
        # An explicit-skills retry also traces its rebuilt first user
        # message, so its slot mapping is not a plain offset: unknown.
        seeded_image_slots=None if explicit and seeded_slots else seeded_slots,
    )
    # A retry has no new input: every seeded slot counts as sent.
    image_view = await _run_images._start_image_view(
        image_runtime,
        store,
        run_id,
        seeded=history,
        sources=[
            *_run_images._replay_sources(replay_runs),
            (original_run_id, source.tenant_id, source.meta),
        ],
        single_full=replay_plan is None,
        input_images=False,
    )
    return replay_plan, history, image_view


async def retry(
    original_run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    **kwargs: Any,
) -> RunResult:
    """Retry a terminal run with approximate prior context.

    Creates a fresh run seeded with the original run's conversation history
    from persisted traces. The LLM sees the prior conversation and can
    continue from context. This is NOT resume (no pause_data, no exact
    state reconstruction) — it's a new run with approximate context.

    Works for any terminal run (error, cancelled, max_iterations, success).
    Source run must have been executed with a retry-capable loop (not
    SingleCall). The retry agent can differ from the original — different
    model, different tools — the only requirement is that the source run
    has persisted traces to seed from.

    Args:
        original_run_id: The terminal run to retry from.
        agent: Agent definition for the retry run. Can differ from the
            original agent (different model, tools, prompt).
        provider: LLM provider for the retry run.
        state_store: Persistence backend (required — retry needs traces).
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop for the retry. Defaults to ReActLoop.
        tenant_id: Optional tenant ID.
        metadata: Optional developer metadata for the retry run.
        extra_notifier: Optional notifier.
        **kwargs: Forwarded to the LLM provider.

    Returns:
        RunResult from the retry run.

    Raises:
        RunNotFoundError: The source run does not exist or belongs to
            another tenant.
        ValueError: If the source run is not terminal, if the source
            run was SingleCall, or if no traces exist.
    """
    # 1. Validate source run is terminal. A missing run and another tenant's
    # run raise the same RunNotFoundError (B13), so retry is no existence oracle.
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "retry", tenant_id, state_store=state_store)
    await ensure_agent_limits_fit(agent, state_store)
    tenant_id = resolve_child_tenant(_context.get_delegation_context(), tenant_id)
    runner_entry_checks(
        agent,
        "retry",
        tenant_id=tenant_id,
        metadata=metadata,
        history=kwargs.get("history"),
        context=kwargs.get("context"),
        provider_kwargs=kwargs,
        provider=provider,
    )
    source_run, explicit = await _retry_source(state_store, original_run_id, tenant_id)
    skill_session = SkillSession.prepare(
        agent, loop=_run_entry._new_run_loop(agent, loop), explicit=explicit
    )
    with skill_session.bind():
        await _check_retry_source(state_store, source_run, original_run_id)
        resolved_loop = _run_entry._new_run_loop(agent, loop)
        _run_tools._validate_loop_skill_compat(agent, resolved_loop)
        # 3. Blocked-content admission for the retried run and its runs= sources
        #    (R1.8), then the same runs, same knobs, before any history is rebuilt.
        replay_req, replay_runs = await _retry_replay(
            source_run, resolved_loop, state_store, tenant_id, agent._limits, original_run_id
        )
        # 3a. Read traces and reconstruct approximate history. Reasoning replay
        #    blocks only seed a retry on the model that made them.
        traces, initial_history = await _retry_history(
            state_store, source_run, agent, provider, kwargs
        )
        # The source's own input follows its persisted seed prefix.
        user_input, source_parts, raw_user, users = _retry_source_input(source_run, traces)
        if explicit:
            retry_context, retry_user = _refresh_retry_user(
                initial_history, users, raw_user, user_input, source_parts, skill_session
            )
        # 5. Create the retry run
        resolved_strategy = strategy or NativeToolCalling()
        provider_kwargs = dict(kwargs)
        run_id = generate_ulid()
        parent_ctx = _context.get_delegation_context()
        parent_run_id, delegation_level = resolve_parent_link(parent_ctx, state_store)
        effective_max_depth = _run_entry._resolve_max_delegation_depth(
            agent.max_delegation_depth, parent_ctx
        )

        _system_prompt = agent.get_system_prompt(loop=resolved_loop)

        run_meta, _ = _run_entry._new_run_metadata(
            agent,
            resolved_loop,
            skill_session,
            metadata,
            effective_max_depth,
            retry_of=original_run_id,
            user_input=user_input,
        )

        image_runtime = _retry_media(
            agent,
            state_store,
            resolved_loop,
            tenant_id,
            run_id,
            source_parts,
            initial_history,
            replay_runs,
        )
        input_edges = await _retry_edges(image_runtime, source_parts)
        ctx_token = None
        mcp_meta: dict[str, Any] = {}
        result: RunResult | None = None
        lifecycle = _run_start._RunLifecycle(
            run_id, state_store, getattr(agent, "_task_manager", None)
        )

        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        try:
            try:
                seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
                    state_store,
                    agent,
                    provider,
                    resolved_loop,
                    resolved_strategy,
                    provider_kwargs,
                    initial_history,
                    user_input=user_input,
                    run_id=run_id,
                    tenant_id=tenant_id,
                )
                create_result = await _run_entry._create_runtime_run(
                    state_store,
                    resolved_loop,
                    run_id,
                    agent.name,
                    input_data=_run_images._input_data(user_input, source_parts),
                    model=provider.model,
                    strategy=type(resolved_strategy).__name__,
                    parent_run_id=parent_run_id,
                    delegation_level=delegation_level,
                    tenant_id=tenant_id,
                    meta=run_meta,
                    retry_of_run_id=original_run_id,
                    **seed_kwargs,
                    **input_edges,
                )
                run_id = create_result.run_id
                state_store = lifecycle._claim(state_store, resolved_loop, create_result)

                trace_link, extra_notifier = await _run_start._start_new_run(
                    lifecycle,
                    agent,
                    provider,
                    tenant_id,
                    seed_kwargs,
                    run_guardrails,
                    extra_notifier,
                    parent_ctx,
                    {
                        "agent_name": agent.name,
                        "system_prompt": _system_prompt,
                        "retry_of": original_run_id,
                    },
                )
                this_ctx = DelegationContext(
                    run_id=run_id,
                    trace_link=trace_link,
                    delegation_level=delegation_level,
                    persisted=True,
                    store_identity=get_store_identity(state_store),
                    max_delegation_depth=effective_max_depth,
                    tenant_id=tenant_id,
                    tenant_explicit=tenant_explicit,
                    state_store=state_store,
                    lease_interval_s=agent.lease_interval.total_seconds(),
                    loop=asyncio.get_running_loop(),
                )
                ctx_token = set_delegation_context(this_ctx)

                await _run_tools._discover_runtime_tools(
                    agent, lifecycle.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
                )

                replay_plan, initial_history, image_view = await _start_retry_history(
                    agent,
                    provider,
                    resolved_loop,
                    provider_kwargs,
                    state_store,
                    run_id,
                    original_run_id,
                    source_run,
                    replay_req,
                    replay_runs,
                    initial_history,
                    image_runtime,
                    explicit,
                )

                if explicit:
                    trace_user = user_trace_message(
                        user_input,
                        [retry_user],
                        render_band(retry_context, "stable"),
                        retry_context,
                        session=skill_session,
                    )
                    await _run_images._record_user_trace(
                        lifecycle.recorder, extra_notifier, run_id, trace_user
                    )

                result = await resolved_loop.run(
                    agent=agent,
                    provider=provider,
                    strategy=resolved_strategy,
                    user_input=user_input,
                    run_id=run_id,
                    recorder=lifecycle.recorder,
                    notifier=extra_notifier,
                    initial_history=initial_history,
                    provider_kwargs=provider_kwargs or None,
                    state_store=state_store,
                    **_run_images._image_loop_kwargs(resolved_loop.run, image_runtime, image_view),
                )

                if replay_plan is not None:
                    result.replay = replay_plan.summary
                _run_observers._attach_mcp_meta(result, mcp_meta)
                result = await _run_outcome._persist_loop_outcome(
                    recorder=lifecycle.recorder,
                    provider_kwargs=provider_kwargs,
                    state_store=state_store,
                    run_id=run_id,
                    result=result,
                )

                await lifecycle._finish(lifecycle.recorder, extra_notifier, result)

                return result

            except RunFencedError:
                return await lifecycle._fenced(lifecycle.recorder, extra_notifier)

            except Exception as exc:
                await lifecycle._failed(exc, mcp_meta, lifecycle.recorder, extra_notifier)
                raise

        except asyncio.CancelledError:
            cancelled_result, swallow = await lifecycle._cancelled(
                result, mcp_meta, lifecycle.recorder, extra_notifier
            )
            if swallow:
                return cancelled_result
            raise

        finally:
            _run_images._release_memory_edges(image_runtime, state_store, run_id)
            lifecycle._release()
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


def _cached_stream_event(cached: RunResult, run_id: str) -> RunEvent:
    from dendrux.types import RunEvent

    event_type = {
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(cached.status, RunEventType.RUN_COMPLETED)
    return RunEvent(
        type=event_type,
        run_id=run_id,
        run_result=cached,
        error=cached.error if event_type == RunEventType.RUN_ERROR else None,
        phase="final_answer" if cached.status == RunStatus.SUCCESS else None,
    )


def _stream_claim(
    store: StateStore,
    loop: Loop,
    created: CreateRunResult,
    child_claim: ClaimResult | None,
    *,
    root_limits: Limits | None,
) -> StateStore:
    """The new run's owned store: a managed child's token comes from its bind."""
    owned = _state.claimed_store(
        store, created.run_id, child_claim or _run_start._creation_claim(loop, created)
    )
    _managed.stamp_root_header(owned, created.run_id, limits=root_limits)
    return owned


def _stream_entry_checks(
    agent: Agent, tenant_id: str | None, shared: dict[str, Any], parent_ctx: Any, **checks: Any
) -> None:
    try:
        entry_tenant = resolve_child_tenant(parent_ctx, tenant_id)
    except ValueError:
        return
    shared["existing_run"] = True
    runner_entry_checks(agent, "stream", tenant_id=entry_tenant, **checks)
    shared["existing_run"] = False


async def _resolve_stream_store(
    store: StateStore | None,
    resolver: Callable[[], Any] | None,
    agent: Agent,
    tenant_id: str | None | _OmittedType,
) -> StateStore | None:
    if store is None and resolver is not None:
        store = await resolver()
        resolve_runner_tenant(agent, "stream", tenant_id, state_store=store)
    return store


async def _stream_replay_runs(
    request: ReplayRequest | None,
    store: StateStore | None,
    tenant_id: str | None,
    loop: Loop,
    shared: dict[str, Any],
    limits: Limits,
) -> tuple[list[Any], tuple[str, ...]]:
    if request is None:
        return [], ()
    shared["existing_run"] = True
    if store is None:
        raise ValueError(
            "runs= requires persistence (database_url or state_store): "
            "replay reads the earlier runs' traces."
        )
    loaded = await _run_entry._load_new_run_replay(
        store, request, tenant_id, loop, limits=limits, method="run_stream"
    )
    shared["existing_run"] = False
    return loaded.runs, loaded.omitted


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    max_delegation_depth: int | None,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


@overload
def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = ...,
    context: list[ContextBlock] | None = ...,
    activate_skills: Sequence[str] | None = ...,
    runs: list[str] | None = ...,
    replay_last: ReplayLast = ...,
    replay_reasoning: bool = ...,
    strategy: Strategy | None = ...,
    loop: Loop | None = ...,
    state_store: StateStore | None = ...,
    state_store_resolver: Callable[[], Any] | None = ...,
    tenant_id: str | None | _OmittedType = ...,
    metadata: dict[str, Any] | None = ...,
    extra_notifier: LoopNotifier | None = ...,
    idempotency_key: str | None = ...,
    output_type: type | None = ...,
    **kwargs: Any,
) -> RunStream: ...


def run_stream(
    agent: Agent,
    *,
    provider: LLMProvider,
    user_input: UserContent,
    history: list[ChatMessage] | None = None,
    context: list[ContextBlock] | None = None,
    activate_skills: Sequence[str] | None = None,
    runs: list[str] | None = None,
    replay_last: ReplayLast = 0,
    replay_reasoning: bool = True,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tenant_id: str | None | _OmittedType = _OMITTED,
    metadata: dict[str, Any] | None = None,
    extra_notifier: LoopNotifier | None = None,
    max_delegation_depth: int | None | _UnsetType = _UNSET_DEPTH,
    idempotency_key: str | None = None,
    output_type: type | None = None,
    _managed_invocation: Any = None,
    **kwargs: Any,
) -> RunStream:
    """Stream an agent run as RunEvents, returning a RunStream.

    Synchronous — returns immediately with run_id available. All async
    setup (DB row, notifiers) runs lazily on first iteration. With
    ``idempotency_key``, the initial run_id is provisional until that setup
    resolves the key. A cached outcome yields one terminal event, without
    replaying text or tool events. Read its answer from ``stream.result``.

    Accepts either an already-resolved ``state_store`` or an async
    ``state_store_resolver`` callable. The resolver is called once in
    the generator preamble.

    Error semantics: exceptions are caught, persisted, and yielded as
    RUN_ERROR events. No exception is raised to the consumer.

    Lifecycle ownership:
      - Runner emits RUN_STARTED, RUN_ERROR, and handles cancellation cleanup.
      - Loop yields execution outcomes (RUN_COMPLETED, RUN_PAUSED) and
        intermediate events (TEXT_DELTA, TOOL_USE_*, TOOL_RESULT).
      - Runner persists loop outcomes before forwarding them.

    ``_managed_invocation`` (internal, 0.7): stream a managed child created by
    its parent's atomic bind instead of ``create_run``. Its id, parent link,
    tenant and depth come from the invocation; a failure before the bind
    raises to the managed driver instead of yielding ``RUN_ERROR``.
    """
    invocation = _managed_invocation
    # Validate depth early — synchronous, before generator starts.
    # Full resolution (parent context, clamping) happens inside _generate()
    # because parent_ctx requires async state_store resolution first.
    # But validation of the raw input can fail fast here.
    requested_tenant_id = tenant_id
    tenant_explicit = _tenant_scope_chosen(tenant_id)
    tenant_id = resolve_runner_tenant(agent, "stream", tenant_id, state_store=state_store)
    if (
        not isinstance(max_delegation_depth, _UnsetType)
        and max_delegation_depth is not None
        and (not isinstance(max_delegation_depth, int) or max_delegation_depth < 0)
    ):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, "
            f"got {max_delegation_depth!r}"
        )

    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    # Validate input + chat history synchronously so TypeError / ValueError
    # surfaces to the caller immediately, not as a RUN_ERROR event. Image
    # ingest is async: it runs on first iteration, before create_run, and
    # its errors are raised from the first ``__anext__`` (no run row).
    entry_user_input = user_input  # as passed, for the runner entry checks
    user_input, input_items = _chat.split_user_content(user_input)
    validate_chat_history(history)
    image_input = bool(input_items) or history_has_images(history)
    normalized_history = [] if image_input else normalize_chat_history(history)
    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = _run_entry._new_run_loop(agent, loop)
    _run_tools._validate_loop_skill_compat(agent, resolved_loop)
    if image_input:
        _run_images._require_image_loop(resolved_loop, "run_stream")
    replay_req = validate_replay_args(
        runs, replay_last, replay_reasoning, loop=resolved_loop, context=context
    )
    # When chat history is provided, build the complete starting message list
    # (loops use initial_history as-is and do not separately append user_input).
    # The new user_input is recorded by the runner itself inside _generate()
    # so seeded prior turns stay out of react_traces. With runs=, the replayed
    # prefix is folded in after expansion.
    skill_session = SkillSession.prepare(agent, loop=resolved_loop, explicit=activate_skills)
    with skill_session.bind():
        if replay_req is None and not image_input:
            initial_seeded = fold_context(
                normalized_history, context, user_input, session=skill_session
            )
        else:
            validate_context(context)
            initial_seeded = None
    provider_kwargs = dict(kwargs)

    # A managed child's id is its parent's bind candidate, minted at the handoff.
    run_id = _managed.new_run_id(invocation)

    # Shared mutable state — generator populates on first iteration,
    # cleanup reads if the consumer abandons after setup.
    _shared: dict[str, Any] = {"state_store": state_store}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext(skill_session)

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        """Inner generator — lazy async setup, then loop stream with lifecycle.

        The entire body is wrapped in try/except so that setup failures
        (state_store_resolver, create_run, notifier init) are also caught
        and yielded as RUN_ERROR — honoring the "no exception to consumer" contract.
        """
        # store is captured in the closure for the error handler.
        # Starts as the pre-provided value (often None for the lazy path).
        nonlocal run_id, extra_notifier
        store = state_store
        startup = _run_start._RunLifecycle(run_id, store, None)
        seed_kwargs: dict[str, Any] = {}
        run_guardrails = None
        seeded_history = initial_seeded
        replay_plan: ReplayPlan | None = None
        image_runtime: ImageRuntime | None = None
        user_parts: tuple[ContentPart, ...] | None = None
        history_messages = normalized_history
        # Entry checks after tenant inheritance, before any I/O. Like image
        # setup they raise (no RUN_ERROR event, no run row, no hooks). A child
        # whose explicit tenant differs from its parent's still fails later, as
        # RUN_ERROR, exactly as before.
        # A managed child never inherits the caller's delegation context.
        entry_parent = _managed.parent_context(invocation, _context.get_delegation_context)
        _stream_entry_checks(
            agent,
            tenant_id,
            _shared,
            entry_parent,
            metadata=metadata,
            user_input=entry_user_input,
            history=history,
            context=context,
            provider_kwargs=provider_kwargs,
            idempotency_key=idempotency_key,
            provider=provider,
        )
        if image_input:
            # Image setup raises (no RUN_ERROR event, no run row, no hooks).
            _shared["existing_run"] = True
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
            _shared["state_store"] = store
            image_runtime, user_parts, history_messages, cached = await _run_images._entry_media(
                agent,
                provider,
                resolved_loop,
                store,
                resolve_child_tenant(_context.get_delegation_context(), tenant_id),
                run_id,
                True,
                user_input,
                input_items,
                history,
                history_messages,
                provider_kwargs,
                idempotency_key,
                output_type,
                context,
                replay_req,
                skill_session,
            )
            if cached is not None:
                run_id = cached.run_id
                stream.run_id = run_id
                yield _cached_stream_event(cached, run_id)
                return
            with skill_session.bind():
                seeded_history = _run_entry._new_run_history(
                    replay_req, history_messages, context, user_input, skill_session, user_parts
                )
            _shared["existing_run"] = False
        try:
            # 1. Resolve state store lazily (may create DB engine)
            store = await _resolve_stream_store(
                store, state_store_resolver, agent, requested_tenant_id
            )
            await ensure_agent_limits_fit(agent, store)
            _shared["state_store"] = store
            # --- Delegation context ---
            parent_ctx = _managed.parent_context(invocation, _context.get_delegation_context)
            parent_run_id, delegation_level = _managed.parent_link(invocation, parent_ctx, store)
            eff_max_depth = _run_entry._resolve_max_delegation_depth(
                max_delegation_depth, parent_ctx
            )
            eff_tenant = resolve_child_tenant(parent_ctx, tenant_id)
            root_limits = await _managed.enter(agent, invocation, store, resolved_loop)
            if image_runtime is None and _run_contracts._accepts_kwarg(
                resolved_loop.run_stream, "image_runtime"
            ):
                image_runtime = _run_images._image_runtime(agent, store, eff_tenant, run_id)
            # Resolve system prompt BEFORE create_run() so skill loading
            # failures don't leave a DB row stuck in running.
            _system_prompt = agent.get_system_prompt(loop=resolved_loop)
            _run_entry._require_idempotency_store(store, idempotency_key)

            replay_runs, replay_omitted = await _stream_replay_runs(
                replay_req, store, eff_tenant, resolved_loop, _shared, agent._limits
            )

            if store is not None:
                run_meta, fingerprint = _run_entry._new_run_metadata(
                    agent,
                    resolved_loop,
                    skill_session,
                    metadata,
                    eff_max_depth,
                    user_input=user_input,
                    idempotency_key=idempotency_key,
                    output_type=output_type,
                    history=history_messages,
                    context=context,
                    replay=replay_req,
                    parts=user_parts,
                    streaming=True,
                )
                try:
                    seed_kwargs, run_guardrails = await _run_entry._runtime_seeds(
                        store,
                        agent,
                        provider,
                        resolved_loop,
                        resolved_strategy,
                        provider_kwargs,
                        seeded_history,
                        user_input=user_input,
                        run_id=run_id,
                        tenant_id=eff_tenant,
                        replay_req=replay_req,
                        replay_runs=replay_runs,
                        replay_omitted=replay_omitted,
                        normalized_history=history_messages,
                        context=context,
                        user_parts=user_parts,
                    )
                    created = await _run_entry._create_projected_run(
                        False,
                        store,
                        resolved_loop,
                        run_id,
                        agent.name,
                        input_data=_run_images._input_data(user_input, user_parts),
                        model=provider.model,
                        strategy=type(resolved_strategy).__name__,
                        parent_run_id=parent_run_id,
                        delegation_level=delegation_level,
                        tenant_id=eff_tenant,
                        meta=run_meta,
                        idempotency_key=idempotency_key,
                        idempotency_fingerprint=fingerprint,
                        **_run_images._input_edge_kwargs(image_runtime, user_parts),
                        **seed_kwargs,
                        invocation=invocation,
                    )
                    # A child bound by another caller reads as existing_active below.
                    create_result, child_claim = _run_entry._managed_creation(created)
                except IdempotencyConflictError:
                    # No row was created and no lifecycle opened; the error
                    # handler must not fire failure hooks for a run that
                    # never started (mirrors run(), which re-raises bare).
                    _shared["existing_run"] = True
                    raise
                _shared["existing_run"] = create_result.outcome != "created"
                run_id = create_result.run_id
                stream.run_id = run_id
                if create_result.outcome == "existing_terminal":
                    cached = await _run_pending._build_cached_result(store, run_id, output_type)
                    yield _cached_stream_event(cached, run_id)
                    return
                if create_result.outcome == "existing_active":
                    raise RunAlreadyActiveError(run_id, create_result.status)
                store = _stream_claim(
                    store, resolved_loop, create_result, child_claim, root_limits=root_limits
                )
                _shared["state_store"] = store
                _shared["claimed"] = True

            # 2. Set delegation context for the generator lifetime
            this_ctx = DelegationContext(
                run_id=run_id,
                delegation_level=delegation_level,
                persisted=store is not None,
                store_identity=get_store_identity(store),
                max_delegation_depth=eff_max_depth,
                tenant_id=eff_tenant,
                tenant_explicit=tenant_explicit,
                state_store=store,
                lease_interval_s=agent.lease_interval.total_seconds(),
                loop=asyncio.get_running_loop(),
            )
            startup.run_id = run_id
            startup.store = store
            trace_link, extra_notifier = await _run_start._start_new_run(
                startup,
                agent,
                provider,
                eff_tenant,
                seed_kwargs,
                run_guardrails,
                extra_notifier,
                parent_ctx,
                {"agent_name": agent.name, "system_prompt": _system_prompt},
                before_event=lambda: stream_context.set_delegation(this_ctx),
            )
            this_ctx = replace(this_ctx, trace_link=trace_link)
            stream_context.set_delegation(this_ctx)
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = startup.recorder
            _shared["notifier"] = extra_notifier

            yield RunEvent(type=RunEventType.RUN_STARTED, run_id=run_id)

            await _run_tools._discover_runtime_tools(
                agent, startup.recorder, extra_notifier, run_id, resolved_loop, mcp_meta
            )

            replay_plan, seeded_history = await _run_entry._apply_replay(
                agent,
                provider,
                resolved_loop,
                provider_kwargs,
                state_store=store,
                run_id=run_id,
                replay_req=replay_req,
                replay_runs=replay_runs,
                replay_omitted=replay_omitted,
                normalized_history=history_messages,
                context=context,
                user_input=user_input,
                seeded_history=seeded_history,
                user_parts=user_parts,
            )
            await _run_images._register_memory_input(image_runtime, store, run_id, user_parts)
            image_view = await _run_entry._new_run_image_view(
                image_runtime, store, run_id, seeded_history, replay_runs, replay_plan, user_parts
            )

            # When chat history was provided, the loop receives a complete
            # message list as initial_history and skips its own user-message
            # recording. Record the new user_input here so it still lands in
            # react_traces (seeded prior turns are NOT recorded).
            await _run_entry._record_seeded_user(
                startup.recorder,
                extra_notifier,
                run_id,
                user_input,
                seeded_history,
                context,
                skill_session,
            )

            # 3. Stream the loop. The interrupt signal lets cancel_run reach
            # an in-flight provider stream owned by this process; the closing
            # wrapper guarantees the loop generator is finalized in this task
            # when the consumer abandons the stream.
            interrupt = agent._task_manager.register_interrupt(run_id)
            loop_stream = resolved_loop.run_stream(
                agent=agent,
                provider=provider,
                strategy=resolved_strategy,
                user_input=user_input,
                run_id=run_id,
                recorder=startup.recorder,
                notifier=extra_notifier,
                initial_history=seeded_history,
                provider_kwargs=provider_kwargs or None,
                output_type=output_type,
                state_store=store,
                interrupt=interrupt,
                **_run_images._image_loop_kwargs(
                    resolved_loop.run_stream, image_runtime, image_view
                ),
            )
            async with aclosing(
                _run_outcome._stream_loop_events(
                    loop_stream,
                    state_store=store,
                    recorder=startup.recorder,
                    notifier=extra_notifier,
                    run_id=run_id,
                    provider_kwargs=provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                    replay_plan=replay_plan,
                    new_run=True,
                )
            ) as events:
                async for event in events:
                    yield event

        except (ManagedRootStoppedError, RunFencedError) as stop:
            result = await startup._stopped(stop, startup.recorder, extra_notifier)
            yield _run_events._fenced_stream_event(run_id, result)

        except Exception as exc:
            # Before its bind a managed child is not a run: the driver decides
            # for its parent, exactly as for run() (0.7).
            _managed.raise_unbound(invocation, _shared.get("claimed"))
            # Persist error, yield RUN_ERROR, return cleanly. No re-raise.
            # Covers both setup failures and loop execution errors.
            await _run_start._stream_run_failure(
                store, _shared, startup.recorder, extra_notifier, run_id, exc
            )
            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=RunStatus.ERROR,
                    error=_run_errors._safe_error(exc),
                    error_info=_run_errors._provider_error_info(exc),
                    meta=_run_outcome._stream_error_meta(mcp_meta, exc),
                ),
                run_id=run_id,
                error=_run_errors._safe_error(exc),
            )
        finally:
            _run_images._release_memory_edges(image_runtime, store, run_id)
            if not _shared.get("existing_run"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Two responsibilities:
          (1) If the run is still RUNNING in the store, finalize it as
              CANCELLED (CAS-guarded so we don't clobber a concurrent
              terminal status set elsewhere).
          (2) If the lifecycle pair was opened locally (recorder/notifier
              stashed by ``_generate`` after ``run_started`` fired), close
              it with a ``run_finished`` event on the LOCAL notifier so
              OTel root spans close. This fires regardless of whether
              the CAS won — if some other caller already finalized the
              run, our local notifier still needs the close event.

        Status resolution for the close event:
          - CAS won → CANCELLED.
          - CAS lost + store available → query store for actual status.
          - No store → CANCELLED (best-effort fallback).
        """
        if not _shared.get("claimed") and _shared.get("state_store") is not None:
            return None
        store = _shared.get("state_store")

        cancel_won = False
        persisted_row: RunRecord | None | _OmittedType = _OMITTED

        def mark_cancelled() -> None:
            nonlocal cancel_won
            cancel_won = True

        partial_answer = _run_outcome._partial_answer(_shared)
        if store is not None:
            try:
                cancel_won, persisted_row = await _run_events._persist_run_cancellation(
                    store,
                    run_id,
                    {
                        "reason": CancelReason.STREAM_CLOSED.value,
                        "partial_output": partial_answer is not None,
                    },
                    on_won=mark_cancelled,
                    answer=partial_answer,
                    emit_event=True,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
                )
            except Exception:
                logger.error(  # noqa: TRY400 - Keep safe_exception_info() to exclude vendor text and exception chains.
                    "Failed to cancel run %s during stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        # Close the lifecycle pair if it was opened.
        if "notifier" in _shared:
            cleanup_recorder = _shared.get("recorder")
            terminal_status = RunStatus.CANCELLED
            cancel_persisted = cancel_won
            if store is not None and not cancel_won:
                try:
                    run_record = (
                        await store.get_run(run_id)
                        if isinstance(persisted_row, _OmittedType)
                        else persisted_row
                    )
                    if run_record is not None:
                        terminal_status = RunStatus(run_record.status)
                        cancel_persisted = terminal_status == RunStatus.CANCELLED
                except Exception:  # noqa: S110 - Cleanup retains CANCELLED fallback when persisted status is unavailable.
                    pass  # CANCELLED is a safe fallback
            cancelled_result = RunResult(
                run_id=run_id,
                status=terminal_status,
                meta={**deepcopy(mcp_meta), CANCEL_PERSISTED_META_KEY: cancel_persisted},
            )
            try:
                await _hooks.record_run_finished(cleanup_recorder, run_id, cancelled_result)
                await _hooks.notify_run_finished(_shared.get("notifier"), run_id, cancelled_result)
            except Exception:
                logger.warning(
                    "Lifecycle finish hooks failed during stream cleanup for run %s",
                    run_id,
                    exc_info=safe_exception_info(),
                )

            return cancelled_result
        return None

    stream = _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _run_events._shielded_cleanup(_cleanup, run_id=run_id),
    )
    return stream


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

    resolved_strategy = strategy or NativeToolCalling()
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
        else _resume_claim_payload(
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
    from dendrux.runtime.submit import (
        DEFAULT_DENIAL_REASON,
        approval_gates_of,
        is_client_call,
    )

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
    _mark_work(on_work)
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
        strategy if strategy is not None else NativeToolCalling(),
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
            _mark_work(on_work)
            await _ntfy_tool(notifier, run_id, tc, tr, iteration, warnings=notifier_warnings)
        result_msg = _format_result(strategy, tr)
        if not executed or tr.meta.get("code") == "max_calls":
            result_msg.meta[NOT_EXECUTED_META_KEY] = True
        target_history.append(result_msg)
        result_msg = await record_message(recorder, run_id, result_msg, iteration)
        _mark_work(on_work)
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
            _mark_work(on_work)
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
            _mark_work(on_work)
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
    validate = submission_validator(
        run_id, raw, agent=agent, expected_pause_id=request.generation, call_results=results
    )
    record = await store.get_run(run_id)
    require_pause_id(
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
        await report_governance_change(agent, store, run_id, raw)
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
        session = await continuation(agent, store, run_id, loop=loop)
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
        require_pause_id(
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
        validate_result_names(run_id, pause, tool_results)
        if not skip_claim:
            tool_results = await prepare_resume_results(agent, store, run_id, pause, tool_results)
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
    released = await _release_claim(store, run_id, status)
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

    async with _release_setup_failure(state_store, run_id, expected_status, enabled=_skip_claim):
        skill_session, user_input, input_items = await _resume_session(
            agent, state_store, run_id, loop, user_input, tool_results, decisions, expected_status
        )
    from dendrux.runtime._continuation_guardrails import _continuation_projection

    with skill_session.bind(), await _continuation_projection(agent, state_store, run_id):
        async with _release_setup_failure(
            state_store, run_id, expected_status, enabled=_skip_claim
        ):
            (
                run_record,
                reads,
                raw_pause,
                pause_state,
                recovering,
                _saved_pii_mapping,
            ) = await _read_resume_state(
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
            ) = await _resume_submission(
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
            stopped = await _stop_marked_continuation(
                state_store,
                run_id,
                run_record,
                pause_state,
                agent=agent,
                claimed_here=_claimed_in_call(state_store, run_id, skip_claim=_skip_claim),
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
            ) = await _project_adopted_submission(
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
            history_results, history_input = _resume_history_inputs(
                recovering, tool_results, user_input
            )
            ctx = await _prepare_resume(
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
            blocked_stop = await _replay_submission_or_stop(
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
                return await _finish_resumed_result(state_store, run_id, ctx, blocked_stop)

            ctx_token = set_delegation_context(
                _resume_delegation(
                    agent, state_store, run_id, run_record, trace_link, _tenant_explicit
                )
            )

            if recovering:
                stop = await _resolve_recovery(
                    agent,
                    state_store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_tenant,
                    on_work=progress._start,
                )
                if stop is not None:
                    return await _finish_resumed_result(state_store, run_id, ctx, stop)

            if not recovering:
                stop = await _resume_approval_work(
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
                    return await _finish_resumed_result(state_store, run_id, ctx, stop)

            result = await _run_resumed_loop(
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
            return await _finish_resumed_result(state_store, run_id, ctx, result)

        except asyncio.CancelledError:
            cancelled_result, swallow = await _cancel_resumed_run(agent, state_store, run_id, ctx)
            if swallow:
                return cancelled_result
            raise

        except (ManagedRootStoppedError, RunFencedError) as stop:
            return await _stopped_resume_result(stop, state_store, run_id, ctx)

        except Exception as exc:
            await _handle_resume_failure(
                state_store,
                run_id,
                expected_status,
                ctx,
                exc,
                progress,
                _resume_warning_meta(ctx, mcp_meta),
            )
            raise

        finally:
            agent._task_manager.release_current(run_id)
            if ctx_token is not None:
                reset_delegation_context(ctx_token)


async def _stream_resume_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore | None,
    state_store_resolver: Callable[[], Any] | None,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    input_items: tuple[Any, ...],
) -> tuple[StateStore, list[ToolResult] | None, tuple[ContentPart, ...] | None, bool]:
    results_prepared = False
    # Image setup raises (the run stays waiting; no RUN_ERROR event).
    if store is None and state_store_resolver is not None:
        store = await state_store_resolver()
    if store is None:
        from dendrux.errors import PersistenceNotConfiguredError

        raise PersistenceNotConfiguredError()
    await ensure_agent_limits_fit(agent, store)
    raw = await store.get_pause_state(run_id)
    if raw is None:
        raise ValueError("Run has no pause state")
    require_pause_id(
        agent, expected_pause_id, pause_id=raw.get("pause_id"), store=store, run_id=run_id
    )
    media_pause = PauseState.from_dict(raw)
    if media_pause.agent_name and media_pause.agent_name != agent.name:
        raise ValueError("Agent name mismatch on resume")
    if stream_results is not None:
        validate_result_names(run_id, media_pause, stream_results)
        stream_results = await prepare_resume_results(
            agent, store, run_id, media_pause, stream_results
        )
        results_prepared = True
    stream_results, user_parts = await _run_images._ingest_resume_images(
        agent,
        provider,
        store,
        run_id,
        loop=None,
        tool_results=stream_results,
        input_items=input_items,
        loop_method="run_stream",
    )
    return store, stream_results, user_parts, results_prepared


async def _stream_resume_state(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    results_prepared: bool,
    stream_context: StreamContext,
) -> tuple[Any, _run_contracts._PauseReads, Any, PauseState, bool, list[ToolResult] | None]:
    skill_session = await continuation(agent, store, run_id)
    stream_context.set_session(skill_session)

    run_record, reads, raw_pause, pause_state, recovering, _ = await _read_resume_state(
        store,
        agent,
        provider,
        run_id,
        False,
        expected_pause_id,
        None,
        None,
        streaming=True,
    )

    if stream_results is not None:
        validate_result_names(run_id, pause_state, stream_results)
        if not results_prepared:
            stream_results = await prepare_resume_results(
                agent, store, run_id, pause_state, stream_results
            )

    return run_record, reads, raw_pause, pause_state, recovering, stream_results


class _StreamClaim(NamedTuple):
    store: StateStore
    record: Any
    reads: _run_contracts._PauseReads
    raw_pause: Any
    pause: Any
    recovering: bool
    results: list[ToolResult] | None
    expected: str
    stopped: RunResult | None = None


async def _stream_resume_claim(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore,
    run_id: str,
    claimed: ClaimResult | None,
    stream_context: StreamContext,
    shared: dict[str, Any],
    *,
    expected_pause_id: str | None,
    stream_results: list[ToolResult] | None,
    results_prepared: bool,
    user_input: str | None,
    user_parts: tuple[ContentPart, ...] | None,
) -> _StreamClaim:
    """Own the run for a streamed resume: claim its pause, or take an owned continuation.

    ``claimed`` (0.7): the run is already owned (a managed parent claimed for
    adoption) and continues from its checkpoint. A continuation stopped by a
    marked run carries its result in ``stopped``.
    """
    from dendrux.types import RunResult

    if claimed is not None:
        owned, record, reads, rebuilt = await _stream_claimed_state(
            agent, store, run_id, claimed, stream_context
        )
        shared["state_store"], shared["claimed"] = owned, True
        if isinstance(rebuilt, RunResult):
            return _StreamClaim(owned, record, reads, None, None, True, None, "", rebuilt)
        return _StreamClaim(
            owned, record, reads, None, rebuilt, True, None, RunStatus.RUNNING.value
        )
    record, reads, raw_pause, pause, recovering, stream_results = await _stream_resume_state(
        agent,
        provider,
        store,
        run_id,
        expected_pause_id,
        stream_results,
        results_prepared,
        stream_context,
    )
    # 4. Determine expected status and claim
    expected = _resume_expected_status(stream_results, user_input)
    store, expected, raw_pause = await _claim_resume(
        store,
        run_id,
        raw_pause,
        agent,
        expected,
        expected_pause_id=expected_pause_id,
        results=stream_results,
        user_input=user_input,
        user_parts=user_parts,
        checkpoint=reads.checkpoint,
        on_release=partial(_remember_stream_release, shared),
    )
    shared["state_store"], shared["claimed"] = store, True
    stopped = await _stop_marked_continuation(
        store, run_id, record, pause, agent=agent, claimed_here=True
    )
    return _StreamClaim(
        store, record, reads, raw_pause, pause, recovering, stream_results, expected, stopped
    )


async def _stream_claimed_state(
    agent: Agent,
    store: StateStore,
    run_id: str,
    claim: ClaimResult,
    stream_context: StreamContext,
) -> tuple[StateStore, Any, _run_contracts._PauseReads, PauseState | RunResult]:
    """An owned run's continuation state, rebuilt from its checkpoint (0.7 adoption).

    The same reads and stop checks as reclaim's continuation: a marked run stops
    before any model call (its stop result is returned instead of a pause); a
    rebuild failure is recorded like a reclaim's.
    """
    from dendrux.errors import ManagedStateError
    from dendrux.skills._session import SkillSession

    session = SkillSession.prepare(agent)
    session.run_id = run_id
    stream_context.set_session(session)
    owned = _state.claimed_store(store, run_id, claim)
    state = claim.loop_state or {}
    if state.get("submission"):
        # Adoption completes its submission atomically; anything else is not ours.
        raise ManagedStateError(reason="binding_outstanding", run_id=run_id)
    history_records: list[Any] = []
    async with _reclaim._reclaim_setup(owned, run_id):
        record = await owned.get_run(run_id)
        stopped = await _stop_marked_continuation(
            owned,
            run_id,
            record,
            agent=agent,
            iteration=state.get("iteration", 0),
            usage=_usage_from_dict(state.get("usage", {})),
        )
        if stopped is not None:
            return owned, record, _run_contracts._PauseReads(record), stopped
        pause = await _reclaim.rebuild_pause(
            agent, owned, run_id, state, history_records=history_records
        )
    return owned, record, _run_contracts._PauseReads(record, history_records=history_records), pause


def _stopped_stream_event(result: RunResult) -> RunEvent:
    from dendrux.types import RunEvent

    return RunEvent(
        type=RunEventType.RUN_CANCELLED
        if result.status == RunStatus.CANCELLED
        else RunEventType.RUN_PAUSED,
        run_result=result,
    )


async def _stream_approval_events(
    agent: Agent,
    store: StateStore,
    run_id: str,
    ctx: _run_contracts._ResumeContext,
    expected: str,
    results: list[ToolResult] | None,
    user_input: str | None,
    image_runtime: ImageRuntime | None,
    meta: dict[str, Any],
    on_work: Callable[[], None] | None = None,
) -> AsyncGenerator[RunEvent, None]:
    from dendrux.types import RunEvent

    approval, approving = await _resume_approval_batch(
        agent,
        store,
        run_id,
        ctx,
        expected,
        results,
        user_input,
        image_runtime,
        streaming=True,
        on_work=on_work,
    )
    if approval is not None:
        for tc, tr in approval.results:
            yield RunEvent(
                type=RunEventType.TOOL_RESULT,
                tool_call=tc,
                tool_result=tr,
                iteration=ctx.pause_state.iteration,
                phase="commentary",
            )
    stop = await _finish_approval_batch(run_id, ctx, expected, results, approval, approving)
    if stop is not None:
        _run_observers._attach_mcp_meta(stop, meta)
        yield _stopped_stream_event(await _finish_resumed_result(store, run_id, ctx, stop))


def _resume_expected_status(results: list[ToolResult] | None, user_input: str | None) -> str:
    if results is not None:
        return RunStatus.WAITING_CLIENT_TOOL.value
    if user_input is not None:
        return RunStatus.WAITING_HUMAN_INPUT.value
    return RunStatus.WAITING_APPROVAL.value


def _mark_stream_recovery_work(shared: dict[str, Any]) -> None:
    shared["work_started"] = True
    shared["progress"]._start()


def _stream_resume_started(shared: dict[str, Any]) -> bool:
    # Work is an independent reason to finalize; it must never be made retryable.
    return bool(shared.get("work_started") or shared.get("claimed"))


async def _stream_resume_failure(
    _shared: dict[str, Any], extra_notifier: LoopNotifier | None, run_id: str, exc: Exception
) -> RunStatus:
    store = _shared.get("state_store")
    err_ctx = _shared.get("ctx")
    progress = _shared.get("progress")
    if (
        store is not None
        and progress is not None
        and progress._can_release(err_ctx, exc)
        and _resume_release_supported(store, _shared["expected_status"])
    ):
        return await _try_resume_release(store, run_id, _shared["expected_status"], err_ctx, exc)
    return await _finish_stream_resume_failure(_shared, store, err_ctx, extra_notifier, run_id, exc)


async def _finish_stream_resume_failure(
    _shared: dict[str, Any],
    store: StateStore | None,
    err_ctx: _run_contracts._ResumeContext | None,
    extra_notifier: LoopNotifier | None,
    run_id: str,
    exc: Exception,
) -> RunStatus:
    if store is not None and _stream_resume_started(_shared):
        await _run_start._persist_run_failure(store, run_id, exc, safe_event=True)
    try:
        if err_ctx is not None:
            await _hooks.record_run_failed(err_ctx.recorder, run_id, exc)
            await _hooks.notify_run_failed(err_ctx.notifier, run_id, exc)
        else:
            # Setup failed before _prepare_resume built the recorder.
            # Fire the notifier-side hook so OTel can surface the error
            # even when no DB-side recorder exists.
            await _hooks.notify_run_failed(extra_notifier, run_id, exc)
    except Exception:
        logger.warning(
            "Lifecycle hooks failed during resume_stream error handler for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )
    prelude = _shared.get("released_prelude")
    if prelude is not None:
        return await _released_resume_status(prelude, run_id)
    return RunStatus.ERROR


def _remember_stream_release(shared: dict[str, Any], store: StateStore) -> None:
    shared["released_prelude"] = store


async def _finish_resume_stream_cleanup(
    store: StateStore | None,
    run_id: str,
    _shared: dict[str, Any],
    mcp_meta: dict[str, Any],
    cancel_won: bool,
    persisted_row: RunRecord | None | _OmittedType,
) -> RunResult:
    from dendrux.types import RunResult

    cleanup_recorder = _shared.get("recorder")
    terminal_status = RunStatus.CANCELLED
    cancel_persisted = cancel_won
    if store is not None and not cancel_won:
        try:
            run_record = (
                await store.get_run(run_id)
                if isinstance(persisted_row, _OmittedType)
                else persisted_row
            )
            if run_record is not None:
                terminal_status = RunStatus(run_record.status)
                cancel_persisted = terminal_status == RunStatus.CANCELLED
        except Exception:  # noqa: S110 - Cleanup retains CANCELLED fallback when persisted status is unavailable.
            pass  # CANCELLED is a safe fallback
    cancelled_result = RunResult(
        run_id=run_id,
        status=terminal_status,
        meta={**deepcopy(mcp_meta), CANCEL_PERSISTED_META_KEY: cancel_persisted},
    )
    try:
        await _hooks.record_run_finished(cleanup_recorder, run_id, cancelled_result)
        await _hooks.notify_run_finished(_shared.get("notifier"), run_id, cancelled_result)
    except Exception:
        logger.warning(
            "Lifecycle finish hooks failed during resume stream cleanup for run %s",
            run_id,
            exc_info=safe_exception_info(),
        )

    return cancelled_result


async def _initial_stream_images(
    agent: Agent,
    provider: LLMProvider,
    store: StateStore | None,
    resolver: Callable[[], Any] | None,
    run_id: str,
    expected_pause_id: str | None,
    results: list[ToolResult] | None,
    items: tuple[Any, ...],
) -> tuple[StateStore | None, list[ToolResult] | None, tuple[ContentPart, ...] | None, bool]:
    if items or any(result.images for result in results or ()):
        return await _stream_resume_images(
            agent, provider, store, resolver, run_id, expected_pause_id, results, items
        )
    return store, results, None, False


def resume_stream(
    run_id: str,
    *,
    agent: Agent,
    provider: LLMProvider,
    state_store: StateStore | None = None,
    state_store_resolver: Callable[[], Any] | None = None,
    tool_results: list[ToolResult] | None = None,
    user_input: UserContent | None = None,
    extra_notifier: LoopNotifier | None = None,
    expected_pause_id: str | None = None,
    _tenant_explicit: bool | None = None,
    _claimed: ClaimResult | None = None,
) -> RunStream:
    """Stream a resumed run as RunEvents, returning a RunStream.

    Synchronous — returns immediately. All async setup (store resolution,
    pause state load, claim, history build) runs lazily on first iteration.

    ``_claimed`` (internal, 0.7): continue a run this caller already owns
    (a managed parent claimed for adoption) from its checkpoint, exactly as
    reclaim's continuation does, instead of loading a pause and claiming it.

    Execution exceptions yield RUN_ERROR events. Before consumption or work,
    cleanup can release the claim; the result reports the observed status and
    the failed attempt's error. Otherwise the run is finalized as ERROR.

    First event is RUN_RESUMED (not RUN_STARTED) carrying the existing run_id.

    Image inputs (``tool_results[i].images`` or a multimodal ``user_input``)
    are ingested before the claim; an image error is raised from the first
    ``__anext__`` and leaves the run waiting.
    """
    from dendrux.types import RunEvent, RunResult
    from dendrux.types import RunStream as _RunStream

    input_items: tuple[Any, ...] = ()
    if user_input is not None:
        user_input, input_items = _chat.split_user_content(user_input)
    _shared: dict[str, Any] = {"state_store": state_store, "ctx": None}

    mcp_meta: dict[str, Any] = {}

    stream_context = StreamContext()

    async def _generate() -> AsyncGenerator[RunEvent, None]:
        store = state_store
        stream_results = tool_results
        user_parts: tuple[ContentPart, ...] | None = None

        store, stream_results, user_parts, results_prepared = await _initial_stream_images(
            agent,
            provider,
            store,
            state_store_resolver,
            run_id,
            expected_pause_id,
            stream_results,
            input_items,
        )

        try:
            store = await _resolve_resume_stream_store(store, state_store_resolver)
            _shared["state_store"] = store

            await ensure_agent_limits_fit(agent, store)

            claim = await _stream_resume_claim(
                agent,
                provider,
                store,
                run_id,
                _claimed,
                stream_context,
                _shared,
                expected_pause_id=expected_pause_id,
                stream_results=stream_results,
                results_prepared=results_prepared,
                user_input=user_input,
                user_parts=user_parts,
            )
            if claim.stopped is not None:
                yield _run_events._fenced_stream_event(run_id, claim.stopped)
                return
            store, run_record, reads, raw_pause = (
                claim.store,
                claim.record,
                claim.reads,
                claim.raw_pause,
            )
            pause_state, recovering, stream_results, expected = (
                claim.pause,
                claim.recovering,
                claim.results,
                claim.expected,
            )
            _shared["expected_status"] = expected
            progress = _shared["progress"] = _run_contracts._ResumeProgress()

            # 5. Prepare (shared helper — history, notifier).
            #    Does NOT fire any lifecycle hooks; that's done below inside
            #    the same try block so failure or released-finish hooks pair
            #    with on_run_started.
            run_tenant = run_record.tenant_id if run_record else None
            ctx = await _prepare_resume(
                run_id,
                pause_state,
                state_store=store,
                agent=agent,
                provider=provider,
                expected_status=expected,
                tool_results=None if recovering else stream_results,
                user_input=None if recovering else user_input,
                extra_notifier=extra_notifier,
                user_parts=user_parts,
                tenant_id=run_tenant,
                history_records=reads.history_records,
                pause_data=raw_pause,
            )
            image_runtime = _run_images._image_runtime(agent, store, run_tenant, run_id)
            _shared["ctx"] = ctx
            progress.submission = getattr(ctx.recorder, "_loop_state", {}).get("submission")

            # 5a. Fire run-level lifecycle hooks. The exception handler closes
            #     the pair with failure or the observed released status.
            await record_run_started(
                ctx.recorder, run_id, agent_name=agent.name, agent_model=provider.model
            )
            ctx.recorder_started = True
            trace_link, ctx.notifier = await _run_observers._start_trace(
                ctx.notifier,
                store,
                run_id,
                agent_name=agent.name,
                agent_model=provider.model,
                saved=getattr(run_record, "trace_link", None),
                on_started=ctx._mark_notifier_started,
            )
            # Lifecycle pair is now open. Stash recorder/notifier so the
            # cleanup callback can close it with a CANCELLED finish if the
            # consumer abandons the stream before a terminal event.
            _shared["recorder"] = ctx.recorder
            _shared["notifier"] = ctx.notifier

            await agent.get_tool_lookups(loop=ctx.resolved_loop)
            mcp_meta.update(_run_tools._snapshot_mcp_meta(agent))

            # 5b. tool/user-input history. Inside the try
            #     so a recorder failure here fires on_run_failed.
            if not recovering:
                await _replay_resume_injected_history(
                    ctx,
                    run_id,
                    tool_results=stream_results,
                    user_input=user_input,
                    on_consumed=progress._start,
                )

            stream_context.set_delegation(
                _resume_delegation(agent, store, run_id, run_record, trace_link, _tenant_explicit)
            )

            # 7. Emit RUN_RESUMED as first event (contract guarantee)
            yield RunEvent(type=RunEventType.RUN_RESUMED, run_id=run_id)

            if recovering:
                stop = await _resolve_recovery(
                    agent,
                    store,
                    run_id,
                    ctx,
                    image_runtime=image_runtime,
                    tenant_id=run_record.tenant_id if run_record else None,
                    on_work=partial(_mark_stream_recovery_work, _shared),
                )
                if stop is not None:
                    yield _stopped_stream_event(
                        await _finish_resumed_result(store, run_id, ctx, stop)
                    )
                    return

            if not recovering:
                async with aclosing(
                    _stream_approval_events(
                        agent,
                        store,
                        run_id,
                        ctx,
                        expected,
                        stream_results,
                        user_input,
                        image_runtime,
                        mcp_meta,
                        progress._start,
                    )
                ) as approval_events:
                    async for event in approval_events:
                        yield event
                        if event.run_result is not None:
                            return

            # 9. Stream the loop (see run_stream for interrupt + closing notes)
            interrupt = agent._task_manager.register_interrupt(run_id)
            progress._start()
            loop_stream = ctx.resolved_loop.run_stream(
                **_resumed_loop_kwargs(
                    agent, provider, store, run_id, ctx, image_runtime, ctx.resolved_loop.run_stream
                ),
                interrupt=interrupt,
            )
            async with aclosing(
                _run_outcome._stream_loop_events(
                    _resume_loop_stream(ctx, loop_stream),
                    state_store=store,
                    recorder=ctx.recorder,
                    notifier=ctx.notifier,
                    run_id=run_id,
                    provider_kwargs=ctx.pause_state.provider_kwargs,
                    shared=_shared,
                    mcp_meta=mcp_meta,
                )
            ) as events:
                async for event in events:
                    yield event

        except (ManagedRootStoppedError, RunFencedError) as stop:
            store = _shared["state_store"]
            result = await _stopped_resume_result(stop, store, run_id, _shared.get("ctx"))
            yield _run_events._fenced_stream_event(run_id, result)

        except Exception as exc:
            failure_status = await _stream_resume_failure(_shared, extra_notifier, run_id, exc)

            yield RunEvent(
                type=RunEventType.RUN_ERROR,
                run_result=RunResult(
                    run_id=run_id,
                    status=failure_status,
                    error=_run_errors._safe_error(exc),
                    error_info=_run_errors._provider_error_info(exc),
                    meta=_resume_warning_meta(
                        _shared.get("ctx"), _run_outcome._stream_error_meta(mcp_meta, exc)
                    ),
                ),
                error=_run_errors._safe_error(exc),
            )

        finally:
            if _shared.get("claimed"):
                agent._task_manager.release_interrupt(run_id)

    async def _cleanup() -> RunResult | None:
        """CAS-guarded cancellation + local lifecycle close.

        Mirrors run_stream._cleanup. The lifecycle close fires whether or
        not the CAS won — if some other caller already finalized the run,
        the local notifier still needs the close event so OTel root spans
        are closed. Status resolution: CAS won → CANCELLED, CAS lost →
        query store for actual status (falling back to CANCELLED).
        """
        if not _shared.get("claimed"):
            return None
        store = _shared.get("state_store")
        # run.cancelled is recorded only once the resume itself was recorded.
        resumed = _shared.get("ctx") is not None

        cancel_won = False
        persisted_row: RunRecord | None | _OmittedType = _OMITTED

        def mark_cancelled() -> None:
            nonlocal cancel_won
            cancel_won = True

        partial_answer = _run_outcome._partial_answer(_shared)
        if store is not None:
            try:
                cancel_won, persisted_row = await _run_events._persist_run_cancellation(
                    store,
                    run_id,
                    {
                        "reason": CancelReason.STREAM_CLOSED.value,
                        "partial_output": partial_answer is not None,
                    },
                    on_won=mark_cancelled,
                    answer=partial_answer,
                    emit_event=resumed,
                    read_on_loss=lambda: "notifier" in _shared,
                    safe_read=True,
                )
            except Exception:
                logger.error(  # noqa: TRY400 - Keep safe_exception_info() to exclude vendor text and exception chains.
                    "Failed to cancel run %s during resume stream cleanup",
                    run_id,
                    exc_info=safe_exception_info(),
                )

        if "notifier" in _shared:
            return await _finish_resume_stream_cleanup(
                store, run_id, _shared, mcp_meta, cancel_won, persisted_row
            )
        return None

    return _RunStream(
        run_id=run_id,
        generator=stream_context.wrap(_generate()),
        cleanup=lambda: _run_events._shielded_cleanup(_cleanup, run_id=run_id),
    )


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

    async with _release_setup_failure(store, run_id, status):
        await _run_events._check_claim(store, run_id)
        raw = await store.get_pause_state(run_id)
        if raw is None:
            raise ValueError("Claimed run has no pause state")
        submission_validator(run_id, raw, agent=agent)
        from dendrux.runtime.submit import submitted_view

        raw = submitted_view(raw)
        submitted = raw.get("submitted_tool_results")
        if submitted is not None:
            validate_result_names(
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
