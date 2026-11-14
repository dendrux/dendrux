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

import asyncio as asyncio
import json as json
from contextlib import aclosing as aclosing
from contextlib import asynccontextmanager as asynccontextmanager
from contextlib import nullcontext as nullcontext
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
from dataclasses import replace as replace
from functools import partial as partial
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Any as Any
from typing import Literal as Literal
from typing import NamedTuple as NamedTuple
from typing import NoReturn as NoReturn
from typing import get_args as get_args
from typing import overload as overload

from dendrux import _codes as _codes
from dendrux._lifecycle import CANCEL_FINALIZE_STATUS_VALUES as CANCEL_FINALIZE_STATUS_VALUES

# Every status a loop result can durably pause in.
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
    STORAGE_FORMAT_CHECKPOINTED as STORAGE_FORMAT_CHECKPOINTED,
)
from dendrux._lifecycle import TERMINAL_STATUSES as TERMINAL_STATUSES
from dendrux._lifecycle import (
    CancelReason as CancelReason,
)
from dendrux._lifecycle import (
    DenialReason as DenialReason,
)
from dendrux._lifecycle import (
    FailureReason as FailureReason,
)
from dendrux._lifecycle import (
    PauseReason as PauseReason,
)
from dendrux._lifecycle import (
    StoredEventType as StoredEventType,
)
from dendrux._meta_keys import BLOCKED_META_KEY as BLOCKED_META_KEY
from dendrux._meta_keys import (
    CANCEL_PERSISTED_META_KEY as CANCEL_PERSISTED_META_KEY,
)
from dendrux._meta_keys import ERROR_INFO_META_KEY as ERROR_INFO_META_KEY
from dendrux._meta_keys import LOOP_META_KEY as LOOP_META_KEY
from dendrux._meta_keys import (
    MAX_DELEGATION_DEPTH_META_KEY as MAX_DELEGATION_DEPTH_META_KEY,
)
from dendrux._meta_keys import (
    NOT_EXECUTED_META_KEY as NOT_EXECUTED_META_KEY,
)
from dendrux._meta_keys import OUTPUT_TYPE_META_KEY as OUTPUT_TYPE_META_KEY
from dendrux._meta_keys import RETRY_OF_META_KEY as RETRY_OF_META_KEY
from dendrux._meta_keys import SEED_PREFIX_META_KEY as SEED_PREFIX_META_KEY
from dendrux._meta_keys import SKILLS_CONTEXT_META_KEY as SKILLS_CONTEXT_META_KEY
from dendrux._meta_keys import (
    STOP_REASON_META_KEY as STOP_REASON_META_KEY,
)
from dendrux._sentinel import _OMITTED as _OMITTED
from dendrux._sentinel import _OmittedType as _OmittedType
from dendrux._sentinel import _UnsetType as _UnsetType
from dendrux.chat import history_has_images as history_has_images
from dendrux.chat import normalize_chat_history as normalize_chat_history
from dendrux.chat import split_user_content as split_user_content
from dendrux.chat import validate_chat_history as validate_chat_history
from dendrux.context_blocks import ContextBlock as ContextBlock
from dendrux.context_blocks import fold_context as fold_context
from dendrux.context_blocks import render_band as render_band
from dendrux.context_blocks import validate_context as validate_context
from dendrux.context_blocks import wire_user_input as wire_user_input
from dendrux.context_blocks import wire_user_parts as wire_user_parts
from dendrux.errors import (
    BlobNotFoundError as BlobNotFoundError,
)
from dendrux.errors import (
    InvalidImageError as InvalidImageError,
)
from dendrux.errors import (
    RunFencedError as RunFencedError,
)
from dendrux.errors import (
    RunNotFoundError as RunNotFoundError,
)
from dendrux.errors import StoreOutcomeUnknown as StoreOutcomeUnknown
from dendrux.errors import (
    safe_exception_info as safe_exception_info,
)
from dendrux.loops._budget import BUDGET_EXCEEDED as BUDGET_EXCEEDED
from dendrux.loops._budget import BudgetCarry as BudgetCarry
from dendrux.loops._helpers import notifier_span_link as notifier_span_link
from dendrux.loops._helpers import notifier_trace_link as notifier_trace_link
from dendrux.loops._helpers import (
    notify_message as notify_message,
)
from dendrux.loops._helpers import notify_run_failed as notify_run_failed
from dendrux.loops._helpers import (
    notify_run_finished as notify_run_finished,
)
from dendrux.loops._helpers import notify_run_started as notify_run_started
from dendrux.loops._helpers import (
    order_observations as order_observations,
)
from dendrux.loops._helpers import (
    project_message as project_message,
)
from dendrux.loops._helpers import (
    record_message as record_message,
)
from dendrux.loops._helpers import record_run_failed as record_run_failed
from dendrux.loops._helpers import (
    record_run_finished as record_run_finished,
)
from dendrux.loops._helpers import (
    record_run_started as record_run_started,
)
from dendrux.loops._helpers import (
    record_tool as record_tool,
)
from dendrux.loops._policy_context import (
    dispatch_policy as dispatch_policy,
)
from dendrux.loops._policy_context import (
    history_call_counts as history_call_counts,
)
from dendrux.loops._policy_context import (
    static_denial as static_denial,
)
from dendrux.loops.react import MANAGED_HANDOFF_KEY as MANAGED_HANDOFF_KEY
from dendrux.loops.react import ReActLoop as ReActLoop
from dendrux.runtime import _managed as _managed
from dendrux.runtime import (
    _managed_budget as _managed_budget,
)
from dendrux.runtime import (
    _reclaim as _reclaim,
)
from dendrux.runtime._entry_checks import resolve_runner_tenant as resolve_runner_tenant
from dendrux.runtime._entry_checks import runner_entry_checks as runner_entry_checks
from dendrux.runtime._managed import (
    DESCEND_KEY as MANAGED_DESCEND_KEY,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux.runtime._managed_cancel import (
    ManagedRootStoppedError as ManagedRootStoppedError,
)
from dendrux.runtime._managed_cancel import ManagedStopLifecycle as ManagedStopLifecycle
from dendrux.runtime._managed_cancel import (
    stopped_resume_result as _stopped_resume_result,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux.runtime._run_approval import _approval_call_counts as _approval_call_counts
from dendrux.runtime._run_approval import _approval_dispatch_inputs as _approval_dispatch_inputs
from dendrux.runtime._run_approval import _approval_outcomes as _approval_outcomes
from dendrux.runtime._run_approval import _approval_policy_denial as _approval_policy_denial
from dendrux.runtime._run_approval import _approval_stop as _approval_stop
from dendrux.runtime._run_approval import _approval_stop_result as _approval_stop_result
from dendrux.runtime._run_approval import _approved_handoff as _approved_handoff
from dendrux.runtime._run_approval import _barrier_pause_result as _barrier_pause_result
from dendrux.runtime._run_approval import _carry_budget_fields as _carry_budget_fields
from dendrux.runtime._run_approval import _decided_payload as _decided_payload
from dendrux.runtime._run_approval import _execute_approved_tools as _execute_approved_tools
from dendrux.runtime._run_approval import _finish_approval_batch as _finish_approval_batch
from dendrux.runtime._run_approval import _hand_off_approved_subagent as _hand_off_approved_subagent
from dendrux.runtime._run_approval import _inject_approved_skill as _inject_approved_skill
from dendrux.runtime._run_approval import _limit_approval_clients as _limit_approval_clients
from dendrux.runtime._run_approval import _pending_decision_map as _pending_decision_map
from dendrux.runtime._run_approval import _record_approval_decided as _record_approval_decided
from dendrux.runtime._run_approval import _rejection_decided_payload as _rejection_decided_payload
from dendrux.runtime._run_approval import _resume_approval_batch as _resume_approval_batch
from dendrux.runtime._run_approval import _resume_approval_work as _resume_approval_work
from dendrux.runtime._run_approval import _resume_barrier as _resume_barrier
from dendrux.runtime._run_approval import _resume_pause_barrier as _resume_pause_barrier
from dendrux.runtime._run_approval import _validated_decisions as _validated_decisions
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
from dendrux.runtime._run_execute import _run_once as _run_once
from dendrux.runtime._run_execute import run as run
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
from dendrux.runtime._run_resume import _owned_claim_view as _owned_claim_view
from dendrux.runtime._run_resume import _resume_claim_status as _resume_claim_status
from dendrux.runtime._run_resume import _resume_core as _resume_core
from dendrux.runtime._run_resume import _validate_owned_pause as _validate_owned_pause
from dendrux.runtime._run_resume import resume as resume
from dendrux.runtime._run_resume import (
    resume_approval_approved_claim as resume_approval_approved_claim,
)
from dendrux.runtime._run_resume import resume_claimed as resume_claimed
from dendrux.runtime._run_resume import resume_paused_claim as resume_paused_claim
from dendrux.runtime._run_resume import resume_with_input as resume_with_input
from dendrux.runtime._run_resume_loop import _CONTINUATION_BLOCKED as _CONTINUATION_BLOCKED
from dendrux.runtime._run_resume_loop import _budget_carry_kwargs as _budget_carry_kwargs
from dendrux.runtime._run_resume_loop import _cancel_resumed_run as _cancel_resumed_run
from dendrux.runtime._run_resume_loop import _continuation_block_error as _continuation_block_error
from dendrux.runtime._run_resume_loop import _fenced_resume_result as _fenced_resume_result
from dendrux.runtime._run_resume_loop import _finish_resumed_result as _finish_resumed_result
from dendrux.runtime._run_resume_loop import _handle_resume_failure as _handle_resume_failure
from dendrux.runtime._run_resume_loop import (
    _initial_call_counts_kwargs as _initial_call_counts_kwargs,
)
from dendrux.runtime._run_resume_loop import _released_resume_hooks as _released_resume_hooks
from dendrux.runtime._run_resume_loop import _released_resume_status as _released_resume_status
from dendrux.runtime._run_resume_loop import _resume_delegation as _resume_delegation
from dendrux.runtime._run_resume_loop import _resume_failure_hooks as _resume_failure_hooks
from dendrux.runtime._run_resume_loop import _resume_history_inputs as _resume_history_inputs
from dendrux.runtime._run_resume_loop import _resume_loop_stream as _resume_loop_stream
from dendrux.runtime._run_resume_loop import _resume_release_supported as _resume_release_supported
from dendrux.runtime._run_resume_loop import _resume_warning_meta as _resume_warning_meta
from dendrux.runtime._run_resume_loop import _resumed_loop_kwargs as _resumed_loop_kwargs
from dendrux.runtime._run_resume_loop import _run_resumed_loop as _run_resumed_loop
from dendrux.runtime._run_resume_loop import _stop_marked_continuation as _stop_marked_continuation
from dendrux.runtime._run_resume_loop import _try_resume_release as _try_resume_release
from dendrux.runtime._run_resume_prep import _attempt_validator as _attempt_validator
from dendrux.runtime._run_resume_prep import (
    _canonicalize_legacy_denials as _canonicalize_legacy_denials,
)
from dendrux.runtime._run_resume_prep import _claim_resume as _claim_resume
from dendrux.runtime._run_resume_prep import _load_pause_state as _load_pause_state
from dendrux.runtime._run_resume_prep import _load_resume_pause as _load_resume_pause
from dendrux.runtime._run_resume_prep import _overlay_pause_checkpoint as _overlay_pause_checkpoint
from dendrux.runtime._run_resume_prep import _prepare_resume as _prepare_resume
from dendrux.runtime._run_resume_prep import _read_resume_pause as _read_resume_pause
from dendrux.runtime._run_resume_prep import _read_resume_state as _read_resume_state
from dendrux.runtime._run_resume_prep import _release_claim as _release_claim
from dendrux.runtime._run_resume_prep import _release_setup_failure as _release_setup_failure
from dendrux.runtime._run_resume_prep import _report_claim_governance as _report_claim_governance
from dendrux.runtime._run_resume_prep import _require_pause_agent as _require_pause_agent
from dendrux.runtime._run_resume_prep import (
    _resolve_resume_stream_store as _resolve_resume_stream_store,
)
from dendrux.runtime._run_resume_prep import _restore_pause_refs as _restore_pause_refs
from dendrux.runtime._run_resume_prep import _resume_claim_fallbacks as _resume_claim_fallbacks
from dendrux.runtime._run_resume_prep import _resume_claim_payload as _resume_claim_payload
from dendrux.runtime._run_resume_prep import _resume_history as _resume_history
from dendrux.runtime._run_resume_prep import _resume_session as _resume_session
from dendrux.runtime._run_resume_prep import _resume_submission as _resume_submission
from dendrux.runtime._run_resume_prep import _resumed_payload as _resumed_payload
from dendrux.runtime._run_resume_replay import _claimed_in_call as _claimed_in_call
from dendrux.runtime._run_resume_replay import (
    _consume_injected_history as _consume_injected_history,
)
from dendrux.runtime._run_resume_replay import _mark_work as _mark_work
from dendrux.runtime._run_resume_replay import (
    _project_adopted_submission as _project_adopted_submission,
)
from dendrux.runtime._run_resume_replay import _record_blocked_replay as _record_blocked_replay
from dendrux.runtime._run_resume_replay import (
    _replay_resume_injected_history as _replay_resume_injected_history,
)
from dendrux.runtime._run_resume_replay import (
    _replay_submission_or_stop as _replay_submission_or_stop,
)
from dendrux.runtime._run_resume_replay import _replay_tool_result as _replay_tool_result
from dendrux.runtime._run_resume_replay import _replay_user_input as _replay_user_input
from dendrux.runtime._run_resume_replay import _replayed_result_blocked as _replayed_result_blocked
from dendrux.runtime._run_resume_replay import _resolve_recovery as _resolve_recovery
from dendrux.runtime._run_resume_replay import _resume_blocked_result as _resume_blocked_result
from dendrux.runtime._run_resume_stream import (
    _finish_resume_stream_cleanup as _finish_resume_stream_cleanup,
)
from dendrux.runtime._run_resume_stream import (
    _finish_stream_resume_failure as _finish_stream_resume_failure,
)
from dendrux.runtime._run_resume_stream import _initial_stream_images as _initial_stream_images
from dendrux.runtime._run_resume_stream import (
    _mark_stream_recovery_work as _mark_stream_recovery_work,
)
from dendrux.runtime._run_resume_stream import _remember_stream_release as _remember_stream_release
from dendrux.runtime._run_resume_stream import _resume_expected_status as _resume_expected_status
from dendrux.runtime._run_resume_stream import _stopped_stream_event as _stopped_stream_event
from dendrux.runtime._run_resume_stream import _stream_approval_events as _stream_approval_events
from dendrux.runtime._run_resume_stream import _stream_claimed_state as _stream_claimed_state
from dendrux.runtime._run_resume_stream import _stream_resume_claim as _stream_resume_claim
from dendrux.runtime._run_resume_stream import _stream_resume_failure as _stream_resume_failure
from dendrux.runtime._run_resume_stream import _stream_resume_images as _stream_resume_images
from dendrux.runtime._run_resume_stream import _stream_resume_started as _stream_resume_started
from dendrux.runtime._run_resume_stream import _stream_resume_state as _stream_resume_state
from dendrux.runtime._run_resume_stream import _StreamClaim as _StreamClaim
from dendrux.runtime._run_resume_stream import resume_stream as resume_stream
from dendrux.runtime._run_retry import _check_retry_source as _check_retry_source
from dendrux.runtime._run_retry import _refresh_retry_user as _refresh_retry_user
from dendrux.runtime._run_retry import _refuse_blocked_retry as _refuse_blocked_retry
from dendrux.runtime._run_retry import _retry_edges as _retry_edges
from dendrux.runtime._run_retry import _retry_history as _retry_history
from dendrux.runtime._run_retry import _retry_media as _retry_media
from dendrux.runtime._run_retry import _retry_replay as _retry_replay
from dendrux.runtime._run_retry import _retry_replay_request as _retry_replay_request
from dendrux.runtime._run_retry import _retry_source as _retry_source
from dendrux.runtime._run_retry import _retry_source_input as _retry_source_input
from dendrux.runtime._run_retry import _start_retry_history as _start_retry_history
from dendrux.runtime._run_retry import _trim_retry_answer as _trim_retry_answer
from dendrux.runtime._run_retry import retry as retry
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
from dendrux.runtime._run_stream import _cached_stream_event as _cached_stream_event
from dendrux.runtime._run_stream import _resolve_stream_store as _resolve_stream_store
from dendrux.runtime._run_stream import _stream_claim as _stream_claim
from dendrux.runtime._run_stream import _stream_entry_checks as _stream_entry_checks
from dendrux.runtime._run_stream import _stream_replay_runs as _stream_replay_runs
from dendrux.runtime._run_stream import run_stream as run_stream
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
from dendrux.runtime._tenant_checks import _tenant_scope_chosen as _tenant_scope_chosen
from dendrux.runtime._tracelink import validate_trace_link as validate_trace_link
from dendrux.runtime.context import (
    DelegationContext as DelegationContext,
)
from dendrux.runtime.context import (
    StreamContext as StreamContext,
)
from dendrux.runtime.context import get_delegation_context as get_delegation_context
from dendrux.runtime.context import (
    get_store_identity as get_store_identity,
)
from dendrux.runtime.context import (
    reset_delegation_context as reset_delegation_context,
)
from dendrux.runtime.context import resolve_child_tenant as resolve_child_tenant
from dendrux.runtime.context import resolve_parent_link as resolve_parent_link
from dendrux.runtime.context import (
    set_delegation_context as set_delegation_context,
)
from dendrux.runtime.context import take_managed_invocation as take_managed_invocation
from dendrux.runtime.replay import PARTS_KEY as PARTS_KEY
from dendrux.runtime.replay import REPLAY_META_KEY as REPLAY_META_KEY
from dendrux.runtime.replay import LoadedReplay as LoadedReplay
from dendrux.runtime.replay import ReplayLast as ReplayLast
from dendrux.runtime.replay import ReplayPlan as ReplayPlan
from dendrux.runtime.replay import ReplayRequest as ReplayRequest
from dendrux.runtime.replay import _load_replay_runs as _load_replay_runs
from dendrux.runtime.replay import _pair_tool_calls as _pair_tool_calls
from dendrux.runtime.replay import build_seed as build_seed
from dendrux.runtime.replay import current_prefix as current_prefix
from dendrux.runtime.replay import expand_replay as expand_replay
from dendrux.runtime.replay import has_images as has_images
from dendrux.runtime.replay import image_slot_count as image_slot_count
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
from dendrux.runtime.replay import rebuild_messages as rebuild_messages
from dendrux.runtime.replay import record_run_prefix as record_run_prefix
from dendrux.runtime.replay import user_trace_message as user_trace_message
from dendrux.runtime.replay import validate_replay_args as validate_replay_args
from dendrux.runtime.state import ClaimResult as ClaimResult
from dendrux.runtime.state import _initial_loop_state as _initial_loop_state
from dendrux.runtime.state import claimed_store as claimed_store
from dendrux.runtime.state import unclaimed_store as unclaimed_store
from dendrux.runtime.stores._limits_fit import ensure_agent_limits_fit as ensure_agent_limits_fit
from dendrux.runtime.submit import GOVERNANCE_KEY as GOVERNANCE_KEY
from dendrux.runtime.submit import _claim_failure_mapping as _claim_failure_mapping
from dendrux.runtime.submit import (
    _claim_request as _claim_request,
)
from dendrux.runtime.submit import (
    _claim_submission as _claim_submission,
)
from dendrux.runtime.submit import (
    _claimed_submission_view as _claimed_submission_view,
)
from dendrux.runtime.submit import (
    _release_supported as _release_supported,
)
from dendrux.runtime.submit import governance_snapshot as governance_snapshot
from dendrux.runtime.submit import (
    legacy_approval_denial as legacy_approval_denial,
)
from dendrux.runtime.submit import (
    prepare_resume_results as prepare_resume_results,
)
from dendrux.runtime.submit import (
    report_governance_change as report_governance_change,
)
from dendrux.runtime.submit import (
    require_pause_id as require_pause_id,
)
from dendrux.runtime.submit import (
    submission_validator as submission_validator,
)
from dendrux.runtime.submit import (
    validate_result_names as validate_result_names,
)
from dendrux.skills._session import META_KEY as META_KEY
from dendrux.skills._session import SkillSession as SkillSession
from dendrux.skills._session import continuation as continuation
from dendrux.skills._session import current as current
from dendrux.skills._session import (
    current as current_skill_session,  # noqa: F401 - Frozen compatibility alias.
)
from dendrux.strategies.native import NativeToolCalling as NativeToolCalling
from dendrux.types import (
    AgentStep as AgentStep,
)
from dendrux.types import (
    ApprovalDecision as ApprovalDecision,
)
from dendrux.types import (
    ErrorInfo as ErrorInfo,
)
from dendrux.types import (
    GovernanceEventType as GovernanceEventType,
)
from dendrux.types import IdempotencyConflictError as IdempotencyConflictError
from dendrux.types import (
    ImagePart as ImagePart,
)
from dendrux.types import (
    ImageRef as ImageRef,
)
from dendrux.types import (
    ImageViewState as ImageViewState,
)
from dendrux.types import (
    Message as Message,
)
from dendrux.types import (
    PauseState as PauseState,
)
from dendrux.types import ReplaySummary as ReplaySummary
from dendrux.types import (
    Role as Role,
)
from dendrux.types import RunAlreadyActiveError as RunAlreadyActiveError
from dendrux.types import (
    RunEventType as RunEventType,
)
from dendrux.types import (
    RunStatus as RunStatus,
)
from dendrux.types import StopReason as StopReason
from dendrux.types import (
    TextPart as TextPart,
)
from dendrux.types import (
    UsageStats as UsageStats,
)
from dendrux.types import (
    _usage_from_dict as _usage_from_dict,
)
from dendrux.types import compute_idempotency_fingerprint as compute_idempotency_fingerprint
from dendrux.types import (
    content_part_from_dict as content_part_from_dict,
)
from dendrux.types import generate_ulid as generate_ulid
from dendrux.types import (
    tool_result_message as tool_result_message,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator as AsyncGenerator,
    )
    from collections.abc import (
        AsyncIterator as AsyncIterator,
    )
    from collections.abc import (
        Callable as Callable,
    )
    from collections.abc import (
        Sequence as Sequence,
    )

    from dendrux.agent import Agent as Agent
    from dendrux.llm.base import LLMProvider as LLMProvider
    from dendrux.loops.base import Loop as Loop
    from dendrux.loops.base import LoopNotifier as LoopNotifier
    from dendrux.loops.base import LoopRecorder as LoopRecorder
    from dendrux.media._catalog import BlobEdge as BlobEdge
    from dendrux.media._image import UserContent as UserContent
    from dendrux.media._runtime import ImageRuntime as ImageRuntime
    from dendrux.runtime._continuation_guardrails import (
        _SubmissionProjection as _SubmissionProjection,
    )
    from dendrux.runtime._entry_guardrails import _EntryProjection as _EntryProjection
    from dendrux.runtime.state import ClaimResult as ClaimResult
    from dendrux.runtime.state import RunRecord as RunRecord
    from dendrux.runtime.state import StateStore as StateStore
    from dendrux.strategies.base import Strategy as Strategy
    from dendrux.types import (
        ContentPart as ContentPart,
    )
    from dendrux.types import (
        RunEvent as RunEvent,
    )
    from dendrux.types import (
        RunResult as RunResult,
    )
    from dendrux.types import (
        RunStream as RunStream,
    )
    from dendrux.types import (
        ToolCall as ToolCall,
    )
    from dendrux.types import (
        ToolResult as ToolResult,
    )
