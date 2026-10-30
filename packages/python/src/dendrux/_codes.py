"""Stable log codes and closed, value-checked structured logging fields."""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Code:
    """A stable diagnostic and the complete set of fields it can project."""

    name: str
    level: int
    kind: str
    fields: frozenset[str]
    fires_when: str
    do: str


_BASE_FIELDS = frozenset({"code", "kind", "run_id", "tenant_id"})
_ULID = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
_PROVIDER_ID = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")
_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_SPAN_ID = re.compile(r"[0-9a-f]{16}")
_NUMBERS = frozenset(
    {
        "attempt",
        "previous_epoch",
        "gap_s",
        "limit",
        "bytes",
        "cap",
        "inspected",
        "count",
        "timeout_seconds",
        "timeout_s",
        "zombie_threads",
        "zombies",
        "pages",
        "status",
        "duration_ms",
    }
)
_ENUMS = {
    "category": frozenset(
        [
            "abandoned_waiting",
            "auth",
            "content_filter",
            "context_length",
            "credits",
            "rate_limit",
            "refusal",
            "rejected",
            "stale_running",
            "stream_incomplete",
            "timeout",
            "unavailable",
            "unknown",
        ]
    ),
    "provider": frozenset(["anthropic", "openai", "openai_responses", "openrouter", "custom"]),
    "timeout_reason": frozenset(
        ["call_deadline", "connect", "deadline_during_backoff", "stream_idle", "stream_total"]
    ),
    "reason": frozenset(
        [
            "agent_mismatch",
            "already_reclaimed",
            "cancel_requested",
            "cancelled",
            "governance_changed",
            "history_too_large",
            "lease_fresh",
            "legacy_run",
            "loop_unknown",
            "loop_unsupported",
            "not_running",
            "reclaim_limit",
            "recovery_call_missing",
            "recovery_state_missing",
            "recovery_state_invalid",
            "sibling_failed",
            "skill_mismatch",
            "store_capability",
        ]
    ),
    "entry": frozenset(
        [
            "add_note",
            "cancel_run",
            "follow",
            "get_event_view",
            "get_pending",
            "get_result",
            "reclaim",
            "request_pause",
            "resume",
            "resume_stream",
            "retry",
            "run",
            "stream",
            "submit_approval",
            "submit_input",
            "submit_note",
            "submit_resume",
            "submit_tool_results",
        ]
    ),
    "argument": frozenset(
        [
            "active_pause",
            "after",
            "after_sequence_index",
            "agent_name",
            "batch_size",
            "before",
            "cursor",
            "event_types",
            "include_payloads",
            "iteration",
            "limit",
            "max_duration_s",
            "max_events",
            "max_runs",
            "metadata_filter",
            "offset",
            "parent_run_id",
            "poll_interval_s",
            "run_id",
            "started_after",
            "started_before",
            "status",
            "tenant_id",
            "timeout_s",
            "undelivered_only",
            "until_terminal",
        ]
    ),
    "field": frozenset(
        [
            "context",
            "decision reasons",
            "history",
            "input",
            "metadata",
            "notes",
            "provider_kwargs",
            "rejection_reason",
            "tool_results",
            "user_input",
        ]
    ),
    "error_type": frozenset(
        [
            "CancelledError",
            "PersistenceError",
            "ProviderError",
            "RunFencedError",
            "RuntimeError",
            "TimeoutError",
            "TypeError",
            "ValueError",
        ]
    ),
    "table": frozenset(["react_traces", "run_events", "tool_calls"]),
    "index": frozenset(
        [
            "ix_agent_runs_created_at",
            "ix_agent_runs_idempotency_key",
            "ix_agent_runs_parent_run_id",
            "ix_agent_runs_retry_of_run_id",
            "ix_agent_runs_status",
            "ix_agent_runs_sweep_running",
            "ix_agent_runs_sweep_waiting",
            "ix_agent_runs_tenant_id",
            "ix_agent_runs_tenant_id_created_at",
            "ix_dendrux_blob_refs_blob",
            "ix_dendrux_blob_refs_run",
            "ix_dendrux_blobs_gc",
            "ix_dendrux_blobs_work",
            "ix_llm_interactions_agent_run_id",
            "ix_react_traces_agent_run_id_order",
            "ix_run_events_agent_run_id_seq",
            "ix_run_events_event_type",
            "ix_run_events_started",
            "ix_run_inbox_run_pending",
            "ix_token_usage_agent_run_id",
            "ix_tool_calls_agent_run_id",
            "uq_agent_runs_tenant_idempotency_key",
            "uq_dendrux_blob_refs_owner",
            "uq_react_traces_run_order",
            "uq_run_events_run_seq",
            "uq_tool_calls_run_call",
        ]
    ),
    "revision": frozenset(["d7e8f9a0b1c2", "e8f9a0b1c2d3", "f9a0b1c2d3e4"]),
    "step": frozenset(
        [
            "add.agent_runs.last_reclaimed_at",
            "add.agent_runs.lease_expired_at",
            "add.agent_runs.loop_state",
            "add.agent_runs.reclaim_attempts",
            "add.agent_runs.reclaim_limit",
            "add.agent_runs.storage_format",
            "add.agent_runs.trace_link",
            "add.react_traces.content_digest",
            "add.react_traces.trace_kind",
            "add.run_events.content_digest",
            "add.tool_calls.content_digest",
            "add.tool_calls.idempotent",
            "add.tool_calls.outcome",
            "add.tool_calls.replayed_at",
            "add_claim_epoch",
            "add_status_changed_at",
            "add_status_clock_check",
            "build_index.ix_agent_runs_created_at",
            "build_index.ix_agent_runs_idempotency_key",
            "build_index.ix_agent_runs_parent_run_id",
            "build_index.ix_agent_runs_retry_of_run_id",
            "build_index.ix_agent_runs_status",
            "build_index.ix_agent_runs_sweep_running",
            "build_index.ix_agent_runs_sweep_waiting",
            "build_index.ix_agent_runs_tenant_id",
            "build_index.ix_agent_runs_tenant_id_created_at",
            "build_index.ix_dendrux_blob_refs_blob",
            "build_index.ix_dendrux_blob_refs_run",
            "build_index.ix_dendrux_blobs_gc",
            "build_index.ix_dendrux_blobs_work",
            "build_index.ix_llm_interactions_agent_run_id",
            "build_index.ix_react_traces_agent_run_id_order",
            "build_index.ix_run_events_agent_run_id_seq",
            "build_index.ix_run_events_event_type",
            "build_index.ix_run_events_started",
            "build_index.ix_run_inbox_run_pending",
            "build_index.ix_token_usage_agent_run_id",
            "build_index.ix_tool_calls_agent_run_id",
            "build_index.uq_agent_runs_tenant_idempotency_key",
            "build_index.uq_dendrux_blob_refs_owner",
            "build_index.uq_react_traces_run_order",
            "build_index.uq_run_events_run_seq",
            "build_index.uq_tool_calls_run_call",
            "create.dendrux_settings",
            "dedupe",
            "done",
            "drop",
            "drop.agent_runs.last_reclaimed_at",
            "drop.agent_runs.lease_expired_at",
            "drop.agent_runs.loop_state",
            "drop.agent_runs.reclaim_attempts",
            "drop.agent_runs.reclaim_limit",
            "drop.agent_runs.storage_format",
            "drop.agent_runs.trace_link",
            "drop.dendrux_settings",
            "drop.react_traces.content_digest",
            "drop.react_traces.trace_kind",
            "drop.run_events.content_digest",
            "drop.tool_calls.content_digest",
            "drop.tool_calls.idempotent",
            "drop.tool_calls.outcome",
            "drop.tool_calls.replayed_at",
            "drop_claim_epoch",
            "drop_constraint.agent_runs_idempotency_key_key",
            "drop_index.ix_agent_runs_created_at",
            "drop_index.ix_agent_runs_idempotency_key",
            "drop_index.ix_agent_runs_parent_run_id",
            "drop_index.ix_agent_runs_retry_of_run_id",
            "drop_index.ix_agent_runs_status",
            "drop_index.ix_agent_runs_sweep_running",
            "drop_index.ix_agent_runs_sweep_waiting",
            "drop_index.ix_agent_runs_tenant_id",
            "drop_index.ix_agent_runs_tenant_id_created_at",
            "drop_index.ix_dendrux_blob_refs_blob",
            "drop_index.ix_dendrux_blob_refs_run",
            "drop_index.ix_dendrux_blobs_gc",
            "drop_index.ix_dendrux_blobs_work",
            "drop_index.ix_llm_interactions_agent_run_id",
            "drop_index.ix_react_traces_agent_run_id_order",
            "drop_index.ix_run_events_agent_run_id_seq",
            "drop_index.ix_run_events_event_type",
            "drop_index.ix_run_events_started",
            "drop_index.ix_run_inbox_run_pending",
            "drop_index.ix_token_usage_agent_run_id",
            "drop_index.ix_tool_calls_agent_run_id",
            "drop_index.uq_agent_runs_tenant_idempotency_key",
            "drop_index.uq_dendrux_blob_refs_owner",
            "drop_index.uq_react_traces_run_order",
            "drop_index.uq_run_events_run_seq",
            "drop_index.uq_tool_calls_run_call",
            "drop_status_changed_at",
            "drop_status_clock_check",
            "format",
            "index",
            "repair_index.ix_agent_runs_created_at",
            "repair_index.ix_agent_runs_idempotency_key",
            "repair_index.ix_agent_runs_parent_run_id",
            "repair_index.ix_agent_runs_retry_of_run_id",
            "repair_index.ix_agent_runs_status",
            "repair_index.ix_agent_runs_sweep_running",
            "repair_index.ix_agent_runs_sweep_waiting",
            "repair_index.ix_agent_runs_tenant_id",
            "repair_index.ix_agent_runs_tenant_id_created_at",
            "repair_index.ix_dendrux_blob_refs_blob",
            "repair_index.ix_dendrux_blob_refs_run",
            "repair_index.ix_dendrux_blobs_gc",
            "repair_index.ix_dendrux_blobs_work",
            "repair_index.ix_llm_interactions_agent_run_id",
            "repair_index.ix_react_traces_agent_run_id_order",
            "repair_index.ix_run_events_agent_run_id_seq",
            "repair_index.ix_run_events_event_type",
            "repair_index.ix_run_events_started",
            "repair_index.ix_run_inbox_run_pending",
            "repair_index.ix_token_usage_agent_run_id",
            "repair_index.ix_tool_calls_agent_run_id",
            "repair_index.uq_agent_runs_tenant_idempotency_key",
            "repair_index.uq_dendrux_blob_refs_owner",
            "repair_index.uq_react_traces_run_order",
            "repair_index.uq_run_events_run_seq",
            "repair_index.uq_tool_calls_run_call",
            "seed.storage_format",
            "set_status_changed_at_not_null",
        ]
    ),
}

FOLLOW_FEED_ERROR = Code(
    "follow_feed_error",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A live event feed fails.",
    "Check store connectivity and retry the feed.",
)
FOLLOW_CAPACITY = Code(
    "follow_capacity",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset(["limit"]),
    "The process follower cap is reached.",
    "Reduce followers or raise the configured cap.",
)
FOLLOW_TENANT_CAPACITY = Code(
    "follow_tenant_capacity",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset(["limit"]),
    "A tenant follower cap is reached.",
    "Reduce tenant followers or raise its cap.",
)
FOLLOW_SHUTDOWN_PENDING = Code(
    "follow_shutdown_pending",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A feed is still stopping after its cleanup deadline.",
    "Inspect slow store operations.",
)
USAGE_INVALID = Code(
    "usage_invalid",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset([]),
    "Provider usage cannot be trusted.",
    "Treat accounting as unknown and inspect the adapter.",
)
DATABASE_URL_ENV_IGNORED = Code(
    "database_url_env_ignored",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "An Agent without persistence ignores DENDRUX_DATABASE_URL.",
    "Pass state_store or database_url explicitly to persist Agent runs.",
)
TENANT_SCOPE_REQUIRED = Code(
    "tenant_scope_required",
    logging.WARNING,
    "governance",
    _BASE_FIELDS | frozenset(["entry"]),
    "A strict tenancy entry has no explicit tenant scope.",
    "Pass the intended tenant scope.",
)
RESERVED_METADATA_KEY = Code(
    "reserved_metadata_key",
    logging.WARNING,
    "governance",
    _BASE_FIELDS | frozenset(["entry"]),
    "User metadata contains a reserved key.",
    "Remove reserved metadata keys.",
)
METADATA_TOO_LARGE = Code(
    "metadata_too_large",
    logging.WARNING,
    "governance",
    _BASE_FIELDS | frozenset(["entry", "bytes", "limit"]),
    "Metadata exceeds its byte bound.",
    "Reduce metadata or raise the bound.",
)
RUN_FAILED = Code(
    "run_failed",
    logging.ERROR,
    "runtime",
    _BASE_FIELDS | frozenset(["error_type"]),
    "A run task fails.",
    "Inspect its safe result and trace.",
)
READ_ARGUMENT_INVALID = Code(
    "read_argument_invalid",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset(["argument"]),
    "A read argument is invalid.",
    "Correct the named argument.",
)
PAUSE_DATA_NOT_JSONB = Code(
    "pause_data_not_jsonb",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A stored pause is not a JSON object.",
    "Inspect the stored pause format and writer versions.",
)
FENCED_FINISH_FAILED = Code(
    "fenced_finish_failed",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A fenced worker cannot finish its lifecycle recorder.",
    "Inspect store connectivity and the winning worker.",
)
CANCEL_PERSIST_FAILED = Code(
    "cancel_persist_failed",
    logging.ERROR,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "Cancellation cannot be persisted.",
    "Retry cancellation after restoring the store.",
)
INVALID_TOOL_RESULT = Code(
    "invalid_tool_result",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset([]),
    "A client tool result fails validation.",
    "Submit a valid result for the pending call.",
)
PAUSE_ID_MISMATCH = Code(
    "pause_id_mismatch",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A submission targets another pause generation.",
    "Reload the pending pause and resubmit.",
)
LEGACY_CLAIM_VALIDATION = Code(
    "legacy_claim_validation",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A legacy store needs claim-time validation.",
    "Implement the current store capabilities when upgrading.",
)
GOVERNANCE_MISMATCH = Code(
    "governance_mismatch",
    logging.WARNING,
    "governance",
    _BASE_FIELDS | frozenset([]),
    "A stored governance policy cannot be continued.",
    "Restore the original policy or start a new run.",
)
GOVERNANCE_LEGACY = Code(
    "governance_legacy",
    logging.INFO,
    "governance",
    _BASE_FIELDS | frozenset([]),
    "A legacy run lacks governance pins.",
    "Review the policy before continuing it.",
)
GOVERNANCE_CHANGED = Code(
    "governance_changed",
    logging.WARNING,
    "governance",
    _BASE_FIELDS | frozenset([]),
    "Continuation detects changed governance in warning mode.",
    "Review the change and use strict policy enforcement.",
)
SIZE_LIMIT = Code(
    "size_limit",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset(["bytes", "limit", "truncated"]),
    "A tool result exceeds its byte cap.",
    "Reduce the result or use payload offload.",
)
WIRE_ANSWER_OMITTED = Code(
    "wire_answer_omitted",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A wire answer exceeds its encoding bound.",
    "Read the bounded result through the store.",
)
NOTE_CAP_UNENFORCED = Code(
    "note_cap_unenforced",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A custom store cannot enforce the note cap.",
    "Implement bounded note writes in the custom store.",
)
INPUT_SIZE_LIMIT = Code(
    "input_size_limit",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset(["entry", "field", "bytes", "limit"]),
    "An input exceeds its configured bound.",
    "Reduce the input or raise the bound deliberately.",
)
IDEMPOTENCY_CONFLICT_LEGACY = Code(
    "idempotency_conflict_legacy",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A legacy write cannot use the current unique index.",
    "Finish the rolling upgrade and activate the schema.",
)
NOTE_LIMIT = Code(
    "note_limit",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset(["limit"]),
    "A run reaches the note cap.",
    "Consume notes or raise the configured cap.",
)
SUBTREE_LIMIT = Code(
    "subtree_limit",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset(["limit"]),
    "A descendant traversal reaches its bound.",
    "Inspect the subtree and raise its bound if appropriate.",
)
RUN_FENCED = Code(
    "run_fenced",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "An old worker writes after another claim wins.",
    "Expected after reclaim; alert on a sustained high rate.",
)
SWEEP_CAP_REACHED = Code(
    "sweep_cap_reached",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["category", "cap", "inspected"]),
    "A sweep reaches its inspection bound.",
    "Run another sweep or raise the bound.",
)
MIGRATION_LOCK_TIMEOUT = Code(
    "migration_lock_timeout",
    logging.WARNING,
    "migration",
    _BASE_FIELDS | frozenset(["revision", "step", "timeout_seconds"]),
    "A migration cannot acquire its lock in time.",
    "Retry when the table is quiet.",
)
SHARED_ENGINE_LOOP_CLOSED = Code(
    "shared_engine_loop_closed",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A pooled engine belongs to a closed event loop.",
    "Use an engine owned by the current loop.",
)
TOOL_ROW_MAY_BE_MISSING = Code(
    "tool_row_may_be_missing",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset([]),
    "A finished tool is interrupted while its row is written.",
    "Check the side effect before retrying.",
)
OUTCOME_UNKNOWN = Code(
    "outcome_unknown",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset(["reason"]),
    "An interrupted tool may have applied its effects.",
    "Check the side effect before retrying.",
)
CANCELLED = Code(
    "cancelled",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset(["reason"]),
    "A sibling tool stops before completion.",
    "Inspect the batch result before retrying.",
)
TIMEOUT_NOT_STARTED = Code(
    "timeout_not_started",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset(["timeout_s", "zombie_threads"]),
    "A queued tool times out before starting.",
    "Retry after reducing tool pool load.",
)
TIMEOUT_OUTCOME_UNKNOWN = Code(
    "timeout_outcome_unknown",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset(["timeout_s", "zombie_threads"]),
    "A running tool exceeds its deadline.",
    "Check the side effect before retrying.",
)
TOOL_POOL_SATURATED = Code(
    "tool_pool_saturated",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset([]),
    "The synchronous tool pool cannot accept work.",
    "Reduce concurrency or raise the pool bound.",
)
TOOL_ZOMBIE_FINISHED = Code(
    "tool_zombie_finished",
    logging.INFO,
    "tool",
    _BASE_FIELDS | frozenset(["zombies"]),
    "An abandoned synchronous tool finally exits.",
    "Inspect repeated long-running tools.",
)
INVALID_TOOL_SCHEMA = Code(
    "invalid_tool_schema",
    logging.WARNING,
    "tool",
    _BASE_FIELDS | frozenset([]),
    "A tool call targets a schema that cannot be validated.",
    "Fix the declared tool schema.",
)
INVALID_SCHEMA = Code(
    "invalid_schema",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset([]),
    "An MCP tool has an invalid schema.",
    "Fix the upstream tool schema.",
)
MCP_VIEW_RELINKED = Code(
    "mcp_view_relinked",
    logging.INFO,
    "mcp",
    _BASE_FIELDS | frozenset([]),
    "An MCP view changes its registered link.",
    "Review the updated MCP configuration.",
)
LLM_FAILED_WRITE = Code(
    "llm_failed_write",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "Best-effort provider-failure evidence cannot be written.",
    "Inspect store connectivity; the safe run result remains authoritative.",
)
RUN_NOT_RECLAIMABLE = Code(
    "run_not_reclaimable",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["reason"]),
    "A reclaim request cannot safely continue the run.",
    "Inspect the refusal reason before retrying.",
)
RUN_CANCELLED = Code(
    "run_cancelled",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "Child admission is refused under a cancelled parent.",
    "Stop creating children for the parent.",
)
INVALID_LIMIT = Code(
    "invalid_limit",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A recovery limit is invalid.",
    "Use a positive limit within the documented bound.",
)
PROVIDER_KWARGS_INVALID = Code(
    "provider_kwargs_invalid",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset([]),
    "Provider options fail closed validation.",
    "Use only supported provider options.",
)
PROVIDER_KWARGS_TOO_LARGE = Code(
    "provider_kwargs_too_large",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset(["bytes", "limit"]),
    "Provider options exceed the checkpoint byte bound.",
    "Reduce provider options.",
)
LEASE_INTERVAL = Code(
    "lease_interval",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "The lease interval is below its floor.",
    "Use a lease interval of at least five seconds.",
)
CHECKPOINT_TOO_LARGE = Code(
    "checkpoint_too_large",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["bytes", "limit"]),
    "A checkpoint exceeds its byte bound.",
    "Reduce run state before continuing.",
)
REWRITE_INVALID = Code(
    "rewrite_invalid",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A conversation rewrite has invalid metadata.",
    "Inspect the conversation writer.",
)
WRITE_CONFLICT = Code(
    "write_conflict",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset(["table", "count"]),
    "A retried identity has conflicting content.",
    "Inspect writer consistency; do not retry blindly.",
)
TRACE_LINK_INVALID = Code(
    "trace_link_invalid",
    logging.WARNING,
    "observability",
    _BASE_FIELDS | frozenset([]),
    "A trace link fails W3C validation.",
    "Fix the notifier trace propagation.",
)
SUBMISSION_MISSING = Code(
    "submission_missing",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A dispatch has no durable approval or submission.",
    "Inspect the pause and durable submission.",
)
RECOVERY_SCAN_LIMIT = Code(
    "recovery_scan_limit",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["limit"]),
    "A recovery scan bound exceeds the supported maximum.",
    "Reduce the scan bound.",
)
LOOP_UNKNOWN = Code(
    "loop_unknown",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A recovery marker has an unknown loop.",
    "Use a supported recovery loop and inspect worker versions.",
)
LEGACY_RUN_SWEPT = Code(
    "legacy_run_swept",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A stale format-1 run is finalised.",
    "Inspect the stopped run; format-1 runs cannot be reclaimed.",
)
STALE_RUNNING = Code(
    "stale_running",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["gap_s"]),
    "A stale run is finalised under the error policy.",
    "Inspect worker health and lease sizing.",
)
CANCEL_REQUESTED = Code(
    "cancel_requested",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset([]),
    "A stale run has a durable cancellation request.",
    "No action unless cancellation is unexpected.",
)
SUBMISSION_GENERATION = Code(
    "submission_generation",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A submission belongs to a different pause generation.",
    "Reload the pending pause and resubmit.",
)
LEASE_RENEW_FAILED = Code(
    "lease_renew_failed",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["attempt"]),
    "A lease renewal fails or exceeds its deadline.",
    "Check store latency before the lease is lost.",
)
LEASE_LOST = Code(
    "lease_lost",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "Consecutive lease renewals fail.",
    "Restore store connectivity and inspect recovery eligibility.",
)
HISTORY_TOO_LARGE = Code(
    "history_too_large",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["bytes", "limit", "count"]),
    "Conversation rebuild exceeds its bound.",
    "Reduce history or raise the documented rebuild bound.",
)
PAUSE_FORMAT_UNSUPPORTED = Code(
    "pause_format_unsupported",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset([]),
    "A stored pause format is unsupported.",
    "Upgrade every worker before continuing the run.",
)
STORE_CAPABILITY = Code(
    "store_capability",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A declared store capability has an incompatible method.",
    "Implement the frozen store contract.",
)
SKILLS_SESSION_UNKNOWN = Code(
    "skills_session_unknown",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset(["count"]),
    "A skill session references unknown state.",
    "Restore the pinned skill definitions.",
)
ACTIVATION_CONFLICT = Code(
    "activation_conflict",
    logging.WARNING,
    "migration",
    _BASE_FIELDS | frozenset(["table", "index"]),
    "Activation detects conflicting duplicate rows.",
    "Resolve the bounded diagnostic before retrying activation.",
)
RECLAIM_LIMIT = Code(
    "reclaim_limit",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS | frozenset(["attempt"]),
    "A run reaches its reclaim attempt limit.",
    "Alert at any rate and inspect worker crashes.",
)
PENDING_LIMIT = Code(
    "pending_limit",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset(["count", "limit"]),
    "A model produces too many pending calls.",
    "Reduce tool fanout or raise the pending call bound.",
)
NOTIFIER_DEGRADED = Code(
    "notifier_degraded",
    logging.WARNING,
    "observability",
    _BASE_FIELDS | frozenset([]),
    "The notification channel is unavailable and readers poll.",
    "Check the listener connection; alert after one minute.",
)
NOTIFIER_RECOVERED = Code(
    "notifier_recovered",
    logging.INFO,
    "observability",
    _BASE_FIELDS | frozenset([]),
    "The notification channel reconnects.",
    "Confirm reader latency returns to normal.",
)
NOTIFIER_UNAVAILABLE = Code(
    "notifier_unavailable",
    logging.WARNING,
    "observability",
    _BASE_FIELDS | frozenset([]),
    "The notification channel cannot start.",
    "Check its direct or session-pooled Postgres connection.",
)
NOTIFIER_MISSED_HINT = Code(
    "notifier_missed_hint",
    logging.WARNING,
    "notifier",
    _BASE_FIELDS | frozenset({"count"}),
    "A healthy feed finds new rows on reconciliation without a wake hint, once per feed.",
    "Enable the channel on every writer and verify matching writer/listener channels.",
)
PROVIDER_UNKNOWN = Code(
    "provider_unknown",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset(["provider", "category", "status", "timeout_reason", "retryable"]),
    "A vendor code has no closed mapping.",
    "Inspect the provider category and status; never log vendor text.",
)
LOG_FIELD_DROPPED = Code(
    "log_field_dropped",
    logging.WARNING,
    "observability",
    _BASE_FIELDS | frozenset(["count"]),
    "A structured log field fails closed validation.",
    "Fix the emitting call; values are deliberately omitted.",
)
PERSISTENCE_ERROR = Code(
    "persistence_error",
    logging.WARNING,
    "store",
    _BASE_FIELDS | frozenset([]),
    "A durable operation fails.",
    "Restore store connectivity and inspect the run result.",
)
MCP_DISCOVERY_REPEATED_CURSOR = Code(
    "mcp_discovery_repeated_cursor",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery repeats its cursor.",
    "Inspect the upstream server or adjust the discovery bound.",
)
MCP_DISCOVERY_PAGE_CAP = Code(
    "mcp_discovery_page_cap",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery exceeds the page cap.",
    "Inspect the upstream server or adjust the discovery bound.",
)
MCP_DISCOVERY_DEADLINE = Code(
    "mcp_discovery_deadline",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery exceeds its deadline.",
    "Inspect the upstream server or adjust the discovery bound.",
)
MCP_DISCOVERY_TOOL_CAP = Code(
    "mcp_discovery_tool_cap",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery exceeds the tool cap.",
    "Inspect the upstream server or adjust the discovery bound.",
)
MCP_DISCOVERY_BYTE_CAP = Code(
    "mcp_discovery_byte_cap",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery exceeds the byte cap.",
    "Inspect the upstream server or adjust the discovery bound.",
)
MCP_DISCOVERY_CURSOR_LENGTH = Code(
    "mcp_discovery_cursor_length",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset(["pages"]),
    "MCP discovery exceeds the cursor bound.",
    "Inspect the upstream server or adjust the discovery bound.",
)

API_ERROR = Code(
    "api_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports an internal failure or unavailable service.",
    "Check provider status and retry only when the error is retryable.",
)
AUTHENTICATION_ERROR = Code(
    "authentication_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects credentials or access permissions.",
    "Check the configured credentials and model permissions.",
)
BILLING_ERROR = Code(
    "billing_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit or usage limit.",
    "Restore provider credits or account quota before retrying.",
)
BILLING_HARD_LIMIT_REACHED = Code(
    "billing_hard_limit_reached",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit or usage limit.",
    "Restore provider credits or account quota before retrying.",
)
CONTENT_POLICY_VIOLATION = Code(
    "content_policy_violation",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider content policy rejects input or output.",
    "Review the request against provider content policy.",
)
CONTEXT_LENGTH_EXCEEDED = Code(
    "context_length_exceeded",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The request exceeds the model context window.",
    "Reduce conversation history, tool results or requested output tokens.",
)
EMPTY_COMPLETION = Code(
    "empty_completion",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider completes without a usable semantic response.",
    "Inspect provider behavior and request configuration before retrying.",
)
IMAGE_CONTENT_POLICY_VIOLATION = Code(
    "image_content_policy_violation",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider content policy rejects input or output.",
    "Review the request against provider content policy.",
)
IMAGE_PARSE_ERROR = Code(
    "image_parse_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
IMAGE_TOO_LARGE = Code(
    "image_too_large",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
IMAGE_TOO_SMALL = Code(
    "image_too_small",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
INSUFFICIENT_CREDITS = Code(
    "insufficient_credits",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit or usage limit.",
    "Restore provider credits or account quota before retrying.",
)
INSUFFICIENT_QUOTA = Code(
    "insufficient_quota",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit or usage limit.",
    "Restore provider credits or account quota before retrying.",
)
INVALID_API_KEY = Code(
    "invalid_api_key",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects credentials or access permissions.",
    "Check the configured credentials and model permissions.",
)
INVALID_BASE64_IMAGE = Code(
    "invalid_base64_image",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider failure projects the declared invalid_base64_image code.",
    "Inspect the safe category and status; apply the provider retry guidance.",
)
INVALID_IMAGE = Code(
    "invalid_image",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
INVALID_IMAGE_FORMAT = Code(
    "invalid_image_format",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
INVALID_IMAGE_URL = Code(
    "invalid_image_url",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
INVALID_PROMPT = Code(
    "invalid_prompt",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
INVALID_REQUEST_ERROR = Code(
    "invalid_request_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
INVALID_VALUE = Code(
    "invalid_value",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
MODEL_NOT_FOUND = Code(
    "model_not_found",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The configured provider model is unavailable to the account.",
    "Check the model identifier and account access.",
)
NOT_FOUND_ERROR = Code(
    "not_found_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
ORGANIZATION_USAGE_LIMIT_EXCEEDED = Code(
    "organization_usage_limit_exceeded",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit or usage limit.",
    "Restore provider credits or account quota before retrying.",
)
OVERLOADED_ERROR = Code(
    "overloaded_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports an internal failure or unavailable service.",
    "Check provider status and retry only when the error is retryable.",
)
PERMISSION_DENIED = Code(
    "permission_denied",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects credentials or access permissions.",
    "Check the configured credentials and model permissions.",
)
PERMISSION_ERROR = Code(
    "permission_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects credentials or access permissions.",
    "Check the configured credentials and model permissions.",
)
PROVIDER_TIMEOUT = Code(
    "provider_timeout",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports a request timeout.",
    "Inspect provider latency and the configured request deadlines.",
)
RATE_LIMIT_ERROR = Code(
    "rate_limit_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rate-limits the request.",
    "Reduce request rate and honor retry-after guidance.",
)
RATE_LIMIT_EXCEEDED = Code(
    "rate_limit_exceeded",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rate-limits the request.",
    "Reduce request rate and honor retry-after guidance.",
)
REQUEST_TOO_LARGE = Code(
    "request_too_large",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
RESPONSE_SIZE_LIMIT = Code(
    "response_size_limit",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider response exceeds its configured byte bound.",
    "Reduce response size or raise the response bound deliberately.",
)
SERVER_ERROR = Code(
    "server_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports an internal failure or unavailable service.",
    "Check provider status and retry only when the error is retryable.",
)
SERVICE_UNAVAILABLE = Code(
    "service_unavailable",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports an internal failure or unavailable service.",
    "Check provider status and retry only when the error is retryable.",
)
STREAM_SIZE_LIMIT = Code(
    "stream_size_limit",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider response exceeds its configured byte bound.",
    "Reduce response size or raise the response bound deliberately.",
)
STREAM_TOOL_CALL_LIMIT = Code(
    "stream_tool_call_limit",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider stream exceeds the distinct tool-call cap.",
    "Reduce tool fanout or raise the stream tool-call bound.",
)
TOKEN_LIMIT_EXCEEDED = Code(
    "token_limit_exceeded",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects request parameters, resources or request limits.",
    "Check request parameters, resource identifiers and provider limits.",
)
UNSUPPORTED_IMAGE = Code(
    "unsupported_image",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects image format, content, size or location.",
    "Check the image descriptor, supported formats and provider size limits.",
)
VECTOR_STORE_TIMEOUT = Code(
    "vector_store_timeout",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider reports an internal failure or unavailable service.",
    "Check provider status and retry only when the error is retryable.",
)

STALE_RUNNING_LEGACY = Code(
    "stale_running_legacy",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "A legacy stale run cannot be reclaimed.",
    "Inspect the stopped run and worker health.",
)

CALL_DEADLINE = Code(
    "call_deadline",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The overall provider call deadline expires.",
    "Reduce retry/response time or raise the call deadline.",
)
CONTENT_FILTER = Code(
    "content_filter",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider content filter rejects the request or response.",
    "Review the input against provider content policy.",
)
DEADLINE_DURING_BACKOFF = Code(
    "deadline_during_backoff",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider call deadline expires during retry backoff.",
    "Inspect the preceding HTTP status and provider availability.",
)
STREAM_IDLE = Code(
    "stream_idle",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider stream exceeds its semantic idle deadline.",
    "Inspect provider responsiveness and idle deadline sizing.",
)
STREAM_TOTAL = Code(
    "stream_total",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider stream exceeds its total deadline.",
    "Reduce response length or raise the total deadline.",
)

AUTH = Code(
    "auth",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects credentials without a vendor code.",
    "Check provider credentials and account permissions.",
)
CREDITS = Code(
    "credits",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider account reaches a credit limit without a vendor code.",
    "Restore provider credits or quota before retrying.",
)
RATE_LIMIT = Code(
    "rate_limit",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rate-limits a request without a vendor code.",
    "Reduce request rate and honor retry-after guidance.",
)
REJECTED = Code(
    "rejected",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider rejects a request without a vendor code.",
    "Check request parameters and provider limits.",
)
UNAVAILABLE = Code(
    "unavailable",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider is unavailable without a vendor code.",
    "Check provider status and retry only when retryable.",
)
TIMEOUT = Code(
    "timeout",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider request times out without a vendor code.",
    "Inspect the timeout reason and configured deadlines.",
)
CONTEXT_LENGTH = Code(
    "context_length",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A request exceeds the context window without a vendor code.",
    "Reduce conversation history or requested output tokens.",
)
REFUSAL = Code(
    "refusal",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The model refuses to answer without a vendor code.",
    "Review the request against provider policy.",
)
STREAM_INCOMPLETE = Code(
    "stream_incomplete",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "A provider stream ends prematurely without a vendor code.",
    "Inspect provider connectivity before retrying.",
)
UNKNOWN = Code(
    "unknown",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "An unclassified provider failure has no vendor code.",
    "Inspect the safe category, status and provider availability.",
)

RESPONSE_ENCODING_UNSUPPORTED = Code(
    "response_encoding_unsupported",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "The provider returns an unsupported response content encoding.",
    "Use gzip or deflate responses; inspect the provider or gateway encoding configuration.",
)

TIMEOUT_ERROR = Code(
    "timeout_error",
    logging.WARNING,
    "provider",
    _BASE_FIELDS | frozenset({"provider", "category", "status", "timeout_reason", "retryable"}),
    "Anthropic reports that the request timed out.",
    "Inspect provider availability and the request deadline before retrying.",
)

LEGACY_RUN = Code(
    "legacy_run",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "The run uses a storage format that cannot be reclaimed.",
    "Allow the legacy run to finish or start a new run.",
)
NOT_RUNNING = Code(
    "not_running",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Reclaim targets a run that is no longer running.",
    "Read the current result before retrying recovery.",
)
LEASE_FRESH = Code(
    "lease_fresh",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Reclaim targets a run whose lease has not expired.",
    "Sweep stale leases before attempting reclaim.",
)
ALREADY_RECLAIMED = Code(
    "already_reclaimed",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Another caller already claimed the recovery generation.",
    "Use the winning worker and avoid duplicate recovery.",
)
AGENT_MISMATCH = Code(
    "agent_mismatch",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "The recovery agent differs from the saved agent.",
    "Recreate the original agent configuration.",
)
SKILL_MISMATCH = Code(
    "skill_mismatch",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery skill pins differ from the saved pins.",
    "Restore the pinned skill versions before retrying.",
)
LOOP_UNSUPPORTED = Code(
    "loop_unsupported",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "The saved loop does not support recovery.",
    "Use a supported loop for new recoverable runs.",
)
RECOVERY_CALL_MISSING = Code(
    "recovery_call_missing",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery cannot locate a checkpointed tool call.",
    "Inspect durable tool rows before restarting the run.",
)
RECOVERY_STATE_MISSING = Code(
    "recovery_state_missing",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery cannot locate required checkpoint state.",
    "Inspect checkpoint persistence before restarting the run.",
)
RECOVERY_STATE_INVALID = Code(
    "recovery_state_invalid",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery cannot rebuild the saved checkpoint.",
    "Inspect checkpoint compatibility and start a new run if necessary.",
)
NEVER_STARTED = Code(
    "never_started",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Sweep finalizes a stale run that never started.",
    "Inspect worker startup and retry with a new run.",
)
RUN_FINALIZED = Code(
    "run_finalized",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery finalizes a run instead of continuing it.",
    "Read the final status and its recovery reason.",
)
RUN_NOT_FOUND = Code(
    "run_not_found",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery cannot find a run in the tenant scope.",
    "Verify the run identifier and tenant scope.",
)
TOOL_REPLAYED = Code(
    "tool_replayed",
    logging.WARNING,
    "recovery",
    _BASE_FIELDS,
    "Recovery replays an interrupted idempotent tool.",
    "Monitor repeated worker crashes and tool idempotence.",
)

CASCADE_TRUNCATED = Code(
    "cascade_truncated",
    logging.WARNING,
    "runtime",
    _BASE_FIELDS | frozenset({"limit", "count"}),
    "Cancellation discovers more descendants than the subtree traversal bound.",
    "Inspect the bounded subtree and cancel remaining subtrees or raise the bound.",
)

MCP_TRANSPORT_FAILED = Code(
    "mcp_transport_failed",
    logging.DEBUG,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "An MCP transport operation fails.",
    "Inspect typed failure metadata and transport configuration.",
)

MCP_TRANSPORT_CLEANUP_FAILED = Code(
    "mcp_transport_cleanup_failed",
    logging.DEBUG,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "Cleanup after an MCP connect failure also fails.",
    "Inspect transport cleanup and cancellation handling.",
)

MCP_CREDENTIAL_RESOLUTION_FAILED = Code(
    "mcp_credential_resolution_failed",
    logging.DEBUG,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "MCP credential resolution fails.",
    "Check the credential provider configuration.",
)

MCP_OBSERVER_DISPOSE_FAILED = Code(
    "mcp_observer_dispose_failed",
    logging.DEBUG,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "Disposing an MCP observer awaitable fails.",
    "Use synchronous observers that return no awaitable.",
)

MCP_OAUTH_CLOSE_FAILED = Code(
    "mcp_oauth_close_failed",
    logging.DEBUG,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "Closing an OAuth response or transport fails.",
    "Inspect the OAuth transport lifecycle.",
)

MCP_CLEANUP_FAILED = Code(
    "mcp_cleanup_failed",
    logging.WARNING,
    "mcp",
    _BASE_FIELDS | frozenset({"count"}),
    "An MCP connection cleanup operation fails.",
    "Inspect transport cleanup and repeated failure counts.",
)

REGISTRY = MappingProxyType(
    {
        code.name: code
        for code in (
            MCP_TRANSPORT_FAILED,
            MCP_TRANSPORT_CLEANUP_FAILED,
            MCP_CREDENTIAL_RESOLUTION_FAILED,
            MCP_OBSERVER_DISPOSE_FAILED,
            MCP_OAUTH_CLOSE_FAILED,
            MCP_CLEANUP_FAILED,
            CASCADE_TRUNCATED,
            LEGACY_RUN,
            NOT_RUNNING,
            LEASE_FRESH,
            ALREADY_RECLAIMED,
            AGENT_MISMATCH,
            SKILL_MISMATCH,
            LOOP_UNSUPPORTED,
            RECOVERY_CALL_MISSING,
            RECOVERY_STATE_MISSING,
            RECOVERY_STATE_INVALID,
            NEVER_STARTED,
            RUN_FINALIZED,
            RUN_NOT_FOUND,
            TOOL_REPLAYED,
            TIMEOUT_ERROR,
            RESPONSE_ENCODING_UNSUPPORTED,
            AUTH,
            CREDITS,
            RATE_LIMIT,
            REJECTED,
            UNAVAILABLE,
            TIMEOUT,
            CONTEXT_LENGTH,
            REFUSAL,
            STREAM_INCOMPLETE,
            UNKNOWN,
            CALL_DEADLINE,
            CONTENT_FILTER,
            DEADLINE_DURING_BACKOFF,
            STREAM_IDLE,
            STREAM_TOTAL,
            STALE_RUNNING_LEGACY,
            API_ERROR,
            AUTHENTICATION_ERROR,
            BILLING_ERROR,
            BILLING_HARD_LIMIT_REACHED,
            CONTENT_POLICY_VIOLATION,
            CONTEXT_LENGTH_EXCEEDED,
            EMPTY_COMPLETION,
            IMAGE_CONTENT_POLICY_VIOLATION,
            IMAGE_PARSE_ERROR,
            IMAGE_TOO_LARGE,
            IMAGE_TOO_SMALL,
            INSUFFICIENT_CREDITS,
            INSUFFICIENT_QUOTA,
            INVALID_API_KEY,
            INVALID_BASE64_IMAGE,
            INVALID_IMAGE,
            INVALID_IMAGE_FORMAT,
            INVALID_IMAGE_URL,
            INVALID_PROMPT,
            INVALID_REQUEST_ERROR,
            INVALID_VALUE,
            MODEL_NOT_FOUND,
            NOT_FOUND_ERROR,
            ORGANIZATION_USAGE_LIMIT_EXCEEDED,
            OVERLOADED_ERROR,
            PERMISSION_DENIED,
            PERMISSION_ERROR,
            PROVIDER_TIMEOUT,
            RATE_LIMIT_ERROR,
            RATE_LIMIT_EXCEEDED,
            REQUEST_TOO_LARGE,
            RESPONSE_SIZE_LIMIT,
            SERVER_ERROR,
            SERVICE_UNAVAILABLE,
            STREAM_SIZE_LIMIT,
            STREAM_TOOL_CALL_LIMIT,
            TOKEN_LIMIT_EXCEEDED,
            UNSUPPORTED_IMAGE,
            VECTOR_STORE_TIMEOUT,
            FOLLOW_FEED_ERROR,
            FOLLOW_CAPACITY,
            FOLLOW_TENANT_CAPACITY,
            FOLLOW_SHUTDOWN_PENDING,
            USAGE_INVALID,
            DATABASE_URL_ENV_IGNORED,
            TENANT_SCOPE_REQUIRED,
            RESERVED_METADATA_KEY,
            METADATA_TOO_LARGE,
            RUN_FAILED,
            READ_ARGUMENT_INVALID,
            PAUSE_DATA_NOT_JSONB,
            FENCED_FINISH_FAILED,
            CANCEL_PERSIST_FAILED,
            INVALID_TOOL_RESULT,
            PAUSE_ID_MISMATCH,
            LEGACY_CLAIM_VALIDATION,
            GOVERNANCE_MISMATCH,
            GOVERNANCE_LEGACY,
            GOVERNANCE_CHANGED,
            SIZE_LIMIT,
            WIRE_ANSWER_OMITTED,
            NOTE_CAP_UNENFORCED,
            INPUT_SIZE_LIMIT,
            IDEMPOTENCY_CONFLICT_LEGACY,
            NOTE_LIMIT,
            SUBTREE_LIMIT,
            RUN_FENCED,
            SWEEP_CAP_REACHED,
            MIGRATION_LOCK_TIMEOUT,
            SHARED_ENGINE_LOOP_CLOSED,
            TOOL_ROW_MAY_BE_MISSING,
            OUTCOME_UNKNOWN,
            CANCELLED,
            TIMEOUT_NOT_STARTED,
            TIMEOUT_OUTCOME_UNKNOWN,
            TOOL_POOL_SATURATED,
            TOOL_ZOMBIE_FINISHED,
            INVALID_TOOL_SCHEMA,
            INVALID_SCHEMA,
            MCP_VIEW_RELINKED,
            LLM_FAILED_WRITE,
            RUN_NOT_RECLAIMABLE,
            RUN_CANCELLED,
            INVALID_LIMIT,
            PROVIDER_KWARGS_INVALID,
            PROVIDER_KWARGS_TOO_LARGE,
            LEASE_INTERVAL,
            CHECKPOINT_TOO_LARGE,
            REWRITE_INVALID,
            WRITE_CONFLICT,
            TRACE_LINK_INVALID,
            SUBMISSION_MISSING,
            RECOVERY_SCAN_LIMIT,
            LOOP_UNKNOWN,
            LEGACY_RUN_SWEPT,
            STALE_RUNNING,
            CANCEL_REQUESTED,
            SUBMISSION_GENERATION,
            LEASE_RENEW_FAILED,
            LEASE_LOST,
            HISTORY_TOO_LARGE,
            PAUSE_FORMAT_UNSUPPORTED,
            STORE_CAPABILITY,
            SKILLS_SESSION_UNKNOWN,
            ACTIVATION_CONFLICT,
            RECLAIM_LIMIT,
            PENDING_LIMIT,
            NOTIFIER_DEGRADED,
            NOTIFIER_RECOVERED,
            NOTIFIER_UNAVAILABLE,
            NOTIFIER_MISSED_HINT,
            PROVIDER_UNKNOWN,
            LOG_FIELD_DROPPED,
            PERSISTENCE_ERROR,
            MCP_DISCOVERY_REPEATED_CURSOR,
            MCP_DISCOVERY_PAGE_CAP,
            MCP_DISCOVERY_DEADLINE,
            MCP_DISCOVERY_TOOL_CAP,
            MCP_DISCOVERY_BYTE_CAP,
            MCP_DISCOVERY_CURSOR_LENGTH,
        )
    }
)


def _valid_field(name: str, value: Any) -> bool:
    if name in {"run_id", "pause_id", "event_id", "row_id"}:
        return isinstance(value, str) and _ULID.fullmatch(value) is not None
    if name in {"call_id", "tool_call_id"}:
        return isinstance(value, str) and _PROVIDER_ID.fullmatch(value) is not None
    if name in {"trace_id", "span_id"}:
        shape = _TRACE_ID if name == "trace_id" else _SPAN_ID
        return isinstance(value, str) and shape.fullmatch(value) is not None and int(value, 16) != 0
    if name == "tenant_id":
        return (
            isinstance(value, str)
            and 1 <= len(value) <= 256
            and all(unicodedata.category(char) != "Cc" for char in value)
        )
    if name in _NUMBERS:
        return type(value) in (int, float) and (type(value) is int or math.isfinite(value))
    if name in {"retryable", "truncated"}:
        return type(value) is bool
    return isinstance(value, str) and value in _ENUMS.get(name, ())


def log_code(
    logger: logging.Logger,
    code: Code,
    *,
    run_id: str | None = None,
    tenant_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit a registered code, refusing undeclared or unsafe field values.

    DENDRUX_STRICT_LOG_FIELDS=1 raises on invalid fields. Otherwise they are
    dropped with one bounded diagnostic containing only a count, never values.
    Provider names outside the built-ins become custom. Other text uses closed
    enums, except tenant identifiers. An unregistered Code always raises.
    """
    if REGISTRY.get(code.name) is not code:
        raise ValueError("Unregistered log code")
    supplied = dict(fields)
    if run_id is not None:
        supplied["run_id"] = run_id
    if tenant_id is not None:
        supplied["tenant_id"] = tenant_id
    if isinstance(supplied.get("provider"), str) and supplied["provider"] not in _ENUMS["provider"]:
        supplied["provider"] = "custom"
    safe = {
        name: value
        for name, value in supplied.items()
        if name in code.fields and _valid_field(name, value)
    }
    dropped = len(supplied) - len(safe)
    if dropped:
        if os.environ.get("DENDRUX_STRICT_LOG_FIELDS") == "1":
            raise ValueError("Invalid structured log fields")
        diagnostic = {
            "code": LOG_FIELD_DROPPED.name,
            "kind": LOG_FIELD_DROPPED.kind,
            "count": dropped,
        }
        diagnostic.update({name: safe[name] for name in ("run_id", "tenant_id") if name in safe})
        logger.log(LOG_FIELD_DROPPED.level, LOG_FIELD_DROPPED.name, extra=diagnostic)
    logger.log(code.level, code.name, extra={"code": code.name, "kind": code.kind, **safe})
