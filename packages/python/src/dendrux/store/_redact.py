"""What a reader without payloads may see of a stored event.

The rule: in a redacted view only developer-declared identifiers (agent, tool,
skill, guardrail and MCP source names), ids, numbers and Dendrux's constant
codes survive. Anything a person, a client, the model, a hook, a tool, a vendor
or an exception wrote is hidden. Trusted readers (``include_payloads=True``)
see everything.

Every event type Dendrux writes is listed in :data:`REVIEWED_EVENT_TYPES` with
its rule; an event type missing from it is shown without any data (fail
closed). A test keeps the list in step with the event types in the source.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args

from dendrux import errors as _errors
from dendrux.errors import PROVIDER_ERROR_CATEGORIES, TIMEOUT_REASONS, ProviderError
from dendrux.guardrails._engine import RESTORE_SKIPPED_CODE
from dendrux.guardrails._pii import _DEFAULT_DETECTORS as _PII_DETECTORS
from dendrux.guardrails._protocol import UNSCANNABLE
from dendrux.guardrails._secrets import _SECRET_DETECTORS
from dendrux.loops._budget import BUDGET_EXCEEDED, COST_USD, TOKENS, UNPRICED_CALL
from dendrux.loops._images import WARN_DROP_WITH_THINKING
from dendrux.loops.react import _CLIENT_APPROVAL_REASON, _HOOK_ERROR_REASON
from dendrux.media._ids import PlaceholderReason
from dendrux.media._policy import Origin as ImageOrigin
from dendrux.pricing import CostSource
from dendrux.runtime._limit_checks import INPUT_SIZE_LIMIT_CODE
from dendrux.runtime._pending import approval_calls, check_read_bool, unknown_tool_calls
from dendrux.runtime.submit import DEFAULT_DENIAL_REASON
from dendrux.tools._limits import MAX_CALLS_CODE, SIZE_LIMIT_CODE
from dendrux.tools._validate import (
    ARGUMENTS_NOT_STABLE_CODE,
    INVALID_JSON_CODE,
    INVALID_PARAMS_CODE,
    PII_PLACEHOLDER_CODE,
)
from dendrux.types import BudgetAction, RunStatus, StopReason

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from dendrux.runtime.state import TenantScope
    from dendrux.store import StoredEvent

UNKNOWN_TOOL = "<unknown_tool>"
"""Shown instead of a tool name the model invented (a call rejected as unknown)."""
UNKNOWN_SKILL = "<unknown_skill>"
"""Shown instead of a skill name the model invented."""

# Pauses whose arguments are shown to a human as the model sent them.
_PLACEHOLDER_PAUSES = frozenset({"waiting_approval", "waiting_human_input"})
# Dendrux's own fixed reasons; any other reason is someone's text.
_CONSTANT_DENIAL_REASONS = frozenset(
    {"denied_by_policy", DEFAULT_DENIAL_REASON, _HOOK_ERROR_REASON, _CLIENT_APPROVAL_REASON}
)
_CONSTANT_APPROVAL_REASONS = frozenset({"requires_approval"})
TOOL_RESULT_CODES = frozenset(
    {
        # tools/_validate.py
        INVALID_PARAMS_CODE,
        INVALID_JSON_CODE,
        PII_PLACEHOLDER_CODE,
        ARGUMENTS_NOT_STABLE_CODE,
        # tools/_limits.py and the result size cap
        SIZE_LIMIT_CODE,
        MAX_CALLS_CODE,
        INPUT_SIZE_LIMIT_CODE,
        # loops/react.py: timeouts, stopped siblings, the thread pool, unknown tools
        "timeout_not_started",
        "timeout_outcome_unknown",
        "cancelled",
        "outcome_unknown",
        "tool_pool_saturated",
        "unknown_tool",
    }
)
"""Every ``tool.completed`` code Dendrux writes; any other code is shown as UNKNOWN_CODE."""
PROVIDER_ERROR_TYPES = frozenset(
    name
    for name, value in vars(_errors).items()
    if isinstance(value, type) and issubclass(value, ProviderError)
)


def is_one_of(value: object, choices: Collection[str]) -> bool:
    """``value`` is one of the string ``choices`` (any other shape is not)."""
    return isinstance(value, str) and value in choices


def _without_keys(item: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(item, dict):
        return item
    return {k: v for k, v in item.items() if k not in keys}


def without_params(call: Any) -> Any:
    """A pending-call dict without its arguments."""
    return _without_keys(call, ("params", "arguments"))


def data_of(event: StoredEvent) -> dict[str, Any]:
    """The event's data as a dict; a malformed (non-object) row reads as empty."""
    return event.data if isinstance(event.data, dict) else {}


def _dicts(value: Any) -> list[dict[str, Any]]:
    """The dict entries of a stored list (other shapes are dropped)."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class _Context:
    """What decides the redacted form beyond the event itself."""

    active: bool
    unknown_calls: frozenset[str] | None


# ---------------------------------------------------------------------------
# Field checks. Each returns the value to show, or _HIDE. A key with no check
# in its event type's schema is hidden: the schema is an allowlist.
# ---------------------------------------------------------------------------

_HIDE: Any = object()
Check = Callable[[Any], Any]
UNKNOWN_CODE = "<unknown_code>"
"""Shown instead of a ``tool.completed`` code Dendrux does not produce."""
CUSTOM_KIND = "<custom_kind>"
"""Shown instead of a guardrail finding kind that is not one of Dendrux's built-in kinds."""

# Enumerations of every constant a shown field may hold, each from its writer.
# A shape check (regex, length) never makes a field safe: a value is shown
# only when it is in one of these sets, or it is a number, a bool, an id
# Dendrux generated, a digest Dendrux computed, or a developer-declared name
# whose writer only ever copies it from configuration.
_RUN_STATUSES = frozenset(s.value for s in RunStatus)
_PAUSE_STATUSES = frozenset(
    {"waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"}
)
_STOP_REASONS = frozenset(get_args(StopReason))
_PAUSE_REASONS = frozenset({"requested", "budget"})  # react pause_reason
_TARGETS = frozenset({"client", "server"})  # tool(target=...)
_CANCEL_REASONS = frozenset(
    {"cancel_requested", "task_cancelled", "stream_closed", BUDGET_EXCEEDED}
)
_FAILURE_REASONS = frozenset(
    {"stale_running", "never_started", "abandoned_waiting", "guardrail_blocked", BUDGET_EXCEEDED}
)
_BUDGET_METRICS = frozenset({TOKENS, COST_USD})
_BUDGET_REASONS = frozenset({"threshold_crossed", BUDGET_EXCEEDED, UNPRICED_CALL})
_BUDGET_ACTIONS = frozenset(get_args(BudgetAction))
_APPROVAL_SOURCES = frozenset({"static", "hook"})
_DECISIONS = frozenset({"approved", "rejected", "mixed"})
_CALL_DECISIONS = frozenset({"approved", "denied"})
_COST_SOURCES = frozenset(get_args(CostSource))
_PHASES = frozenset({"commentary", "final_answer"})
_DIRECTIONS = frozenset({"incoming", "outgoing", "incoming_image"})
_GUARDRAIL_CODES = frozenset({"guardrail_blocked"})
_RESTORE_CODES = frozenset({RESTORE_SKIPPED_CODE})
_VERDICT_ACTIONS = frozenset({"pass", "redact", "block", "warn"})
_IMAGE_ORIGINS = frozenset(get_args(ImageOrigin))
_PLACEHOLDER_REASONS = frozenset(get_args(PlaceholderReason))
_IMAGE_WARNINGS = frozenset({WARN_DROP_WITH_THINKING})
_SKILL_DELIVERY = frozenset({"inlined", "progressive"})
_SKILL_ORIGINS = frozenset({"explicit", "inlined", "model"})
_SKILL_FAILURES = frozenset({"timeout", "executor_unavailable", "invalid_result"})
_SKILL_ACTIVATION = frozenset({"denied", "not_found"})
_MCP_CODES = frozenset(
    {
        "authorization_required",
        "authentication_failed",
        "credential_error",
        "destination_denied",
        "circuit_open",
        "capacity",
        "connection_failed",
        "connection_state",
        "tool_name_collision",
        "tool_not_found",
        "invalid_tool_names",
        "unknown",
    }
)  # mcp/_errors.error_code_for
_MCP_FAILURE_MODES = frozenset({"strict", "best_effort"})  # mcp/_source.MCPFailureMode
_MCP_AUTH_REASONS = frozenset(
    {"not_signed_in", "expired", "revoked", "insufficient_scope", "resource_mismatch"}
)  # mcp/_errors.MCPAuthorizationReason
_MCP_SKIP_REASONS = frozenset({"name_unsafe", "duplicate_name", "unknown_tool"})
_GOVERNANCE_PARTS = frozenset(
    {"guardrails", "before_tool", "deny", "require_approval", "limits", "budget"}
)  # runtime/submit.governance_snapshot
BUILTIN_GUARDRAIL_KINDS = frozenset(
    {d.name for d in _PII_DETECTORS} | {d.name for d in _SECRET_DETECTORS} | {UNSCANNABLE}
)
"""Finding kinds of Dendrux's built-in detectors; any other kind is shown as CUSTOM_KIND."""
_MAX_RETRY_AFTER = 86_400
_RETRY_AFTER = re.compile(r"\d{1,6}")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_HEX = re.compile(r"[0-9a-f]{8,128}")


def _enum(choices: Collection[str]) -> Check:
    return lambda value: value if value is None or is_one_of(value, choices) else _HIDE


def _num(value: Any) -> Any:
    ok = value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
    return value if ok else _HIDE


def _bool(value: Any) -> Any:
    return value if value is None or isinstance(value, bool) else _HIDE


def _generated_id(value: Any) -> Any:
    """An id Dendrux generated (run, call, pause, note, image id). Provenance: the
    writer passes a ``generate_ulid()`` / ``image_id()`` value; the check only
    rejects malformed rows."""
    return value if value is None or (isinstance(value, str) and _ID.fullmatch(value)) else _HIDE


def _digest(value: Any) -> Any:
    """A digest Dendrux computed (sha256, governance and skill digests)."""
    return value if value is None or (isinstance(value, str) and _HEX.fullmatch(value)) else _HIDE


def _declared(value: Any) -> Any:
    """A developer-declared identifier. Provenance: the writer copies it from the
    developer's configuration, or only after checking it against that
    configuration (a declared tool, skill, source or guardrail)."""
    ok = value is None or (isinstance(value, str) and 0 < len(value) <= 255)
    return value if ok else _HIDE


def _list(check: Check) -> Check:
    def checked(value: Any) -> Any:
        if not isinstance(value, list):
            return _HIDE
        return [item for item in map(check, value) if item is not _HIDE]

    return checked


def _obj(schema: dict[str, Check]) -> Check:
    return lambda value: _apply(schema, value) if isinstance(value, dict) else _HIDE


def _kind(value: Any) -> str:
    return value if is_one_of(value, BUILTIN_GUARDRAIL_KINDS) else CUSTOM_KIND


def _kinds(value: Any) -> Any:
    """Finding kinds as a list or a ``{kind: count}`` map; custom kinds merged
    into CUSTOM_KIND (their counts are summed)."""
    if isinstance(value, dict):
        merged: dict[str, Any] = {}
        for key, count in value.items():
            if _num(count) is not _HIDE and count is not None:
                merged[_kind(key)] = merged.get(_kind(key), 0) + count
        return merged
    if isinstance(value, list):
        return sorted({_kind(item) for item in value})
    return _HIDE


def _retry_after(value: Any) -> Any:
    """The retry-after header only as a bounded whole number of seconds."""
    if isinstance(value, str) and _RETRY_AFTER.fullmatch(value) and int(value) <= _MAX_RETRY_AFTER:
        return int(value)
    return _HIDE


def _apply(schema: dict[str, Check], data: dict[str, Any]) -> dict[str, Any]:
    shown = {}
    for key, value in data.items():
        check = schema.get(key)
        if check is not None and (kept := check(value)) is not _HIDE:
            shown[key] = kept
    return shown


_EXCEEDED = _obj({"metric": _enum(_BUDGET_METRICS), "used": _num, "max": _num})
_ERROR_INFO = _obj(
    {
        "category": _enum(PROVIDER_ERROR_CATEGORIES),
        "retryable": _bool,
        "error_type": _enum(PROVIDER_ERROR_TYPES),
        "provider": _declared,  # the adapter's provider name (approved, round 2)
        "status_code": _num,
        "timeout_reason": _enum(TIMEOUT_REASONS),
        # hidden: safe_message (embeds provider_code), model, provider_code
    }
)
_PENDING_CALL: dict[str, Check] = {
    "id": _generated_id,
    "name": _declared,  # a pending call is a declared tool (unknown tools never pause)
    "target": _enum(_TARGETS),
    "args_hidden": _bool,
}
_SCHEMAS: dict[str, dict[str, Check]] = {
    # runtime/runner.py: Agent(name=...) and the retried run's id
    "run.started": {"agent_name": _declared, "retry_of": _generated_id},
    # runner._pause_event_data; params only for the active pause (in _special);
    # message (a person's request_pause reason) hidden
    "run.paused": {
        "status": _enum(_PAUSE_STATUSES),
        "pause_id": _generated_id,
        "reason": _enum(_PAUSE_REASONS),
        "exceeded": _list(_EXCEEDED),
        "pending_tool_calls": _list(_obj(_PENDING_CALL)),
    },
    # runner resume; submitted names were checked against the pause's calls
    "run.resumed": {
        "resumed_from": _enum(_PAUSE_STATUSES),
        "image_count": _num,
        "submitted_results": _list(
            _obj(
                {
                    "call_id": _generated_id,
                    "name": _declared,
                    "success": _bool,
                    "image_count": _num,
                }
            )
        ),
    },
    "run.completed": {
        "status": _enum(_RUN_STATUSES),
        "stop_reason": _enum(_STOP_REASONS),
        "truncated": _bool,
    },
    "run.error": {
        "status": _enum(_RUN_STATUSES),
        "stop_reason": _enum(_STOP_REASONS),
        "truncated": _bool,
        "error_info": _ERROR_INFO,
    },
    "run.cancelled": {
        "reason": _enum(_CANCEL_REASONS),
        "interrupted": _bool,
        "partial_output": _bool,
        "exceeded": _list(_EXCEEDED),
        "failure_reason": _enum(_FAILURE_REASONS),
    },
    "run.interrupted": {"failure_reason": _enum(_FAILURE_REASONS)},
    "run.abandoned": {"failure_reason": _enum(_FAILURE_REASONS)},
    # loops/react.py: gated calls are declared tools; params in _special
    "approval.requested": {
        "tool_name": _declared,
        "call_id": _generated_id,
        "source": _enum(_APPROVAL_SOURCES),
        "reason": _enum(_CONSTANT_APPROVAL_REASONS),
    },
    "approval.decided": {
        "decision": _enum(_DECISIONS),
        "run_id": _generated_id,
        "decisions": _list(
            _obj(
                {
                    "call_id": _generated_id,
                    "tool_name": _declared,
                    "decision": _enum(_CALL_DECISIONS),
                    "reason": _enum(_CONSTANT_DENIAL_REASONS),
                }
            )
        ),
    },
    # react policy paths: hooks run on declared tools; error_type (a class name) hidden
    "policy.denied": {
        "tool_name": _declared,
        "call_id": _generated_id,
        "source": _enum(_APPROVAL_SOURCES),
        "reason": _enum(_CONSTANT_DENIAL_REASONS),
    },
    "policy.substituted": {"tool_name": _declared, "call_id": _generated_id},
    "policy.hook_error": {"tool_name": _declared, "call_id": _generated_id},
    "policy.hook_warning": {
        "tool_name": _declared,
        "call_id": _generated_id,
        "warning": _enum({_CLIENT_APPROVAL_REASON}),
    },
    # runtime/persistence.py: numbers and enumerations; model hidden
    "llm.completed": {
        "input_tokens": _num,
        "output_tokens": _num,
        "cache_read_input_tokens": _num,
        "cache_creation_input_tokens": _num,
        "reasoning_tokens": _num,
        "cost_usd": _num,
        "cost_source": _enum(_COST_SOURCES),
        "usage_reported": _bool,
        "has_tool_calls": _bool,
        "phase": _enum(_PHASES),
    },
    "llm.failed": {
        "error_info": _ERROR_INFO,
        "timeout_reason": _enum(TIMEOUT_REASONS),
        "duration_ms": _num,
    },
    # persistence: tool_name and code decided in _special; parse_error hidden
    # (a provider adapter may set it), raw_length kept
    "tool.completed": {
        "tool_name": _declared,
        "target": _enum(_TARGETS),
        "success": _bool,
        "duration_ms": _num,
        "zombie_threads": _num,
        "raw_length": _num,
        "image_count": _num,
    },
    # llm/_retry_telemetry.py: endpoint (request path) hidden; retry_after a number
    "provider.retry": {"status_code": _num, "attempt": _num, "retry_after": _retry_after},
    # guardrail paths: kinds are a scanner's output, so only built-in kinds are named
    "guardrail.blocked": {
        "direction": _enum(_DIRECTIONS),
        "code": _enum(_GUARDRAIL_CODES),
        "kinds": _kinds,
        "counts": _kinds,
    },
    "guardrail.detected": {
        "direction": _enum(_DIRECTIONS),
        "findings_count": _num,
        "entities": _kinds,
    },
    "guardrail.redacted": {"direction": _enum(_DIRECTIONS), "entities": _kinds},
    # runner: placeholder-shaped tokens the model wrote: only their count (in _special)
    "guardrail.unmapped_placeholder": {},
    "guardrail.restore_skipped": {
        "tool_name": _declared,
        "call_id": _generated_id,
        "entities": _kinds,
        "count": _num,
        "code": _enum(_RESTORE_CODES),
    },
    "guardrail.image_unscanned": {"count": _num},
    # skills/_session.py: names, revisions and paths are checked against the
    # developer's skills before they are written
    "skill.catalog_resolved": {
        "delivery": _enum(_SKILL_DELIVERY),
        "skills": _list(_obj({"name": _declared, "revision": _declared, "digest": _digest})),
        "denied": _list(_declared),
    },
    "skill.activated": {"name": _declared, "digest": _digest, "origin": _enum(_SKILL_ORIGINS)},
    "skill.activation_failed": {"name": _declared, "reason": _enum(_SKILL_ACTIVATION)},
    "skill.resource_read": {
        "name": _declared,
        "path": _declared,
        "chars": _num,
        "truncated": _bool,
    },
    "skill.script_requested": {"name": _declared, "path": _declared, "argc": _num},
    "skill.script_completed": {
        "name": _declared,
        "path": _declared,
        "exit_code": _num,
        "duration_ms": _num,
        "stdout_chars": _num,
        "stderr_chars": _num,
        "truncated": _bool,
    },
    "skill.script_failed": {
        "name": _declared,
        "path": _declared,
        "error_type": _enum(_SKILL_FAILURES),
    },
    # runtime/runner.py MCP: source names are configuration; exposed tool names
    # are the run's tool set (approved, round 2); error text and class hidden
    "mcp.connected": {
        "source_name": _declared,
        "namespace": _declared,
        "tool_count": _num,
        "tool_names": _list(_declared),
    },
    "mcp.error": {
        "source_name": _declared,
        "namespace": _declared,
        "code": _enum(_MCP_CODES),
        "failure_mode": _enum(_MCP_FAILURE_MODES),
        "reason": _enum(_MCP_AUTH_REASONS),
    },
    "mcp.tools_skipped": {
        "skipped": _list(
            _obj(
                {
                    "source_name": _declared,
                    "namespace": _declared,
                    "tool": _declared,  # kept only for unknown_tool (_special)
                    "reason": _enum(_MCP_SKIP_REASONS),
                }
            )
        )
    },
    # loops/_budget.py
    "budget.unpriced": {
        "metric": _enum(_BUDGET_METRICS),
        "used": _num,
        "max": _num,
        "unpriced_calls": _num,
        "known_cost_usd": _num,
    },
    "budget.threshold": {
        "metric": _enum(_BUDGET_METRICS),
        "fraction": _num,
        "used": _num,
        "max": _num,
        "reason": _enum(_BUDGET_REASONS),
    },
    "budget.exceeded": {
        "metric": _enum(_BUDGET_METRICS),
        "used": _num,
        "max": _num,
        "reason": _enum(_BUDGET_REASONS),
        "action": _enum(_BUDGET_ACTIONS),
    },
    # runtime/submit.py
    "governance.changed": {
        "expected_digest": _digest,
        "actual_digest": _digest,
        "changed": _list(_enum(_GOVERNANCE_PARTS)),
    },
    "inbox.delivered": {"note_id": _generated_id},
    # loops/_images.py: model hidden; tenant_scope is the run's own (approved)
    "image.rejected": {
        "image_id": _generated_id,
        "sha256": _digest,
        "tenant_scope": _declared,
        "provider": _declared,
        "status_code": _num,
        "located": _bool,
        "retried": _bool,
    },
    "image.view_changed": {"epoch": _num, "dropped_through": _num, "visible": _num},
    "image.degraded": {
        "reason": _enum(_PLACEHOLDER_REASONS),
        "count": _num,
        "slots": _list(_num),
    },
    "image.warning": {"warning": _enum(_IMAGE_WARNINGS)},
    "image.state": {
        "tenant_scope": _declared,
        "described": _list(_obj({"sha256": _digest})),
        "guardrail_verdicts": _list(
            _obj(
                {
                    "sha256": _digest,
                    "guardrail_id": _declared,
                    "guardrail_version": _declared,
                    "origin": _enum(_IMAGE_ORIGINS),
                    "tool_name": _declared,
                    "action": _enum(_VERDICT_ACTIONS),
                    "entities": _kinds,
                }
            )
        ),
        "placeholders": _list(_obj({"slot": _num, "reason": _enum(_PLACEHOLDER_REASONS)})),
    },
}


def _special(event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context) -> None:
    """Rules that need more than one field or the reader's context."""
    kind = event.event_type
    if kind == "run.paused" and ctx.active:
        for call, original in zip(
            shown.get("pending_tool_calls", []), _dicts(raw.get("pending_tool_calls")), strict=False
        ):
            if "params" in original:
                call["params"] = original["params"]  # the active pause's arguments (B2)
    elif kind == "approval.requested" and ctx.active and "params" in raw:
        shown["params"] = raw["params"]
    elif kind == "tool.completed":
        if "code" in raw:
            shown["code"] = (
                raw["code"] if is_one_of(raw["code"], TOOL_RESULT_CODES) else UNKNOWN_CODE
            )
        if _unknown_tool(raw, event, ctx):
            shown["tool_name"] = UNKNOWN_TOOL
    elif kind == "skill.activation_failed" and "name" in raw:
        declared = raw.get("reason") == "denied" and isinstance(raw.get("name"), str)
        if not declared:  # a denied name is the developer's; any other is the model's
            shown["name"] = UNKNOWN_SKILL
    elif kind == "guardrail.unmapped_placeholder" and isinstance(raw.get("placeholders"), list):
        shown["count"] = len(raw["placeholders"])
    elif kind == "mcp.tools_skipped":
        for item in shown.get("skipped", []):
            if item.get("reason") != "unknown_tool":  # only then the developer's name
                item.pop("tool", None)


def _unknown_tool(data: dict[str, Any], event: StoredEvent, ctx: _Context) -> bool:
    """True unless the call is shown to be one of the agent's declared tools."""
    if not isinstance(data.get("tool_name"), str) or data.get("code") == "unknown_tool":
        return True
    if data.get("success") is True:
        return False  # it ran (or a hook substituted a result): a declared tool
    if ctx.unknown_calls is None:
        return True  # no evidence: fail closed
    return event.correlation_id in ctx.unknown_calls


REVIEWED_EVENT_TYPES: frozenset[str] = frozenset(_SCHEMAS)
"""Every event type with a reviewed allowlist (a test keeps it complete)."""


def redact_event(
    event: StoredEvent,
    *,
    include_payloads: bool = False,
    active_pause: bool = False,
    pii_mapping: Mapping[str, str] | None = None,
    approval_params: Mapping[str, dict[str, Any]] | None = None,
) -> StoredEvent:
    """The event as a reader may see it (B2 and the approval-arguments rule).

    ``include_payloads`` and ``active_pause`` must be real ``bool`` values
    (a string such as ``"false"`` raises ``ValueError``).

    In every mode, the arguments of an **approval or input** ``run.paused``
    use ``approval.requested.params``, exactly as the model sent them. Without
    those arguments, ``params`` is hidden. Client-tool pauses keep the 0.4.1 form.

    With ``include_payloads=True`` nothing else changes. Otherwise only
    developer-declared identifiers, ids, numbers and Dendrux's constant codes
    remain:

    - the system prompt, provider bodies and any ``pii_mapping`` are removed;
    - a pause's arguments (``run.paused``) and an approval request's
      ``params`` are removed unless ``active_pause`` is true; a pause's
      ``message`` is always removed;
    - people's and hooks' text is removed: ``run.resumed.user_input``, denial
      and hook reasons (Dendrux's constant reasons stay), exception text
      (``run.error.error``, ``mcp.error.error``; ``error_info`` stays), image
      descriptions, non-numeric ``provider.retry.retry_after`` and skipped
      MCP tool names the server supplied;
    - a model-invented name becomes ``"<unknown_tool>"`` / ``"<unknown_skill>"``.
      Standalone, a failed ``tool.completed`` cannot be shown to name a
      declared tool, so its name is ``"<unknown_tool>"``; ``RunStore`` views
      check the run's rejected calls and keep declared names.

    Tool results and server-tool arguments never appear in event data; read
    them with ``RunStore.get_tool_invocations`` (trusted readers only).
    ``pii_mapping`` is accepted for compatibility and not used.
    """
    check_read_bool("include_payloads", include_payloads)
    check_read_bool("active_pause", active_pause)
    return _redact(
        event,
        include_payloads=include_payloads,
        active_pause=active_pause,
        approval_params=approval_params,
        unknown_calls=None,
    )


def _redact(
    event: StoredEvent,
    *,
    include_payloads: bool,
    active_pause: bool,
    approval_params: Mapping[str, dict[str, Any]] | None,
    unknown_calls: frozenset[str] | None,
) -> StoredEvent:
    if not isinstance(event.data, dict):  # malformed or empty row: nothing to show
        return event if include_payloads is True else dataclasses.replace(event, data={})
    if not event.data:
        return event
    data = dict(event.data)
    if (
        event.event_type == "run.paused"
        and is_one_of(data.get("status"), _PLACEHOLDER_PAUSES)
        and isinstance(data.get("pending_tool_calls"), list)
    ):
        data["pending_tool_calls"] = approval_calls(data["pending_tool_calls"], approval_params)
    if include_payloads is True:
        return dataclasses.replace(event, data=data) if data != event.data else event
    schema = _SCHEMAS.get(event.event_type)
    if schema is None:  # an event type nobody reviewed shows no data
        return dataclasses.replace(event, data={})
    context = _Context(active=active_pause is True, unknown_calls=unknown_calls)
    shown = _apply(schema, data)
    _special(event, data, shown, context)
    return dataclasses.replace(event, data=shown)


# ---------------------------------------------------------------------------
# Views that need the database: the active pause, past pause arguments and
# the run's rejected (unknown-tool) calls.
# ---------------------------------------------------------------------------

# Event types whose view depends on the run's active pause.
PAUSE_VIEW_EVENTS = frozenset({"run.paused", "approval.requested"})
# Event types after which a cached view of the active pause may be out of date.
PAUSE_CHANGING_EVENTS = PAUSE_VIEW_EVENTS | {
    "run.resumed",
    "run.completed",
    "run.error",
    "run.cancelled",
    "run.interrupted",
    "run.abandoned",
}
_PAUSE_EVENT_TYPES = ["approval.requested", "run.paused", "run.resumed"]
_PAGE = 1000
_PAST_WINDOW = 1024


@dataclass(slots=True)
class PauseView:
    """The run's active pause when it was read, valid for events up to ``through``.

    ``active`` is the index of the active ``run.paused`` event (None when the
    run is not paused); ``active_params`` its recorded model-sent arguments by
    call id (the ``approval.requested`` events of that pause).
    """

    through: int
    active: int | None
    active_params: dict[str, dict[str, Any]]


async def read_pause_view(
    store: Any, run_id: str, tenant_id: TenantScope, through: int
) -> PauseView:
    """Two bounded reads: the active pause's index, then its recorded arguments."""
    active = await store._active_pause_sequence(run_id, tenant_id)
    params = {} if active is None else await store._approval_params(run_id, active, tenant_id)
    return PauseView(through, active, params)


@dataclass(slots=True)
class PastParams:
    """Recorded arguments of past pauses, read forward in pages of 1000 events.

    One reader (a trusted follower, or one ``get_pause_page`` call) asks for
    pauses in increasing order, so the cost is one query per 1000
    pause-related events, never one per pause. A small window of results is
    kept; asking for an older pause again starts over from the beginning.
    Stores without recorded arguments show none (fail closed).
    """

    cursor: int | None = None
    current: dict[str, dict[str, Any]] = field(default_factory=dict)
    found: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)

    async def get(
        self, store: Any, run_id: str, seq: int, tenant_id: TenantScope
    ) -> dict[str, dict[str, Any]]:
        if seq in self.found:
            return self.found[seq]
        if getattr(getattr(store, "_state", None), "get_approval_params", None) is None:
            return {}
        if self.cursor is not None and seq <= self.cursor:
            self.cursor, self.current, self.found = None, {}, {}
        while self.cursor is None or self.cursor < seq:
            rows = await store.get_events(
                run_id,
                after_sequence_index=self.cursor,
                limit=_PAGE,
                event_types=_PAUSE_EVENT_TYPES,
                tenant_id=tenant_id,
            )
            for row in rows:
                self._consume(row)
            if len(rows) < _PAGE:
                break
        return self.found.get(seq, {})

    def _consume(self, event: StoredEvent) -> None:
        data = data_of(event)
        if event.event_type == "approval.requested":
            call_id, params = data.get("call_id"), data.get("params")
            if isinstance(call_id, str) and isinstance(params, dict):
                self.current[call_id] = params
        else:
            if event.event_type == "run.paused":
                self.found[event.sequence_index] = self.current
                if len(self.found) > _PAST_WINDOW:
                    del self.found[next(iter(self.found))]
            self.current = {}
        self.cursor = event.sequence_index


async def view_with(
    store: Any,
    run_id: str,
    event: StoredEvent,
    view: PauseView,
    *,
    tenant_id: TenantScope,
    include_payloads: bool,
    past: PastParams,
    unknown_calls: frozenset[str] | None = None,
) -> StoredEvent:
    """``event`` as a reader sees it, given the run's active pause ``view``."""
    data = data_of(event)
    seq = event.sequence_index
    active = False
    params = None
    if event.event_type == "approval.requested":
        call_id = data.get("call_id")
        active = (
            view.active is not None
            and seq < view.active
            and isinstance(call_id, str)
            and call_id in view.active_params
        )
    elif event.event_type == "run.paused":
        active = seq == view.active
        if (include_payloads or active) and is_one_of(data.get("status"), _PLACEHOLDER_PAUSES):
            params = view.active_params if active else await past.get(store, run_id, seq, tenant_id)
    return _redact(
        event,
        include_payloads=include_payloads,
        active_pause=active,
        approval_params=params,
        unknown_calls=unknown_calls,
    )


def needs_unknown_calls(event: StoredEvent, include_payloads: bool) -> bool:
    """A redacted failed ``tool.completed``: its name needs the run's rejected calls."""
    return (
        not include_payloads
        and event.event_type == "tool.completed"
        and data_of(event).get("success") is not True
    )


async def read_unknown_calls(
    store: Any, run_id: str, tenant_id: TenantScope
) -> frozenset[str] | None:
    return await unknown_tool_calls(getattr(store, "_state", None), run_id, tenant_id)
