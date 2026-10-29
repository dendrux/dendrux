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
from dendrux.runtime._pending import (
    check_read_bool,
    declared_tool_names,
    pause_display_calls,
    pause_window_start,
    unknown_tool_calls,
)
from dendrux.runtime.submit import DEFAULT_DENIAL_REASON
from dendrux.skills._session import NAMES as SKILL_TOOL_NAMES
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

MCP_TOOL = "<mcp_tool>"
"""Shown instead of a tool name an MCP server supplied."""
CUSTOM_PROVIDER = "<custom_provider>"
"""Shown instead of a provider label that is not one of Dendrux's built-in adapters."""
BUILTIN_PROVIDERS = frozenset({"anthropic", "openai", "openai_responses", "openrouter"})
"""Provider labels Dendrux's own adapters set (``llm/anthropic.py``, ``openai.py``,
``openai_responses.py``, ``openrouter.py``); a test keeps it in step with them."""
UNKNOWN_TOOL = "<unknown_tool>"
"""Shown instead of a tool name the model invented (a call rejected as unknown)."""
UNKNOWN_SKILL = "<unknown_skill>"
"""Shown instead of a skill name the model invented."""

# Pauses whose arguments are shown to a human as the model sent them.
_PLACEHOLDER_PAUSES = frozenset({"waiting_approval", "waiting_human_input", "paused"})
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
    declared_tools: frozenset[str] | None = None
    tools: ToolEvidence | None = None


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
    """An id Dendrux generated (run, pause, note, image id; never a tool call id,
    which a custom provider may choose). Provenance: the writer passes a
    ``generate_ulid()`` / ``image_id()`` value; the check only rejects malformed rows."""
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


def _provider(value: Any) -> Any:
    """A provider label: Dendrux's built-in adapters by name, any other as CUSTOM_PROVIDER."""
    if value is None:
        return None
    return value if is_one_of(value, BUILTIN_PROVIDERS) else CUSTOM_PROVIDER


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
        "provider": _provider,  # a built-in adapter's label, else CUSTOM_PROVIDER
        "status_code": _num,
        "timeout_reason": _enum(TIMEOUT_REASONS),
        # hidden: safe_message (embeds provider_code), model, provider_code
    }
)
_PENDING_CALL: dict[str, Check] = {
    # "id" is a ToolCall.id a custom provider may choose: active pause only (_special)
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
    "run.lease_expired": {"gap_s": _num},
    "run.reclaimed": {
        "attempt": _num,
        "previous_epoch": _num,
        "gap_s": _num,
        "governance_changed": _bool,
    },
    "run.interrupted": {"failure_reason": _enum(_FAILURE_REASONS)},
    "run.abandoned": {"failure_reason": _enum(_FAILURE_REASONS)},
    # loops/react.py: gated calls are declared tools; params in _special
    "approval.requested": {
        "tool_name": _declared,
        "source": _enum(_APPROVAL_SOURCES),
        "reason": _enum(_CONSTANT_APPROVAL_REASONS),
    },
    "approval.decided": {
        "decision": _enum(_DECISIONS),
        "run_id": _generated_id,
        "decisions": _list(
            _obj(
                {
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
        "source": _enum(_APPROVAL_SOURCES),
        "reason": _enum(_CONSTANT_DENIAL_REASONS),
    },
    "policy.substituted": {"tool_name": _declared},
    "policy.hook_error": {"tool_name": _declared},
    "policy.hook_warning": {
        "tool_name": _declared,
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
    "tool.started": {"tool_name": _declared, "target": _enum(_TARGETS)},
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
        "entities": _kinds,
        "count": _num,
        "code": _enum(_RESTORE_CODES),
    },
    "guardrail.image_unscanned": {"count": _num},
    # the run's declared non-MCP tool names (positive evidence for readers):
    # redacted readers see only how many
    "tools.declared": {},
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
    # are the run's declared tool set; error text and class hidden
    "mcp.connected": {
        "source_name": _declared,
        "namespace": _declared,
        "tool_count": _num,
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
        "provider": _provider,
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


Special = Callable[["StoredEvent", dict[str, Any], dict[str, Any], _Context], None]


def _active_paused(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    if ctx.active:  # the active pause's call ids, names and arguments a client acts on (B2)
        for call, original in zip(
            shown.get("pending_tool_calls", []), _dicts(raw.get("pending_tool_calls")), strict=False
        ):
            call.update(
                {k: original[k] for k in ("id", "name", "params", "args_hidden") if k in original}
            )


def _active_request(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    if ctx.active:
        shown.update({k: raw[k] for k in ("call_id", "tool_name", "params") if k in raw})


def _image_names(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    for verdict in shown.get("guardrail_verdicts", []):
        name = verdict.get("tool_name")
        if name is not None and (ctx.declared_tools is None or name not in ctx.declared_tools):
            verdict["tool_name"] = UNKNOWN_TOOL  # a strategy-built name, unverified


def _tool_result(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    if "code" in raw:
        shown["code"] = raw["code"] if is_one_of(raw["code"], TOOL_RESULT_CODES) else UNKNOWN_CODE
    if _unknown_tool(raw, event, ctx):
        shown["tool_name"] = UNKNOWN_TOOL


def _skill_name(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    declared = raw.get("reason") == "denied" and isinstance(raw.get("name"), str)
    if "name" in raw and not declared:  # a denied name is the developer's; others the model's
        shown["name"] = UNKNOWN_SKILL


def _count_of(key: str) -> Special:
    def count(
        event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
    ) -> None:
        if isinstance(raw.get(key), list):
            shown["count"] = len(raw[key])

    return count


def _skipped_names(
    event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context
) -> None:
    for item in shown.get("skipped", []):
        if item.get("reason") != "unknown_tool":  # only then the developer's name
            item.pop("tool", None)


_SPECIAL: dict[str, Special] = {
    "run.paused": _active_paused,
    "approval.requested": _active_request,
    "image.state": _image_names,
    "tool.completed": _tool_result,
    "skill.activation_failed": _skill_name,
    "guardrail.unmapped_placeholder": _count_of("placeholders"),
    "tools.declared": _count_of("local_tool_names"),
    "mcp.tools_skipped": _skipped_names,
}


def _special(event: StoredEvent, raw: dict[str, Any], shown: dict[str, Any], ctx: _Context) -> None:
    """Rules that need more than one field or the reader's context."""
    rule = _SPECIAL.get(event.event_type)
    if rule is not None:
        rule(event, raw, shown, ctx)
    if event.event_type in TOOL_NAME_EVENTS:
        _verify_tool_names(event.event_type, shown, ctx)


# Threat model: redacted readers protect against untrusted DATA (text from
# models, provider responses, tool results, MCP servers, hook return values and
# clients). They do not protect against hostile in-process code: a custom
# provider, strategy, store or hook runs with full process privileges and
# could write any record (``tools.declared`` included) or bypass this module.
#
# Events that name tools, and where. A tool's name can come from an MCP server
# (or a custom provider or strategy), so a redacted reader sees a name only with
# positive evidence that it is a developer-declared, non-MCP tool: one of
# Dendrux's own tool names, or a name the run recorded in ``tools.declared``.
# Anything else is "<mcp_tool>" when the run's ``mcp.connected`` records list
# it, and "<unknown_tool>" otherwise (missing or partial records, custom
# stores, older runs, standalone ``redact_event``).
_TOOL_NAME_PATHS: dict[str, tuple[str | None, str]] = {
    "tool.started": (None, "tool_name"),
    "tool.completed": (None, "tool_name"),
    "approval.requested": (None, "tool_name"),
    "policy.denied": (None, "tool_name"),
    "policy.substituted": (None, "tool_name"),
    "policy.hook_error": (None, "tool_name"),
    "policy.hook_warning": (None, "tool_name"),
    "guardrail.restore_skipped": (None, "tool_name"),
    "approval.decided": ("decisions", "tool_name"),
    "run.paused": ("pending_tool_calls", "name"),
    "run.resumed": ("submitted_results", "name"),
    "image.state": ("guardrail_verdicts", "tool_name"),
}
TOOL_NAME_EVENTS = frozenset(_TOOL_NAME_PATHS)
DENDRUX_TOOL_NAMES = SKILL_TOOL_NAMES
"""Tool names Dendrux itself defines (skills); reserved, so never an MCP tool's."""


@dataclass(frozen=True, slots=True)
class ToolEvidence:
    """What the run's records prove about tool names (each None when unknown)."""

    local: frozenset[str] | None
    mcp: frozenset[str] | None


def _shown_tool_name(name: Any, ctx: _Context) -> Any:
    if not isinstance(name, str) or name in (UNKNOWN_TOOL, MCP_TOOL):
        return name
    tools = ctx.tools
    if tools is not None and tools.mcp is not None and name in tools.mcp:
        return MCP_TOOL  # MCP evidence first, even for a reserved name
    if name in DENDRUX_TOOL_NAMES:
        return name
    if tools is not None and tools.local is not None and name in tools.local:
        return name  # positively declared, non-MCP (collisions are rejected at build)
    return UNKNOWN_TOOL


def _verify_tool_names(kind: str, shown: dict[str, Any], ctx: _Context) -> None:
    container, key = _TOOL_NAME_PATHS[kind]
    if ctx.active and kind in ("run.paused", "approval.requested"):
        return  # the active pause's names, which a client acts on (B2)
    items = [shown] if container is None else shown.get(container, [])
    for item in items:
        if key in item:
            item[key] = _shown_tool_name(item[key], ctx)


def _unknown_tool(data: dict[str, Any], event: StoredEvent, ctx: _Context) -> bool:
    """True unless the call is shown to be one of the agent's declared tools."""
    if not isinstance(data.get("tool_name"), str) or data.get("code") == "unknown_tool":
        return True
    if data.get("success") is True:
        return False  # it ran (or a hook substituted a result): a declared tool
    if ctx.unknown_calls is None:
        return True  # no evidence: fail closed
    return event.correlation_id in ctx.unknown_calls


# Event types whose correlation_id is an id Dendrux generates (an LLM
# interaction, an inbox note). Every other correlation id is a ToolCall.id.
_DENDRUX_CORRELATED = frozenset({"llm.completed", "inbox.delivered"})


def _without_call_correlation(event: StoredEvent, active: bool) -> StoredEvent:
    """Hide a tool call id carried as ``correlation_id`` (a custom provider may
    choose it), except on the active pause's approval request."""
    keep = event.event_type in _DENDRUX_CORRELATED or (
        active and event.event_type == "approval.requested"
    )
    if keep or event.correlation_id is None:
        return event
    return dataclasses.replace(event, correlation_id=None)


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
    declared_tools: frozenset[str] | None = None,
    tools: ToolEvidence | None = None,
) -> StoredEvent:
    if include_payloads is True:
        return _trusted(event, approval_params)
    shown_event = _without_call_correlation(event, active_pause is True)
    schema = _SCHEMAS.get(event.event_type)
    if not isinstance(event.data, dict) or schema is None:
        # A malformed row, or an event type nobody reviewed: nothing to show.
        return dataclasses.replace(shown_event, data={})
    if not event.data:
        return shown_event
    data = _with_recorded_args(event, approval_params)
    context = _Context(
        active=active_pause is True,
        unknown_calls=unknown_calls,
        declared_tools=declared_tools,
        tools=tools,
    )
    shown = _apply(schema, data)
    _special(event, data, shown, context)  # the original event: its ids are evidence
    return dataclasses.replace(shown_event, data=shown)


def _trusted(
    event: StoredEvent, approval_params: Mapping[str, dict[str, Any]] | None
) -> StoredEvent:
    """A payload reader's view: everything, with approval pauses' recorded arguments."""
    if not isinstance(event.data, dict) or not event.data:
        return event
    data = _with_recorded_args(event, approval_params)
    return dataclasses.replace(event, data=data) if data != event.data else event


def _with_recorded_args(
    event: StoredEvent, approval_params: Mapping[str, dict[str, Any]] | None
) -> dict[str, Any]:
    data = dict(event.data or {})
    if (
        event.event_type == "run.paused"
        and is_one_of(data.get("status"), _PLACEHOLDER_PAUSES)
        and isinstance(data.get("pending_tool_calls"), list)
    ):
        data["pending_tool_calls"] = pause_display_calls(data, approval_params)
    return data


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
    window_start: int | None = None
    """The boundary before the active pause; its events lie after it. None: unknown."""


async def read_pause_view(
    store: Any, run_id: str, tenant_id: TenantScope, through: int
) -> PauseView:
    """Bounded reads: the active pause's index, its recorded arguments and window start."""
    active = await store._active_pause_sequence(run_id, tenant_id)
    if active is None:
        return PauseView(through, None, {})
    params = await store._approval_params(run_id, active, tenant_id)
    start = await pause_window_start(getattr(store, "_state", None), run_id, active, tenant_id)
    return PauseView(through, active, params, start)


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
    declared_tools: frozenset[str] | None = None,
    tools: ToolEvidence | None = None,
) -> StoredEvent:
    """``event`` as a reader sees it, given the run's active pause ``view``."""
    data = data_of(event)
    seq = event.sequence_index
    active = False
    params = None
    if event.event_type == "approval.requested":
        # Inside the active pause's own event window, never by call id (a
        # custom provider may reuse one across pauses).
        active = (
            view.active is not None
            and view.window_start is not None
            and view.window_start < seq < view.active
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
        declared_tools=declared_tools,
        tools=tools,
    )


def needs_tool_evidence(event: StoredEvent, include_payloads: bool) -> bool:
    """A redacted event that names tools: the run's tool records decide what shows."""
    return not include_payloads and event.event_type in TOOL_NAME_EVENTS


_TOOL_EVIDENCE_EVENTS = ["tools.declared", "mcp.connected"]
TOOL_EVIDENCE_EVENTS = frozenset(_TOOL_EVIDENCE_EVENTS)


async def read_tool_evidence(store: Any, run_id: str, tenant_id: TenantScope) -> ToolEvidence:
    """Declared local and MCP tool names from the run's ``tools.declared`` and
    ``mcp.connected`` records (one page of at most 1000). A full page may be
    partial, so it proves nothing (both None). A run with no ``tools.declared``
    record has no local evidence (None): its names stay hidden."""
    rows = await store.get_events(
        run_id, event_types=_TOOL_EVIDENCE_EVENTS, limit=_PAGE, tenant_id=tenant_id
    )
    if len(rows) >= _PAGE:
        return ToolEvidence(None, None)
    found: dict[str, set[str]] = {}
    for row in rows:
        key = "local_tool_names" if row.event_type == "tools.declared" else "tool_names"
        listed = data_of(row).get(key)
        names = found.setdefault(row.event_type, set())
        if isinstance(listed, list):
            names.update(n for n in listed if isinstance(n, str))
    local = found.get("tools.declared")
    mcp = found.get("mcp.connected")
    return ToolEvidence(
        frozenset(local) if local is not None else None,
        frozenset(mcp) if mcp is not None else None,
    )


def needs_declared_tools(event: StoredEvent, include_payloads: bool) -> bool:
    """A redacted ``image.state``: its verdicts' tool names need the run's recorded tools."""
    return not include_payloads and event.event_type == "image.state"


async def read_declared_tools(
    store: Any, run_id: str, tenant_id: TenantScope
) -> frozenset[str] | None:
    return await declared_tool_names(getattr(store, "_state", None), run_id, tenant_id)


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
