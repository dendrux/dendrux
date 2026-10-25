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
from typing import TYPE_CHECKING, Any

from dendrux.loops.react import _CLIENT_APPROVAL_REASON, _HOOK_ERROR_REASON
from dendrux.runtime._pending import approval_calls, check_read_bool, unknown_tool_calls
from dendrux.runtime.submit import DEFAULT_DENIAL_REASON

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from dendrux.runtime.state import TenantScope
    from dendrux.store import StoredEvent

UNKNOWN_TOOL = "<unknown_tool>"
"""Shown instead of a tool name the model invented (a call rejected as unknown)."""
UNKNOWN_SKILL = "<unknown_skill>"
"""Shown instead of a skill name the model invented."""

# Event-data keys that are payloads in every event type.
_HIDDEN_EVENT_KEYS = frozenset(
    {"system_prompt", "provider_request", "provider_response", "pii_mapping"}
)
# Pauses whose arguments are shown to a human as the model sent them.
_PLACEHOLDER_PAUSES = frozenset({"waiting_approval", "waiting_human_input"})
# Dendrux's own fixed reasons; any other reason is someone's text.
_CONSTANT_DENIAL_REASONS = frozenset(
    {"denied_by_policy", DEFAULT_DENIAL_REASON, _HOOK_ERROR_REASON, _CLIENT_APPROVAL_REASON}
)
_CONSTANT_APPROVAL_REASONS = frozenset({"requires_approval"})
_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")
_RETRY_AFTER = re.compile(r"\d{1,10}")


def is_one_of(value: object, choices: Collection[str]) -> bool:
    """``value`` is one of the string ``choices`` (any other shape is not)."""
    return isinstance(value, str) and value in choices


def _without_keys(item: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(item, dict):
        return item
    return {k: v for k, v in item.items() if k not in keys}


def _omit(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in keys}


def without_params(call: Any) -> Any:
    """A pending-call dict without its arguments."""
    return _without_keys(call, ("params", "arguments"))


def _dicts(value: Any) -> list[dict[str, Any]]:
    """The dict entries of a stored list (other shapes are dropped)."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class _Context:
    """What decides the redacted form beyond the event itself."""

    active: bool
    unknown_calls: frozenset[str] | None


Rule = Callable[[dict[str, Any], "StoredEvent", _Context], dict[str, Any]]


def _keep(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    return data


def _drop(*keys: str) -> Rule:
    def rule(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
        return _omit(data, keys)

    return rule


def _constant(key: str, allowed: frozenset[str], data: dict[str, Any]) -> dict[str, Any]:
    if key in data and not is_one_of(data[key], allowed):
        return _omit(data, (key,))
    return data


def _paused(data: dict[str, Any], _event: StoredEvent, ctx: _Context) -> dict[str, Any]:
    data = _omit(data, ("message",))  # a requested pause's free-text reason
    if not ctx.active and "pending_tool_calls" in data:
        data["pending_tool_calls"] = [without_params(c) for c in _dicts(data["pending_tool_calls"])]
    return data


def _approval_requested(data: dict[str, Any], _event: StoredEvent, ctx: _Context) -> dict[str, Any]:
    if not ctx.active:
        data = _omit(data, ("params",))
    return _constant("reason", _CONSTANT_APPROVAL_REASONS, data)


def _decided(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    if "decisions" in data:
        data["decisions"] = [
            _constant("reason", _CONSTANT_DENIAL_REASONS, d) for d in _dicts(data["decisions"])
        ]
    return data


def _policy_denied(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    return _constant("reason", _CONSTANT_DENIAL_REASONS, data)


def _tool_completed(data: dict[str, Any], event: StoredEvent, ctx: _Context) -> dict[str, Any]:
    code = data.get("code")
    if "code" in data and not (isinstance(code, str) and _CODE.fullmatch(code)):
        data = _omit(data, ("code",))
    if _unknown_tool(data, event, ctx):
        data["tool_name"] = UNKNOWN_TOOL
    return data


def _unknown_tool(data: dict[str, Any], event: StoredEvent, ctx: _Context) -> bool:
    """True unless the call is shown to be one of the agent's declared tools."""
    if not isinstance(data.get("tool_name"), str) or data.get("code") == "unknown_tool":
        return True
    if data.get("success") is True:
        return False  # it ran (or a hook substituted a result): a declared tool
    if ctx.unknown_calls is None:
        return True  # no evidence: fail closed
    return event.correlation_id in ctx.unknown_calls


def _skill_failed(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    data = _constant("reason", frozenset({"denied", "not_found"}), data)
    declared = data.get("reason") == "denied" and isinstance(data.get("name"), str)
    if "name" in data and not declared:  # denied names are the developer's
        data["name"] = UNKNOWN_SKILL
    return data


def _tools_skipped(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    if "skipped" in data:
        data["skipped"] = [
            d if d.get("reason") == "unknown_tool" else _without_keys(d, ("tool",))
            for d in (_constant("reason", _SKIP_REASONS, d) for d in _dicts(data["skipped"]))
        ]
    return data


_SKIP_REASONS = frozenset({"name_unsafe", "duplicate_name", "unknown_tool"})


def _retry(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    value = data.get("retry_after")
    if "retry_after" in data and not (isinstance(value, str) and _RETRY_AFTER.fullmatch(value)):
        return _omit(data, ("retry_after",))
    return data


def _image_state(data: dict[str, Any], _event: StoredEvent, _ctx: _Context) -> dict[str, Any]:
    if "described" in data:
        data["described"] = [_without_keys(d, ("text",)) for d in _dicts(data["described"])]
    return data


REVIEWED_EVENT_TYPES: dict[str, Rule] = {
    # Run lifecycle
    "run.started": _keep,  # agent_name, retry_of (system_prompt is a payload key)
    "run.paused": _paused,
    "run.resumed": _drop("user_input"),  # submitted_results carry validated names
    "run.completed": _keep,
    "run.error": _drop("error"),  # error_info is codes and a fixed safe message
    "run.cancelled": _keep,
    "run.interrupted": _keep,
    "run.abandoned": _keep,
    # Approvals and policy
    "approval.requested": _approval_requested,
    "approval.decided": _decided,
    "policy.denied": _policy_denied,
    "policy.substituted": _keep,
    "policy.hook_error": _keep,  # error_type is an exception class name
    "policy.hook_warning": _keep,
    # Model and tools
    "llm.completed": _keep,
    "llm.failed": _keep,
    "tool.completed": _tool_completed,
    "provider.retry": _retry,
    # Guardrails (messages are built from guardrail ids and entity kinds)
    "guardrail.blocked": _keep,
    "guardrail.detected": _keep,
    "guardrail.redacted": _keep,
    "guardrail.unmapped_placeholder": _keep,  # placeholder grammar only
    "guardrail.restore_skipped": _keep,
    "guardrail.image_unscanned": _keep,
    # Skills (paths and names are validated against declared skills)
    "skill.catalog_resolved": _keep,
    "skill.activated": _keep,
    "skill.activation_failed": _skill_failed,
    "skill.resource_read": _keep,
    "skill.script_requested": _keep,
    "skill.script_completed": _keep,
    "skill.script_failed": _keep,
    # MCP (source names are developer config; exposed names passed the safe-name check)
    "mcp.connected": _keep,
    "mcp.error": _drop("error"),
    "mcp.tools_skipped": _tools_skipped,
    # Budget, governance, inbox and images
    "budget.unpriced": _keep,
    "budget.threshold": _keep,
    "budget.exceeded": _keep,
    "governance.changed": _keep,
    "inbox.delivered": _keep,
    "image.rejected": _keep,
    "image.view_changed": _keep,
    "image.degraded": _keep,
    "image.warning": _keep,
    "image.state": _image_state,
}


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
    rule = REVIEWED_EVENT_TYPES.get(event.event_type)
    if rule is None:  # an event type nobody reviewed shows no data
        return dataclasses.replace(event, data={})
    data = {k: v for k, v in data.items() if k not in _HIDDEN_EVENT_KEYS}
    context = _Context(active=active_pause is True, unknown_calls=unknown_calls)
    return dataclasses.replace(event, data=rule(data, event, context))


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
        data = event.data or {}
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
    data = event.data or {}
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
        and (event.data or {}).get("success") is not True
    )


async def read_unknown_calls(
    store: Any, run_id: str, tenant_id: TenantScope
) -> frozenset[str] | None:
    return await unknown_tool_calls(getattr(store, "_state", None), run_id, tenant_id)
