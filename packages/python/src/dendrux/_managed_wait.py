"""Bounded, strict codecs for persisted managed waits (v3 and v4)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux.errors import ManagedStateError

if TYPE_CHECKING:
    from dendrux.types import PauseState

MAX_PARALLEL_SUBAGENTS = 32
MANAGED_FANOUT_SCHEMA_VERSION = 4

_ULID_PATTERN = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
MANAGED_PROTOCOL_VERSION = 1
MANAGED_WAIT_REASON = "managed_wait"
MANAGED_WAIT_SCHEMA_VERSION = 3
# Matches types.ULID_CHARS without importing the higher-level types module.
_MANAGED_CALL_ID_MAX_CHARS = 26


@dataclass(frozen=True)
class ManagedWait:
    """Internal: one immutable managed tool-call binding, used by v3 and v4 waits.

    A v3 wait saves this envelope directly. A v4 wait saves the two ids as an
    entry in its ordered bindings list. Never derived from model arguments.
    """

    tool_call_id: str
    child_run_id: str
    version: int = MANAGED_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tool_call_id, str)
            or not self.tool_call_id
            or len(self.tool_call_id) > _MANAGED_CALL_ID_MAX_CHARS
        ):
            raise ValueError("ManagedWait.tool_call_id must be a stable call id")
        if not isinstance(self.child_run_id, str) or not _ULID_PATTERN.fullmatch(self.child_run_id):
            raise ValueError("ManagedWait.child_run_id must be a run ULID")
        if type(self.version) is not int or self.version != MANAGED_PROTOCOL_VERSION:
            raise ValueError("Unsupported managed protocol version")

    def to_dict(self) -> dict[str, Any]:
        """The stored binding envelope."""
        return {
            "v": self.version,
            "tool_call_id": self.tool_call_id,
            "child_run_id": self.child_run_id,
        }


MANAGED_WAIT_MAX_BYTES = 32 * 1024
"""Upper bound of a stored v3 or v4 wait (references only, never history)."""
_MANAGED_WAIT_ABSENT = ("history", "steps", "image_view")
_MANAGED_WAIT_NONE = ("approval_gates", "pending_decisions", "clarification")


def _compact_json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _managed_wait_binding(data: Any, run_id: str | None) -> ManagedWait:
    """Validate a stored v3 managed wait; return its binding or refuse safely."""
    reason, binding = _managed_wait_check(data)
    if reason is not None or binding is None:
        raise ManagedStateError(reason=reason or "wait_invalid", run_id=run_id)
    return binding


def _managed_wait_check(data: Any) -> tuple[str | None, ManagedWait | None]:
    """Shared by the v3 encoder and decoder: ``(refusal reason, binding)``."""
    if not isinstance(data, dict) or "managed_fanout" in data:
        return "wait_invalid", None
    try:
        size = _compact_json_bytes(data)
    except (TypeError, ValueError, RecursionError):
        return "wait_invalid", None
    if size > MANAGED_WAIT_MAX_BYTES:
        return "wait_too_large", None
    binding = _managed_wait_envelope(data) if _managed_wait_fields_ok(data) else None
    if (
        binding is None
        or not _managed_wait_call_ok(data, binding)
        or not _managed_wait_continuation_ok(data)
    ):
        return "wait_invalid", None
    return None, binding


def _managed_wait_fields_ok(
    data: dict[str, Any], *, schema_version: int = MANAGED_WAIT_SCHEMA_VERSION
) -> bool:
    return (
        type(data.get("schema_version")) is int
        and data["schema_version"] == schema_version
        and data.get("pause_reason") == MANAGED_WAIT_REASON
        and isinstance(data.get("agent_name"), str)
        and bool(data["agent_name"])
        and isinstance(data.get("pause_id"), str)
        and _ULID_PATTERN.fullmatch(data["pause_id"]) is not None
        and data.get("loop_state_ref") is True
        and not any(key in data for key in _MANAGED_WAIT_ABSENT)
        and all(data.get(key) is None for key in _MANAGED_WAIT_NONE)
    )


def _managed_wait_envelope(data: dict[str, Any]) -> ManagedWait | None:
    envelope = data.get("managed_wait")
    if not isinstance(envelope, dict) or set(envelope) != {"v", "tool_call_id", "child_run_id"}:
        return None
    try:
        return ManagedWait(
            tool_call_id=envelope["tool_call_id"],
            child_run_id=envelope["child_run_id"],
            version=envelope["v"],
        )
    except ValueError:
        return None


def _managed_wait_call_ok(data: dict[str, Any], binding: ManagedWait) -> bool:
    calls = data.get("pending_tool_calls")
    return (
        isinstance(calls, list)
        and len(calls) == 1
        and isinstance(calls[0], dict)
        and calls[0].get("id") == binding.tool_call_id
        and isinstance(calls[0].get("name"), str)
        and calls[0].get("params") == {}
        and _optional_text(calls[0].get("provider_tool_call_id"))
        and data.get("pending_targets") == {binding.tool_call_id: "agent"}
    )


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _money(value: Any) -> bool:
    """Total predicate: a finite, non-negative int or float; never raises."""
    if type(value) is int:
        try:
            value = float(value)
        except OverflowError:
            return False
    return type(value) is float and math.isfinite(value) and value >= 0


def _optional_text(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _usage_ok(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    counts = ("input_tokens", "output_tokens", "total_tokens")
    optional = ("cache_read_input_tokens", "cache_creation_input_tokens", "reasoning_tokens")
    return (
        all(_count(usage.get(key, 0)) for key in counts)
        and all(usage.get(key) is None or _count(usage[key]) for key in optional)
        and (usage.get("cost_usd") is None or _money(usage["cost_usd"]))
        and _optional_text(usage.get("cost_source"))
        and type(usage.get("usage_reported", True)) is bool
        and type(usage.get("cost_unknown", False)) is bool
    )


def _budget_ok(data: dict[str, Any]) -> bool:
    fired = data.get("budget_fired")
    exceeded = data.get("budget_exceeded")
    known = data.get("budget_known_cost_usd")
    return (
        (
            fired is None
            or isinstance(fired, list)
            and all(
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and _money(item[1])
                for item in fired
            )
        )
        and (exceeded is None or isinstance(exceeded, list))
        and all(isinstance(item, dict) for item in exceeded or ())
        and (known is None or _money(known))
        and _count(data.get("budget_unpriced_calls", 0))
        and type(data.get("budget_unpriced_reported", False)) is bool
    )


def _managed_wait_continuation_ok(data: dict[str, Any]) -> bool:
    """The continuation fields the managed protocol reads, with exact types."""
    return (
        _count(data.get("iteration"))
        and _count(data.get("trace_order_offset"))
        and _usage_ok(data.get("usage"))
        and _optional_text(data.get("image_view_ref"))
        and isinstance(data.get("provider_kwargs", {}), dict)
        and _optional_text(data.get("pause_message"))
        and _budget_ok(data)
    )


def encode_managed_wait(pause: PauseState) -> dict[str, Any]:
    """Format-2 references plus the binding; the brief stays on the tool row."""
    binding = pause.managed_wait
    if binding is None:
        raise ValueError("A v3 managed wait needs its managed binding")
    call_id = binding.tool_call_id
    if (
        [tc.id for tc in pause.pending_tool_calls] != [call_id]
        or pause.pending_targets != {call_id: "agent"}
        or pause.pause_reason != MANAGED_WAIT_REASON
        or not isinstance(pause.pause_id, str)
        or not _ULID_PATTERN.fullmatch(pause.pause_id)
        or pause.approval_gates is not None
        or pause.pending_decisions is not None
        or pause.clarification is not None
        or not pause.loop_state_ref
    ):
        raise ValueError("Incomplete managed wait")
    references = replace(
        pause,
        managed_wait=None,
        managed_fanout=None,
        schema_version=2,
        pending_tool_calls=[replace(pause.pending_tool_calls[0], params={})],
    )
    d = references.to_dict()
    d["schema_version"] = MANAGED_WAIT_SCHEMA_VERSION
    d["managed_wait"] = binding.to_dict()
    reason, _ = _managed_wait_check(d)
    if reason is not None:
        # The encoder never writes what the strict decoder would refuse.
        raise ValueError(f"Invalid managed wait ({reason})")
    return d


@dataclass(frozen=True)
class ManagedFanout:
    """Immutable ordered bindings for one persisted v4 wait generation."""

    bindings: tuple[ManagedWait, ...]
    version: int = MANAGED_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or self.version != MANAGED_PROTOCOL_VERSION
            or not isinstance(self.bindings, tuple)
            or not 2 <= len(self.bindings) <= MAX_PARALLEL_SUBAGENTS
            or not all(isinstance(binding, ManagedWait) for binding in self.bindings)
        ):
            raise ValueError("Invalid managed fan-out bindings")
        calls = {binding.tool_call_id for binding in self.bindings}
        children = {binding.child_run_id for binding in self.bindings}
        if len(calls) != len(self.bindings) or len(children) != len(self.bindings):
            raise ValueError("Managed fan-out bindings must be unique")

    def to_dict(self) -> dict[str, Any]:
        """The stored v4 binding envelope, without caller-owned mutable values."""
        return {
            "v": self.version,
            "n": len(self.bindings),
            "bindings": [
                {"tool_call_id": binding.tool_call_id, "child_run_id": binding.child_run_id}
                for binding in self.bindings
            ],
        }

    def to_checkpoint(self, *, pause_id: str) -> dict[str, Any]:
        """The matching loop-state wait record, with a fresh bindings mapping.

        Call order comes from the pause envelope's ``bindings`` list or
        ``managed_admitted``, never this mapping: JSONB may reorder its keys.
        """
        if not isinstance(pause_id, str) or not _ULID_PATTERN.fullmatch(pause_id):
            raise ValueError("A managed fan-out checkpoint needs a pause ULID")
        return {
            "v": self.version,
            "pause_id": pause_id,
            "bindings": {b.tool_call_id: b.child_run_id for b in self.bindings},
        }


def _fanout_envelope(data: dict[str, Any]) -> ManagedFanout | None:
    envelope = data.get("managed_fanout")
    if not isinstance(envelope, dict) or set(envelope) != {"v", "n", "bindings"}:
        return None
    n, bindings = envelope["n"], envelope["bindings"]
    if (
        type(n) is not int
        or not 2 <= n <= MAX_PARALLEL_SUBAGENTS
        or not isinstance(bindings, list)
        or len(bindings) != n
        or not all(
            isinstance(b, dict) and set(b) == {"tool_call_id", "child_run_id"} for b in bindings
        )
    ):
        return None
    try:
        return ManagedFanout(
            tuple(ManagedWait(b["tool_call_id"], b["child_run_id"]) for b in bindings),
            version=envelope["v"],
        )
    except ValueError:
        return None


def _fanout_calls_ok(data: dict[str, Any], fanout: ManagedFanout) -> bool:
    calls = data.get("pending_tool_calls")
    if not isinstance(calls, list) or len(calls) != len(fanout.bindings):
        return False
    return all(
        isinstance(call, dict)
        and call.get("id") == binding.tool_call_id
        and isinstance(call.get("name"), str)
        and call.get("params") == {}
        and _optional_text(call.get("provider_tool_call_id"))
        for call, binding in zip(calls, fanout.bindings, strict=True)
    ) and data.get("pending_targets") == {b.tool_call_id: "agent" for b in fanout.bindings}


def _managed_fanout_check(data: Any) -> ManagedFanout | None:
    if not isinstance(data, dict):
        return None
    try:
        if _compact_json_bytes(data) > MANAGED_WAIT_MAX_BYTES:
            return None
    except (TypeError, ValueError, RecursionError):
        return None
    if not _managed_wait_fields_ok(data, schema_version=4) or "managed_wait" in data:
        return None
    fanout = _fanout_envelope(data)
    if (
        fanout is None
        or not _fanout_calls_ok(data, fanout)
        or not _managed_wait_continuation_ok(data)
    ):
        return None
    return fanout


def decode_managed_binding(data: Any, *, run_id: str | None) -> ManagedWait | ManagedFanout:
    """Validate persisted managed JSON before the PauseState decoder reads it."""
    # Persisted JSON is a dynamic boundary; validate before constructing typed records.
    if isinstance(data, dict) and data.get("schema_version") == MANAGED_FANOUT_SCHEMA_VERSION:
        fanout = _managed_fanout_check(data)
        if fanout is None:
            raise ManagedStateError(reason="wait_invalid", run_id=run_id)
        return fanout
    return _managed_wait_binding(data, run_id)


def encode_managed_fanout(pause: PauseState) -> dict[str, Any]:
    """Serialize reference-only v4 state, applying the strict decoder's check."""
    fanout = pause.managed_fanout
    if fanout is None or pause.managed_wait is not None or pause.loop_state_ref is not True:
        raise ValueError("A v4 managed wait needs only its fan-out binding and references")
    references = replace(
        pause,
        managed_wait=None,
        managed_fanout=None,
        schema_version=2,
        pending_tool_calls=[replace(call, params={}) for call in pause.pending_tool_calls],
    )
    data = references.to_dict()
    data["schema_version"] = MANAGED_FANOUT_SCHEMA_VERSION
    data["managed_fanout"] = fanout.to_dict()
    if _managed_fanout_check(data) is None:
        raise ValueError("Invalid managed fan-out wait")
    return data
