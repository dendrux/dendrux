"""Normalize vendor stop reasons to :data:`dendrux.types.StopReason`.

Each helper returns ``(stop_reason, raw_stop_reason)``. A missing vendor
field gives ``(None, None)``: Dendrux never guesses a stop reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.types import StopReason

_ANTHROPIC_VALUES: frozenset[str] = frozenset(
    {"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal", "pause_turn"}
)

# Anthropic values that mean something other than their own name. Sonnet 4.5
# and newer stop with ``model_context_window_exceeded`` when the output fills
# the context window before ``max_tokens``: the output is cut off, so it is a
# truncation like ``max_tokens``.
_ANTHROPIC_MAP: dict[str, StopReason] = {
    "model_context_window_exceeded": "max_tokens",
}

_CHAT_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "content_filter",
}

_RESPONSES_INCOMPLETE_MAP: dict[str, StopReason] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "content_filter",
}


def anthropic_stop_reason(raw: Any) -> tuple[StopReason | None, str | None]:
    """Anthropic ``stop_reason``: identity for the SDK values, else ``"other"``.

    ``model_context_window_exceeded`` maps to ``"max_tokens"`` (output cut off
    at the context window); ``raw_stop_reason`` keeps the vendor string.
    """
    if not isinstance(raw, str) or not raw:
        return None, None
    if raw in _ANTHROPIC_VALUES:
        return raw, raw  # type: ignore[return-value]
    return _ANTHROPIC_MAP.get(raw, "other"), raw


def chat_stop_reason(raw: Any) -> tuple[StopReason | None, str | None]:
    """Chat Completions ``finish_reason`` (OpenAI and compatible gateways)."""
    if not isinstance(raw, str) or not raw:
        return None, None
    return _CHAT_MAP.get(raw, "other"), raw


def responses_stop_reason(response: Any) -> tuple[StopReason | None, str | None]:
    """Responses API ``status`` + ``incomplete_details.reason``.

    ``completed`` maps to ``tool_use`` when the output holds a function call,
    else ``end_turn``. ``raw_stop_reason`` is the incomplete reason when the
    response is incomplete and reports one, otherwise the status.
    """
    status = getattr(response, "status", None)
    if not isinstance(status, str) or not status:
        return None, None
    if status == "completed":
        has_call = any(
            getattr(item, "type", None) == "function_call"
            for item in getattr(response, "output", None) or []
        )
        return ("tool_use" if has_call else "end_turn"), status
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if isinstance(reason, str) and reason:
            return _RESPONSES_INCOMPLETE_MAP.get(reason, "other"), reason
        return "other", status
    return "other", status
