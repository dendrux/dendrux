"""Shared helpers for LLM provider adapters.

Internal module — not part of the public API. Contains small, pure functions
that all three providers (Anthropic, OpenAI, OpenAI Responses) share:
  - Tool call JSON parsing (lossy for streams, strict for batch)
  - Error factories for timeout and connection failures
  - Message history helpers (call index, tool message resolution)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.types import Message, ToolCall

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Tool call JSON parsing
# ------------------------------------------------------------------


def parse_tool_json_lossy(
    raw_json: str,
    *,
    provider: str,
    model: str,
    tool_name: str,
    call_id: str,
) -> dict[str, Any]:
    """Parse tool call JSON from a stream, falling back to ``{}`` on failure.

    Used in streaming paths where a malformed tool call should be logged
    and degraded gracefully rather than crashing the stream.
    """
    if not raw_json:
        return {}
    try:
        params = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning(
            "Malformed tool call JSON in stream — "
            "provider=%s model=%s tool=%s call_id=%s raw_len=%d",
            provider,
            model,
            tool_name,
            call_id,
            len(raw_json),
        )
        return {}
    if not isinstance(params, dict):
        return {}
    return params


def parse_tool_json_strict(
    raw_json: str,
    *,
    tool_name: str,
    call_id: str,
) -> dict[str, Any]:
    """Parse tool call JSON from a batch response, raising on failure.

    Used in batch (non-streaming) normalization where invalid JSON
    indicates a provider-side issue that should surface immediately.
    """
    if not raw_json:
        return {}
    try:
        params = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Tool call '{tool_name}' (id={call_id}) returned "
            f"invalid JSON arguments: {raw_json!r}"
        ) from exc
    if not isinstance(params, dict):
        return {}
    return params


# ------------------------------------------------------------------
# Error factories
# ------------------------------------------------------------------


def timeout_error(provider_class: str, timeout: float) -> TimeoutError:
    """Build a TimeoutError with a helpful message pointing to the fix."""
    return TimeoutError(
        f"LLM request timed out after {timeout}s. "
        f"The model may need more time for large outputs. "
        f"Increase timeout: {provider_class}(model=..., timeout=300)"
    )


def connection_error(
    api_name: str,
    model: str,
    exc: Exception,
    *,
    streaming: bool = False,
) -> ConnectionError:
    """Build a ConnectionError with model context."""
    verb = "failed during streaming" if streaming else "failed"
    return ConnectionError(
        f"Connection to {api_name} {verb}. "
        f"Model: {model}. Original error: {exc}"
    )


# ------------------------------------------------------------------
# Message history helpers
# ------------------------------------------------------------------


def build_call_index(messages: list[Message]) -> dict[str, ToolCall]:
    """Build a Dendrux call_id -> ToolCall index from message history.

    Used by all providers to correlate TOOL messages with the ASSISTANT
    tool calls that triggered them. Raises ValueError on duplicate IDs.
    """
    call_index: dict[str, ToolCall] = {}
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id in call_index:
                    raise ValueError(
                        f"Duplicate Dendrux call_id '{tc.id}' in conversation "
                        f"history. Tool calls must have unique IDs."
                    )
                call_index[tc.id] = tc
    return call_index


def resolve_tool_message_call(
    msg: Message, call_index: dict[str, ToolCall]
) -> ToolCall:
    """Resolve a TOOL message's call_id to the original ToolCall.

    Raises ValueError if call_id is missing or references an unknown call.
    """
    if msg.call_id is None:
        raise ValueError(
            "TOOL message missing call_id — this violates Message.__post_init__"
        )
    original_call = call_index.get(msg.call_id)
    if original_call is None:
        raise ValueError(
            f"TOOL message references call_id '{msg.call_id}' "
            f"but no matching ToolCall found in conversation history."
        )
    return original_call
