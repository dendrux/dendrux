"""Shared helpers for LLM provider adapters.

Internal module — not part of the public API. Contains small, pure functions
that all three providers (Anthropic, OpenAI, OpenAI Responses) share:
  - Tool call JSON parsing (one never-raising parser for batch and stream)
  - Error factories for timeout and connection failures
  - Message history helpers (call index, tool message resolution)
  - Reasoning replay selection (which stored blocks go back on the wire)
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

from dendrux.types import REASONING_MODEL_META_KEY, ToolCall

if TYPE_CHECKING:
    from dendrux.types import Message

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Tool call JSON parsing
# ------------------------------------------------------------------


NOT_AN_OBJECT_ERROR = "arguments must be a JSON object"


def parse_tool_json(raw_json: str | dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's arguments into ``(params, parse_error)``.

    Never raises. On failure ``params`` is ``{}`` and ``parse_error`` is a
    short, model-readable description built from the decoder error (for
    example ``"Expecting ',' delimiter at char 1234"``); it never contains
    the raw text. Empty or whitespace-only input and JSON ``null`` (sent by
    some backends for argument-less calls) are ``{}`` with no error.

    Accepts an already-parsed dict as-is: some OpenAI-compatible backends
    (a few OpenRouter upstreams) return ``arguments`` as an object instead
    of a JSON string.
    """
    if isinstance(raw_json, dict):
        return raw_json, None
    if not raw_json or not raw_json.strip():
        return {}, None
    try:
        params = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return {}, f"{exc.msg} at char {exc.pos}"
    if params is None:
        return {}, None
    if not isinstance(params, dict):
        return {}, NOT_AN_OBJECT_ERROR
    return params, None


def build_tool_call(
    name: str,
    raw_json: str | dict[str, Any],
    *,
    provider_tool_call_id: str | None,
    provider: str,
    model: str,
) -> ToolCall:
    """Build a ``ToolCall`` from raw provider arguments, never raising.

    Malformed arguments yield ``params == {}`` with ``parse_error`` and
    ``raw_arguments`` set, so the loop can reject the call and let the model
    retry. The warning logged here carries the raw length, never the text.
    """
    params, parse_error = parse_tool_json(raw_json)
    if parse_error is None:
        return ToolCall(name=name, params=params, provider_tool_call_id=provider_tool_call_id)
    raw = raw_json if isinstance(raw_json, str) else ""
    logger.warning(
        "Malformed tool call JSON — provider=%s model=%s tool=%s call_id=%s raw_len=%d error=%s",
        provider,
        model,
        name,
        provider_tool_call_id,
        len(raw),
        parse_error,
    )
    return ToolCall(
        name=name,
        params={},
        provider_tool_call_id=provider_tool_call_id,
        raw_arguments=raw,
        parse_error=parse_error,
    )


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
        f"Connection to {api_name} {verb}. Model: {model}. Original error: {exc}"
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


def resolve_tool_message_call(msg: Message, call_index: dict[str, ToolCall]) -> ToolCall:
    """Resolve a TOOL message's call_id to the original ToolCall.

    Raises ValueError if call_id is missing or references an unknown call.
    """
    if msg.call_id is None:
        raise ValueError("TOOL message missing call_id — this violates Message.__post_init__")
    original_call = call_index.get(msg.call_id)
    if original_call is None:
        raise ValueError(
            f"TOOL message references call_id '{msg.call_id}' "
            f"but no matching ToolCall found in conversation history."
        )
    return original_call


def replayable_reasoning_blocks(
    msg: Message,
    *,
    model: str | None,
    block_types: frozenset[str],
) -> list[dict[str, Any]]:
    """Return deep copies of ``msg.reasoning_blocks`` safe to send to ``model``.

    Empty when replay is disabled (``model`` is None), when the blocks were
    produced by a different model (per ``REASONING_MODEL_META_KEY``), or for
    blocks whose ``type`` this provider does not accept. Unmarked blocks are
    trusted, so developer-supplied history still replays.
    """
    if model is None or not msg.reasoning_blocks:
        return []
    producer = msg.meta.get(REASONING_MODEL_META_KEY)
    if producer is not None and producer != model:
        return []
    return [
        copy.deepcopy(block)
        for block in msg.reasoning_blocks
        if isinstance(block, dict) and block.get("type") in block_types
    ]
