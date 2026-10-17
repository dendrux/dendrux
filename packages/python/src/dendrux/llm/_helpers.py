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
import re
from dataclasses import dataclass
from enum import Enum
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
# Context-length and stream-rejection detection (OpenAI-compatible wire)
# ------------------------------------------------------------------

# Provider error codes (lower-cased) that mean "request exceeds the context window".
CONTEXT_LENGTH_CODES: frozenset[str] = frozenset(
    {"context_length_exceeded", "context_length_exceeded_error"}
)

# Statuses an endpoint uses for an oversized request. 403 covers Together's
# documented-but-odd labelling; 413/422 cover gateways and FastAPI/TGI servers.
# An in-stream error event has no status (None) and is judged on code/text alone.
CONTEXT_LENGTH_STATUSES: frozenset[int] = frozenset({400, 403, 413, 422})

# Provider codes that name a different failure. They veto the message-pattern
# path (an explicit context-length code still wins first), so a rate limit or
# auth error whose text mentions tokens or a context window is never an overflow.
NON_CONTEXT_CODES: frozenset[str] = frozenset(
    {
        "rate_limit_exceeded",
        "rate_limit",
        "rate_limited",
        "too_many_requests",
        "insufficient_quota",
        "quota_exceeded",
        "billing_hard_limit_reached",
        "permission_denied",
        "forbidden",
        "unauthorized",
        "authentication_error",
        "invalid_api_key",
        "access_denied",
        "model_not_found",
        "model_not_available",
        "invalid_value",
    }
)

# ``param`` values that name the output-token cap. An error about the cap is a
# request-configuration error unless its text also counts input/prompt tokens
# (vLLM: "... maximum context length is N tokens and your request has M input tokens").
OUTPUT_CAP_PARAMS: frozenset[str] = frozenset(
    {"max_tokens", "max_completion_tokens", "max_new_tokens", "max_output_tokens"}
)
OUTPUT_CAP_INPUT_EVIDENCE = re.compile(
    r"\binput tokens?\b|\bin the (messages?|prompt)\b|\byour (messages?|prompt)\b|"
    r"\bprompt tokens?\b",
    re.IGNORECASE,
)

# On 403 the generic phrases are too weak (an auth error can describe a model's
# context length). Only Together's documented formula is accepted there.
FORBIDDEN_CONTEXT_PATTERN = re.compile(
    r"input token count\s*\+\s*`?max_(new_)?tokens`?.{0,80}\bcontext[ _-]?length",
    re.IGNORECASE,
)

# Streaming-unsupported is only ever a plain 400.
STREAM_REJECTED_STATUSES: frozenset[int] = frozenset({400})

# Server-templated phrases (OpenAI, vLLM, SGLang, Ollama, Groq, Together,
# Fireworks, DeepInfra, TGI), matched case-insensitively. Loose by design but
# every pattern is about the context window or input length, never about rate
# limits (a Groq "request too large ... tokens per minute" 413 is deliberately
# not matched). Never pin exact provider wording beyond these fragments.
CONTEXT_LENGTH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"maximum context length",
        r"context[ _-]?(length|window|size)\b.{0,40}\b(exceed|too (long|large)|limit|only)",
        r"\b(exceed\w*|longer than|over|above)\b.{0,40}\bcontext[ _-]?(length|window|size)",
        r"\binput (length|token count|tokens?)\b.{0,60}\b(exceed\w*|too (long|large)|must be)",
        r"\b(prompt|input|messages?)\s{1,3}(is|are) too (long|large)",
        r"reduce the length of (the |your )?(messages?|prompt|input)",
        r"`?inputs`? (tokens )?\+ `?max_(new_)?tokens`?",
        r"`?inputs`? must have less than \d+ tokens",
        r"\btoken length too long\b",
        r"context overflow",
    )
)

# The error's ``param`` naming the stream flag is itself a rejection signal.
STREAM_REJECTED_PARAMS: frozenset[str] = frozenset({"stream"})

STREAM_REJECTED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bstream(ing)?\b.{0,60}\b(not (be )?(supported|allowed|available|enabled|implemented)"
        r"|unsupported|isn'?t supported|disabled)",
        r"\b(does ?n[o']t|do not|cannot|can'?t|not) (support|allow|stream)\w*\b.{0,40}\bstream",
        r"\bunsupported (value|parameter)\b.{0,30}\bstream\b",
    )
)

_ERROR_TEXT_LIMIT = 2000


class GatewayErrorKind(Enum):
    """Outcome of classifying an OpenAI-compatible endpoint error."""

    CONTEXT_LENGTH = "context_length"
    STREAMING_UNSUPPORTED = "streaming_unsupported"
    OTHER = "other"


@dataclass(frozen=True)
class GatewayErrorClass:
    """Classification of one endpoint error, with normalized status and provider code."""

    kind: GatewayErrorKind
    status_code: int | None
    provider_code: str | None


def _error_text(message: str, body: object) -> str:
    """Searchable error text: the parsed body fields, else the SDK message.

    The SDK message for a JSON error is a repr of the whole body, so it is only
    used when no readable field could be extracted.
    """
    parts: list[str] = []
    if isinstance(body, dict):
        for key in ("message", "detail", "error"):
            value = body.get(key)
            if value is not None:
                parts.append(value if isinstance(value, str) else str(value))
    elif isinstance(body, str) and body.strip():
        parts.append(body)
    if not parts:
        parts.append(message)
    return "\n".join(parts)[:_ERROR_TEXT_LIMIT]


def _body_field(body: object, key: str) -> str | None:
    if isinstance(body, dict):
        value = body.get(key)
        if value is not None:
            return str(value)
    return None


def _body_provider_code(body: object, status_code: int | None) -> str | None:
    """Body ``code`` as a string, except a number repeating the HTTP status.

    Some gateways (OpenRouter) send the HTTP status as a numeric ``code``;
    that is not a provider code. A string such as ``"400"`` is kept.
    """
    if isinstance(body, dict):
        value = body.get("code")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return None if value == status_code else str(value)
    return _body_field(body, "code")


def _is_context_length(
    status_code: int | None, code: str | None, param: str | None, text: str
) -> bool:
    if code in CONTEXT_LENGTH_CODES:
        return True
    if code in NON_CONTEXT_CODES:
        return False
    if status_code == 403:
        return FORBIDDEN_CONTEXT_PATTERN.search(text) is not None
    if not any(p.search(text) for p in CONTEXT_LENGTH_PATTERNS):
        return False
    if param in OUTPUT_CAP_PARAMS:
        return OUTPUT_CAP_INPUT_EVIDENCE.search(text) is not None
    return True


def classify_gateway_error(
    *,
    status_code: int | None,
    code: str | None,
    param: str | None,
    message: str,
    body: object = None,
) -> GatewayErrorClass:
    """Classify an endpoint error. Pure; the single decision point for all call sites.

    Precedence is fixed: context-length overflow first, streaming-unsupported
    second, otherwise ``OTHER`` (the caller re-raises the original exception
    untouched). Context length matches an accepted provider code or message
    pattern on an accepted status (see the constants above), unless a
    non-context provider code or an output-cap ``param`` vetoes the text path
    (403 accepts only Together's formula); streaming
    rejection requires a 400 whose ``param`` is ``stream`` or whose text says
    streaming is unsupported.
    """
    provider_code = code or _body_provider_code(body, status_code)
    text = _error_text(message, body)

    lowered_code = provider_code.lower() if provider_code is not None else None
    if (status_code is None or status_code in CONTEXT_LENGTH_STATUSES) and _is_context_length(
        status_code, lowered_code, param, text
    ):
        return GatewayErrorClass(GatewayErrorKind.CONTEXT_LENGTH, status_code, provider_code)

    if status_code in STREAM_REJECTED_STATUSES and (
        param in STREAM_REJECTED_PARAMS or any(p.search(text) for p in STREAM_REJECTED_PATTERNS)
    ):
        return GatewayErrorClass(GatewayErrorKind.STREAMING_UNSUPPORTED, status_code, provider_code)

    return GatewayErrorClass(GatewayErrorKind.OTHER, status_code, provider_code)


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
