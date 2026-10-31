"""Closed provider-code projection; classification always happens before this boundary."""

from __future__ import annotations

from types import MappingProxyType

from dendrux import errors

# These are package-issued codes, independent of the vendor identity.
SELF_RAISED_CODES = frozenset(
    {
        "empty_completion",
        "stream_size_limit",
        "stream_tool_call_limit",
        "response_size_limit",
        "response_encoding_unsupported",
        "content_filter",
        "call_deadline",
        "stream_idle",
        "stream_total",
        "deadline_during_backoff",
    }
)

_ANTHROPIC = {
    code: code
    for code in (
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
        "timeout_error",
    )
}
_OPENAI = {
    code: code
    for code in (
        "invalid_api_key",
        "insufficient_quota",
        "rate_limit_exceeded",
        "context_length_exceeded",
        "server_error",
        "invalid_request_error",
        "content_policy_violation",
        "invalid_image",
        "invalid_image_format",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "unsupported_image",
        "invalid_base64_image",
        "invalid_prompt",
        "vector_store_timeout",
        "invalid_value",
        "model_not_found",
        "permission_denied",
        "billing_hard_limit_reached",
        "organization_usage_limit_exceeded",
    )
}
_OPENROUTER = {
    **_OPENAI,
    "400": "invalid_request_error",
    "401": "invalid_api_key",
    "402": "insufficient_credits",
    "403": "content_policy_violation",
    "408": "provider_timeout",
    "429": "rate_limit_exceeded",
    "500": "server_error",
    "502": "server_error",
    "503": "service_unavailable",
    "504": "provider_timeout",
    "insufficient_credits": "insufficient_credits",
    "service_unavailable": "service_unavailable",
    "provider_timeout": "provider_timeout",
    "token_limit_exceeded": "token_limit_exceeded",
}

PROVIDER_CODES = MappingProxyType(
    {
        "anthropic": MappingProxyType(_ANTHROPIC),
        "openai": MappingProxyType(_OPENAI),
        "openai_responses": MappingProxyType(_OPENAI),
        "openrouter": MappingProxyType(_OPENROUTER),
    }
)


def map_provider_code(provider: str | None, raw: str | None) -> str | None:
    """Project only declared vendor or package codes, never arbitrary vendor text."""
    if not isinstance(raw, str) or errors._SAFE_PROVIDER_CODE.fullmatch(raw) is None:
        return None
    if raw in SELF_RAISED_CODES or raw == "provider_unknown":
        return raw
    if provider is None:
        return "provider_unknown"
    table = PROVIDER_CODES.get(provider)
    return table.get(raw, "provider_unknown") if table is not None else "provider_unknown"


errors._provider_code_mapper = map_provider_code
