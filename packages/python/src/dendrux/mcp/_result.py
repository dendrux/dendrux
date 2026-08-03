"""MCP tool-result normalization and output boundaries."""

from __future__ import annotations

import json
from typing import Any

from dendrux.mcp._errors import MCPResultTooLargeError


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    data: dict[str, Any] = {"type": getattr(value, "type", type(value).__name__)}
    for field_name in ("text", "data", "mime_type", "uri", "name", "description", "size"):
        field_value = getattr(value, field_name, None)
        if field_value is not None:
            data[field_name] = field_value
    return data


def _enforce_size(value: Any, max_result_bytes: int) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > max_result_bytes:
        raise MCPResultTooLargeError(
            f"MCP tool result is {len(encoded)} bytes; configured limit is "
            f"{max_result_bytes} bytes."
        )


def normalize_mcp_result(result: Any, *, max_result_bytes: int = 1_000_000) -> Any:
    """Return a JSON-serializable result while preserving MCP content blocks.

    Structured content remains the preferred application value. Text-only
    results retain the legacy string shape. Mixed or non-text results are
    returned as a content envelope rather than being silently discarded.
    """
    # Pydantic v2 models store the snake_case field in ``__dict__``. The
    # camelCase fallback keeps the former private helper testable with simple
    # protocol-shaped objects without letting MagicMock fabricate a value.
    values = getattr(result, "__dict__", {})
    structured = values.get("structured_content", values.get("structuredContent"))
    if structured is not None:
        _enforce_size(structured, max_result_bytes)
        return structured

    content = list(getattr(result, "content", []))
    if all(getattr(block, "type", None) == "text" for block in content):
        value = "\n".join(str(block.text) for block in content)
        _enforce_size(value, max_result_bytes)
        return value

    envelope: dict[str, Any] = {"content": [_model_dump(block) for block in content]}
    meta = getattr(result, "meta", None)
    if meta:
        envelope["_meta"] = meta
    _enforce_size(envelope, max_result_bytes)
    return envelope
