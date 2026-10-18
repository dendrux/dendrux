"""MCP tool-result normalization and output boundaries."""

from __future__ import annotations

import json
from typing import Any, Literal

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


def _is_image(block: Any) -> bool:
    return getattr(block, "type", None) == "image"


def _mime_type(block: Any) -> str | None:
    value = getattr(block, "mimeType", None)
    if value is None:
        value = getattr(block, "mime_type", None)
    return value if isinstance(value, str) else None


def _image_input(block: Any) -> Any:
    """``Image.from_base64`` for one MCP image block; never raises (ingest judges it)."""
    from dendrux.media._image import Image

    data = getattr(block, "data", None)
    text = data if isinstance(data, str) else ""
    try:
        return Image.from_base64(text, media_type=_mime_type(block))
    except (TypeError, ValueError):
        return Image.from_base64(text)


def normalize_mcp_result(
    result: Any,
    *,
    max_result_bytes: int = 1_000_000,
    images: Literal["parts", "inline"] = "inline",
) -> Any:
    """Return a JSON-serializable result while preserving MCP content blocks.

    Structured content remains the preferred application value. Text-only
    results retain the legacy string shape. Mixed or non-text results are
    returned as a content envelope rather than being silently discarded.

    ``images="inline"`` (default; ``MCPConnection.call_tool``) keeps image
    blocks, base64 included, inside the envelope exactly as before.
    ``images="parts"`` (Agent runs) returns a ``ToolOutput`` when the result
    has image blocks: each block becomes an image input and is replaced in the
    envelope by ``{"type": "image", "image_index": i, "mime_type": ...}``;
    structured content keeps its value and gains the images. The size limit
    applies to the text value only.
    """
    # Pydantic v2 models store the snake_case field in ``__dict__``. The
    # camelCase fallback keeps the former private helper testable with simple
    # protocol-shaped objects without letting MagicMock fabricate a value.
    values = getattr(result, "__dict__", {})
    structured = values.get("structured_content", values.get("structuredContent"))
    content = list(getattr(result, "content", []) or [])
    if images == "parts" and any(_is_image(block) for block in content):
        return _normalize_with_parts(result, structured, content, max_result_bytes)

    if structured is not None:
        _enforce_size(structured, max_result_bytes)
        return structured

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


def _normalize_with_parts(
    result: Any, structured: Any, content: list[Any], max_result_bytes: int
) -> Any:
    from dendrux.media._image import ToolOutput

    image_inputs = [_image_input(block) for block in content if _is_image(block)]
    if structured is not None:
        _enforce_size(structured, max_result_bytes)
        return ToolOutput(structured, images=image_inputs)
    blocks: list[Any] = []
    index = 0
    for block in content:
        if _is_image(block):
            blocks.append({"type": "image", "image_index": index, "mime_type": _mime_type(block)})
            index += 1
        else:
            blocks.append(_model_dump(block))
    envelope: dict[str, Any] = {"content": blocks}
    meta = getattr(result, "meta", None)
    if meta:
        envelope["_meta"] = meta
    _enforce_size(envelope, max_result_bytes)
    return ToolOutput(envelope, images=image_inputs)


def error_text(result: Any) -> str:
    """Message for an ``is_error`` result with images: text blocks only.

    Each image block renders as ``"[image omitted]"``; other blocks are skipped,
    so neither base64 nor a ``ToolOutput`` repr can reach the error.
    """
    lines: list[str] = []
    for block in list(getattr(result, "content", []) or []):
        kind = getattr(block, "type", None)
        if kind == "text":
            lines.append(str(getattr(block, "text", "")))
        elif kind == "image":
            lines.append("[image omitted]")
    return "\n".join(lines)


def has_image_blocks(result: Any) -> bool:
    """Whether an MCP result carries image content blocks."""
    return any(_is_image(block) for block in list(getattr(result, "content", []) or []))
