"""Allowlisted browser result output and bounded untrusted tool-result input."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.types import Limits, RunResult, ToolResult

WIRE_VERSION = 1
_PAUSED = {"paused", "waiting_approval", "waiting_human_input", "waiting_client_tool"}


def _redacted(value: Any, mapping: Any) -> Any:
    """Copy only bounded JSON data, replacing known PII with placeholders."""
    replacements = {}
    if isinstance(mapping, dict):
        if len(mapping) > 1000:
            return None
        for key, real in mapping.items():
            if isinstance(real, str) and real and isinstance(key, str):
                replacements[real] = key if re.fullmatch(r"<<[A-Z0-9_:-]+>>", key) else "[redacted]"
    parts: list[str] = []
    pattern_size = 0
    for real in sorted(replacements, key=len, reverse=True):
        if len(real) > 65536:
            return None
        escaped = re.escape(real)
        pattern_size += len(escaped) + bool(parts)
        if pattern_size > 65536:
            return None
        parts.append(escaped)
    pattern_text = "|".join(parts)
    pattern = re.compile(pattern_text) if pattern_text else None
    nodes = size = 0

    def copy(item: Any, depth: int = 0) -> Any:
        nonlocal nodes, size
        nodes += 1
        if depth > 32 or nodes > 10000 or size > 1048576:
            raise ValueError("Wire value exceeds its display bound")
        if isinstance(item, str):
            if len(item) > 1048576:
                raise ValueError("Wire text exceeds its display bound")
            size += len(item.encode("utf-8"))
            if size > 1048576:
                raise ValueError("Wire text exceeds its display bound")
            return pattern.sub(lambda m: replacements[m.group(0)], item) if pattern else item
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            # Below Python's default JSON integer conversion guard; also count
            # digits against the aggregate display budget before copying it.
            if item.bit_length() > 14000:
                raise ValueError("Wire integer exceeds its display bound")
            size += len(str(item))
            if size > 1048576:
                raise ValueError("Wire value exceeds its display bound")
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {
                copy(k, depth + 1): copy(v, depth + 1)
                for k, v in item.items()
                if isinstance(k, str)
            }
        if isinstance(item, (list, tuple)):
            return [copy(v, depth + 1) for v in item]
        return None  # never call repr/str/model serializers on arbitrary objects

    try:
        return copy(value)
    except (ValueError, UnicodeError, RecursionError):
        return None


def result_to_dict(result: RunResult) -> dict[str, Any]:
    """Browser DTO, deliberately independent of dataclass/package versions."""
    mapping = result.meta.get("pii_mapping")
    pending = result.pending if result.status.value in _PAUSED else None
    return {
        "wire_version": WIRE_VERSION,
        "run_id": result.run_id,
        "status": result.status.value,
        "pending": _redacted(pending.to_dict(), mapping) if pending else None,
        "answer": _redacted(result.answer, mapping) if isinstance(result.answer, str) else None,
        # App output is unrestricted and may contain tool payloads or private objects.
        "output": None,
        "error_info": _redacted(result.error_info.to_dict(), mapping)
        if result.error_info
        else None,
    }


def _invalid_constant(value: str) -> Any:
    raise ValueError("Tool result payload must be finite JSON")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Tool result payload must be finite JSON")
    return parsed


def tool_result_from_dict(data: Any, limits: Limits) -> ToolResult:
    from dendrux.types import ImageRef, ToolResult

    allowed = {"name", "call_id", "payload", "success", "error", "duration_ms", "images"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Tool result must be an object with only supported fields")
    for key in ("name", "call_id", "payload"):
        if not isinstance(data.get(key), str) or (key != "payload" and not data[key].strip()):
            raise ValueError(f"Tool result {key} must be a nonempty string")
    success, error, duration = (
        data.get("success", True),
        data.get("error"),
        data.get("duration_ms", 0),
    )
    if not isinstance(success, bool) or (error is not None and not isinstance(error, str)):
        raise ValueError("Tool result success/error have invalid types")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
        raise ValueError("Tool result duration_ms must be a nonnegative integer")
    images = data.get("images", [])
    if not isinstance(images, list) or len(images) > 100:
        raise ValueError("Tool result images must be a bounded descriptor list")
    # Metadata cannot be used to bypass server-owned annotations or size caps.
    metadata = {k: v for k, v in data.items() if k != "payload"}
    try:
        if len(json.dumps(metadata, allow_nan=False).encode()) > limits.max_metadata_bytes:
            raise ValueError("Tool result metadata exceeds its cap")
        size = len(data["payload"].encode("utf-8"))
        if size > limits.max_tool_result_bytes:
            return ToolResult(
                data["name"],
                data["call_id"],
                json.dumps({"error": "result too large", "code": "size_limit", "bytes": size}),
                success=False,
                error="result too large",
                meta={"code": "size_limit", "bytes": size},
            )
        json.loads(data["payload"], parse_constant=_invalid_constant, parse_float=_finite_float)
        refs = tuple(ImageRef.from_dict(image) for image in images)
    except (TypeError, UnicodeError, RecursionError, KeyError) as exc:
        raise ValueError("Tool result must contain valid JSON and image descriptors") from exc
    return ToolResult(
        data["name"],
        data["call_id"],
        data["payload"],
        success=success,
        error=error,
        duration_ms=duration,
        images=refs,
    )


def tool_results_from_dict(data: Any, limits: Limits, max_results: int) -> list[ToolResult]:
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 10000
    ):
        raise ValueError("max_results must be between 1 and 10000")
    if not isinstance(data, list) or len(data) > max_results:
        raise ValueError("Tool results must be a bounded list")
    results = []
    seen = set()
    total = 0
    for item in data:
        result = tool_result_from_dict(item, limits)
        if result.call_id in seen:
            raise ValueError("Tool result call IDs must be unique")
        seen.add(result.call_id)
        total += len(result.payload.encode())
        if total > limits.max_input_bytes:
            raise ValueError("Tool result batch exceeds max_input_bytes")
        results.append(result)
    return results
