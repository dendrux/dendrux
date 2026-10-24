"""Per-run call limits and tool-result size caps.

``enforce_call_limits`` is the one ``max_calls_per_run`` check: the loop runs
it before ``before_tool`` and the approval pause, and the approval-resume path
runs it again before executing approved calls.

``cap_tool_result`` / ``cap_value_payload`` apply
``Limits.max_tool_result_bytes`` to local, MCP and client results. An
oversized string result is truncated with a marker; anything else is replaced
by ``{"error": "result too large", "code": "size_limit", "bytes": N}``. Either
way the payload stays valid JSON.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from dendrux.types import ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

SIZE_LIMIT_CODE = "size_limit"
MAX_CALLS_CODE = "max_calls"

_TRUNCATION_MARKER = "\n[truncated by Dendrux: the result was {size} bytes; the limit is {limit}]"

# The result cap of the tool call running in this context. The MCP HTTP
# layer reads it to stop reading an oversized response early.
_result_cap: ContextVar[int | None] = ContextVar("dendrux_tool_result_cap", default=None)
# Wire bytes allowed for image blocks in the running call's result: zero unless
# the run can ingest images (see ``result_cap``).
_image_allowance: ContextVar[int] = ContextVar("dendrux_tool_image_allowance", default=0)


class ToolResultTooLargeError(Exception):
    """A tool's result passed its size cap before it could be serialised.

    Raised by tool sources that enforce the cap while reading (MCP). The
    loop turns it into the ``size_limit`` result. ``size`` is the byte count
    seen when the cap tripped, if known.
    """

    size: int | None = None


def current_result_cap() -> int | None:
    """``max_tool_result_bytes`` for the tool call running in this context, if any."""
    return _result_cap.get()


def current_image_allowance() -> int:
    """Extra wire bytes for image blocks in the running tool call (0: no images)."""
    return _image_allowance.get()


@contextmanager
def result_cap(limit: int | None, *, image_bytes: int = 0) -> Iterator[None]:
    """Make ``limit`` (and an image allowance) current while a tool runs."""
    token = _result_cap.set(limit)
    image_token = _image_allowance.set(image_bytes)
    try:
        yield
    finally:
        _image_allowance.reset(image_token)
        _result_cap.reset(token)


def limit_result(tool_call: ToolCall, limit: int) -> ToolResult:
    """The failed result for a call over its ``max_calls_per_run``."""
    message = f"Tool '{tool_call.name}' has reached its maximum of {limit} calls for this run."
    return ToolResult(
        name=tool_call.name,
        call_id=tool_call.id,
        payload=json.dumps({"limit": message}),
        success=False,
        error=message,
        meta={"code": MAX_CALLS_CODE},
    )


def enforce_call_limits(
    calls: Sequence[ToolCall],
    *,
    max_calls: Mapping[str, int | None],
    call_counts: Mapping[str, int],
) -> tuple[list[ToolCall], list[tuple[ToolCall, ToolResult]]]:
    """Split ``calls`` into those within ``max_calls_per_run`` and rejections.

    ``call_counts`` holds the calls already made this run (it is not
    mutated). Calls earlier in the same batch count against later ones, in
    order, so a batch of three calls to a tool limited to one lets only the
    first through. Rejections carry their failed ``ToolResult``.
    """
    allowed: list[ToolCall] = []
    rejected: list[tuple[ToolCall, ToolResult]] = []
    in_batch: dict[str, int] = {}
    for tc in calls:
        limit = max_calls.get(tc.name)
        used = call_counts.get(tc.name, 0) + in_batch.get(tc.name, 0)
        if limit is not None and used >= limit:
            rejected.append((tc, limit_result(tc, limit)))
            continue
        in_batch[tc.name] = in_batch.get(tc.name, 0) + 1
        allowed.append(tc)
    return allowed, rejected


def _encoded_size(payload: str) -> int:
    return len(payload.encode("utf-8"))


def size_limit_payload(size: int | None) -> tuple[str, dict[str, Any]]:
    """The ``size_limit`` replacement payload and meta (``bytes`` when known)."""
    body: dict[str, Any] = {"error": "result too large", "code": SIZE_LIMIT_CODE}
    meta: dict[str, Any] = {"code": SIZE_LIMIT_CODE}
    if size is not None:
        body["bytes"] = size
        meta["bytes"] = size
    return json.dumps(body), meta


def _truncate(text: str, size: int, limit: int) -> str | None:
    """The longest prefix of ``text`` whose JSON payload, with the marker, fits ``limit``."""
    marker = _TRUNCATION_MARKER.format(size=size, limit=limit)
    if _encoded_size(json.dumps(marker)) > limit:
        return None
    # Every character costs at least one byte, so no prefix longer than
    # ``limit`` characters fits: the search is bounded by log2(limit) dumps
    # of at most ``limit`` characters, whatever the original size.
    low, high = 0, min(len(text), limit)
    best: str | None = None
    while low <= high:
        mid = (low + high) // 2
        candidate = json.dumps(text[:mid] + marker)
        if _encoded_size(candidate) <= limit:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def cap_value_payload(value: Any, payload: str, limit: int) -> tuple[str, dict[str, Any] | None]:
    """Apply the cap to a serialised tool value; returns ``(payload, meta)``.

    ``meta`` is None when the payload fits. A ``str`` value is truncated with
    a marker (meta ``truncated=True``); any other value is replaced by the
    ``size_limit`` error payload.
    """
    size = _encoded_size(payload)
    if size <= limit:
        return payload, None
    if isinstance(value, str):
        truncated = _truncate(value, size, limit)
        if truncated is not None:
            return truncated, {"code": SIZE_LIMIT_CODE, "bytes": size, "truncated": True}
    return size_limit_payload(size)


def cap_tool_result(result: ToolResult, limit: int) -> ToolResult:
    """Apply ``Limits.max_tool_result_bytes`` to a finished ``ToolResult``.

    Used for results that arrive already serialised (client tool results).
    A JSON string payload is truncated with a marker; anything else over the
    cap, including a payload that is not valid JSON, is replaced by the
    ``size_limit`` error and the result is marked failed.
    """
    size = _encoded_size(result.payload)
    if size <= limit:
        return result
    try:
        value = json.loads(result.payload)
    except ValueError:
        value = None
    payload, meta = cap_value_payload(value, result.payload, limit)
    assert meta is not None
    merged = {**result.meta, **meta}
    if meta.get("truncated"):
        return replace(result, payload=payload, meta=merged)
    return replace(result, payload=payload, success=False, error="result too large", meta=merged)
