"""Transport helpers for SSE bridge hardening.

Provides _offer_sse_event() — the single chokepoint for all event
enqueuing. Handles payload truncation, size validation, and overflow
detection. All producers (TransportNotifier, _resume_task, cancel_run)
use this instead of direct queue.put().
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from dendrux.bridge.notifier import ServerEvent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MAX_QUEUE_SIZE = 1000
"""Per-run queue bound. Overflow triggers client disconnect."""

MAX_EVENT_BYTES = 65_536
"""Maximum JSON data payload size in bytes (64KB)."""

MAX_STRING_TRUNCATE = 2000
"""Max length for individual string values during recursive truncation."""

MAX_CONNECTIONS_TOTAL = 100
"""Global SSE connection limit. Reject with 503 when reached."""


# ------------------------------------------------------------------
# Payload truncation
# ------------------------------------------------------------------


def _truncate_value(value: Any, *, max_str: int = MAX_STRING_TRUNCATE) -> Any:
    """Recursively truncate string values in dicts/lists."""
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + "..."
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_str=max_str) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, max_str=max_str) for item in value]
    return value


def _enforce_payload_limit(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure serialized data fits within MAX_EVENT_BYTES.

    1. Try original data — if it fits, return as-is.
    2. Recursively truncate strings — if it fits, return with truncated flag.
    3. Replace with minimal fallback.
    """
    serialized = json.dumps(data, default=str)
    if len(serialized.encode()) <= MAX_EVENT_BYTES:
        return data

    # Truncate recursively
    truncated = _truncate_value(data)
    if not isinstance(truncated, dict):
        truncated = {"value": truncated}
    truncated["truncated"] = True

    serialized = json.dumps(truncated, default=str)
    if len(serialized.encode()) <= MAX_EVENT_BYTES:
        return truncated

    # Still too large — minimal fallback
    return {"truncated": True, "reason": "payload_too_large"}


# ------------------------------------------------------------------
# Offer helper
# ------------------------------------------------------------------


def _offer_sse_event(
    run_id: str,
    event: ServerEvent,
    queue: asyncio.Queue[ServerEvent],
    overflow_flags: dict[str, bool],
) -> None:
    """Non-blocking event offer — single chokepoint for all enqueuing.

    1. Enforce payload size limit (truncate if needed).
    2. put_nowait() into queue.
    3. If QueueFull: set overflow flag for the run_id.
       SSE generator checks the flag and closes the stream.
    """
    # Enforce payload limit
    event = ServerEvent(event=event.event, data=_enforce_payload_limit(event.data))

    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        overflow_flags[run_id] = True
        logger.warning(
            "SSE queue full for run %s — overflow flagged, client will be disconnected",
            run_id,
        )
