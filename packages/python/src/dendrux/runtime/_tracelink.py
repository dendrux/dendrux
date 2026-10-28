"""Validate W3C version-00 traceparents at the persistence boundary."""

from __future__ import annotations

import re

_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})")


def validate_trace_link(value: object) -> str | None:
    """Return a valid, non-zero traceparent, otherwise discard it without echoing it."""
    if not isinstance(value, str):
        return None
    match = _TRACEPARENT.fullmatch(value)
    if match is None or int(match[1], 16) == 0 or int(match[2], 16) == 0:
        return None
    return value
