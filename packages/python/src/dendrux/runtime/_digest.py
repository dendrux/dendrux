"""Canonical content digests for durable retry identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_digest(payload: Any) -> str:
    """Hash client-supplied fields, preserving JSON types and Unicode."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def trace_digest(
    role: str, content: str, order_index: int, trace_kind: str, meta: dict[str, Any] | None
) -> str:
    """Digest the conversation write, excluding database identity and timestamps."""
    return content_digest(
        dict(role=role, content=content, order_index=order_index, trace_kind=trace_kind, meta=meta)
    )


def event_digest(
    event_type: str, iteration_index: int, correlation_id: str | None, data: dict[str, Any] | None
) -> str:
    """Digest an event before its sequence number is allocated."""
    return content_digest(
        dict(
            event_type=event_type,
            iteration_index=iteration_index,
            correlation_id=correlation_id,
            data=data,
        )
    )
