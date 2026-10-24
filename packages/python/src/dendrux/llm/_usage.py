"""OpenAI-wire usage normalisation (Chat Completions, Responses, OpenRouter).

``input_tokens`` in :class:`~dendrux.types.UsageStats` always means fresh
input: the vendor's prompt count minus cache reads and cache writes. A field
the vendor did not report stays ``None``; a reported zero stays ``0``.
"""

from __future__ import annotations

import logging
from typing import Any

from dendrux.types import UsageStats

logger = logging.getLogger(__name__)


def extra_field(obj: Any, name: str) -> Any:
    """A usage field that may be typed, untyped (``model_extra``) or a dict key."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def _has_long_ttl_writes(details: Any) -> bool:
    """True when any ``ttl_12h_tokens`` under the input details is above zero.

    Long-TTL cache-write rates are not in :class:`~dendrux.pricing.PriceTable`
    until 0.6, so such a call is marked ``cost_unknown``. Read defensively:
    the field may sit on the details or one level down.
    """
    if details is None:
        return False
    if (_count(extra_field(details, "ttl_12h_tokens")) or 0) > 0:
        return True
    children: list[Any] = []
    if isinstance(details, dict):
        children = list(details.values())
    else:
        dumped = getattr(details, "model_dump", None)
        if callable(dumped):
            try:
                children = list(dumped().values())
            except Exception:
                children = []
    return any(
        isinstance(child, dict) and (_count(child.get("ttl_12h_tokens")) or 0) > 0
        for child in children
    )


def openai_wire_usage(
    usage: Any,
    *,
    input_field: str,
    output_field: str,
    details_field: str,
    output_details_field: str,
) -> UsageStats:
    """UsageStats from a Chat (``prompt_tokens``) or Responses (``input_tokens``) usage.

    Cache reads come from ``<details>.cached_tokens`` and cache writes from
    ``<details>.cache_write_tokens`` (typed or not). Fresh input is the
    vendor count minus both, clamped at zero.
    """
    raw_input = _count(extra_field(usage, input_field)) or 0
    output_tokens = _count(extra_field(usage, output_field)) or 0
    details = extra_field(usage, details_field)
    cache_read = _count(extra_field(details, "cached_tokens"))
    cache_write = _count(extra_field(details, "cache_write_tokens"))
    reasoning = _count(extra_field(extra_field(usage, output_details_field), "reasoning_tokens"))
    fresh_input = max(0, raw_input - (cache_read or 0) - (cache_write or 0))
    long_ttl = _has_long_ttl_writes(details)
    if long_ttl:
        logger.info("provider.usage long-TTL cache writes reported; call marked cost_unknown")
    return UsageStats(
        input_tokens=fresh_input,
        output_tokens=output_tokens,
        total_tokens=fresh_input + output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        cost_unknown=long_ttl,
    )
