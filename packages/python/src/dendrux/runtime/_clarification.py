"""Bound the display copy of a human-input question without failing its pause."""

from __future__ import annotations

import json
import logging
from typing import Any

from dendrux import _codes

DEFAULT_PAUSE_SUMMARY_BYTES = 65536


def _text(value: str, budget: int) -> tuple[str, int, bool]:
    candidate = value[:budget]
    encoded = json.dumps(candidate)
    if len(candidate) == len(value) and len(encoded) <= budget:
        return candidate, len(encoded), False
    suffix = "[truncated]"
    lo, hi = 0, len(candidate)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(json.dumps(candidate[:mid] + suffix)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    result = candidate[:lo] + suffix
    return result, len(json.dumps(result)), True


def clarification_summary(
    question: str,
    options: list[str],
    *,
    max_bytes: int = DEFAULT_PAUSE_SUMMARY_BYTES,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Keep a bounded question and option prefix, with a visible truncation marker."""
    remaining = max_bytes - 64
    question, used, truncated = _text(question, remaining)
    remaining -= used
    kept = []
    for option in options:
        if remaining < 16:
            break
        value, used, cut = _text(option, remaining - 2)
        kept.append(value)
        remaining -= used + 2
        truncated |= cut
    truncated |= len(kept) != len(options)
    if truncated:
        _codes.log_code(logging.getLogger(__name__), _codes.PAUSE_SUMMARY_TRUNCATED, run_id=run_id)
    return {"question": question, "options": kept}
