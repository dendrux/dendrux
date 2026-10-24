"""Placeholder view of restored tool arguments (what approvers and events see).

PII placeholders (``<<EMAIL_1>>``) are restored to real values before
validation, and lax coercion can then change a value's form: ``"5551234567"``
becomes ``5551234567``, a validator lowercases an email. Substituting
placeholders back into the result cannot undo that, so the view is built from
the arguments **as the model sent them**:

- a leaf whose sent form holds a known placeholder is shown as sent;
- a leaf unchanged by restore and coercion is shown as is;
- any other leaf that contains a restored value in any form (case-insensitive
  text, or the same digits) is shown as ``<<REDACTED>>``. This fails closed.

Only the executed function and the stored execution record carry real values.
"""

from __future__ import annotations

import re
from typing import Any

from dendrux.tools._coerce import _PLACEHOLDER_SHAPE, contains_placeholder

REDACTED = "<<REDACTED>>"

_MISSING = object()
_DIGITS = re.compile(r"\d")
_MAX_DEPTH = 50
# A restored value with fewer digits than this is not matched by digits alone.
_MIN_DIGIT_MATCH = 4


def placeholder_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Only real ``<<ENTITY_N>>`` entries (drops bookkeeping keys)."""
    return {
        k: v
        for k, v in mapping.items()
        if isinstance(k, str) and isinstance(v, str) and _PLACEHOLDER_SHAPE.fullmatch(k)
    }


def _digits(text: str) -> str:
    return "".join(_DIGITS.findall(text))


def _leaks(value: Any, reals: list[tuple[str, str]]) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    text = str(value)
    folded = text.casefold()
    digits = _digits(text)
    for real_folded, real_digits in reals:
        if real_folded and real_folded in folded:
            return True
        if len(real_digits) >= _MIN_DIGIT_MATCH and real_digits in digits:
            return True
    return False


def placeholder_view(normalized: Any, sent: Any, mapping: dict[str, str]) -> Any:
    """``normalized`` arguments with every restored value masked (see module doc)."""
    entries = placeholder_mapping(mapping)
    if not entries:
        return normalized
    placeholders = frozenset(entries)
    reals = [(real.strip().casefold(), _digits(real)) for real in entries.values() if real]

    def walk(value: Any, original: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return REDACTED
        if isinstance(value, dict):
            return {
                k: walk(
                    v,
                    original.get(k, _MISSING) if isinstance(original, dict) else _MISSING,
                    depth + 1,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                walk(
                    v,
                    original[i] if isinstance(original, list) and i < len(original) else _MISSING,
                    depth + 1,
                )
                for i, v in enumerate(value)
            ]
        if original is not _MISSING and contains_placeholder(original, placeholders):
            return original
        if original is not _MISSING and type(original) is type(value) and original == value:
            return value
        return REDACTED if _leaks(value, reals) else value

    return walk(normalized, sent, 0)
