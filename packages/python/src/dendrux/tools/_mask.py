"""The approval view of a tool call: what approvers and events see.

The view is the arguments **as the model sent them** (after JSON parsing,
before PII restore and before coercion), so it only ever holds placeholders
such as ``<<EMAIL_1>>``. Coercion is deterministic from these arguments, so
what is approved still determines what runs; only the executed function and
the stored execution record carry real values.

As defence in depth, any key or value that nevertheless contains a value from
the run's PII mapping (as a whole token, case-insensitively) is replaced with
``<<REDACTED>>``.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from dendrux.tools._coerce import _PLACEHOLDER_SHAPE

REDACTED = "<<REDACTED>>"
_MAX_DEPTH = 50


def placeholder_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Only real ``<<ENTITY_N>>`` entries (drops bookkeeping keys)."""
    return {
        k: v
        for k, v in mapping.items()
        if isinstance(k, str) and isinstance(v, str) and _PLACEHOLDER_SHAPE.fullmatch(k)
    }


def approval_view(sent: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """A copy of the model-sent arguments with any real PII value redacted."""
    reals = [r.strip().casefold() for r in placeholder_mapping(mapping).values() if r.strip()]
    if not reals:
        return copy.deepcopy(sent)
    # Whole-token matches only, so a short value ("Al") never hits "Always".
    pattern = re.compile(
        "|".join(rf"(?<!\w){re.escape(r)}(?!\w)" for r in sorted(reals, key=len, reverse=True))
    )

    def leaks(text: str) -> bool:
        return pattern.search(text.casefold()) is not None

    def walk(value: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return REDACTED
        if isinstance(value, dict):
            return {
                (REDACTED if leaks(str(k)) else k): walk(v, depth + 1) for k, v in value.items()
            }
        if isinstance(value, list):
            return [walk(v, depth + 1) for v in value]
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return REDACTED if leaks(str(value)) else value
        return value

    result: dict[str, Any] = walk(sent, 0)
    return result
