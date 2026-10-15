"""Internal helpers for the Dendrux runtime. Not public API; may change without notice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux.errors import SkillContinuityError, SkillMismatchError, SkillMissingError
from dendrux.skills._executor import validate_executor
from dendrux.skills._package import (
    escape_body,
    format_size,
    package_bytes,
    render_activation_envelope,
)
from dendrux.skills._paths import is_script, validate_file_path

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dendrux.errors import SkillDigestMismatch


def continuity_error(
    run_id: str,
    *,
    missing: Iterable[str] = (),
    mismatched: Iterable[SkillDigestMismatch] = (),
) -> SkillContinuityError | None:
    """Return the error to raise for a resume check, or ``None`` when skills match.

    ``SkillMissingError`` when anything is missing (mismatches included),
    ``SkillMismatchError`` when only digests differ.
    """
    missing_t = tuple(missing)
    mismatched_t = tuple(mismatched)
    if missing_t:
        return SkillMissingError(run_id, missing=missing_t, mismatched=mismatched_t)
    if mismatched_t:
        return SkillMismatchError(run_id, mismatched=mismatched_t)
    return None


__all__ = [
    "continuity_error",
    "escape_body",
    "format_size",
    "is_script",
    "package_bytes",
    "render_activation_envelope",
    "validate_executor",
    "validate_file_path",
]
