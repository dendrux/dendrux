"""Skill package file path rules."""

from __future__ import annotations

from dendrux.errors import SkillValidationError

MAX_PATH_LEN = 512
SCRIPTS_PREFIX = "scripts/"
_FORBIDDEN_CHARS = frozenset('<>"\\')


def validate_file_path(path: str) -> None:
    """Raise :class:`SkillValidationError` unless ``path`` is a valid file key.

    A valid key is a non-empty relative POSIX path of at most 512 chars,
    with ``/`` separators, no ASCII control characters, none of ``<>"\\``,
    no ``.``/``..``/empty segments, and not the reserved ``SKILL.md``.
    Runtime tools must resolve paths by exact match against ``Skill.files``
    after this check, never by joining onto a filesystem path.
    """
    if not isinstance(path, str):
        raise SkillValidationError(
            f"Skill file path must be a string, got {type(path).__name__}.", field="files"
        )
    reason = _path_problem(path)
    if reason is not None:
        raise SkillValidationError(f"Invalid skill file path {path!r}: {reason}.", field="files")


def _path_problem(path: str) -> str | None:
    if not path:
        return "path is empty"
    if len(path) > MAX_PATH_LEN:
        return f"longer than {MAX_PATH_LEN} characters"
    if path.startswith("/"):
        return "must be relative"
    for ch in path:
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return "contains a control character"
        if ch in _FORBIDDEN_CHARS:
            return f"contains forbidden character {ch!r}"
    if any(segment in ("", ".", "..") for segment in path.split("/")):
        return "contains an empty, '.' or '..' segment"
    if path == "SKILL.md":
        return "'SKILL.md' is reserved for the skill body"
    return None


def is_script(path: str) -> bool:
    """Return True if ``path`` names a script (any file under ``scripts/``).

    Scripts are runnable through the executor and never readable into context.
    """
    return path.startswith(SCRIPTS_PREFIX)
