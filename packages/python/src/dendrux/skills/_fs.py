"""Symlink-safe reading of a skill directory."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

from dendrux.errors import SkillValidationError
from dendrux.skills._paths import validate_file_path

_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
)


def read_skill_md(root: Path) -> tuple[Path, str]:
    """Return ``(SKILL.md path, its text)`` after checking the root and file."""
    if root.is_symlink():
        raise SkillValidationError(f"{root}: skill directory must not be a symlink.", field="path")
    if not root.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {root}")
    skill_md = root / "SKILL.md"
    if skill_md.is_symlink():
        raise SkillValidationError(f"{skill_md}: SKILL.md must not be a symlink.", field="path")
    if not skill_md.is_file():
        raise FileNotFoundError(
            f"SKILL.md not found in {root}. A valid skill directory must contain a SKILL.md file."
        )
    raw = _read_regular_file(skill_md)
    try:
        return skill_md, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillValidationError(
            f"{skill_md}: SKILL.md is not valid UTF-8.", field="body"
        ) from exc


def read_package_files(root: Path) -> dict[str, str | bytes]:
    """Read every non-hidden regular file under ``root`` except the top-level SKILL.md.

    Traversal never follows symlinks; any symlink met (hidden or not) raises.
    Files decoding as strict UTF-8 without a NUL byte become ``str``.
    """
    files: dict[str, str | bytes] = {}
    _walk(root, root, (), files)
    return files


def _walk(
    root: Path, directory: Path, prefix: tuple[str, ...], out: dict[str, str | bytes]
) -> None:
    with os.scandir(directory) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        entry_path = Path(entry.path)
        if entry.is_symlink():
            raise SkillValidationError(
                f"{entry_path}: symlinks are not allowed in a skill directory.", field="files"
            )
        if entry.name.startswith("."):
            continue
        segments = (*prefix, entry.name)
        if entry.is_dir(follow_symlinks=False):
            _walk(root, entry_path, segments, out)
            continue
        if not entry.is_file(follow_symlinks=False):
            raise SkillValidationError(
                f"{entry_path}: only regular files are allowed in a skill directory.",
                field="files",
            )
        if not prefix and entry.name == "SKILL.md":
            continue
        key = "/".join(segments)
        try:
            validate_file_path(key)
        except SkillValidationError as exc:
            raise SkillValidationError(f"{root}: {exc}", field="files") from exc
        out[key] = _decode(_read_regular_file(entry_path))


def _read_regular_file(path: Path) -> bytes:
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SkillValidationError(
                f"{path}: symlinks are not allowed in a skill directory.", field="files"
            ) from exc
        raise
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise SkillValidationError(
                f"{path}: only regular files are allowed in a skill directory.", field="files"
            )
        return fh.read()


def _decode(raw: bytes) -> str | bytes:
    if b"\0" in raw:
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
