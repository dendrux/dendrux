"""Symlink-safe reading of a skill directory.

On platforms where ``os.open`` supports ``dir_fd`` and ``os.scandir`` accepts
a descriptor (Linux, macOS, BSDs), traversal works on descriptors only: the
root is opened with ``O_DIRECTORY | O_NOFOLLOW``, each child is opened
relative to its verified parent descriptor with ``O_NOFOLLOW`` and its type
re-checked with ``fstat``, and no checked pathname is ever resolved again.
Swapping an entry for a symlink between enumeration and open is refused.

Elsewhere (e.g. Windows) a path-based fallback runs: symlinks present at
enumeration are rejected and files are opened with ``O_NOFOLLOW`` where it
exists, but a directory swapped for a link mid-load cannot be detected.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from dendrux.errors import SkillValidationError
from dendrux.skills._paths import validate_file_path

if TYPE_CHECKING:
    from types import TracebackType

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = (
    os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
)
_DIR_FLAGS = os.O_RDONLY | _NOFOLLOW | _CLOEXEC | getattr(os, "O_DIRECTORY", 0)

_FD_MODE = (
    os.open in os.supports_dir_fd
    and os.scandir in os.supports_fd
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


def _symlink_error(display: Path) -> SkillValidationError:
    return SkillValidationError(
        f"{display}: symlinks are not allowed in a skill directory.", field="files"
    )


def _not_regular_error(display: Path) -> SkillValidationError:
    return SkillValidationError(
        f"{display}: only regular files are allowed in a skill directory.", field="files"
    )


class SkillDir:
    """An open skill directory. Use as a context manager; reads SKILL.md, then files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._fd: int | None = None
        if root.is_symlink():
            raise SkillValidationError(
                f"{root}: skill directory must not be a symlink.", field="path"
            )
        if not _FD_MODE:
            if not root.is_dir():
                raise FileNotFoundError(f"Skill directory not found: {root}")
            return
        try:
            self._fd = os.open(root, _DIR_FLAGS)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SkillValidationError(
                    f"{root}: skill directory must not be a symlink.", field="path"
                ) from exc
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                raise FileNotFoundError(f"Skill directory not found: {root}") from exc
            raise

    def __enter__(self) -> SkillDir:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the root descriptor (no-op in path mode)."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def read_skill_md(self) -> tuple[Path, str]:
        """Return ``(SKILL.md path, its text)``.

        Raises:
            FileNotFoundError: SKILL.md is missing or not a regular file.
            SkillValidationError: SKILL.md is a symlink or not UTF-8.
        """
        skill_md = self.root / "SKILL.md"
        missing = FileNotFoundError(
            f"SKILL.md not found in {self.root}. "
            "A valid skill directory must contain a SKILL.md file."
        )
        if self._fd is None:
            if skill_md.is_symlink():
                raise _symlink_md_error(skill_md)
            if not skill_md.is_file():
                raise missing
            raw = _read_regular_file(skill_md)
        else:
            try:
                fd = _open_entry(self._fd, "SKILL.md", directory=False, display=skill_md)
            except FileNotFoundError:
                raise missing from None
            except SkillValidationError as exc:
                raise _symlink_md_error(skill_md) from exc
            try:
                raw = _read_fd(fd, skill_md)
            except SkillValidationError:
                raise missing from None
        try:
            return skill_md, raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillValidationError(
                f"{skill_md}: SKILL.md is not valid UTF-8.", field="body"
            ) from exc

    def read_files(self) -> dict[str, str | bytes]:
        """Read every non-hidden regular file except the top-level SKILL.md.

        Any symlink met (hidden or not) raises. Files decoding as strict UTF-8
        without a NUL byte become ``str``, others ``bytes``.
        """
        files: dict[str, str | bytes] = {}
        if self._fd is None:
            _walk_path(self.root, self.root, (), files)
        else:
            _walk_fd(self.root, self._fd, self.root, (), files)
        return files


def _symlink_md_error(skill_md: Path) -> SkillValidationError:
    return SkillValidationError(f"{skill_md}: SKILL.md must not be a symlink.", field="path")


def _file_key(root: Path, segments: tuple[str, ...]) -> str:
    key = "/".join(segments)
    try:
        validate_file_path(key)
    except SkillValidationError as exc:
        raise SkillValidationError(f"{root}: {exc}", field="files") from exc
    return key


def _walk_fd(
    root: Path,
    dir_fd: int,
    display_dir: Path,
    prefix: tuple[str, ...],
    out: dict[str, str | bytes],
) -> None:
    with os.scandir(dir_fd) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        display = display_dir / entry.name
        if entry.is_symlink():
            raise _symlink_error(display)
        if entry.name.startswith("."):
            continue
        segments = (*prefix, entry.name)
        if entry.is_dir(follow_symlinks=False):
            child = _open_entry(dir_fd, entry.name, directory=True, display=display)
            try:
                _walk_fd(root, child, display, segments, out)
            finally:
                os.close(child)
            continue
        if not entry.is_file(follow_symlinks=False):
            raise _not_regular_error(display)
        if not prefix and entry.name == "SKILL.md":
            continue
        key = _file_key(root, segments)
        fd = _open_entry(dir_fd, entry.name, directory=False, display=display)
        out[key] = _decode(_read_fd(fd, display))


def _open_entry(dir_fd: int, name: str, *, directory: bool, display: Path) -> int:
    """Open ``name`` relative to ``dir_fd`` without following a final symlink.

    Directories are opened with ``O_DIRECTORY`` and verified with ``fstat``;
    the caller owns the returned descriptor.
    """
    try:
        fd = os.open(name, _DIR_FLAGS if directory else _FILE_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _symlink_error(display) from exc
        if directory and exc.errno == errno.ENOTDIR:
            raise SkillValidationError(
                f"{display}: entry changed while loading (symlinks are not allowed "
                "in a skill directory).",
                field="files",
            ) from exc
        raise
    if directory and not stat.S_ISDIR(os.fstat(fd).st_mode):  # pragma: no cover - O_DIRECTORY
        os.close(fd)
        raise SkillValidationError(f"{display}: entry changed while loading.", field="files")
    return fd


def _read_fd(fd: int, display: Path) -> bytes:
    """Read a whole regular file from ``fd`` (always closed)."""
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except BaseException:
        os.close(fd)
        raise
    if not regular:
        os.close(fd)
        raise _not_regular_error(display)
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def _walk_path(
    root: Path, directory: Path, prefix: tuple[str, ...], out: dict[str, str | bytes]
) -> None:
    with os.scandir(directory) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        entry_path = Path(entry.path)
        if entry.is_symlink():
            raise _symlink_error(entry_path)
        if entry.name.startswith("."):
            continue
        segments = (*prefix, entry.name)
        if entry.is_dir(follow_symlinks=False):
            _walk_path(root, entry_path, segments, out)
            continue
        if not entry.is_file(follow_symlinks=False):
            raise _not_regular_error(entry_path)
        if not prefix and entry.name == "SKILL.md":
            continue
        key = _file_key(root, segments)
        out[key] = _decode(_read_regular_file(entry_path))


def _read_regular_file(path: Path) -> bytes:
    try:
        fd = os.open(path, _FILE_FLAGS)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _symlink_error(path) from exc
        raise
    return _read_fd(fd, path)


def _decode(raw: bytes) -> str | bytes:
    if b"\0" in raw:
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
