"""The ``Skill`` value: a validated, immutable instruction package."""

from __future__ import annotations

import os
import re
import unicodedata
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field, fields
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dendrux.errors import SkillValidationError
from dendrux.skills._digest import compute_digest
from dendrux.skills._fs import SkillDir
from dendrux.skills._markdown import parse_frontmatter, render_markdown
from dendrux.skills._paths import validate_file_path

_NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
MAX_COMPATIBILITY_LEN = 500
MAX_REVISION_LEN = 256
PACKAGE_WARN_BYTES = 5 * 1024 * 1024


def _empty_metadata() -> Mapping[str, str]:
    return MappingProxyType({})


def _empty_files() -> Mapping[str, str | bytes]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Skill:
    """A validated, immutable skill package.

    Build one directly, from SKILL.md text (:meth:`from_markdown`) or from a
    directory (:meth:`from_dir`). Every construction path validates; any
    violation raises :class:`~dendrux.errors.SkillValidationError`.

    ``files`` maps relative POSIX paths to content: ``str`` is a text file,
    ``bytes`` a binary file; paths under ``scripts/`` are scripts.
    ``revision`` is a non-secret app label stored in run metadata and events.
    ``path`` is provenance only. ``digest`` identifies the content.
    """

    name: str
    description: str
    body: str = field(default="", repr=False)
    _: KW_ONLY
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, str] = field(default_factory=_empty_metadata)
    allowed_tools: tuple[str, ...] = ()
    files: Mapping[str, str | bytes] = field(default_factory=_empty_files, repr=False)
    revision: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        _set(self, "name", _validate_name(self.name))
        _set(self, "description", _validate_description(self.description))
        _set(self, "body", _text(self.body, "body"))
        _set(self, "license", _optional_text(self.license, "license"))
        compatibility = _optional_text(self.compatibility, "compatibility")
        if compatibility is not None and len(compatibility) > MAX_COMPATIBILITY_LEN:
            raise SkillValidationError(
                f"'compatibility' exceeds {MAX_COMPATIBILITY_LEN} characters "
                f"(got {len(compatibility)}).",
                field="compatibility",
            )
        _set(self, "compatibility", compatibility)
        _set(self, "metadata", _freeze_metadata(self.metadata))
        _set(self, "allowed_tools", _freeze_allowed_tools(self.allowed_tools))
        _set(self, "files", _freeze_files(self.files))
        _set(self, "revision", _validate_revision(self.revision))
        _set(self, "path", _validate_path(self.path))

        size = self.package_size
        if size > PACKAGE_WARN_BYTES:
            warnings.warn(
                f"Skill '{self.name}' package is {size} bytes (over 5 MiB). Large packages "
                "are held in memory and copied to the script executor; consider trimming.",
                UserWarning,
                stacklevel=3,
            )

    def __hash__(self) -> int:
        return hash(self.digest)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        state: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            state[f.name] = dict(value) if isinstance(value, Mapping) else value
        return (_restore, (state,))

    @cached_property
    def digest(self) -> str:
        """Content digest ``"sha256:<hex>"`` (excludes ``revision`` and ``path``)."""
        return compute_digest(self)

    @property
    def package_size(self) -> int:
        """Body plus file sizes in bytes (text counted as UTF-8)."""
        total = len(self.body.encode())
        for value in self.files.values():
            total += len(value.encode()) if isinstance(value, str) else len(value)
        return total

    @classmethod
    def from_markdown(
        cls,
        markdown: str,
        *,
        files: Mapping[str, str | bytes] | None = None,
        revision: str | None = None,
    ) -> Skill:
        """Build a skill from SKILL.md text (YAML frontmatter + Markdown body).

        Raises:
            SkillValidationError: On malformed frontmatter, unknown keys,
                non-string values for string fields, or any field violation.
        """
        kwargs, body = parse_frontmatter(markdown)
        return cls(
            body=body,
            files=files if files is not None else {},
            revision=revision,
            **kwargs,
        )

    @classmethod
    def from_dir(cls, path: str | os.PathLike[str], *, revision: str | None = None) -> Skill:
        """Load a skill directory: ``SKILL.md`` plus every regular file below it.

        The directory name must equal the skill name. Symlinks (the root,
        ``SKILL.md`` or any traversed entry) are rejected, including entries
        swapped for a symlink mid-load where the OS supports ``dir_fd`` (see
        ``dendrux.skills._fs`` for the fallback); hidden files and
        directories are skipped. Files that decode as strict UTF-8 without a
        NUL byte become ``str``, others ``bytes``.

        Raises:
            FileNotFoundError: If the directory or its SKILL.md is missing.
            SkillValidationError: If SKILL.md or any file fails validation.
        """
        root = Path(path)
        with SkillDir(root) as skill_dir:
            skill_md, text = skill_dir.read_skill_md()
            with _prefixed(skill_md):
                kwargs, body = parse_frontmatter(text)
                name = kwargs["name"]
                _validate_name(name)
                if name != root.name:
                    raise SkillValidationError(
                        f"Skill name '{name}' does not match directory name '{root.name}'. "
                        "The name field must match the parent directory name.",
                        field="name",
                    )
            files = skill_dir.read_files()
        with _prefixed(root):
            return cls(body=body, files=files, revision=revision, path=root, **kwargs)

    @classmethod
    def scan_dir(cls, path: str | os.PathLike[str]) -> list[Skill]:
        """Load every subdirectory of ``path`` that contains a SKILL.md, sorted by name.

        Subdirectories without SKILL.md are skipped; invalid skills raise.

        Raises:
            FileNotFoundError: If ``path`` is not a directory.
        """
        parent = Path(path)
        if not parent.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {parent}")
        skills = [
            cls.from_dir(child)
            for child in sorted(parent.iterdir())
            if child.is_dir() and (child / "SKILL.md").is_file()
        ]
        return sorted(skills, key=lambda s: s.name)

    def to_markdown(self) -> str:
        """Return canonical SKILL.md text.

        ``Skill.from_markdown(s.to_markdown())`` equals ``s`` except for
        ``files``, ``revision`` and ``path``.
        """
        return render_markdown(self)


def _restore(state: dict[str, Any]) -> Skill:
    return Skill(**state)


@contextmanager
def _prefixed(where: Path) -> Iterator[None]:
    """Re-raise SkillValidationError with a location prefix."""
    try:
        yield
    except SkillValidationError as exc:
        raise SkillValidationError(f"{where}: {exc}", field=exc.field) from exc


def _set(skill: Skill, name: str, value: object) -> None:
    object.__setattr__(skill, name, value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SkillValidationError(
            f"'{name}' must be a string, got {type(value).__name__}.", field=name
        )
    try:
        value.encode()
    except UnicodeEncodeError as exc:
        raise SkillValidationError(
            f"'{name}' is not valid Unicode text: {exc}", field=name
        ) from exc
    return str(value)


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name)
    if not text:
        raise SkillValidationError(f"'{name}' must be a non-empty string or None.", field=name)
    return text


def _validate_name(name: object) -> str:
    value = _text(name, "name")
    if not value:
        raise SkillValidationError("Skill name cannot be empty.", field="name")
    if len(value) > MAX_NAME_LEN:
        raise SkillValidationError(
            f"Skill name '{value}' exceeds {MAX_NAME_LEN} characters (got {len(value)}).",
            field="name",
        )
    if _NAME_RE.fullmatch(value):
        return value
    if "--" in value and re.fullmatch(r"[a-z0-9-]+", value):
        reason = "contains consecutive hyphens"
    elif value.startswith("-") or value.endswith("-"):
        reason = "cannot start or end with a hyphen"
    else:
        reason = "must contain only lowercase letters, numbers, and hyphens"
    raise SkillValidationError(f"Skill name '{value}' is invalid: {reason}.", field="name")


def _validate_description(description: object) -> str:
    value = _text(description, "description")
    stripped = value.strip()
    if not stripped:
        raise SkillValidationError("Skill description cannot be empty.", field="description")
    if len(stripped) > MAX_DESCRIPTION_LEN:
        raise SkillValidationError(
            f"Description exceeds {MAX_DESCRIPTION_LEN} characters (got {len(stripped)}).",
            field="description",
        )
    return value


def _freeze_metadata(metadata: object) -> Mapping[str, str]:
    if not isinstance(metadata, Mapping):
        raise SkillValidationError(
            f"'metadata' must be a mapping, got {type(metadata).__name__}.", field="metadata"
        )
    copy: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SkillValidationError(
                f"'metadata' keys and values must be strings; got {key!r}: "
                f"{type(value).__name__}. Quote the value in YAML.",
                field="metadata",
            )
        copy[_text(key, "metadata")] = _text(value, "metadata")
    return MappingProxyType(dict(sorted(copy.items())))


def _freeze_allowed_tools(tools: object) -> tuple[str, ...]:
    if isinstance(tools, str | bytes) or not isinstance(tools, Iterable):
        raise SkillValidationError(
            "'allowed_tools' must be a sequence of tool names (not a single string).",
            field="allowed_tools",
        )
    result: list[str] = []
    for tool in tools:
        if not isinstance(tool, str) or not tool or any(ch.isspace() for ch in tool):
            raise SkillValidationError(
                f"'allowed_tools' entries must be non-empty strings without whitespace; "
                f"got {tool!r}.",
                field="allowed_tools",
            )
        result.append(_text(tool, "allowed_tools"))
    return tuple(result)


def _freeze_files(files: object) -> Mapping[str, str | bytes]:
    if not isinstance(files, Mapping):
        raise SkillValidationError(
            f"'files' must be a mapping of path to str or bytes, got {type(files).__name__}.",
            field="files",
        )
    copy: dict[str, str | bytes] = {}
    for path, content in files.items():
        validate_file_path(path)
        if isinstance(content, str):
            copy[path] = _text(content, "files")
        elif isinstance(content, bytes):
            copy[path] = bytes(content)
        else:
            raise SkillValidationError(
                f"File {path!r} content must be str (text) or bytes (binary), "
                f"got {type(content).__name__}.",
                field="files",
            )
    return MappingProxyType(dict(sorted(copy.items())))


def _validate_revision(revision: object) -> str | None:
    if revision is None:
        return None
    value = _text(revision, "revision")
    if not value or len(value) > MAX_REVISION_LEN:
        raise SkillValidationError(
            f"'revision' must be 1-{MAX_REVISION_LEN} characters (got {len(value)}).",
            field="revision",
        )
    if any(unicodedata.category(ch) == "Cc" for ch in value):
        raise SkillValidationError(
            "'revision' must not contain control characters.", field="revision"
        )
    return value


def _validate_path(path: object) -> Path | None:
    if path is None:
        return None
    if isinstance(path, str | os.PathLike):
        return Path(path)
    raise SkillValidationError(
        f"'path' must be a path or None, got {type(path).__name__}.", field="path"
    )
