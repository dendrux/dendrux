"""SKILL.md parsing and rendering (YAML frontmatter + Markdown body)."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, cast

import yaml

from dendrux.errors import SkillValidationError

if TYPE_CHECKING:
    from dendrux.skills._model import Skill

FRONTMATTER_KEYS = ("name", "description", "license", "compatibility", "allowed-tools", "metadata")
_BOM = "﻿"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_strict_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        # Stub versions differ on whether construct_object is typed; cast keeps both happy.
        key = cast("Any", loader).construct_object(key_node, deep=True)
        if isinstance(key, Hashable):
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_strict_mapping,
)


def _is_delimiter(line: str) -> bool:
    return line.rstrip(" \t\r") == "---"


def split_frontmatter(markdown: str) -> tuple[str, str]:
    """Split SKILL.md text into ``(frontmatter_yaml, body)``.

    The frontmatter is the block between the first two ``---`` lines. The
    body is everything after the closing line with exactly one leading
    newline removed; nothing else is stripped.
    """
    if not isinstance(markdown, str):
        raise SkillValidationError(
            f"SKILL.md content must be a string, got {type(markdown).__name__}.",
            field="frontmatter",
        )
    text = markdown.removeprefix(_BOM)
    first_end = text.find("\n")
    first_line = text if first_end == -1 else text[:first_end]
    if not _is_delimiter(first_line):
        raise SkillValidationError(
            "SKILL.md must start with YAML frontmatter (a '---' line).", field="frontmatter"
        )
    fm_start = pos = first_end + 1 if first_end != -1 else len(text)
    while pos < len(text):
        nl = text.find("\n", pos)
        line_end = len(text) if nl == -1 else nl
        if _is_delimiter(text[pos:line_end]):
            body = "" if nl == -1 else text[nl + 1 :]
            if body.startswith("\r\n"):
                body = body[2:]
            elif body.startswith("\n"):
                body = body[1:]
            return text[fm_start:pos], body
        pos = line_end + 1
    raise SkillValidationError(
        "SKILL.md has an opening '---' but no closing '---' for its frontmatter.",
        field="frontmatter",
    )


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Parse SKILL.md text into ``(Skill keyword arguments, body)``.

    Strict: unknown keys, duplicate keys, non-mapping frontmatter and
    non-string scalars for string fields raise :class:`SkillValidationError`.
    """
    fm_text, body = split_frontmatter(markdown)
    try:
        data = yaml.load(fm_text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise SkillValidationError(
            f"Invalid YAML in frontmatter: {exc}", field="frontmatter"
        ) from exc
    if not isinstance(data, dict):
        raise SkillValidationError(
            f"Frontmatter must be a YAML mapping, got {type(data).__name__}.",
            field="frontmatter",
        )
    unknown = [k for k in data if k not in FRONTMATTER_KEYS]
    if unknown:
        raise SkillValidationError(
            f"Unknown frontmatter key(s): {', '.join(sorted(repr(k) for k in unknown))}. "
            f"Allowed: {', '.join(FRONTMATTER_KEYS)}.",
            field="frontmatter",
        )

    kwargs: dict[str, Any] = {}
    for key in ("name", "description"):
        if key not in data:
            raise SkillValidationError(f"Missing required '{key}' field in frontmatter.", field=key)
        kwargs[key] = _require_str(data[key], key)
    for key in ("license", "compatibility"):
        if key in data:
            kwargs[key] = _require_str(data[key], key)
    if "allowed-tools" in data:
        kwargs["allowed_tools"] = tuple(
            _require_str(data["allowed-tools"], "allowed-tools").split()
        )
    if "metadata" in data:
        metadata = data["metadata"]
        if not isinstance(metadata, Mapping):
            raise SkillValidationError(
                f"'metadata' must be a YAML mapping, got {type(metadata).__name__}.",
                field="metadata",
            )
        kwargs["metadata"] = metadata
    return kwargs, body


def _require_str(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise SkillValidationError(
            f"'{key}' must be a string, got {type(value).__name__}. Quote the value in YAML.",
            field=key.replace("-", "_"),
        )
    return value


def render_markdown(skill: Skill) -> str:
    """Render the canonical SKILL.md text for ``skill`` (see ``Skill.to_markdown``)."""
    fm: dict[str, Any] = {"name": skill.name, "description": skill.description}
    if skill.license:
        fm["license"] = skill.license
    if skill.compatibility:
        fm["compatibility"] = skill.compatibility
    if skill.allowed_tools:
        fm["allowed-tools"] = " ".join(skill.allowed_tools)
    if skill.metadata:
        fm["metadata"] = {k: skill.metadata[k] for k in sorted(skill.metadata)}
    dumped = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"---\n{dumped}---\n\n{skill.body}"
