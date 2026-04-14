"""SKILL.md parsing and skill loading.

Implements the Agent Skills open standard (agentskills.io).
Validates frontmatter, name format, and directory structure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# agentskills.io spec: lowercase alphanumeric + hyphens, no leading/trailing
# hyphen, no consecutive hyphens, max 64 chars.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 1024


def _validate_skill_name(name: str, dir_name: str) -> None:
    """Validate skill name per agentskills.io spec."""
    if not name:
        raise ValueError("Skill name cannot be empty.")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"Skill name '{name}' exceeds {_MAX_NAME_LEN} characters (got {len(name)})."
        )
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f"Skill name '{name}' is invalid. Must contain only lowercase "
            f"letters, numbers, and hyphens. Cannot start or end with a hyphen."
        )
    if "--" in name:
        raise ValueError(f"Skill name '{name}' contains consecutive hyphens, which is not allowed.")
    if name != dir_name:
        raise ValueError(
            f"Skill name '{name}' does not match directory name '{dir_name}'. "
            f"The name field must match the parent directory name."
        )


def _parse_skill_md(path: Path) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_string).

    Uses line-based parsing: first line must be exactly ``---``,
    then scans for the next line that is exactly ``---``.
    Everything between is YAML frontmatter, everything after is body.

    Raises ValueError if frontmatter is missing or malformed.
    """
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"{path}: SKILL.md must start with YAML frontmatter (--- delimiter). "
            f"See agentskills.io/specification for the required format."
        )

    # Find the closing --- line
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break

    if close_idx is None:
        raise ValueError(f"{path}: SKILL.md has opening --- but no closing --- for frontmatter.")

    frontmatter_str = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :]).strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: Invalid YAML in frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
        raise ValueError(
            f"{path}: Frontmatter must be a YAML mapping, got {type(frontmatter).__name__}."
        )

    return frontmatter, body


@dataclass(frozen=True)
class Skill:
    """A loaded skill definition.

    Parsed from a SKILL.md file following the agentskills.io spec.
    Immutable after creation.
    """

    name: str
    description: str
    body: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    compatibility: str | None = None

    @classmethod
    def from_dir(cls, path: str | Path) -> Skill:
        """Load a skill from a directory containing SKILL.md.

        Validates frontmatter, name format, and name-directory match.

        Args:
            path: Path to the skill directory.

        Raises:
            FileNotFoundError: If path doesn't exist or has no SKILL.md.
            ValueError: If SKILL.md is malformed or fails validation.
        """
        skill_dir = Path(path)
        if not skill_dir.is_dir():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise FileNotFoundError(
                f"SKILL.md not found in {skill_dir}. "
                f"A valid skill directory must contain a SKILL.md file."
            )

        frontmatter, body = _parse_skill_md(skill_md)
        dir_name = skill_dir.name

        # Validate required fields
        name = frontmatter.get("name")
        if not name:
            raise ValueError(f"{skill_md}: Missing required 'name' field in frontmatter.")

        description = frontmatter.get("description")
        if not description:
            raise ValueError(f"{skill_md}: Missing required 'description' field in frontmatter.")

        if len(str(description)) > _MAX_DESCRIPTION_LEN:
            raise ValueError(
                f"{skill_md}: Description exceeds {_MAX_DESCRIPTION_LEN} characters "
                f"(got {len(str(description))})."
            )

        # Validate name format and directory match
        _validate_skill_name(str(name), dir_name)

        # Validate optional field types
        raw_metadata = frontmatter.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, dict):
            raise ValueError(
                f"{skill_md}: 'metadata' must be a YAML mapping, got {type(raw_metadata).__name__}."
            )

        raw_license = frontmatter.get("license")
        if raw_license is not None and not isinstance(raw_license, str):
            raise ValueError(
                f"{skill_md}: 'license' must be a string, got {type(raw_license).__name__}."
            )

        raw_compat = frontmatter.get("compatibility")
        if raw_compat is not None and not isinstance(raw_compat, str):
            raise ValueError(
                f"{skill_md}: 'compatibility' must be a string, got {type(raw_compat).__name__}."
            )

        return cls(
            name=str(name),
            description=str(description),
            body=body,
            path=skill_dir,
            metadata=raw_metadata or {},
            license=raw_license,
            compatibility=raw_compat,
        )

    @classmethod
    def scan_dir(cls, path: str | Path) -> list[Skill]:
        """Scan a directory for skill subdirectories.

        Finds all subdirectories containing a SKILL.md file and loads
        them. Subdirectories without SKILL.md are silently skipped.
        Invalid skills raise — fail fast, don't silently degrade.

        Args:
            path: Path to the parent directory containing skill folders.

        Returns:
            List of loaded Skill objects, sorted by name.

        Raises:
            FileNotFoundError: If path doesn't exist.
        """
        parent = Path(path)
        if not parent.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {parent}")

        skills: list[Skill] = []
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                skills.append(cls.from_dir(child))

        return skills
