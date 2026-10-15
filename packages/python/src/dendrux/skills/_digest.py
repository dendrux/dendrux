"""Content digest for a skill (independent of YAML and ``to_markdown``)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dendrux.skills._model import Skill

DIGEST_PREFIX = b"dendrux-skill-v1\n"


def compute_digest(skill: Skill) -> str:
    """Return ``"sha256:<hex>"`` over the skill's content (spec §2.5).

    ``revision`` and ``path`` are excluded, so equal content gives an equal
    digest regardless of how the skill was built.
    """
    header = {
        "name": skill.name,
        "description": skill.description,
        "body": skill.body,
        "license": skill.license,
        "compatibility": skill.compatibility,
        "allowed_tools": list(skill.allowed_tools),
        "metadata": dict(skill.metadata),
    }
    h = hashlib.sha256()
    h.update(DIGEST_PREFIX)
    h.update(json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    h.update(b"\n")
    _update_files(h, skill.files)
    return f"sha256:{h.hexdigest()}"


def _update_files(h: hashlib._Hash, files: Mapping[str, str | bytes]) -> None:
    for path in sorted(files):
        value = files[path]
        content = value.encode() if isinstance(value, str) else value
        kind = b"t" if isinstance(value, str) else b"b"
        h.update(path.encode() + b"\0" + kind + b"\0" + len(content).to_bytes(8, "big"))
        h.update(content)
