"""Extract the Python code blocks in the docs so tests can run them.

A block runs by default. To keep a block out of the run, put a Markdown reference
note on the line above its fence (blank lines between are fine)::

    [//]: # "docs-test: skip - needs a live provider"

The reason is required. The same ``[//]: # "..."`` form is already used for reviewer
notes (see docs/README.md); only notes starting ``docs-test:`` mean anything here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

MARKER_PREFIX = "docs-test:"
_MARKER = re.compile(r'^\s*\[//\]:\s*#\s*"(docs-test:[^"]*)"\s*$')
_SKIP = re.compile(r"^docs-test:\s*skip\s+-\s+(\S.*)$")
_FENCE_OPEN = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)(.*)$")
_PYTHON = frozenset({"python", "py"})
_EXCLUDED_PARTS = frozenset({"node_modules", ".git", ".venv", "venv", "site-packages"})


@dataclass(frozen=True)
class Snippet:
    file: str
    line: int
    code: str
    skip_reason: str | None

    @property
    def id(self) -> str:
        return f"{self.file}:{self.line}"


def _parse_marker(body: str, where: str) -> str:
    match = _SKIP.match(body.strip())
    if not match:
        raise ValueError(
            f'{where}: bad docs-test marker {body!r}; use [//]: # "docs-test: skip - <reason>"'
        )
    return match.group(1).strip()


def extract_snippets(path: Path, root: Path) -> list[Snippet]:
    rel = path.relative_to(root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets: list[Snippet] = []
    pending: tuple[int, str] | None = None  # (line number, reason)
    i = 0
    while i < len(lines):
        raw = lines[i]
        marker = _MARKER.match(raw)
        fence = _FENCE_OPEN.match(raw)
        if marker:
            if pending is not None:
                raise ValueError(f"{rel}:{pending[0]}: docs-test marker is not followed by a fence")
            pending = (i + 1, _parse_marker(marker.group(1), f"{rel}:{i + 1}"))
        elif fence:
            indent, ticks, lang = fence.group(1), fence.group(2), fence.group(3).lower()
            start = i + 1
            body: list[str] = []
            i += 1
            closer = re.compile(rf"^\s*{re.escape(ticks[0])}{{{len(ticks)},}}\s*$")
            while i < len(lines) and not closer.match(lines[i]):
                body.append(
                    lines[i][len(indent) :] if lines[i].startswith(indent) else lines[i].lstrip()
                )
                i += 1
            if i >= len(lines):
                raise ValueError(f"{rel}:{start}: unclosed code fence")
            if lang in _PYTHON:
                reason = pending[1] if pending else None
                snippets.append(Snippet(rel, start + 1, "\n".join(body) + "\n", reason))
                pending = None
            elif pending is not None:
                raise ValueError(
                    f"{rel}:{pending[0]}: docs-test marker is above a non-Python fence"
                )
        elif raw.strip() and pending is not None:
            raise ValueError(
                f"{rel}:{pending[0]}: docs-test marker must sit directly above a fence"
            )
        i += 1
    if pending is not None:
        raise ValueError(f"{rel}:{pending[0]}: docs-test marker is not followed by a fence")
    return snippets


def discover_files(root: Path) -> list[Path]:
    """Every ``docs/**/*.mdx`` plus every ``README.md`` under ``root``, sorted."""
    found = list((root / "docs").rglob("*.mdx")) + list(root.rglob("README.md"))
    keep = {p for p in found if not _EXCLUDED_PARTS & set(p.relative_to(root).parts)}
    return sorted(keep)
