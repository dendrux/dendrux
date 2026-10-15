"""Runtime views of a skill package: executor bytes and the activation envelope."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING

from dendrux.skills._paths import is_script

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dendrux.skills._model import Skill

_CLOSE_TAG_RE = re.compile(r"</(skill_content)", re.IGNORECASE)


def package_bytes(skill: Skill) -> Mapping[str, bytes]:
    """Return a read-only ``{path: bytes}`` view of the whole package.

    Includes ``SKILL.md`` (= ``skill.to_markdown()``) and every file; text
    is UTF-8 encoded. Build it once per run session and share it across
    executor calls; it is not copied per call.
    """
    view: dict[str, bytes] = {"SKILL.md": skill.to_markdown().encode()}
    for path, content in skill.files.items():
        view[path] = content.encode() if isinstance(content, str) else content
    return MappingProxyType(dict(sorted(view.items())))


def escape_body(body: str) -> str:
    """Neutralise any ``</skill_content`` (any case) so the body cannot close its envelope."""
    return _CLOSE_TAG_RE.sub(r"<\\/\1", body)


def render_activation_envelope(skill: Skill, *, scripts_runnable: bool) -> str:
    """Render the model-facing ``<skill_content>`` block for an activated skill.

    Lists files as text (readable), binary (not readable) or script. Script
    lines and the run-scripts hint appear only when ``scripts_runnable``.
    The envelope is framing for the model, not an authorization boundary.
    """
    lines = [f'<skill_content name="{skill.name}" digest="{skill.digest}">']
    body = escape_body(skill.body).rstrip("\n")
    if body:
        lines.append(body)
    listing = _file_listing(skill, scripts_runnable=scripts_runnable)
    if listing:
        hint = "read with read_skill_resource"
        if scripts_runnable:
            hint += "; run scripts with run_skill_script"
        lines.append("")
        lines.append(f"Files ({hint}):")
        lines.extend(listing)
    lines.append("</skill_content>")
    return "\n".join(lines)


def _file_listing(skill: Skill, *, scripts_runnable: bool) -> list[str]:
    text: list[str] = []
    binary: list[str] = []
    scripts: list[str] = []
    for path in sorted(skill.files):
        content = skill.files[path]
        if is_script(path):
            if scripts_runnable:
                scripts.append(f"- {path} (script)")
        elif isinstance(content, str):
            text.append(f"- {path} (text, {format_size(len(content.encode()))})")
        else:
            binary.append(f"- {path} (binary, {format_size(len(content))}, not readable)")
    return text + binary + scripts


def format_size(size: int) -> str:
    """Human size: ``512 B``, ``2.1 KB``, ``48 KB``, ``3.5 MB``."""
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if value < 10 else f"{round(value)} {unit}"
    raise AssertionError("unreachable")  # pragma: no cover
