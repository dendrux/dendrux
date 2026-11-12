"""Find the patch call sites in a test module (shared by patch_codemod and stale_patch_lint).

Recognised calls: ``<x>.setattr`` / ``<x>.delattr`` (pytest ``monkeypatch``), ``patch``,
``<x>.patch`` and ``patch.object`` (``unittest.mock``, ``mocker``). Two target forms:

* ``string``: ``patch("pkg.mod.name")`` / ``monkeypatch.setattr("pkg.mod.name", value)``;
* ``alias``: ``patch.object(mod, "name")`` / ``monkeypatch.setattr(mod, "name", value)``
  where ``mod`` is a name or attribute chain, resolved through the file's imports.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PatchSite:
    kind: Literal["string", "alias"]
    dotted: str  # string form: the whole target; alias form: the resolved object dotted path
    attr: str  # alias form: the attribute name (first component); "" for the string form
    node: ast.expr  # the target expression to rewrite
    lineno: int


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> dotted path for every import in the file, at any depth."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    aliases[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _resolve(node: ast.expr, aliases: dict[str, str]) -> str | None:
    name = dotted_name(node)
    if name is None:
        return None
    head, _, tail = name.partition(".")
    base = aliases.get(head)
    if base is None:
        return None
    return f"{base}.{tail}" if tail else base


def _call_shape(func: ast.expr) -> str | None:
    """'attr' for setattr/delattr methods, 'string' for patch(), 'object' for patch.object()."""
    if isinstance(func, ast.Attribute):
        if func.attr in {"setattr", "delattr"}:
            return "attr"
        if func.attr == "patch":
            return "string"
        if func.attr == "object":
            owner = func.value
            if (isinstance(owner, ast.Name) and owner.id == "patch") or (
                isinstance(owner, ast.Attribute) and owner.attr == "patch"
            ):
                return "object"
    elif isinstance(func, ast.Name) and func.id == "patch":
        return "string"
    return None


def _str_arg(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _site(call: ast.Call, shape: str, aliases: dict[str, str]) -> PatchSite | None:
    first = call.args[0] if call.args else None
    if first is None:
        return None
    as_string = _str_arg(first)
    positional = len(call.args)
    if shape == "string" or (shape == "attr" and as_string is not None and positional <= 2):
        if as_string is not None and "." in as_string:
            return PatchSite("string", as_string, "", first, first.lineno)
        return None
    attr = _str_arg(call.args[1]) if positional > 1 else None
    if attr is None or as_string is not None:
        return None
    target = _resolve(first, aliases)
    if target is None:
        return None
    return PatchSite("alias", target, attr.split(".")[0], first, first.lineno)


def find_sites(tree: ast.AST) -> list[PatchSite]:
    aliases = import_aliases(tree)
    sites: list[PatchSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            shape = _call_shape(node.func)
            if shape is not None and (site := _site(node, shape, aliases)) is not None:
                sites.append(site)
    return sorted(sites, key=lambda s: (s.lineno, s.node.col_offset))
