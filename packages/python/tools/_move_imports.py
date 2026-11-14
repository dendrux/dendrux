"""Explain executable imports introduced by a pure module move."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from tools._move_typechecking import check_runtime_imports, guarded_imports
from tools._scopes import ScopeIndex, dotted, resolve_dotted, resolve_package

if TYPE_CHECKING:
    from collections.abc import Mapping

_IMPORTS = (ast.Import, ast.ImportFrom)


def _bindings(node: ast.Import | ast.ImportFrom, module: str) -> list[tuple[str, str, str]]:
    if isinstance(node, ast.Import):
        return [
            (
                "import-package" if "." in alias.name and not alias.asname else "import",
                alias.name,
                alias.asname or alias.name.split(".")[0],
            )
            for alias in node.names
        ]
    source = resolve_package(module.rpartition(".")[0], node.level, node.module)
    return [("from", f"{source}.{alias.name}", alias.asname or alias.name) for alias in node.names]


def _type_guards(tree: ast.Module, module: str) -> set[str]:
    index = ScopeIndex(tree, module.rpartition(".")[0])
    guards = set()
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and not node.orelse
            and all(isinstance(n, _IMPORTS) for n in node.body)
        ):
            spelling = dotted(node.test) or ""
            if resolve_dotted(spelling, index.module) == "typing.TYPE_CHECKING":
                guards.add(spelling.split(".")[0])
    return guards


def _scaffold_names(manifest: Mapping[str, Any], key: str) -> set[str]:
    return {
        node.id
        for entry in manifest.get("scaffolding", {}).values()
        if entry["module"] == key
        for node in ast.walk(ast.parse(entry["source"]))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _explained(
    binding: tuple[str, str, str],
    original: set[tuple[str, str, str]],
    owned: Mapping[str, str],
    lexical_targets: set[str],
    hoisted: set[tuple[str, str, str]],
    typing_names: set[str],
) -> bool:
    kind, target, local = binding
    if binding in original or binding in hoisted:
        return True
    if kind == "from" and owned.get(local) == target:
        return True
    if any(expected.startswith(target + ".") for expected in lexical_targets):
        return True
    return local in typing_names and (
        (kind == "from" and target.startswith("typing."))
        or (kind == "import" and target == "typing")
    )


def check_imports(
    old: ast.Module,
    trees: Mapping[str, ast.Module],
    *,
    old_key: str,
    modules: Mapping[str, str],
    owners: Mapping[str, str],
    hosts: Mapping[str, str],
    manifest: Mapping[str, Any] | None = None,
) -> list[str]:
    """Reject stars and explain runtime and TYPE_CHECKING imports without hiding runtime reads."""
    modules = {key: modules.get(key, key) for key in [old_key, *trees]}
    evidence = manifest or {}
    original = {
        b for n in old.body if isinstance(n, _IMPORTS) for b in _bindings(n, modules[old_key])
    }
    original_typed = {
        binding
        for node in guarded_imports(old, modules[old_key])
        for binding in _bindings(node, modules[old_key])
    }
    owned = {}
    for name, key in dict(owners, **hosts).items():
        for local in name.split(","):
            if local.isidentifier():
                owned[local] = f"{modules[key]}.{local}"
    problems = []
    for key, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                problems.append(f"star-import: {key}:{node.lineno}: {ast.unparse(node)}")
        targets = {e["target"] for e in evidence.get("lexical", []) if e["module"] == key}
        hoisted = {
            b
            for e in evidence.get("hoists", [])
            if e["module"] == key
            for n in ast.parse(e["statement"]).body
            if isinstance(n, _IMPORTS)
            for b in _bindings(n, modules[key])
        }
        typing_names = _scaffold_names(evidence, key) | _type_guards(tree, modules[key])
        peer_owned = {
            name: target
            for name, target in owned.items()
            if target.rpartition(".")[0] != modules[key]
        }
        guarded = guarded_imports(tree, modules[key])
        runtime_nodes = [node for node in tree.body if isinstance(node, _IMPORTS)]
        for node in [*runtime_nodes, *guarded]:
            explained_original = original | original_typed if node in guarded else original
            for binding in _bindings(node, modules[key]):
                scaffold_abstract = (
                    node in guarded
                    and binding == ("from", "abc.abstractmethod", "abstractmethod")
                    and binding[2] in _scaffold_names(evidence, key)
                )
                if not scaffold_abstract and not _explained(
                    binding, explained_original, peer_owned, targets, hoisted, typing_names
                ):
                    problems.append(
                        f"extra-import: {key}:{node.lineno}: {ast.unparse(node)} "
                        f"(unexplained {binding[1]})"
                    )
        problems.extend(check_runtime_imports(tree, modules[key], key))
    return problems
