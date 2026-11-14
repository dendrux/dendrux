"""Reject copied callable bindings at consumers of moved or imported patch seams."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from tools._move_manifest import roots
from tools._scopes import ScopeIndex

if TYPE_CHECKING:
    from collections.abc import Mapping


def check_bindings(source: str, module: str, targets: Mapping[str, str]) -> list[str]:
    """Bare imported callables must be read through their defining owner module."""
    tree = ast.parse(source)
    index = ScopeIndex(tree, module.rpartition(".")[0])
    forbidden = set(targets.values())
    retained = "dendrux.runtime._run_entry.current_skill_session"
    retained_alias = retained in forbidden and module != "dendrux.runtime._run_entry"
    problems = []
    for node, scope in index.names:
        if not isinstance(node.ctx, ast.Load):
            continue
        binding = scope.resolve(node.id)
        copied_alias = (
            retained_alias
            and node.id != "current"
            and binding is not None
            and binding.kind == "import"
            and binding.value == "dendrux.skills._session.current"
        )
        if binding and binding.kind == "import" and (binding.value in forbidden or copied_alias):
            owner = binding.value.rpartition(".")[0]
            if owner != module:
                problems.append(
                    f"owner-binding: {module}:{node.lineno} copies {binding.value}; "
                    "qualify its owner"
                )
        elif scope.hides_import(node.id) and any(
            b.kind == "import"
            and (
                b.value in forbidden
                or retained_alias
                and node.id != "current"
                and b.value == "dendrux.skills._session.current"
            )
            for s in index.scopes
            for b in s.bindings.get(node.id, ())
        ):
            problems.append(
                f"owner-binding: {module}:{node.lineno} has ambiguous/shadowed seam {node.id}"
            )
    return problems


def check_globals(
    old: str,
    after: Mapping[str, str],
    old_module: str,
    modules: Mapping[str, str],
    owners: Mapping[str, str],
) -> list[str]:
    """Unchanged bare global reads retain the exact original dependency binding."""
    tree = ast.parse(old)
    before = ScopeIndex(tree, old_module.rpartition(".")[0])
    problems = []
    for root, key in owners.items():
        original = roots(tree, root)
        if not original:
            continue
        current = ast.parse(after[key])
        index = ScopeIndex(current, modules[key].rpartition(".")[0])
        old_names = {
            n.id
            for node in original
            for n in ast.walk(node)
            if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and before.scope_of(n).find(n.id) is before.module
        }
        for node in roots(current, root):
            for ref in ast.walk(node):
                if (
                    not isinstance(ref, ast.Name)
                    or not isinstance(ref.ctx, ast.Load)
                    or ref.id not in old_names
                ):
                    continue
                scope = index.scope_of(ref)
                if scope.find(ref.id) is not index.module:
                    if scope.find(ref.id) is None:
                        problems.append(f"owner-binding: unbound global {modules[key]}.{ref.id}")
                    continue
                expected = before.module.resolve(ref.id)
                actual = scope.resolve(ref.id)
                if expected is None or actual is None:
                    problems.append(f"owner-binding: ambiguous global {ref.id}")
                elif expected.kind == "import":
                    if actual.kind != "import" or actual.value != expected.value:
                        problems.append(f"owner-binding: changed dependency {ref.id}")
                elif ref.id in owners:
                    owner = owners[ref.id]
                    if owner != key and (
                        actual.kind != "import"
                        or actual.value != f"{modules[owner]}.{ref.id}"
                        or not actual.direct
                    ):
                        problems.append(f"owner-binding: changed shared object {ref.id}")
    return problems
