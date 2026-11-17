"""Static C3 and bound-provider provenance for operation mixins."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from tools._scopes import ScopeIndex, dotted, resolve_dotted

if TYPE_CHECKING:
    from collections.abc import Mapping

BUILTINS = {"object", "Exception", "BaseException"}


def _members(node: ast.ClassDef) -> set[str]:
    found = set()
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(child.name)
        elif isinstance(child, ast.Assign):
            found.update(
                n.id
                for target in child.targets
                for n in ast.walk(target)
                if isinstance(n, ast.Name)
            )
        elif (
            isinstance(child, ast.AnnAssign)
            and child.value is not None
            and isinstance(child.target, ast.Name)
        ):
            found.add(child.target.id)
    return found


def _linearize(
    graph: Mapping[str, list[str]],
    name: str,
    cache: dict[str, list[str]],
    visiting: frozenset[str] = frozenset(),
) -> list[str]:
    if name in visiting:
        raise ValueError(f"cyclic provider {name}")
    if name in cache:
        return cache[name]
    bases = graph.get(name, [])
    if len(set(bases)) != len(bases):
        raise ValueError(f"duplicate provider bases for {name}")
    sequences = [_linearize(graph, b, cache, visiting | {name})[:] for b in bases] + [bases[:]]
    result = [name]
    while any(sequences):
        sequences = [s for s in sequences if s]
        candidate = next(
            (s[0] for s in sequences if not any(s[0] in other[1:] for other in sequences)), None
        )
        if candidate is None:
            raise ValueError(f"inconsistent C3 for {name}")
        result.append(candidate)
        for sequence in sequences:
            if sequence[0] == candidate:
                sequence.pop(0)
    cache[name] = result
    return result


def _provider_classes(providers: Mapping[str, Any]) -> dict[str, ast.ClassDef]:
    classes = {}
    for identity, entry in providers.items():
        if set(entry) != {"path", "sha256", "class"}:
            raise ValueError("external provider evidence schema mismatch")
        nodes = ast.parse(entry["class"]).body
        if (
            len(nodes) != 1
            or not isinstance(nodes[0], ast.ClassDef)
            or nodes[0].name != identity.rsplit(".", 1)[-1]
            or nodes[0].bases
        ):
            raise ValueError("external provider needs one exact base-free class snapshot")
        classes[identity] = nodes[0]
    return classes


def validate_providers(
    old: ast.Module,
    trees: Mapping[str, ast.Module],
    modules: Mapping[str, str],
    hosts: Mapping[str, tuple[str, ast.ClassDef]],
    providers: Mapping[str, Any],
    old_key: str,
) -> list[str]:
    """Compare actual member and zero-argument super winners through Python C3.

    Imported original bases require reviewed class snapshots. The CLI/CI verifies
    each snapshot and its file hash against the immutable base and current source.
    """
    try:
        return _validate(old, trees, modules, hosts, providers, old_key)
    except ValueError as exc:
        return [f"provider: {exc}"]


def _validate(
    old: ast.Module,
    trees: Mapping[str, ast.Module],
    modules: Mapping[str, str],
    hosts: Mapping[str, tuple[str, ast.ClassDef]],
    providers: Mapping[str, Any],
    old_key: str,
) -> list[str]:
    composed = {
        n.name
        for tree in trees.values()
        for n in tree.body
        if isinstance(n, ast.ClassDef) and any(dotted(b) in hosts for b in n.bases)
    }
    original = {n.name: n for n in old.body if isinstance(n, ast.ClassDef) and n.name in composed}
    external = _provider_classes(providers)
    nodes = {name: node for name, (_, node) in hosts.items()} | external
    locations = {name: module for name, (module, _) in hosts.items()}
    contract_names = set()
    for key, tree in trees.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in original:
                nodes[node.name] = node
                locations[node.name] = key
            elif isinstance(node, ast.ClassDef) and node.name == "_StoreHost":
                contract_names.add(node.name)
                nodes[node.name] = node
                locations[node.name] = key
    old_index = ScopeIndex(old, modules[old_key].rpartition(".")[0])
    used_external: set[str] = set()

    def base_id(base: ast.expr, key: str, index: ScopeIndex) -> str:
        text = dotted(base)
        if text is None:
            raise ValueError("unknown provider expression")
        if text in hosts or text in contract_names or text in original:
            source = locations.get(text, old_key)
            if source != key:
                bound = index.module.resolve(text)
                if not bound or not bound.direct or bound.value != f"{modules[source]}.{text}":
                    raise ValueError(f"{text} has wrong/type-only/rebound provider identity")
            return text
        identity = resolve_dotted(text, index.module)
        if identity in external:
            bound = index.module.resolve(text.split(".")[0])
            if not bound or not bound.direct:
                raise ValueError("original provider is not unconditionally bound")
            used_external.add(identity)
            return identity
        if text in BUILTINS and index.module.find(text) is None:
            return text
        raise ValueError(f"unknown provider {text}; require immutable original-base evidence")

    before_graph = {
        name: [base_id(b, old_key, old_index) for b in node.bases]
        for name, node in original.items()
    }
    before_graph.update({name: [] for name in external})
    after_graph: dict[str, list[str]] = {}
    for name, node in nodes.items():
        if name in external:
            after_graph[name] = []
        else:
            key = locations[name]
            index = ScopeIndex(trees[key], modules[key].rpartition(".")[0])
            after_graph[name] = [base_id(b, key, index) for b in node.bases]
    if set(external) != used_external:
        raise ValueError("unused original provider evidence")
    before_nodes = original | external
    before_cache: dict[str, list[str]] = {}
    after_cache: dict[str, list[str]] = {}
    for name in after_graph:
        _linearize(after_graph, name, after_cache)
    problems = []
    for name in original:
        if name not in nodes:
            continue
        before = _linearize(before_graph, name, before_cache)
        after = _linearize(after_graph, name, after_cache)
        if [p for p in after if p in external] != [p for p in before if p in external]:
            raise ValueError(f"original-base order changed for {name}")
        names = set().union(*(_members(before_nodes[p]) for p in before if p in before_nodes))

        def winner(
            order: list[str],
            member: str,
            definitions: Mapping[str, ast.ClassDef],
            concrete: str = name,
        ) -> str | None:
            result = next(
                (p for p in order if p in definitions and member in _members(definitions[p])), None
            )
            return (
                concrete if result in hosts and member in _members(original[concrete]) else result
            )

        for member in names:
            if winner(before, member, before_nodes) != winner(after, member, nodes):
                problems.append(f"provider: changed member winner {name}.{member}")
        for provider in before:
            if provider not in before_nodes:
                continue
            targets = {
                n.attr
                for n in ast.walk(before_nodes[provider])
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "super"
                and not n.value.args
                and not n.value.keywords
            }
            for target in targets:
                old_tail = before[before.index(provider) + 1 :]
                new_tail = after[after.index(provider) + 1 :]
                if winner(old_tail, target, before_nodes) != winner(new_tail, target, nodes):
                    problems.append(f"provider: changed super winner {provider}.{target}")
    return problems
