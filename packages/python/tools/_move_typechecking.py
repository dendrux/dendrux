"""Separate guarded type imports from the bindings Python needs at runtime."""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING

from tools._scopes import ScopeIndex, dotted, resolve_dotted

if TYPE_CHECKING:
    from tools._scopes import Scope

Import = ast.Import | ast.ImportFrom


def _guards(tree: ast.Module, index: ScopeIndex) -> set[int]:
    return {
        id(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and resolve_dotted(dotted(node.test) or "", index.scope_of(node.test))
        == "typing.TYPE_CHECKING"
    }


def _guarded_imports(tree: ast.Module, guards: set[int]) -> list[Import]:
    result: list[Import] = []

    def walk(node: ast.AST, guarded: bool) -> None:
        if isinstance(node, ast.If) and id(node) in guards:
            for child in node.body:
                walk(child, True)
            for child in node.orelse:
                walk(child, guarded)
        else:
            if guarded and isinstance(node, ast.Import | ast.ImportFrom):
                result.append(node)
            for descendant in ast.iter_child_nodes(node):
                walk(descendant, guarded)

    walk(tree, False)
    return result


def guarded_imports(tree: ast.Module, module: str) -> list[Import]:
    """Return imports inside real TYPE_CHECKING bodies, including class declarations."""
    index = ScopeIndex(tree, module.rpartition(".")[0])
    return _guarded_imports(tree, _guards(tree, index))


class _RuntimeTree(ast.NodeTransformer):
    def __init__(self, guards: set[tuple[int, int]]) -> None:
        self.guards = guards

    def visit_If(self, node: ast.If) -> ast.AST:
        if (node.lineno, node.col_offset) in self.guards:
            node.body = [ast.copy_location(ast.Pass(), node)]
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if isinstance(node.target, ast.Name) and node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


class _RuntimeReads(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[ast.Name] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expression is not None:
                self.visit(expression)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)


def _chain(scope: Scope) -> list[Scope]:
    chain = []
    current: Scope | None = scope
    while current is not None:
        if current is scope or current.kind != "class":
            chain.append(current)
        current = current.parent
    return chain


def check_runtime_imports(tree: ast.Module, module: str, key: str) -> list[str]:
    """Reject runtime reads of names whose lexical binding exists only under TYPE_CHECKING."""
    index = ScopeIndex(tree, module.rpartition(".")[0])
    guards = _guards(tree, index)
    typed = set()
    for node in _guarded_imports(tree, guards):
        for alias in node.names:
            local = alias.asname or (
                alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
            )
            scope = index.scope_of(node).find(local)
            if scope is not None:
                typed.add((id(scope), local))
    locations = {
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if id(node) in guards and isinstance(node, ast.If)
    }
    runtime_tree = _RuntimeTree(locations).visit(copy.deepcopy(tree))
    if not isinstance(runtime_tree, ast.Module):
        raise ValueError("runtime import analysis requires a module")
    runtime = ScopeIndex(runtime_tree, module.rpartition(".")[0])
    reads = _RuntimeReads()
    reads.visit(runtime_tree)
    original = {(node.lineno, node.col_offset, node.id): scope for node, scope in index.names}
    problems = []
    for read in reads.names:
        before = original[(read.lineno, read.col_offset, read.id)]
        owner = before.find(read.id)
        if owner is None or (id(owner), read.id) not in typed:
            continue
        depth = _chain(before).index(owner)
        runtime_scope = _chain(runtime.scope_of(read))[depth]
        if read.id not in runtime_scope.bindings:
            problems.append(
                f"type-only-runtime: {key}:{read.lineno}: {read.id} "
                "is bound only under TYPE_CHECKING"
            )
    return problems
