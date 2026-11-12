"""Lexical scopes of a Python module: what a name means at a given use site.

Shared by ``_patch_sites`` (which object does a patch target name?) and
``stale_patch_lint`` (is a module-level import read when the module runs?).
Conservative on purpose: a name bound more than one way in a scope is *ambiguous* and
resolves to nothing, so callers leave such sites alone instead of guessing.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Literal

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Binding:
    """``import``: value is the dotted target. ``param``: value is the annotation text.
    ``mp``: a ``pytest.MonkeyPatch`` object. ``other``: any other binding (def, class, ...)."""

    kind: Literal["import", "param", "mp", "other"]
    value: str = ""


@dataclass
class Scope:
    kind: Literal["module", "function", "class"]
    parent: Scope | None = None
    bindings: dict[str, set[Binding]] = field(default_factory=dict)

    def add(self, name: str, binding: Binding) -> None:
        self.bindings.setdefault(name, set()).add(binding)

    def _chain(self) -> list[Scope]:
        out: list[Scope] = []
        scope: Scope | None = self
        while scope is not None:
            if scope is self or scope.kind != "class":  # class bodies are not closures
                out.append(scope)
            scope = scope.parent
        return out

    def find(self, name: str) -> Scope | None:
        """The scope whose binding of ``name`` is visible here (None if unbound)."""
        return next((s for s in self._chain() if name in s.bindings), None)

    def resolve(self, name: str) -> Binding | None:
        """The one binding of ``name`` visible here; None when unbound or ambiguous."""
        scope = self.find(name)
        if scope is None:
            return None
        found = scope.bindings[name]
        return next(iter(found)) if len(found) == 1 else None

    def hides_import(self, name: str) -> bool:
        """True when ``name`` is an import somewhere that a closer or mixed binding hides here."""
        chain = self._chain()
        found = self.find(name)
        if found is None:
            return False
        if any(b.kind == "import" for b in found.bindings[name]) and len(found.bindings[name]) > 1:
            return True
        return any(
            b.kind == "import"
            for s in chain[chain.index(found) + 1 :]
            for b in s.bindings.get(name, ())
        )

    def deferred(self) -> bool:
        """True when code here runs at call time (inside a function), not at import."""
        scope: Scope | None = self
        while scope is not None:
            if scope.kind == "function":
                return True
            scope = scope.parent
        return False


def dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_monkeypatch_factory(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    text = dotted(node.func) or ""
    return (
        text.endswith("MonkeyPatch")
        or text.endswith("MonkeyPatch.context")
        or (text.startswith("monkeypatch.") and text.endswith(".context"))
    )


class _Collector(ast.NodeVisitor):
    """Names bound directly in one scope; does not enter nested scopes."""

    def __init__(self, scope: Scope, module: Scope) -> None:
        self.scope = scope
        self.module = module

    def _other(self, name: str) -> None:
        self.scope.add(name, Binding("other"))

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.asname:
                self.scope.add(a.asname, Binding("import", a.name))
            else:
                head = a.name.split(".")[0]
                self.scope.add(head, Binding("import", head))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            local = a.asname or a.name
            if node.module and not node.level and a.name != "*":
                self.scope.add(local, Binding("import", f"{node.module}.{a.name}"))
            else:
                self._other(local)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._other(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]  # noqa: N815 - ast visitor name

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._other(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._other(name)
            self.module.add(name, Binding("other"))

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._other(name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._other(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self._other(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self._other(node.name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self._other(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_monkeypatch_factory(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.scope.add(target.id, Binding("mp"))
                else:
                    self.visit(target)
            self.visit(node.value)
        else:
            self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name) and _is_monkeypatch_factory(
                item.context_expr
            ):
                self.scope.add(item.optional_vars.id, Binding("mp"))
                self.visit(item.context_expr)
            else:
                self.visit(item)
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncWith = visit_With  # type: ignore[assignment]  # noqa: N815 - ast visitor name


def _params(args: ast.arguments) -> list[ast.arg]:
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    every += [a for a in (args.vararg, args.kwarg) if a is not None]
    return every


class ScopeIndex:
    """Scopes of a parsed module, and the scope each ``Name`` and ``Call`` is evaluated in."""

    def __init__(self, tree: ast.Module) -> None:
        self.module = Scope("module")
        self.scopes: list[Scope] = [self.module]
        self._at: dict[int, Scope] = {}
        self.names: list[tuple[ast.Name, Scope]] = []
        self.calls: list[tuple[ast.Call, Scope]] = []
        collector = _Collector(self.module, self.module)
        for stmt in tree.body:
            collector.visit(stmt)
        for stmt in tree.body:
            self._visit(stmt, self.module)

    def scope_of(self, node: ast.AST) -> Scope:
        return self._at[id(node)]

    def all_names(self) -> set[str]:
        """Every identifier that could clash with a new module-level name."""
        used = set(dir(builtins))
        for scope in self.scopes:
            used.update(scope.bindings)
        used.update(n.id for n, _ in self.names)
        return used

    def _enter(self, kind: Literal["function", "class"], parent: Scope) -> Scope:
        scope = Scope(kind, parent)
        self.scopes.append(scope)
        return scope

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: Scope) -> None:
        for expr in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self._visit(expr, outer)
        for arg in _params(node.args):
            if arg.annotation is not None:
                self._visit(arg.annotation, outer)
        if node.returns is not None:
            self._visit(node.returns, outer)
        inner = self._enter("function", outer)
        for arg in _params(node.args):
            annotation = (dotted(arg.annotation) or "") if arg.annotation else ""
            inner.add(arg.arg, Binding("param", annotation))
        collector = _Collector(inner, self.module)
        for stmt in node.body:
            collector.visit(stmt)
        for stmt in node.body:
            self._visit(stmt, inner)

    def _class(self, node: ast.ClassDef, outer: Scope) -> None:
        for expr in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
            self._visit(expr, outer)
        inner = self._enter("class", outer)
        collector = _Collector(inner, self.module)
        for stmt in node.body:
            collector.visit(stmt)
        for stmt in node.body:
            self._visit(stmt, inner)

    def _lambda(self, node: ast.Lambda, outer: Scope) -> None:
        for expr in [*node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self._visit(expr, outer)
        inner = self._enter("function", outer)
        for arg in _params(node.args):
            inner.add(arg.arg, Binding("param", ""))
        self._visit(node.body, inner)

    def _visit(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, _FUNCTIONS):
            self._function(node, scope)
            return
        if isinstance(node, ast.ClassDef):
            self._class(node, scope)
            return
        if isinstance(node, ast.Lambda):
            self._lambda(node, scope)
            return
        if isinstance(node, ast.Name):
            self.names.append((node, scope))
        elif isinstance(node, ast.Call):
            self.calls.append((node, scope))
        self._at[id(node)] = scope
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope)
