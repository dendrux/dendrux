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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Binding:
    """``import``: value is the dotted target ("" when a relative import cannot be resolved).
    ``param``: value is the resolved annotation (dotted, "" when unknown).
    ``mp``: a ``pytest.MonkeyPatch`` object. ``other``: any other binding (def, class, ...).

    ``relative`` and ``direct`` describe an import statement: written with dots, and a direct
    child of its scope's body (not under ``if``/``try``/``with``). ``line`` is where it ends.
    None of them takes part in equality: two imports of one target are one binding.
    """

    kind: Literal["import", "param", "mp", "other"]
    value: str = ""
    relative: bool = field(default=False, compare=False)
    direct: bool = field(default=False, compare=False)
    line: int = field(default=0, compare=False)


@dataclass
class Scope:
    kind: Literal["module", "function", "class", "comprehension"]
    parent: Scope | None = None
    bindings: dict[str, set[Binding]] = field(default_factory=dict)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)

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
        chain = self._chain()
        for i, scope in enumerate(chain):
            if name in scope.global_names:
                return chain[-1] if name in chain[-1].bindings else None
            if name in scope.nonlocal_names:
                return next(
                    (s for s in chain[i + 1 :] if s.kind == "function" and name in s.bindings),
                    None,
                )
            if name in scope.bindings:
                return scope
        return None

    def declared(self, name: str) -> bool:
        """Whether lookup of ``name`` follows a global or nonlocal declaration."""
        for scope in self._chain():
            if name in scope.global_names or name in scope.nonlocal_names:
                return True
            if name in scope.bindings:
                break
        return False

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


MONKEYPATCH = frozenset({"pytest.MonkeyPatch", "_pytest.monkeypatch.MonkeyPatch"})


def resolve_dotted(text: str, scope: Scope) -> str | None:
    """``text`` with its first name replaced by what that name imports, if it is an import."""
    head, _, tail = text.partition(".")
    binding = scope.resolve(head)
    if binding is None or binding.kind != "import" or not binding.value:
        return None
    return f"{binding.value}.{tail}" if tail else binding.value


def is_fixture_param(name: str, binding: Binding) -> bool:
    """A parameter that is the pytest ``monkeypatch`` fixture (by name or by annotation)."""
    return binding.kind == "param" and (name == "monkeypatch" or binding.value in MONKEYPATCH)


def _is_monkeypatch_factory(node: ast.expr | None, scope: Scope) -> bool:
    """A call that makes a ``pytest.MonkeyPatch``: the class, its ``context()``, or the
    fixture's ``context()``; resolved through imports, never guessed from spelling."""
    if not isinstance(node, ast.Call):
        return False
    text = dotted(node.func)
    if text is None:
        return False
    full = resolve_dotted(text, scope)
    if full is not None:
        return full in MONKEYPATCH or full.removesuffix(".context") in MONKEYPATCH
    head, _, tail = text.partition(".")
    binding = scope.resolve(head)
    return binding is not None and is_fixture_param(head, binding) and tail == "context"


def _annotation_text(annotation: ast.expr | None) -> str:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return ""
    return (dotted(annotation) or "") if annotation is not None else ""


def resolve_package(package: str | None, level: int, module: str | None) -> str | None:
    """The absolute dotted name of ``from <dots><module> import``; None when it cannot be known."""
    if not level:
        return module
    if package is None:
        return None
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.append(module)
    return ".".join(base)


class _Collector(ast.NodeVisitor):
    """Names bound directly in one scope; does not enter nested scopes."""

    def __init__(
        self, scope: Scope, module: Scope, body: Sequence[ast.stmt] = (), package: str | None = None
    ) -> None:
        self.scope = scope
        self.module = module
        self.package = package
        self.direct = {id(stmt) for stmt in body}
        self.global_names = scope.global_names
        self.nonlocal_names = scope.nonlocal_names

    def _bind(self, name: str, binding: Binding) -> None:
        """Bind ``name`` where Python does: here, or the module / enclosing function it declares."""
        if name in self.global_names:
            self.module.add(name, binding)
            return
        if name in self.nonlocal_names:
            outer = self.scope.parent
            while outer is not None:
                if outer.kind != "class" and name in outer.bindings:
                    outer.add(name, binding)
                    return
                outer = outer.parent
        self.scope.add(name, binding)

    def _other(self, name: str) -> None:
        self._bind(name, Binding("other"))

    def _import(self, local: str, target: str, node: ast.stmt, *, relative: bool = False) -> None:
        binding = Binding(
            "import",
            target,
            relative=relative,
            direct=id(node) in self.direct,
            line=node.end_lineno or node.lineno,
        )
        self._bind(local, binding)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.asname:
                self._import(a.asname, a.name, node)
            else:
                head = a.name.split(".")[0]
                self._import(head, head, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = resolve_package(self.package, node.level, node.module)
        for a in node.names:
            local = a.asname or a.name
            if a.name == "*":
                self._other(local)
            elif base is None:
                self._import(local, "", node, relative=bool(node.level))
            else:
                target = f"{base}.{a.name}" if base else a.name
                self._import(local, target, node, relative=bool(node.level))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._other(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]  # noqa: N815 - ast visitor name

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._other(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def _comprehension(self, node: ast.expr) -> None:
        """Targets stay inside; an assignment expression binds in the enclosing scope."""
        for n in ast.walk(node):
            if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
                self._other(n.target.id)

    visit_ListComp = _comprehension  # noqa: N815 - ast visitor name
    visit_SetComp = _comprehension  # noqa: N815 - ast visitor name
    visit_DictComp = _comprehension  # noqa: N815 - ast visitor name
    visit_GeneratorExp = _comprehension  # noqa: N815 - ast visitor name

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

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

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self._other(node.rest)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self._other(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_monkeypatch_factory(node.value, self.scope):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._bind(target.id, Binding("mp"))
                else:
                    self.visit(target)
            self.visit(node.value)
        else:
            self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name) and _is_monkeypatch_factory(
                item.context_expr, self.scope
            ):
                self._bind(item.optional_vars.id, Binding("mp"))
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

    def __init__(self, tree: ast.Module, package: str | None = None) -> None:
        """``package`` is the dotted package the module sits in (for relative imports);
        None leaves a relative import unresolved (its binding value is empty)."""
        self.package = package
        self.module = Scope("module")
        self.scopes: list[Scope] = [self.module]
        self._at: dict[int, Scope] = {}
        self.names: list[tuple[ast.Name, Scope]] = []
        self.calls: list[tuple[ast.Call, Scope]] = []
        collector = _Collector(self.module, self.module, tree.body, package)
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

    def _enter(self, kind: Literal["function", "class", "comprehension"], parent: Scope) -> Scope:
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
            text = _annotation_text(arg.annotation)
            inner.add(
                arg.arg, Binding("param", (resolve_dotted(text, outer) or "") if text else "")
            )
        collector = _Collector(inner, self.module, node.body, self.package)
        for stmt in node.body:
            collector.visit(stmt)
        for stmt in node.body:
            self._visit(stmt, inner)

    def _class(self, node: ast.ClassDef, outer: Scope) -> None:
        for expr in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
            self._visit(expr, outer)
        inner = self._enter("class", outer)
        collector = _Collector(inner, self.module, node.body, self.package)
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

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp, outer: Scope
    ) -> None:
        """Targets live in a scope of their own; only the first iterable is read outside."""
        self._at[id(node)] = outer
        self._visit(node.generators[0].iter, outer)
        inner = self._enter("comprehension", outer)
        collector = _Collector(inner, self.module, (), self.package)
        for gen in node.generators:
            collector.visit(gen.target)
        for i, gen in enumerate(node.generators):
            if i:
                self._visit(gen.iter, inner)
            self._visit(gen.target, inner)
            for cond in gen.ifs:
                self._visit(cond, inner)
        elements = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        for element in elements:
            self._visit(element, inner)

    def _visit(self, node: ast.AST, scope: Scope) -> None:
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            self._comprehension(node, scope)
            return
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
