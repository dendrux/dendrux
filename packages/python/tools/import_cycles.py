"""Import-cycle gate: runtime import cycles in ``dendrux``, ratcheted.

The module graph uses **top-level, runtime** imports only. Importing ``a.b.c`` first runs
the ``__init__`` of ``a`` and ``a.b``, so those are edges too (not for a module that sits
inside that package, which is already running). Imports under ``if TYPE_CHECKING:`` (also
through ``from typing import TYPE_CHECKING as TC``) and imports inside functions are
ignored. A function-level import is a deliberate way to break a cycle, but it is also hidden
coupling, so ``report`` lists the cycles that include them too, for information only. A
cycle is a strongly connected component of two or more modules.

Baseline ``tools/import_cycles_baseline.json``: ``{"cycles": [[module, ...], ...]}``.
A cycle that is not inside a baseline cycle fails, so does a module that joins one and so
does a new edge that merges two baseline cycles. A cycle may shrink or disappear.

    python tools/import_cycles.py check
    python tools/import_cycles.py report         # also shows the function-level view
    python tools/import_cycles.py stamp          # after improving; refuses to add cycles
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

BASELINE = Path(__file__).resolve().parent / "import_cycles_baseline.json"
Graph = dict[str, set[str]]


_TYPING_MODULES = frozenset({"typing", "typing_extensions"})


def _type_checking_names(tree: ast.Module) -> set[str]:
    """Names this module binds to ``typing.TYPE_CHECKING``, including ``as`` aliases."""
    names = {"TYPE_CHECKING"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "TYPE_CHECKING"
            )
    return names


def _is_type_checking(test: ast.expr, names: set[str]) -> bool:
    if isinstance(test, ast.Name):
        return test.id in names
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


class _Imports(ast.NodeVisitor):
    """Import statements split into module-level ones and ones nested in functions."""

    def __init__(self, type_checking_names: set[str]) -> None:
        self.top: list[ast.Import | ast.ImportFrom] = []
        self.lazy: list[ast.Import | ast.ImportFrom] = []
        self._depth = 0
        self._names = type_checking_names

    def _add(self, node: ast.Import | ast.ImportFrom) -> None:
        (self.lazy if self._depth else self.top).append(node)

    visit_Import = _add  # noqa: N815
    visit_ImportFrom = _add  # noqa: N815

    def _function(self, node: ast.AST) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_FunctionDef = _function  # noqa: N815
    visit_AsyncFunctionDef = _function  # noqa: N815
    visit_Lambda = _function  # noqa: N815

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if _is_type_checking(node.test, self._names):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)


def _targets(
    node: ast.Import | ast.ImportFrom, module: str, is_package: bool, known: set[str]
) -> set[str]:
    found: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in known:
                found.add(alias.name)
        return found
    base = node.module or ""
    if node.level:
        package = module.split(".") if is_package else module.split(".")[:-1]
        package = package[: len(package) - (node.level - 1)]
        base = ".".join(package + ([node.module] if node.module else []))
    for alias in node.names:
        submodule = f"{base}.{alias.name}"
        if alias.name != "*" and submodule in known:
            found.add(submodule)
        elif base in known:
            found.add(base)
    return found


def _ancestors(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _package_initializers(targets: set[str], module: str, known: set[str]) -> set[str]:
    """Importing ``a.b.c`` runs ``a`` and ``a.b`` first. The packages ``module`` itself sits
    in are already executing, so they are not a new dependency (that would also add the
    root package, which every module sits in)."""
    running = {module, *_ancestors(module)}
    return {
        pkg
        for target in targets
        for pkg in _ancestors(target)
        if pkg in known and pkg not in running
    }


def build_graph(src: Path = r.SRC, include_function_level: bool = False) -> Graph:
    files = {}
    for path in r.iter_sources(src):
        files[r.module_name(path.resolve())] = path
    known = set(files)
    graph: Graph = {}
    for module, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _Imports(_type_checking_names(tree))
        visitor.visit(tree)
        nodes = visitor.top + (visitor.lazy if include_function_level else [])
        edges: set[str] = set()
        for node in nodes:
            edges |= _targets(node, module, path.name == "__init__.py", known)
        edges |= _package_initializers(edges, module, known)
        edges.discard(module)
        graph[module] = edges
    return graph


def find_cycles(graph: Graph) -> list[list[str]]:
    """Strongly connected components of two or more modules (iterative Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []
    counter = 0
    for root in sorted(graph):
        if root in index:
            continue
        work: list[tuple[str, Any]] = [(root, iter(sorted(graph.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    cycles.append(sorted(component))
    return sorted(cycles)


def compare(current: list[list[str]], baseline: list[list[str]]) -> r.CountComparison:
    result = r.CountComparison()
    allowed = [set(cycle) for cycle in baseline]
    for cycle in current:
        members = set(cycle)
        if any(members <= known for known in allowed):
            continue
        joined = [known for known in allowed if members & known]
        if not joined:
            result.new.append("new cycle: " + " -> ".join(cycle))
        else:
            extra = sorted(members - set().union(*joined)) or sorted(members)
            result.new.append("cycle grew or merged, via: " + ", ".join(extra))
    seen = [set(c) for c in current]
    for known in baseline:
        size = len(set(known))
        covered = [c for c in seen if c <= set(known)]
        if not covered or max(len(c) for c in covered) < size:
            result.improved.append(f"cycle of {size} modules shrank or went: {known[0]}, ...")
    return result


def _print_report(src: Path) -> None:
    for title, lazy in (("runtime (top-level) imports", False), ("including function-level", True)):
        cycles = find_cycles(build_graph(src, include_function_level=lazy))
        sizes = ", ".join(str(len(c)) for c in sorted(cycles, key=len, reverse=True)) or "none"
        print(f"{title}: {len(cycles)} cycle(s); sizes: {sizes}")
        if lazy:
            for cycle in cycles:
                print(f"  [{len(cycle)}] " + ", ".join(cycle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser, ("check", "stamp", "report"))
    parser.add_argument("--src", type=Path, default=r.SRC)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    src = args.src.resolve()
    if args.command == "report":
        _print_report(src)
        return 0
    cycles = find_cycles(build_graph(src))
    old = r.baseline_or_none(args.baseline)
    comparison = None if old is None else compare(cycles, old["cycles"])
    code = r.finish(
        args.command,
        label="import cycles",
        comparison=comparison,
        new_baseline={"cycles": cycles},
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=str,
    )
    if args.command == "check":
        print(f"{len(cycles)} runtime cycle(s) now; run 'import_cycles.py report' for the rest")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
