"""Stale-patch guard: a test that patches a pure re-export silently patches nothing.

``monkeypatch.setattr("dendrux.runtime.runner.X", fake)`` replaces ``X`` in the *runner*
module only. If the runner merely re-exports ``X`` (imports it and never uses it), no
code reads that binding, the fake is never called and the test still passes. After a
module split this is the worst failure mode for failure-injection tests.

A target is stale when its module imports the name, does not define it, and never
references it in code (``__all__`` does not count). Patching an import the module does use
is the normal patch-where-used pattern and is fine. Pure AST work; the library is not
imported. Shrink-only baseline: ``stale_patch_baseline.json`` (count per test file and target).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from tools import _patch_sites as ps
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _patch_sites as ps
    from tools import _ratchet as r

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "stale_patch_baseline.json"
LIBRARY = "dendrux"


@dataclass(frozen=True)
class StaleSite:
    path: Path
    lineno: int
    module: str
    attr: str


@dataclass(frozen=True)
class _Module:
    defined: frozenset[str]
    reexported: frozenset[str]  # imported, not defined, never used in code


def _module_file(src: Path, module: str) -> Path | None:
    base = src.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_level(body: Sequence[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for node in body:
        out.append(node)
        if isinstance(node, ast.If | ast.Try):
            out.extend(_top_level(node.body))
            out.extend(_top_level(node.orelse))
            if isinstance(node, ast.Try):
                out.extend(_top_level(node.finalbody))
                for handler in node.handlers:
                    out.extend(_top_level(handler.body))
    return out


def _targets(node: ast.stmt) -> list[str]:
    raw: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        raw = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        raw = [node.target]
    names: list[str] = []
    while raw:
        t = raw.pop()
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Tuple | ast.List):
            raw.extend(t.elts)
    return names


def analyse(source: str) -> _Module:
    tree = ast.parse(source)
    defined: set[str] = set()
    imported: set[str] = set()
    for node in _top_level(tree.body):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            imported.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
        else:
            defined.update(_targets(node))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return _Module(frozenset(defined), frozenset(imported - defined - used))


class _Modules:
    def __init__(self, src: Path) -> None:
        self.src = src
        self._cache: dict[str, _Module | None] = {}

    def get(self, module: str) -> _Module | None:
        if module not in self._cache:
            path = _module_file(self.src, module)
            self._cache[module] = analyse(path.read_text(encoding="utf-8")) if path else None
        return self._cache[module]

    def split(self, dotted: str) -> tuple[str, str] | None:
        """Longest library module prefix of ``dotted`` and the next component (the attribute)."""
        parts = dotted.split(".")
        if parts[0] != LIBRARY:
            return None
        for cut in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:cut])
            if self.get(module) is not None:
                return module, parts[cut]
        return None


def find_stale(tests: Path, src: Path) -> list[StaleSite]:
    """Every patch site under ``tests`` whose target is a re-export in ``src``."""
    modules = _Modules(src)
    found: list[StaleSite] = []
    for path in sorted(tests.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for site in ps.find_sites(tree):
            hit = modules.split(site.dotted) if site.kind == "string" else None
            if site.kind == "alias" and modules.get(site.dotted) is not None:
                hit = (site.dotted, site.attr)  # only a module alias; a class alias is not
            info = modules.get(hit[0]) if hit else None
            if hit and info and hit[1] in info.reexported:
                found.append(StaleSite(path, site.lineno, hit[0], hit[1]))
    return found


def _key(site: StaleSite, root: Path) -> str:
    try:
        rel = site.path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = site.path
    return f"{rel.as_posix()}::{site.module}.{site.attr}"


def _counts(sites: Sequence[StaleSite], root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in sites:
        key = _key(site, root)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser, ("check", "stamp", "report"))
    parser.add_argument("--tests", type=Path, default=PACKAGE_ROOT / "tests")
    parser.add_argument("--src", type=Path, default=PACKAGE_ROOT / "src")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    sites = find_stale(args.tests, args.src)
    root = args.tests.resolve().parent
    counts = _counts(sites, root)
    print(f"stale patches: {len(sites)} site(s) in {len(counts)} file/target pair(s)")
    for site in sites:
        print(
            f"{_key(site, root).split('::')[0]}:{site.lineno}: patches {site.module}.{site.attr}, "
            "a re-export the module never uses; patch the defining module instead"
        )
    if args.command == "report":
        return 0
    # A missing baseline file means "none allowed": today's tree is clean, so none is committed.
    old = r.baseline_or_none(args.baseline) or {}
    comparison = r.compare_counts(counts, old)
    return r.finish(
        args.command,
        label="stale-patch",
        comparison=comparison,
        new_baseline=counts,
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda key: f"{key} (re-export patch: it silently does nothing)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
