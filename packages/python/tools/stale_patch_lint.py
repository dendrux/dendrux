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
    from tools import _scopes as sc
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _patch_sites as ps
    from tools import _ratchet as r
    from tools import _scopes as sc

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
class Scan:
    stale: list[StaleSite]
    unresolved: list[tuple[Path, int, str]]  # path, line, note: a human must look at these


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


def analyse(source: str, package: str | None = None) -> _Module:
    """Names the module only imports, and none of its code reads when it runs.

    A read counts when it sits inside a function (so it happens at call time) and resolves to
    the module-level binding, not to a parameter or local of the same spelling. An import-time
    copy (``Y = X``, a decorator, a default argument, a base class) is not a read of ``X``
    later, so replacing ``X`` afterwards reaches nothing.
    """
    index = sc.ScopeIndex(ast.parse(source), package)
    module = index.module
    imported = {n for n, b in module.bindings.items() if all(x.kind == "import" for x in b)}
    used = {
        node.id
        for node, scope in index.names
        if isinstance(node.ctx, ast.Load) and scope.deferred() and scope.find(node.id) is module
    }
    return _Module(frozenset(module.bindings) - imported, frozenset(imported - used))


class _Modules:
    def __init__(self, src: Path) -> None:
        self.src = src
        self._cache: dict[str, _Module | None] = {}

    def get(self, module: str) -> _Module | None:
        if module not in self._cache:
            path = _module_file(self.src, module)
            package = module if path and path.name == "__init__.py" else module.rpartition(".")[0]
            self._cache[module] = (
                analyse(path.read_text(encoding="utf-8"), package) if path else None
            )
        return self._cache[module]

    def split(self, dotted: str) -> tuple[str, str, str] | None:
        """Longest library module prefix of ``dotted``, the next component, and what follows."""
        parts = dotted.split(".")
        if parts[0] != LIBRARY:
            return None
        for cut in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:cut])
            if self.get(module) is not None:
                return module, parts[cut], ".".join(parts[cut + 1 :])
        return None


def _package_of(path: Path) -> str | None:
    """Dotted package of a test file: the directories above it that hold an ``__init__.py``."""
    parts: list[str] = []
    folder = path.resolve().parent
    while (folder / "__init__.py").is_file():
        parts.append(folder.name)
        folder = folder.parent
    return ".".join(reversed(parts)) or None


def scan(tests: Path, src: Path) -> Scan:
    """Every stale patch site under ``tests`` (target is a re-export in ``src``), and every
    site the model cannot resolve."""
    modules = _Modules(src)
    found: list[StaleSite] = []
    unresolved: list[tuple[Path, int, str]] = []
    for path in sorted(tests.rglob("*.py")):
        try:
            index = sc.ScopeIndex(ast.parse(path.read_text(encoding="utf-8")), _package_of(path))
        except SyntaxError:
            continue
        for site in ps.find_sites(index):
            if site.kind == "unresolved":
                unresolved.append((path, site.lineno, site.note))
            hit: tuple[str, str] | None = None
            if site.kind == "string":
                split = modules.split(site.dotted)
                # A tail means an attribute *below* the name is patched: that mutates the shared
                # object, which the defining module sees, so it is not a stale facade patch.
                if split and not split[2]:
                    hit = split[:2]
            elif site.kind == "alias" and modules.get(site.dotted) is not None:
                hit = (site.dotted, site.attr)  # only a module alias; a class alias is not
            info = modules.get(hit[0]) if hit else None
            if hit and info and hit[1] in info.reexported:
                found.append(StaleSite(path, site.lineno, hit[0], hit[1]))
    return Scan(found, unresolved)


def find_stale(tests: Path, src: Path) -> list[StaleSite]:
    """Every patch site under ``tests`` whose target is a re-export in ``src``."""
    return scan(tests, src).stale


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
    result = scan(args.tests, args.src)
    sites = result.stale
    root = args.tests.resolve().parent
    counts = _counts(sites, root)
    print(f"stale patches: {len(sites)} site(s) in {len(counts)} file/target pair(s)")
    for site in sites:
        print(
            f"{_key(site, root).split('::')[0]}:{site.lineno}: patches {site.module}.{site.attr}, "
            "a re-export the module never uses; patch the defining module instead"
        )
    for path, lineno, note in result.unresolved:
        print(f"{path}:{lineno}: unresolved patch site, manual review needed: {note}")
    if args.command == "report":
        return 0
    if result.unresolved:
        print(f"stale-patch: FAILED, {len(result.unresolved)} site(s) could not be resolved")
        return 1
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
