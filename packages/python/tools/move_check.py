"""AST pure-move checker: prove a refactor only moved code.

Parses the old module at a base git ref and the module(s) as they are now (the old
path, if it still exists, plus the new paths). Passes only when:

* every top-level definition (function, class, constant, other statement) of the old
  module is present in exactly one module afterwards, with an identical AST
  (``ast.dump``, no locations, docstrings included);
* nothing extra appears (new names need ``--allow-new``);
* a method may move out of a class into a *mixin class* that is a new top-level class
  and a direct base of the original class; the method body must be identical.

Ignored on purpose: import statements (including ``if TYPE_CHECKING`` blocks that only
import), the module docstring and ``__all__``. Everything else is compared.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

_IMPORTS = (ast.Import, ast.ImportFrom)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass
class MoveReport:
    """``problems`` is empty when the refactor is a pure move; ``moved`` maps name -> module."""

    problems: list[str] = field(default_factory=list)
    moved: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class _Member:
    key: str
    node: ast.stmt


def _dump(node: ast.AST) -> str:
    return ast.dump(node, indent=1)


def _only_imports(node: ast.stmt) -> bool:
    if isinstance(node, _IMPORTS):
        return True
    if isinstance(node, ast.If | ast.Try):
        branches = [node.body, node.orelse]
        if isinstance(node, ast.Try):
            branches += [h.body for h in node.handlers] + [node.finalbody]
        return all(_only_imports(s) for branch in branches for s in branch)
    return False


def _target_names(node: ast.stmt) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    names: list[str] = []
    stack = list(targets)
    while stack:
        t = stack.pop(0)
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Tuple | ast.List):
            stack.extend(t.elts)
    return names


def _key(node: ast.stmt) -> str | None:
    """Stable name of a statement; None when it is ignored."""
    if _only_imports(node):
        return None
    if isinstance(node, _DEFS):
        return node.name
    names = _target_names(node)
    if names:
        if names == ["__all__"]:
            return None
        return ",".join(names)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return None  # a stray string / docstring
    return f"<statement {_dump(node)[:60]!r}>"


def _members(body: Iterable[ast.stmt]) -> list[_Member]:
    out: list[_Member] = []
    for node in body:
        key = _key(node)
        if key is not None:
            out.append(_Member(key, node))
    return out


def _grouped(members: Iterable[_Member]) -> dict[str, list[ast.stmt]]:
    groups: dict[str, list[ast.stmt]] = defaultdict(list)
    for m in members:
        groups[m.key].append(m.node)
    return groups


def _unified(label: str, old: str, new: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"{label} (base)", f"{label} (now)", lineterm="", n=2
    )
    return "\n".join(lines)


def _name_of(base: ast.expr) -> str | None:
    return base.id if isinstance(base, ast.Name) else None


def _body_members(cls: ast.ClassDef) -> list[_Member]:
    """Class-body statements keyed by name; the docstring is a member too."""
    out: list[_Member] = []
    for node in cls.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            out.append(_Member("<docstring>", node))
        elif isinstance(node, _DEFS):
            out.append(_Member(node.name, node))
        elif names := _target_names(node):
            out.append(_Member(",".join(names), node))
        else:
            out.append(_Member(f"<statement {_dump(node)[:60]!r}>", node))
    return out


def _header(node: ast.ClassDef, bases: list[ast.expr]) -> str:
    """The class without its body: bases, keywords, decorators."""
    shell = copy.copy(node)
    shell.bases = bases
    shell.body = [ast.Pass()]
    return _dump(shell)


@dataclass
class _Host:
    module: str
    node: ast.ClassDef


def _compare_class(
    report: MoveReport,
    old: ast.ClassDef,
    new: ast.ClassDef,
    module: str,
    hosts: Mapping[str, _Host],
) -> None:
    """Compare a class whose methods may have moved into direct mixin bases."""
    mixin_bases = [n for b in new.bases if (n := _name_of(b)) is not None and n in hosts]
    plain = [b for b in new.bases if _name_of(b) not in hosts]
    if _header(old, list(old.bases)) != _header(new, plain):
        report.problems.append(
            f"changed: class {old.name} header (bases, keywords, decorators) in {module}\n"
            + _unified(old.name, _header(old, list(old.bases)), _header(new, plain))
        )
    here = {m.key: m.node for m in _body_members(new)}
    moved: dict[str, tuple[str, ast.stmt]] = {}
    for host_name in mixin_bases:
        host = hosts[host_name]
        for member in _body_members(host.node):
            if member.key == "<docstring>":
                continue
            if member.key in moved or member.key in here:
                report.problems.append(f"duplicated: {old.name}.{member.key} (mixin {host_name})")
            else:
                moved[member.key] = (host_name, member.node)
    old_keys = set()
    for member in _body_members(old):
        old_keys.add(member.key)
        qual = f"{old.name}.{member.key}"
        if member.key in here:
            now, where = here[member.key], module
        elif member.key in moved:
            where, now = moved[member.key]
            report.moved[qual] = hosts[where].module
        else:
            report.problems.append(f"dropped: {qual} is in neither {module} nor its mixins")
            continue
        if _dump(member.node) != _dump(now):
            report.problems.append(
                f"changed: {qual} in {where}\n" + _unified(qual, _dump(member.node), _dump(now))
            )
    for key in [*here, *moved]:
        if key not in old_keys:
            report.problems.append(f"extra: {old.name}.{key} is not in the base class")


def _find_hosts(
    after_groups: Mapping[str, Mapping[str, list[ast.stmt]]],
    old_groups: Mapping[str, list[ast.stmt]],
    allow_new: Collection[str],
    report: MoveReport,
) -> dict[str, _Host]:
    """New top-level classes that an old class names as a direct base are mixin hosts."""
    bases: set[str] = set()
    for groups in after_groups.values():
        for key, nodes in groups.items():
            if key in old_groups:
                bases.update(
                    n
                    for node in nodes
                    if isinstance(node, ast.ClassDef)
                    for b in node.bases
                    if (n := _name_of(b))
                )
    hosts: dict[str, _Host] = {}
    for module, groups in after_groups.items():
        for key, nodes in groups.items():
            node = nodes[0]
            if key in old_groups or key not in bases or not isinstance(node, ast.ClassDef):
                continue
            stray = [n for b in node.bases if (n := _name_of(b)) not in allow_new]
            if stray or len(nodes) > 1:
                report.problems.append(f"extra: mixin {key} in {module} has other bases or copies")
            hosts[key] = _Host(module, node)
    return hosts


def check_move(
    old_source: str,
    after: Mapping[str, str],
    *,
    old_key: str = "old",
    allow_new: Collection[str] = (),
    require_reexports: bool = False,
) -> MoveReport:
    """Compare the base source of one module with its pieces now (module key -> source).

    ``old_key`` names the entry of ``after`` that is the old module itself (it may be absent).
    """
    report = MoveReport()
    old_groups = _grouped(_members(ast.parse(old_source).body))
    after_trees = {name: ast.parse(src) for name, src in after.items()}
    after_groups = {name: _grouped(_members(tree.body)) for name, tree in after_trees.items()}
    hosts = _find_hosts(after_groups, old_groups, allow_new, report)
    homes: dict[str, list[str]] = defaultdict(list)
    for module, groups in after_groups.items():
        for key in groups:
            homes[key].append(module)

    for key, old_nodes in old_groups.items():
        found = homes.get(key, [])
        if len(found) != 1:
            word = "dropped" if not found else "duplicated"
            where = "no module" if not found else ", ".join(sorted(found))
            report.problems.append(f"{word}: {key} is in {where} after the move")
            continue
        module = found[0]
        new_nodes = after_groups[module][key]
        if len(old_nodes) != len(new_nodes):
            report.problems.append(
                f"changed: {key} has {len(old_nodes)} definition(s) at base, {len(new_nodes)} now"
            )
            continue
        for old_node, new_node in zip(old_nodes, new_nodes, strict=True):
            if isinstance(old_node, ast.ClassDef) and isinstance(new_node, ast.ClassDef):
                _compare_class(report, old_node, new_node, module, hosts)
            elif _dump(old_node) != _dump(new_node):
                report.problems.append(
                    f"changed: {key} in {module}\n"
                    + _unified(key, _dump(old_node), _dump(new_node))
                )
        if module != old_key:
            report.moved[key] = module
    for module, groups in after_groups.items():
        report.problems.extend(
            f"extra: {key} in {module} is not in the base module"
            for key in groups
            if key not in old_groups and key not in allow_new and key not in hosts
        )
    if require_reexports:
        _check_reexports(report, after_trees.get(old_key))
    return report


def _bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    for node in tree.body:
        if (key := _key(node)) is not None:
            names.update(key.split(","))
    return names


def _check_reexports(report: MoveReport, old_tree: ast.Module | None) -> None:
    bound = _bound_names(old_tree) if old_tree is not None else set()
    for key, module in sorted(report.moved.items()):
        if "." in key or key.startswith("_") or key in bound:
            continue
        report.problems.append(
            f"re-export: public name {key} moved to {module}; old module lost it"
        )


def _git_show(base: str, path: str) -> str:
    proc = subprocess.run(  # noqa: S603 - argument list, fixed git subcommand
        ["git", "show", f"{base}:./{path}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise SystemExit(f"move_check: cannot read {path} at {base}: {proc.stderr.strip()}")
    return proc.stdout


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="git ref holding the original module")
    parser.add_argument("--old", required=True, type=Path, help="old module path (relative to cwd)")
    parser.add_argument("--new", nargs="+", default=[], type=Path, help="new module paths")
    parser.add_argument("--allow-new", nargs="*", default=[], metavar="NAME")
    parser.add_argument("--require-reexports", action="store_true")
    parser.add_argument("--emit-map", type=Path, help="write {old_module: {name: new_module}} JSON")
    return parser.parse_args(argv)


def _patch_map(old: Path, moved: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Top-level moved names only, keyed by module (methods are not patch targets)."""
    names = {k: r.module_name(Path(v)) for k, v in moved.items() if "." not in k}
    return {r.module_name(old): dict(sorted(names.items()))} if names else {}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    old_key = str(args.old)
    after: dict[str, str] = {str(p): p.read_text(encoding="utf-8") for p in args.new}
    if args.old.is_file():
        after[old_key] = args.old.read_text(encoding="utf-8")
    report = check_move(
        _git_show(args.base, old_key),
        after,
        old_key=old_key,
        allow_new=args.allow_new,
        require_reexports=args.require_reexports,
    )
    for key, module in sorted(report.moved.items()):
        print(f"moved: {key} -> {module}")
    for problem in report.problems:
        print(problem)
    if report.problems:
        print(f"move_check: FAILED, {len(report.problems)} problem(s); this is not a pure move")
        return 1
    if args.emit_map:
        args.emit_map.write_text(json.dumps(_patch_map(args.old, report.moved), indent=1) + "\n")
    print(f"move_check: ok, {len(report.moved)} definition(s) moved, nothing else changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
