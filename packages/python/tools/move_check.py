"""AST pure-move checker: prove a refactor only moved code. It fails closed.

Compares the old module at a base git ref with the old module as it is now plus the new
modules. It passes only when, apart from the small explicit list of allowed edits below,
nothing changed:

* every top-level statement of the old module is present exactly once afterwards, in
  exactly one module, with an identical ``ast.dump`` (docstrings, decorators, defaults);
* the statements of each module keep the relative order they had at base, and repeated
  names (overloads, property setters, redefinitions) keep their count and order;
* only function and class definitions, and constant assignments, may move. Any other
  statement (calls, ``if``/``try``/``for``/``with``, augmented assignments, ``__all__``) must
  stay in the old module, unchanged. An assignment whose value calls something is such a
  statement unless its target is named in ``--allow-effects``;
* ``from __future__`` imports are the same in every module;
* methods may leave a class for a *mixin*: a new top-level class with no decorator, keyword
  or base other than ``--allow-new`` names, listed **before** the original bases of the
  class (so method resolution keeps its order). The remaining class body must equal the
  base body minus the moved methods, in order.

Allowed edits (everything else is a mismatch): import statements and ``if TYPE_CHECKING:``
blocks that only import; the module docstring; a literal ``__all__`` list in a *new* module
(the old module's ``__all__`` must stay identical and every name in it must still be bound
at module scope). ``--require-reexports`` also checks that each moved public name is imported
by an unconditional, absolute, module-level ``from <new module> import name`` in the old module.

Not proved: import order and cycles (see ``make arch`` / ``make cycles``), the effect of
decorators that register things, and the contents of ``--allow-new`` definitions (review them).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

_IMPORTS = (ast.Import, ast.ImportFrom)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class MoveReport:
    """``problems`` is empty when the refactor is a pure move; ``moved`` maps name -> module."""

    problems: list[str] = field(default_factory=list)
    moved: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class _Item:
    key: str
    dump: str | None  # None: compared separately (classes at module level)
    node: ast.stmt
    movable: bool


@dataclass(frozen=True)
class _Host:
    module: str
    node: ast.ClassDef


def _dump(node: ast.AST) -> str:
    return ast.dump(node, indent=1)


def _unified(label: str, old: str, new: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"{label} (base)", f"{label} (now)", lineterm="", n=2
    )
    return "\n".join(lines)


def _name_of(base: ast.expr) -> str | None:
    return base.id if isinstance(base, ast.Name) else None


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_type_checking_imports(node: ast.stmt) -> bool:
    """``if TYPE_CHECKING:`` whose body only imports; the only guard ignored."""
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    named = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )
    return named and all(isinstance(n, _IMPORTS) for n in node.body)


def _target_names(node: ast.stmt) -> list[str] | None:
    """Names assigned by a plain assignment; None when it is not one (or targets more)."""
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return None
    names: list[str] = []
    while targets:
        t = targets.pop(0)
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Tuple | ast.List):
            targets.extend(t.elts)
        else:
            return None
    return names


def _has_effect(expr: ast.expr | None) -> bool:
    return expr is not None and any(
        isinstance(n, ast.Call | ast.Await | ast.Yield | ast.YieldFrom | ast.NamedExpr)
        for n in ast.walk(expr)
    )


def _literal_strings(expr: ast.expr | None) -> list[str] | None:
    if isinstance(expr, ast.List | ast.Tuple) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in expr.elts
    ):
        return [str(e.value) for e in expr.elts if isinstance(e, ast.Constant)]
    return None


def _module_items(tree: ast.Module, *, is_old: bool, allow_effects: Collection[str]) -> list[_Item]:
    items: list[_Item] = []
    for i, node in enumerate(tree.body):
        if isinstance(node, _IMPORTS) or _is_type_checking_imports(node):
            continue
        if i == 0 and _is_docstring(node):
            continue
        if isinstance(node, _FUNCTIONS):
            items.append(_Item(node.name, _dump(node), node, True))
        elif isinstance(node, ast.ClassDef):
            items.append(_Item(node.name, None, node, True))
        elif (names := _target_names(node)) is not None:
            key = ",".join(names)
            value = getattr(node, "value", None)
            if key == "__all__" and not is_old and _literal_strings(value) is not None:
                continue  # a destination module's own export list is its business
            safe = not _has_effect(value) or set(names) <= set(allow_effects)
            items.append(_Item(key, _dump(node), node, safe and key != "__all__"))
        else:
            items.append(_Item(f"<{type(node).__name__}>", _dump(node), node, False))
    return items


def _class_items(cls: ast.ClassDef) -> list[_Item]:
    items: list[_Item] = []
    for i, node in enumerate(cls.body):
        if i == 0 and _is_docstring(node):
            items.append(_Item("<docstring>", _dump(node), node, False))
        elif isinstance(node, _FUNCTIONS):
            items.append(_Item(node.name, _dump(node), node, True))
        elif isinstance(node, ast.ClassDef):
            items.append(_Item(node.name, _dump(node), node, False))
        elif (names := _target_names(node)) is not None:
            items.append(_Item(",".join(names), _dump(node), node, False))
        else:
            items.append(_Item(f"<{type(node).__name__}>", _dump(node), node, False))
    return items


def _place(
    label: str,
    old: Sequence[_Item],
    containers: Mapping[str, Sequence[_Item]],
    home: str,
    report: MoveReport,
    allowed_extra: Collection[str] = (),
) -> dict[str, str]:
    """Match base items to their new containers; returns key -> container.

    Reports dropped, duplicated, changed, extra, immovable-but-moved and reordered items.
    """
    old_at: dict[str, list[tuple[int, _Item]]] = defaultdict(list)
    for i, it in enumerate(old):
        old_at[it.key].append((i, it))
    new_at: dict[str, dict[str, list[_Item]]] = defaultdict(dict)
    for name, items in containers.items():
        for it in items:
            new_at[it.key].setdefault(name, []).append(it)
    where: dict[str, str] = {}
    for key, olds in old_at.items():
        homes = new_at.get(key, {})
        if len(homes) != 1:
            word = "dropped" if not homes else "duplicated"
            places = ", ".join(sorted(homes)) or "no module"
            report.problems.append(f"{word}: {label}{key} is in {places} after the change")
            continue
        name, news = next(iter(homes.items()))
        if len(news) != len(olds):
            report.problems.append(
                f"changed: {label}{key} has {len(olds)} definition(s) at base, {len(news)} now"
            )
            continue
        for (_, o), n in zip(olds, news, strict=True):
            if o.dump is not None and o.dump != n.dump:
                report.problems.append(
                    f"changed: {label}{key} in {name}\n"
                    + _unified(label + key, o.dump, n.dump or "")
                )
        if name != home and not all(o.movable for _, o in olds):
            report.problems.append(f"immovable: {label}{key} must stay in {home}, found in {name}")
        where[key] = name
    for key, homes in new_at.items():
        if key not in old_at and key not in allowed_extra:
            report.problems.extend(f"extra: {label}{key} in {n} is not in the base" for n in homes)
    _check_order(label, old_at, containers, where, report)
    return where


def _check_order(
    label: str,
    old_at: Mapping[str, Sequence[tuple[int, _Item]]],
    containers: Mapping[str, Sequence[_Item]],
    where: Mapping[str, str],
    report: MoveReport,
) -> None:
    for name, items in containers.items():
        seen: dict[str, int] = defaultdict(int)
        last = -1
        for it in items:
            if where.get(it.key) != name:
                continue
            index = old_at[it.key][seen[it.key]][0]
            seen[it.key] += 1
            if index < last:
                report.problems.append(
                    f"reordered: {label}{it.key} in {name} now comes before a statement "
                    "that preceded it at base"
                )
                break
            last = index


def _header(node: ast.ClassDef) -> list[str]:
    return [
        *(_dump(d) for d in node.decorator_list),
        *(_dump(k) for k in node.keywords),
        *(_dump(t) for t in getattr(node, "type_params", [])),
    ]


def _compare_class(
    old: ast.ClassDef,
    new: ast.ClassDef,
    module: str,
    hosts: Mapping[str, _Host],
    used: dict[str, str],
    report: MoveReport,
) -> None:
    leading: list[str] = []
    for base in new.bases:
        name = _name_of(base)
        if name is None or name not in hosts:
            break
        leading.append(name)
    rest = [_dump(b) for b in new.bases[len(leading) :]]
    if rest != [_dump(b) for b in old.bases] or len(set(leading)) != len(leading):
        report.problems.append(
            f"changed: class {old.name} bases in {module}: mixins must come first, once each, "
            "and the original bases stay unchanged"
        )
    if _header(old) != _header(new):
        report.problems.append(f"changed: class {old.name} decorators/keywords in {module}")
    here, base_items = _class_items(new), _class_items(old)
    if (
        leading
        and [it.key for it in here] == ["<Pass>"]
        and "<Pass>" not in {it.key for it in base_items}
    ):
        here = []  # the `pass` that keeps a class whose methods all moved syntactically valid
    containers: dict[str, list[_Item]] = {"<class>": here}
    owner = {"<class>": module}
    for name in leading:
        if name in used and used[name] != old.name:
            report.problems.append(f"extra: mixin {name} is a base of {used[name]} and {old.name}")
        used[name] = old.name
        host_items = _class_items(hosts[name].node)
        if host_items and host_items[0].key == "<docstring>":
            host_items = host_items[1:]
        containers[name] = host_items
        owner[name] = hosts[name].module
    where = _place(f"{old.name}.", base_items, containers, "<class>", report)
    for key, name in where.items():
        if name != "<class>":
            report.moved[f"{old.name}.{key}"] = owner[name]


def _find_hosts(
    old_keys: Collection[str],
    modules: Mapping[str, list[_Item]],
    allow_new: Collection[str],
    report: MoveReport,
) -> dict[str, _Host]:
    """New top-level classes that an old class names as a base are mixin hosts."""
    base_names = {
        n
        for items in modules.values()
        for it in items
        if it.key in old_keys and isinstance(it.node, ast.ClassDef)
        for b in it.node.bases
        if (n := _name_of(b))
    }
    defined: dict[str, list[_Host]] = defaultdict(list)
    for module, items in modules.items():
        for it in items:
            node = it.node
            if (
                isinstance(node, ast.ClassDef)
                and it.key not in old_keys
                and it.key in base_names
                and it.key not in allow_new
            ):
                defined[it.key].append(_Host(module, node))
    hosts: dict[str, _Host] = {}
    for name, found in defined.items():
        if len(found) > 1:
            places = ", ".join(sorted(h.module for h in found))
            report.problems.append(
                f"extra: mixin {name} is defined in {places}; the owner is ambiguous"
            )
            continue
        node = found[0].node
        stray = [b for b in node.bases if _name_of(b) not in allow_new]
        if node.decorator_list or node.keywords or stray or getattr(node, "type_params", []):
            report.problems.append(
                f"extra: mixin {name} in {found[0].module} has decorators, keywords or bases "
                "(list a plain base with --allow-new)"
            )
            continue
        hosts[name] = found[0]
    return hosts


def _future(tree: ast.Module) -> frozenset[str]:
    return frozenset(
        a.name
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
        for a in n.names
    )


def _runtime_bindings(tree: ast.Module) -> dict[str, tuple[str, str] | None]:
    """Unconditional module-scope bindings: name -> (source module, name) for imports."""
    out: dict[str, tuple[str, str] | None] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = None
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                src = f"{'.' * node.level}{node.module or ''}"
                out[a.asname or a.name] = (src, a.name)
        elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            out[node.name] = None
        else:
            for name in _target_names(node) or []:
                out[name] = None
    return out


def _check_exports(
    report: MoveReport,
    old_tree: ast.Module,
    now: ast.Module,
    moved: Mapping[str, str],
    modules: Mapping[str, str],
    require_reexports: bool,
) -> None:
    bound = _runtime_bindings(now)
    for node in old_tree.body:
        names = _target_names(node)
        exports = _literal_strings(getattr(node, "value", None)) if names == ["__all__"] else None
        for name in exports or []:
            if name not in bound:
                report.problems.append(
                    f"export: {name!r} is in __all__ but not bound at module scope"
                )
    if not require_reexports:
        return
    for key, module in sorted(moved.items()):
        for name in key.split(","):
            if "." in key or name.startswith("_"):
                continue
            want = (modules.get(module, module), name)
            got = bound.get(name)
            if got is None or got != want:
                detail = (
                    "relative import; use an absolute one"
                    if got and got[0].startswith(".")
                    else (
                        "not imported at module scope"
                        if name not in bound
                        else "imported from elsewhere"
                    )
                )
                report.problems.append(
                    f"re-export: public name {name} moved to {want[0]}; old module: {detail}"
                )


def check_move(
    old_source: str,
    after: Mapping[str, str],
    *,
    old_key: str = "old",
    allow_new: Collection[str] = (),
    allow_effects: Collection[str] = (),
    require_reexports: bool = False,
    modules: Mapping[str, str] | None = None,
) -> MoveReport:
    """Compare the base source of one module with its pieces now (module key -> source).

    ``old_key`` names the entry of ``after`` that is the old module itself (it may be absent).
    ``modules`` maps a key to its dotted module name (default: the key).
    """
    report = MoveReport()
    old_tree = ast.parse(old_source)
    trees = {name: ast.parse(src) for name, src in after.items()}
    old_items = _module_items(old_tree, is_old=True, allow_effects=allow_effects)
    new_items = {
        name: _module_items(tree, is_old=name == old_key, allow_effects=allow_effects)
        for name, tree in trees.items()
    }
    for name, tree in trees.items():
        if _future(tree) != _future(old_tree):
            report.problems.append(f"changed: `from __future__` imports differ in {name}")
    old_keys = {it.key for it in old_items}
    hosts = _find_hosts(old_keys, new_items, allow_new, report)
    where = _place("", old_items, new_items, old_key, report, {*allow_new, *hosts})
    for key, name in where.items():
        if name != old_key:
            report.moved[key] = name
    used: dict[str, str] = {}
    for key in dict.fromkeys(it.key for it in old_items if isinstance(it.node, ast.ClassDef)):
        if key not in where:
            continue
        olds = [it.node for it in old_items if it.key == key]
        news = [it.node for it in new_items[where[key]] if it.key == key]
        for old_node, new_node in zip(olds, news, strict=False):
            if isinstance(old_node, ast.ClassDef) and isinstance(new_node, ast.ClassDef):
                _compare_class(old_node, new_node, where[key], hosts, used, report)
    if old_key in trees:
        _check_exports(
            report, old_tree, trees[old_key], report.moved, modules or {}, require_reexports
        )
    return report


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
    parser.add_argument(
        "--allow-effects",
        nargs="*",
        default=[],
        metavar="NAME",
        help="assignments whose value calls something may move only if their target is listed",
    )
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
        allow_effects=args.allow_effects,
        require_reexports=args.require_reexports,
        modules={str(p): r.module_name(p) for p in [args.old, *args.new]},
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
