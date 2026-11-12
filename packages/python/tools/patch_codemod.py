"""Rewrite test patch targets after a pure move: old module -> the module that now defines it.

``mapping`` is ``{old_module: {name: new_module}}`` (``move_check.py --emit-map`` writes it).
Rewritten, in files under the given paths (default ``tests``):

* string targets: ``patch("old.mod.name")``, ``monkeypatch.setattr("old.mod.name", v)``;
* module aliases: ``patch.object(old_alias, "name")``, ``monkeypatch.setattr(old_alias, "name", v)``
  (the alias argument becomes the new module; its import is added or reused).

Nothing else changes: prose, constants, imports and the unmapped names stay. A dry run
(default) prints a unified diff; ``--write`` applies it. Unused old imports are left for
``ruff check --select F401 --fix``.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

try:
    from tools import _patch_sites as ps
    from tools import _scopes as sc
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _patch_sites as ps
    from tools import _scopes as sc


@dataclass
class Rewrite:
    source: str
    notes: list[str] = field(default_factory=list)
    edits: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.edits)


class _Text:
    """Offsets into the source; ``ast`` columns are UTF-8 byte offsets."""

    def __init__(self, source: str) -> None:
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, col: int) -> int:
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode()[:col].decode(errors="ignore"))

    def span(self, node: ast.expr) -> tuple[int, int]:
        if node.end_lineno is None or node.end_col_offset is None:
            raise ValueError("node has no end position")
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )


def _new_module_for(
    dotted: str, mapping: Mapping[str, Mapping[str, str]]
) -> tuple[str, str, str] | None:
    """(old module, name, new module) when ``dotted`` starts with a mapped module + name."""
    for old in sorted(mapping, key=len, reverse=True):
        if dotted.startswith(old + "."):
            name = dotted[len(old) + 1 :].split(".")[0]
            if name in mapping[old]:
                return old, name, mapping[old][name]
    return None


def _is_type_checking_block(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and not node.orelse
        and (sc.dotted(node.test) or "").split(".")[-1] == "TYPE_CHECKING"
        and all(isinstance(n, ast.Import | ast.ImportFrom) for n in node.body)
    )


def _import_insert_line(tree: ast.Module) -> int:
    """Line after which a new import goes: the end of the leading docstring and import block.

    Everything after that block (decorators included) runs later, so a name imported here is
    bound before any rewritten patch target can use it.
    """
    last = 0
    for i, node in enumerate(tree.body):
        is_docstring = (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not (
            is_docstring
            or isinstance(node, ast.Import | ast.ImportFrom)
            or _is_type_checking_block(node)
        ):
            break
        last = node.end_lineno or node.lineno
    return last


class _Rewriter:
    def __init__(self, source: str, mapping: Mapping[str, Mapping[str, str]]) -> None:
        self.source = source
        self.mapping = mapping
        self.tree = ast.parse(source)
        self.index = sc.ScopeIndex(self.tree)
        self.text = _Text(source)
        self.taken = self.index.all_names()
        self.new_imports: dict[str, str] = {}  # new module -> fresh module-level name
        self.edits: list[tuple[int, int, str]] = []
        self.notes: list[str] = []

    def _fresh(self, leaf: str) -> str:
        for candidate in (leaf, f"{leaf}_mod", *(f"{leaf}_mod{n}" for n in range(2, 100))):
            if candidate not in self.taken:
                self.taken.add(candidate)
                return candidate
        raise ValueError(f"no free name for {leaf}")

    def _reference(self, module: str, scope: sc.Scope) -> str:
        """A name that is bound to ``module`` at this scope."""
        seen: set[str] = set()
        s: sc.Scope | None = scope
        while s is not None:
            for name in s.bindings:
                binding = scope.resolve(name)
                if (
                    name not in seen
                    and binding is not None
                    and binding == sc.Binding("import", module)
                ):
                    return name
                seen.add(name)
            s = s.parent
        if module not in self.new_imports:
            self.new_imports[module] = self._fresh(module.rpartition(".")[2])
        return self.new_imports[module]

    def _string(self, site: ps.PatchSite) -> None:
        hit = _new_module_for(site.dotted, self.mapping)
        if hit is None:
            return
        old, _name, new = hit
        start, end = self.text.span(site.node)
        segment = self.source[start:end]
        if site.dotted not in segment:
            self.notes.append(f"line {site.lineno}: target is not one plain literal; left")
            return
        tail = site.dotted[len(old) + 1 :]
        self.edits.append((start, end, segment.replace(site.dotted, f"{new}.{tail}", 1)))

    def _alias(self, site: ps.PatchSite) -> None:
        hit = _new_module_for(f"{site.dotted}.{site.attr}", self.mapping)
        if hit is None or hit[0] != site.dotted:
            return
        start, end = self.text.span(site.node)
        self.edits.append((start, end, self._reference(hit[2], site.scope)))

    def run(self) -> Rewrite:
        for site in ps.find_sites(self.index):
            if site.kind == "string":
                self._string(site)
            elif site.kind == "alias":
                self._alias(site)
            else:
                self.notes.append(site.note)
        result = Rewrite(self.source, self.notes)
        if not self.edits:
            return result
        line = _import_insert_line(self.tree)
        at = self.text.starts[line] if line < len(self.text.starts) else len(self.source)
        prefix = (
            "\n"
            if at == len(self.source) and self.source and not self.source.endswith("\n")
            else ""
        )
        for module, name in self.new_imports.items():
            parent, _, leaf = module.rpartition(".")
            suffix = "" if name == leaf else f" as {name}"
            self.edits.append((at, at, f"{prefix}from {parent} import {leaf}{suffix}\n"))
        out = self.source
        order = sorted(enumerate(self.edits), key=lambda e: (e[1][0], e[1][1], -e[0]), reverse=True)
        for _, (start, end, new_text) in order:
            out = out[:start] + new_text + out[end:]
        result.source, result.edits = out, len(self.edits)
        return result


def rewrite_source(source: str, mapping: Mapping[str, Mapping[str, str]]) -> Rewrite:
    """Return the rewritten source and notes about sites that were left alone."""
    try:
        return _Rewriter(source, mapping).run()
    except SyntaxError as exc:
        return Rewrite(source, [f"skipped: syntax error at line {exc.lineno}"])


def _load_mapping(path: Path) -> dict[str, dict[str, str]]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    ok = isinstance(data, dict) and all(
        isinstance(k, str) and isinstance(v, dict) and all(isinstance(x, str) for x in v.values())
        for k, v in data.items()
    )
    if not ok:
        raise SystemExit(f"patch_codemod: {path} must map old module -> {{name: new module}}")
    return cast("dict[str, dict[str, str]]", data)


def _files(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map", required=True, type=Path, help="JSON {old_module: {name: new_module}}"
    )
    parser.add_argument("--write", action="store_true", help="apply the rewrite (default: dry run)")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("tests")])
    args = parser.parse_args(argv)
    mapping = _load_mapping(args.map)
    changed = 0
    for path in _files(args.paths):
        source = path.read_text(encoding="utf-8")
        rewritten = rewrite_source(source, mapping)
        for note in rewritten.notes:
            print(f"{path}: {note}")
        if rewritten.source == source:
            continue
        changed += 1
        diff = difflib.unified_diff(
            source.splitlines(),
            rewritten.source.splitlines(),
            f"a/{path}",
            f"b/{path}",
            lineterm="",
        )
        print("\n".join(diff))
        if args.write:
            path.write_text(rewritten.source, encoding="utf-8")
    verb = "rewrote" if args.write else "would rewrite"
    print(f"patch_codemod: {verb} {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
