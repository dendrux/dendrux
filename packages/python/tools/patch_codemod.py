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
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _patch_sites as ps


@dataclass
class Rewrite:
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.edits)

    edits: int = 0


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
        assert node.end_lineno is not None and node.end_col_offset is not None  # noqa: S101
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


def _last_import_end(tree: ast.Module) -> int:
    """Line after which a new top-level import goes (0 = top of the file)."""
    last = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import | ast.ImportFrom) or (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            last = node.end_lineno or node.lineno
    return last


def _names_in_use(tree: ast.AST) -> set[str]:
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for n in ast.walk(tree):
        if isinstance(n, ast.alias):
            used.add(n.asname or n.name.split(".")[0])
        elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            used.add(n.name)
    return used


def rewrite_source(source: str, mapping: Mapping[str, Mapping[str, str]]) -> Rewrite:
    """Return the rewritten source and notes about sites that were left alone."""
    result = Rewrite(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.notes.append(f"skipped: syntax error at line {exc.lineno}")
        return result
    text = _Text(source)
    aliases = ps.import_aliases(tree)
    in_use = _names_in_use(tree)
    new_imports: dict[str, str] = {}  # new module -> local name
    edits: list[tuple[int, int, str]] = []

    def local_name(module: str) -> str:
        for local, target in aliases.items():
            if target == module:
                return local
        if module in new_imports:
            return new_imports[module]
        parent, _, leaf = module.rpartition(".")
        name = leaf if leaf not in in_use else f"{leaf}_mod"
        in_use.add(name)
        new_imports[module] = name
        return name

    for site in ps.find_sites(tree):
        if site.kind == "string":
            hit = _new_module_for(site.dotted, mapping)
            if hit is None:
                continue
            old, _name, new = hit
            start, end = text.span(site.node)
            segment = source[start:end]
            if site.dotted not in segment:
                result.notes.append(f"line {site.lineno}: target is not one plain literal; left")
                continue
            tail = site.dotted[len(old) + 1 :]
            edits.append((start, end, segment.replace(site.dotted, f"{new}.{tail}", 1)))
        else:
            hit = _new_module_for(f"{site.dotted}.{site.attr}", mapping)
            if hit is None or hit[0] != site.dotted:
                continue
            start, end = text.span(site.node)
            edits.append((start, end, local_name(hit[2])))
    if not edits:
        return result
    for module, name in new_imports.items():
        parent, _, leaf = module.rpartition(".")
        suffix = "" if name == leaf else f" as {name}"
        line = _last_import_end(tree)
        at = text.starts[line] if line < len(text.starts) else len(source)
        prefix = "\n" if at == len(source) and source and not source.endswith("\n") else ""
        edits.append((at, at, f"{prefix}from {parent} import {leaf}{suffix}\n"))
    out = source
    for start, end, new_text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        out = out[:start] + new_text + out[end:]
    result.source = out
    result.edits = len(edits)
    return result


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
