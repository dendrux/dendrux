"""Public API break check: compare ``dendrux`` against a released git tag with griffe.

Every break griffe finds must be listed in ``tools/api_break_allowlist.json`` with the
migration-note B-item (spec section 2) that documents it. A break with no B-item can be
listed as a *gap* (``"b_item": null`` plus a ``"gap"`` reason): it passes the normal check
but is reported loudly, and ``--strict`` (use it at release time) fails on any gap.

Run locally (from packages/python)::

    python tools/api_breaks.py                   # the CI check
    python tools/api_breaks.py --strict          # release gate: no unmapped breaks
    python tools/api_breaks.py --report breaks.md

Needs the comparison tag in the clone (``git fetch --tags``; CI checks out full history).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import griffe

HERE = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = HERE / "api_break_allowlist.json"
DEFAULT_REPO = HERE.parents[2]
DEFAULT_SEARCH_PATH = "packages/python/src"
DEFAULT_AGAINST = "v0.4.1"
_B_ITEM = re.compile(r"^B\d+$")


@dataclass(frozen=True)
class Break:
    kind: str
    path: str
    param: str | None
    detail: str


@dataclass(frozen=True)
class AllowEntry:
    kind: str
    paths: tuple[str, ...]
    param: str | None
    b_item: str | None
    note: str


@dataclass
class Classification:
    mapped: list[tuple[Break, AllowEntry]] = field(default_factory=list)
    gaps: list[tuple[Break, AllowEntry]] = field(default_factory=list)
    unlisted: list[Break] = field(default_factory=list)
    stale: list[tuple[AllowEntry, str]] = field(default_factory=list)


def _to_break(breakage: Any) -> Break:
    kind = breakage.kind.name.lower().replace("_", "-")
    old, new = breakage.old_value, breakage.new_value
    param = next((v.name for v in (new, old) if isinstance(v, griffe.Parameter)), None)
    detail = breakage.kind.value
    if kind == "parameter-changed-default" and param is not None:
        detail = f"{detail}: {old.default} -> {new.default}"
    return Break(kind=kind, path=breakage.obj.path, param=param, detail=detail)


def compute_breaks(package: str, repo: Path, ref: str, search_path: str) -> list[Break]:
    """Breaking changes in ``package`` between git ``ref`` and the working tree."""
    try:
        old = griffe.load_git(package, ref=ref, repo=repo, search_paths=[search_path])
    except Exception as exc:  # griffe raises GitError / OSError / ValueError
        raise ValueError(f"cannot load {package!r} at git ref {ref!r}: {exc}") from exc
    new = griffe.load(package, search_paths=[repo / search_path])
    found = {_to_break(b) for b in griffe.find_breaking_changes(old, new)}
    return sorted(found, key=lambda b: (b.path, b.kind, b.param or ""))


def _parse_entry(index: int, raw: dict[str, Any], known: set[str]) -> AllowEntry:
    where = f"entries[{index}]"
    kind = raw.get("kind")
    paths = raw.get("paths")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{where}: 'kind' is required")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError(f"{where}: 'paths' must be a non-empty list of strings")
    if "b_item" not in raw:
        raise ValueError(f"{where}: 'b_item' is required (a B-item id, or null for a gap)")
    b_item = raw["b_item"]
    if b_item is None:
        note = raw.get("gap")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{where}: a gap entry (b_item null) needs a 'gap' reason")
    else:
        if not isinstance(b_item, str) or not _B_ITEM.match(b_item) or b_item not in known:
            raise ValueError(f"{where}: unknown B-item {b_item!r}")
        note = raw.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{where}: 'note' (why this B-item covers the break) is required")
    return AllowEntry(
        kind=kind,
        paths=tuple(paths),
        param=raw.get("param"),
        b_item=b_item,
        note=note,
    )


def load_allowlist(path: Path) -> list[AllowEntry]:
    data = json.loads(path.read_text())
    known = set(data.get("known_b_items", []))
    return [_parse_entry(i, raw, known) for i, raw in enumerate(data.get("entries", []))]


def _matches(entry: AllowEntry, brk: Break) -> bool:
    return (
        entry.kind == brk.kind
        and brk.path in entry.paths
        and (entry.param is None or entry.param == brk.param)
    )


def classify(breaks: list[Break], entries: list[AllowEntry]) -> Classification:
    result = Classification()
    used: set[int] = set()
    for brk in breaks:
        hit = next((i for i, e in enumerate(entries) if _matches(e, brk)), None)
        if hit is None:
            result.unlisted.append(brk)
            continue
        used.add(hit)
        (result.gaps if entries[hit].b_item is None else result.mapped).append((brk, entries[hit]))
    for i, entry in enumerate(entries):
        if i not in used:
            result.stale.append((entry, ", ".join(entry.paths)))
    return result


def _line(brk: Break) -> str:
    param = f" [{brk.param}]" if brk.param else ""
    return f"{brk.path}{param}: {brk.detail}"


def format_report(result: Classification, against: str) -> str:
    total = len(result.mapped) + len(result.gaps) + len(result.unlisted)
    lines = [f"Public API breaks against {against}: {total}"]
    if result.mapped:
        lines.append(f"\nDocumented ({len(result.mapped)}):")
        lines.extend(f"  {e.b_item}  {_line(b)}" for b, e in result.mapped)
    if result.gaps:
        lines.append(f"\nGAP: allowed but not documented by any B-item ({len(result.gaps)}):")
        lines.extend(f"  {_line(b)}  -- {e.note}" for b, e in result.gaps)
    if result.unlisted:
        lines.append(f"\nUNLISTED: new breaks, not in the allowlist ({len(result.unlisted)}):")
        lines.extend(f"  {_line(b)}" for b in result.unlisted)
    if result.stale:
        lines.append(f"\nStale allowlist entries, no longer breaking ({len(result.stale)}):")
        lines.extend(f"  {e.kind}: {paths}" for e, paths in result.stale)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--package", default="dendrux")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--search-path", default=DEFAULT_SEARCH_PATH)
    parser.add_argument("--against", default=DEFAULT_AGAINST, help="git ref of the last release")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--strict", action="store_true", help="also fail on unmapped gaps")
    parser.add_argument("--report", type=Path, help="write the report to this file too")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        entries = load_allowlist(args.allowlist)
        breaks = compute_breaks(args.package, args.repo, args.against, args.search_path)
    except (ValueError, OSError) as exc:
        print(f"api check error: {exc}", file=sys.stderr)
        return 2
    result = classify(breaks, entries)
    report = format_report(result, args.against)
    print(report)
    if args.report:
        args.report.write_text(report + "\n")
    if result.unlisted:
        print(f"\nFAIL: {len(result.unlisted)} break(s) are not in {args.allowlist.name}.")
        print("Document each in the migration notes, then add it with its B-item.")
        return 1
    if args.strict and result.gaps:
        print(f"\nFAIL (--strict): {len(result.gaps)} break(s) have no migration-note B-item.")
        return 1
    print("\nAPI break check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
