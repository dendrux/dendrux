"""Helpers shared by the quality-gate ratchets (see tools/QUALITY.md).

A ratchet records today's problems in a committed baseline and fails only on a new
problem or an existing one getting worse. Baselines are small, sorted JSON keyed by
``module:qualified_name`` so unrelated edits never churn them.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC = PACKAGE_ROOT / "src" / "dendrux"
EXCLUDED_DIRS = frozenset({"migrations", "__pycache__"})


class BaselineMissingError(Exception):
    pass


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineMissingError(f"baseline not found: {path} (run 'make quality-stamp')")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def write_baseline(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=1, sort_keys=True) + "\n")


@dataclass
class CountComparison:
    new: list[str] = field(default_factory=list)
    worse: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.new or self.worse)


def compare_counts(
    current: Mapping[str, int], baseline: Mapping[str, int], limit: int | None = None
) -> CountComparison:
    """New keys fail (when above ``limit``, if given); known keys may not grow."""
    result = CountComparison()
    for key, value in sorted(current.items()):
        old = baseline.get(key)
        if old is None:
            if limit is None or value > limit:
                result.new.append(key)
        elif value > old:
            result.worse.append(key)
        elif value < old:
            result.improved.append(key)
    result.improved.extend(sorted(k for k in baseline if k not in current))
    result.improved.sort()
    return result


def module_name(path: str | Path) -> str:
    """``.../src/dendrux/runtime/runner.py`` -> ``dendrux.runtime.runner`` (``__init__`` drops).

    Works on the path text alone (the last ``src/`` component anchors it), so it needs no
    knowledge of where the tree lives.
    """
    parts = list(Path(path).with_suffix("").parts)
    anchors = [i for i, part in enumerate(parts) if part == "src"]
    parts = parts[anchors[-1] + 1 :] if anchors else parts
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_sources(src: Path = SRC) -> list[Path]:
    return sorted(p for p in src.rglob("*.py") if not EXCLUDED_DIRS & set(p.relative_to(src).parts))


def add_common_args(parser: Any, commands: tuple[str, ...] = ("check", "stamp")) -> None:
    parser.add_argument("command", choices=list(commands))
    parser.add_argument(
        "--allow-regress",
        action="store_true",
        help="stamp: record violations too (coordinator approval only; needs --reason)",
    )
    parser.add_argument("--reason", help="stamp --allow-regress: why; printed in the log")


def finish(
    command: str,
    *,
    label: str,
    comparison: CountComparison | None,
    new_baseline: Mapping[str, Any],
    path: Path,
    allow_regress: bool = False,
    reason: str | None = None,
    describe: Any = str,
) -> int:
    """Shared check/stamp ending. ``comparison`` is None when no baseline file exists yet."""
    failed = comparison is not None and comparison.failed
    if comparison is not None:
        for key in comparison.new:
            print(f"{label} NEW: {describe(key)}")
        for key in comparison.worse:
            print(f"{label} WORSE: {describe(key)}")
    if command == "check":
        if comparison is None:
            print(f"baseline not found: {path} (run 'make quality-stamp')")
            return 2
        if failed:
            n = len(comparison.new) + len(comparison.worse)
            print(f"{label}: {n} violation(s); fix them, do not stamp them (tools/QUALITY.md)")
            return 1
        if comparison.improved:
            print(f"{label}: {len(comparison.improved)} improved; run 'make quality-stamp'")
        print(f"{label} ok")
        return 0
    if failed and not allow_regress:
        print(f"{label}: refusing to stamp over new or worse entries (needs --allow-regress)")
        return 1
    if failed and not (reason and reason.strip()):
        print(f"{label}: --allow-regress needs --reason")
        return 2
    if failed:
        print(f"{label} WAIVED: {reason}")
    write_baseline(path, new_baseline)
    print(f"{label} baseline written: {len(new_baseline)} entries")
    return 0


def baseline_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return read_baseline(path)
    except BaselineMissingError:
        return None


def transfer_args(
    argv: list[str] | None, *, description: str | None, baseline: Path
) -> argparse.Namespace:
    """Parse the shrink-only tools' check/stamp/report/move interface."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=["check", "stamp", "report", "move"])
    parser.add_argument(
        "keys", nargs="*", metavar="KEY", help="move: repeated OLD_KEY NEW_KEY pairs"
    )
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--baseline", type=Path, default=baseline)
    args = parser.parse_args(argv)
    if args.command == "move" and (not args.keys or len(args.keys) % 2):
        parser.error("move requires complete OLD_KEY NEW_KEY pairs")
    if args.command != "move" and args.keys:
        parser.error("only move accepts key pairs")
    return args


def transfer_counts(
    scores: Mapping[str, int],
    baseline: Mapping[str, int],
    pairs: Sequence[tuple[str, str]],
    *,
    present: Collection[str],
    retained: Mapping[str, int],
    minimum: Callable[[str], int],
    indivisible: Collection[str] = (),
) -> dict[str, int]:
    """Validate a move batch and return a new mapping, conserving each source budget.

    Tool-specific wrappers validate key kinds. Present sources must have a smaller
    retained measurement; that measurement consumes their allowance before transfers.
    """
    if not pairs:
        raise ValueError("move requires at least one OLD_KEY NEW_KEY pair")
    destinations: set[str] = set()
    groups: dict[str, list[str]] = {}
    for old, new in pairs:
        if old not in baseline or baseline[old] <= minimum(old):
            raise ValueError(f"{old}: source has no offender allowance")
        if old in present:
            if old not in retained:
                raise ValueError(f"{old}: source still exists in the tree")
            if retained[old] >= baseline[old]:
                raise ValueError(f"{old}: retained source did not shrink")
        if new in baseline or new in destinations:
            raise ValueError(f"{new}: destination already has or claims an allowance")
        if scores.get(new, 0) <= minimum(new):
            raise ValueError(f"{new}: destination is not a current offender")
        destinations.add(new)
        groups.setdefault(old, []).append(new)
    for old, news in groups.items():
        if old in indivisible and len(news) != 1:
            raise ValueError(f"{old}: function allowances cannot split")
        if sum(scores[new] for new in news) + retained.get(old, 0) > baseline[old]:
            raise ValueError(f"{old}: destination total plus retained value exceeds the allowance")
    updated = dict(baseline)
    for old, news in groups.items():
        if old in retained:
            updated[old] = retained[old]
        else:
            del updated[old]
        updated.update({new: scores[new] for new in news})
    return updated


def finish_transfer(
    keys: Sequence[str],
    *,
    label: str,
    scores: Mapping[str, int],
    baseline: Mapping[str, int] | None,
    path: Path,
    prepare: Callable[[Mapping[str, int], Sequence[tuple[str, str]]], Mapping[str, int]],
) -> int:
    """Prepare the whole transfer before writing, then log each moved allowance."""
    if baseline is None:
        print(f"{label}: move requires an existing baseline: {path}")
        return 2
    pairs = list(zip(keys[::2], keys[1::2], strict=True))
    try:
        updated = prepare(baseline, pairs)
    except ValueError as exc:
        print(f"{label}: refusing move: {exc}")
        return 1
    write_baseline(path, updated)
    for old, new in pairs:
        print(f"{label} MOVED: {old} -> {new}: {baseline[old]} -> {scores[new]}")
    return 0


def function_spans(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, qualified name) for every function, outermost first."""
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qual = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno or child.lineno, qual))
                walk(child, f"{qual}.<locals>.")
            else:
                walk(child, prefix)

    walk(ast.parse(source), "")
    return spans


def enclosing(spans: list[tuple[int, int, str]], line: int) -> str:
    best = "<module>"
    best_size = None
    for first, last, qual in spans:
        if first <= line <= last and (best_size is None or last - first < best_size):
            best, best_size = qual, last - first
    return best
