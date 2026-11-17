"""Shrink-only physical size gate: files > 1,000 lines, functions > 80 lines.

Keys are modules and module:qualname; only offenders are baselined. Implicit
renames fail closed; coordinators can explicitly transfer allowances with move.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

BASELINE = Path(__file__).resolve().parent / "size_baseline.json"
FILE_LIMIT = 1000
FUNCTION_LIMIT = 80
# Owner-approved code/type tables; exact module names, not a basename wildcard.
DATA_MODULES = frozenset({"dendrux._codes", "dendrux.types"})


def _definition_starts(source: str) -> dict[int, int]:
    starts: dict[int, int] = {}
    first_decorator: int | None = None
    beginning = True
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NEWLINE:
            beginning = True
        elif token.type not in {tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT}:
            if beginning:
                if token.string == "@":
                    first_decorator = first_decorator or token.start[0]
                else:
                    if token.string in {"def", "async"}:
                        starts[token.start[0]] = first_decorator or token.start[0]
                    first_decorator = None
            beginning = False
    return starts


def _overload_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "typing_extensions"}:
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "overload"
            )
        elif isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"typing", "typing_extensions"}
            )
    return names, modules


def function_sizes(source: str) -> dict[str, int]:
    """Measure all implementations, including decorators and nested functions.

    Blank/comment lines inside the AST span count. Trailing lines outside the
    definition do not. Repeated qualnames (e.g. property setters) keep the maximum.
    """
    tree = ast.parse(source)
    starts = _definition_starts(source)
    overloads, typing_modules = _overload_names(tree)
    sizes: dict[str, int] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                is_overload = any(
                    (isinstance(d, ast.Name) and d.id in overloads)
                    or (
                        isinstance(d, ast.Attribute)
                        and isinstance(d.value, ast.Name)
                        and d.value.id in typing_modules
                        and d.attr == "overload"
                    )
                    for d in child.decorator_list
                )
                if not is_overload:
                    qual = f"{prefix}{child.name}"
                    first = starts[child.lineno]
                    length = (child.end_lineno or child.lineno) - first + 1
                    sizes[qual] = max(sizes.get(qual, 0), length)
                    walk(child, f"{qual}.<locals>.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return sizes


def collect_scores(src: Path = r.SRC) -> dict[str, int]:
    """Count library modules and functions, excluding migrations and data tables."""
    scores: dict[str, int] = {}
    for path in r.iter_sources(src):
        module = r.module_name(path)
        if module in DATA_MODULES:
            continue
        source = path.read_text(encoding="utf-8")
        scores[module] = source.count("\n") + int(bool(source) and not source.endswith("\n"))
        scores.update({f"{module}:{qual}": value for qual, value in function_sizes(source).items()})
    return scores


def _limit(key: str) -> int:
    return FUNCTION_LIMIT if ":" in key else FILE_LIMIT


def baseline_from_scores(scores: Mapping[str, int]) -> dict[str, int]:
    """Record only over-limit modules/functions at their measured physical length."""
    return {key: value for key, value in sorted(scores.items()) if value > _limit(key)}


def compare(scores: Mapping[str, int], baseline: Mapping[str, int]) -> r.CountComparison:
    """New offenders fail; a recorded offender may only stay the same or shrink."""
    offenders = baseline_from_scores(scores)
    # Keep below-limit scores for recorded keys so an improvement is still visible.
    current = offenders | {key: scores[key] for key in baseline if key in scores}
    return r.compare_counts(current, baseline)


def _present_keys(src: Path) -> set[str]:
    # Presence includes exempt modules and overload stubs, even though size does not.
    keys: set[str] = set()
    for path in r.iter_sources(src):
        module = r.module_name(path)
        keys.add(module)
        keys.update(
            f"{module}:{qual}" for _, _, qual in r.function_spans(path.read_text(encoding="utf-8"))
        )
    return keys


def moved_baseline(
    scores: Mapping[str, int],
    baseline: Mapping[str, int],
    pairs: Sequence[tuple[str, str]],
    present: Collection[str],
) -> dict[str, int]:
    """Validate a whole move batch before returning a new, independently owned baseline.

    Functions move one-to-one. Files may split, conserving retained plus moved size.
    Unlisted entries are preserved; this operation does not stamp other improvements.
    """
    for old, new in pairs:
        if (":" in old) != (":" in new):
            raise ValueError(f"{old} -> {new}: cannot transfer between files and functions")
    return r.transfer_counts(
        scores,
        baseline,
        pairs,
        present=present,
        retained={key: scores[key] for key in present if ":" not in key and key in scores},
        minimum=_limit,
        indivisible={key for key in baseline if ":" in key},
    )


def _summary(scores: Mapping[str, int]) -> None:
    files = [value for key, value in scores.items() if ":" not in key]
    functions = [value for key, value in scores.items() if ":" in key]
    print(
        f"size: {len(files)} modules, {len(functions)} functions; "
        f"files >1000: {sum(v > 1000 for v in files)}, "
        f">2000: {sum(v > 2000 for v in files)}, >3000: {sum(v > 3000 for v in files)}; "
        f"functions >80: {sum(v > 80 for v in functions)}, "
        f">100: {sum(v > 100 for v in functions)}"
    )


def main(argv: list[str] | None = None) -> int:
    """Check, stamp, report, or explicitly transfer allowances for coordinator-reviewed moves."""
    args = r.transfer_args(argv, description=__doc__, baseline=BASELINE)
    scores = collect_scores(args.src.resolve())
    _summary(scores)
    offenders = baseline_from_scores(scores)
    if args.command == "report":
        for key, value in offenders.items():
            print(f"{key}: {value} lines (limit {_limit(key)})")
        return 0
    old = r.baseline_or_none(args.baseline)
    if args.command == "move":
        return r.finish_transfer(
            args.keys,
            label="size",
            scores=scores,
            baseline=old,
            path=args.baseline,
            prepare=lambda old, pairs: moved_baseline(
                scores, old, pairs, _present_keys(args.src.resolve())
            ),
        )
    comparison = None if old is None else compare(scores, old)
    if args.command == "stamp" and comparison is not None and comparison.failed:
        print("size: refusing to stamp new or worse entries; fix regressions (tools/QUALITY.md)")
        return 1
    return r.finish(
        args.command,
        label="size",
        comparison=comparison,
        new_baseline=offenders,
        path=args.baseline,
        describe=lambda key: f"{key} has {scores[key]} lines (limit {_limit(key)})",
    )


if __name__ == "__main__":
    raise SystemExit(main())
