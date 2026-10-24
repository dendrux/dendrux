"""CRAP ratchet: fail CI when function risk (complexity vs. test coverage) grows.

CRAP = complexity**2 * (1 - coverage)**3 + complexity, per function, using radon
cyclomatic complexity and the per-line coverage that coverage.py wrote to JSON.

Rules enforced by ``check``:
  * a new function must score at most the threshold (default 30);
  * a changed function (its AST differs from the baseline) must score at most the threshold;
  * no existing function may score higher than its baseline (small tolerance for noise).

Run locally (from packages/python)::

    pytest tests --cov=src/dendrux --cov-report=json:coverage.json
    python tools/crap_ratchet.py check --coverage coverage.json
    python tools/crap_ratchet.py update --coverage coverage.json   # after improving scores
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radon.visitors import ComplexityVisitor

DEFAULT_THRESHOLD = 30.0
DEFAULT_TOLERANCE = 0.5
HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "crap_baseline.json"
DEFAULT_REPO_ROOT = HERE.parent
DEFAULT_SRC = DEFAULT_REPO_ROOT / "src" / "dendrux"
EXCLUDED_DIRS = frozenset({"migrations", "__pycache__"})


@dataclass(frozen=True)
class FunctionScore:
    key: str
    complexity: int
    coverage: float
    crap: float
    digest: str


@dataclass(frozen=True)
class Violation:
    key: str
    kind: str
    message: str


def crap(complexity: int, coverage: float) -> float:
    """CRAP score for one function; ``coverage`` is a fraction in [0, 1]."""
    return complexity**2 * (1.0 - coverage) ** 3 + complexity


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _canon(node: object) -> object:
    """Version-stable structure of an AST: no positions, no docstrings, no empty fields."""
    if isinstance(node, list):
        return [_canon(item) for item in node]
    if not isinstance(node, ast.AST):
        return node
    fields = []
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, list) and name == "body" and value and _is_docstring(value[0]):
            value = value[1:]
        if value is None or value == []:
            continue
        fields.append((name, _canon(value)))
    return (type(node).__name__, fields)


def _digest(node: ast.AST) -> str:
    """Hash of a function's code that ignores docstrings, positions and Python version."""
    return hashlib.sha1(repr(_canon(node)).encode()).hexdigest()[:12]


def _function_nodes(tree: ast.AST) -> dict[int, ast.AST]:
    return {
        n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _is_overload(node: ast.AST | None) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in decorators
    )


def _flatten(blocks: list[Any], prefix: str = "") -> list[tuple[str, Any]]:
    """(qualified name, radon block) for every function, method and closure."""
    out: list[tuple[str, Any]] = []
    for block in blocks:
        if type(block).__name__ == "Class":
            out.extend(_flatten(list(block.methods), f"{prefix}{block.name}."))
            continue
        qual = f"{prefix}{block.name}"
        out.append((qual, block))
        out.extend(_flatten(list(block.closures), f"{qual}.<locals>."))
    return out


def _line_coverage(first: int, last: int, executed: set[int], missing: set[int]) -> float:
    run = sum(1 for n in executed if first <= n <= last)
    miss = sum(1 for n in missing if first <= n <= last)
    total = run + miss
    return 1.0 if total == 0 else run / total


def _iter_sources(src_root: Path) -> list[Path]:
    return sorted(p for p in src_root.rglob("*.py") if not EXCLUDED_DIRS & set(p.parts))


def collect_scores(
    src_root: Path, repo_root: Path, coverage: dict[str, Any]
) -> dict[str, FunctionScore]:
    """Score every function under ``src_root``; keys are ``relpath::Qual.name``."""
    files = coverage.get("files", {})
    scores: dict[str, FunctionScore] = {}
    for path in _iter_sources(src_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        visitor = ComplexityVisitor.from_code(text)
        nodes = _function_nodes(ast.parse(text))
        cov = files.get(rel, {})
        executed = set(cov.get("executed_lines", []))
        missing = set(cov.get("missing_lines", []))
        seen: dict[str, int] = {}
        blocks = _flatten(list(visitor.functions) + list(visitor.classes))
        for qual, block in sorted(blocks, key=lambda item: item[1].lineno):
            node = nodes.get(block.lineno)
            if _is_overload(node):
                continue
            seen[qual] = seen.get(qual, 0) + 1
            suffix = "" if seen[qual] == 1 else f"#{seen[qual]}"
            key = f"{rel}::{qual}{suffix}"
            fraction = _line_coverage(block.lineno, block.endline, executed, missing)
            scores[key] = FunctionScore(
                key=key,
                complexity=block.complexity,
                coverage=fraction,
                crap=crap(block.complexity, fraction),
                digest=_digest(node) if node is not None else "",
            )
    return scores


def compare(
    scores: dict[str, FunctionScore],
    baseline: dict[str, dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Violation]:
    violations: list[Violation] = []
    for key, score in sorted(scores.items()):
        old = baseline.get(key)
        if old is None:
            if score.crap > threshold:
                violations.append(
                    Violation(
                        key,
                        "new-over-threshold",
                        f"new function scores CRAP {score.crap:.1f} (limit {threshold:g})",
                    )
                )
            continue
        if score.crap > old["crap"] + tolerance:
            violations.append(
                Violation(
                    key,
                    "worse",
                    f"CRAP rose from {old['crap']:.1f} to {score.crap:.1f}",
                )
            )
        elif score.digest != old["digest"] and score.crap > threshold:
            violations.append(
                Violation(
                    key,
                    "changed-over-threshold",
                    f"changed function scores CRAP {score.crap:.1f} (limit {threshold:g})",
                )
            )
    return violations


def _entry(score: FunctionScore) -> dict[str, Any]:
    return {
        "complexity": score.complexity,
        "coverage": round(score.coverage, 4),
        "crap": round(score.crap, 2),
        "digest": score.digest,
    }


def build_baseline(
    scores: dict[str, FunctionScore],
    previous: dict[str, dict[str, Any]],
    allow_regress: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, dict[str, Any]]:
    """Baseline for ``scores``. Refuses to record a regression or a new hotspot."""
    if not allow_regress:
        problems = [
            v
            for v in compare(scores, previous, threshold, tolerance)
            if v.kind != "changed-over-threshold"
        ]
        if problems:
            raise ValueError("; ".join(f"{v.key}: {v.message}" for v in problems))
    return {key: _entry(score) for key, score in sorted(scores.items())}


def format_report(scores: dict[str, FunctionScore], top: int = 20) -> str:
    worst = sorted(scores.values(), key=lambda s: (-s.crap, s.key))[:top]
    lines = [f"Top {len(worst)} CRAP hotspots", "  CRAP   cc   cov  function"]
    lines.extend(
        f"{s.crap:6.1f} {s.complexity:4d} {s.coverage * 100:4.0f}%  {s.key}" for s in worst
    )
    return "\n".join(lines)


def _load_baseline(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    data: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["check", "update", "report"])
    parser.add_argument("--coverage", type=Path, default=Path("coverage.json"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--allow-regress",
        action="store_true",
        help="update: accept worse scores and new hotspots (bootstrap)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.coverage.exists():
        print(f"coverage file not found: {args.coverage}", file=sys.stderr)
        return 2
    coverage = json.loads(args.coverage.read_text())
    scores = collect_scores(args.src.resolve(), args.repo_root.resolve(), coverage)
    print(format_report(scores, args.top))

    baseline = _load_baseline(args.baseline)
    if args.command == "report":
        return 0
    if args.command == "update":
        try:
            new = build_baseline(
                scores, baseline or {}, args.allow_regress, args.threshold, args.tolerance
            )
        except ValueError as exc:
            print(f"refusing to update the baseline: {exc}", file=sys.stderr)
            return 1
        args.baseline.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {len(new)} functions")
        return 0
    if baseline is None:
        print(f"baseline not found: {args.baseline} (run 'update' first)", file=sys.stderr)
        return 2
    violations = compare(scores, baseline, args.threshold, args.tolerance)
    for v in violations:
        print(f"CRAP {v.kind}: {v.key}: {v.message}")
    if violations:
        print(f"{len(violations)} CRAP violation(s)")
        return 1
    print("CRAP ratchet ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
