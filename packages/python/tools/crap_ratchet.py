"""CRAP ratchet: fail CI when function risk (complexity vs. test coverage) grows.

CRAP = complexity**2 * (1 - coverage)**3 + complexity, per function, using radon
cyclomatic complexity and the per-line coverage that coverage.py wrote to JSON.

Coverage is attributed to each function's own executable body: the ``def`` line and
decorators (which run when the enclosing scope is imported) and the bodies of nested
functions (scored on their own) are left out. Lines excluded with ``# pragma: no cover``
count as uncovered. Missing evidence fails the run instead of counting as covered: a
source file absent from the report, a body with statements but no measured lines, or a
report that does not match the source. Functions that have no statements to cover (a
docstring or ``...`` body) need no measurement.

Rules enforced by ``check``:
  * a new function must score at most the threshold (default 30);
  * a changed function (its AST differs from the baseline) must score at most the threshold;
  * no existing function may score higher than its baseline (small tolerance for noise).

Run locally (from packages/python)::

    pytest tests --cov=src/dendrux --cov-report=json:coverage.json
    python tools/crap_ratchet.py check --coverage coverage.json
    python tools/crap_ratchet.py update --coverage coverage.json   # after improving scores
    python tools/crap_ratchet.py update --allow-regress ...        # waive; listed as WAIVED
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


class CoverageDataError(Exception):
    """The coverage report cannot be trusted to score the source tree."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _is_overload(node: ast.AST | None) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in decorators
    )


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class _Scopes(ast.NodeVisitor):
    """Every function and method, at any nesting, with a stable qualified name."""

    def __init__(self) -> None:
        self.prefix = ""
        self.found: list[tuple[str, _FunctionNode]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        saved, self.prefix = self.prefix, f"{self.prefix}{node.name}."
        self.generic_visit(node)
        self.prefix = saved

    def _function(self, node: _FunctionNode) -> None:
        qual = f"{self.prefix}{node.name}"
        self.found.append((qual, node))
        saved, self.prefix = self.prefix, f"{qual}.<locals>."
        self.generic_visit(node)
        self.prefix = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._function(node)


def _complexity(node: _FunctionNode) -> int:
    return int(ComplexityVisitor.from_ast(node).functions[0].complexity)


def _has_statements(node: _FunctionNode) -> bool:
    """False for a body that is only a docstring or ``...`` (nothing to cover)."""
    return any(
        not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        for stmt in node.body
    )


def _own_lines(node: _FunctionNode) -> set[int]:
    """Lines of ``node``'s body that execute when ``node`` runs, not when it is defined.

    Nested function bodies are excluded (they have their own score); nested ``def`` lines,
    class bodies and decorators belong to the enclosing function.
    """
    first, last = node.body[0].lineno, node.end_lineno or node.body[0].lineno
    own = set(range(first, last + 1))
    for inner in ast.walk(node):
        if inner is node or not isinstance(inner, _FunctionNode):
            continue
        if inner.body[0].lineno > inner.lineno:  # an inline body shares the def line
            own -= set(range(inner.body[0].lineno, (inner.end_lineno or 0) + 1))
    return own


def _body_coverage(
    node: _FunctionNode, executed: set[int], missing: set[int], excluded: set[int]
) -> tuple[float, bool]:
    """(coverage fraction, measured). ``measured`` is False when statements have no data."""
    if not _has_statements(node):
        return 1.0, True
    if node.body[0].lineno <= node.lineno:
        # The body shares the def line, which runs at import: execution cannot be seen.
        return 0.0, True
    own = _own_lines(node)
    ran = len(own & executed)
    unrun = len(own & (missing | excluded))
    if ran + unrun == 0:
        return 0.0, False
    return ran / (ran + unrun), True


def _iter_sources(src_root: Path) -> list[Path]:
    return sorted(p for p in src_root.rglob("*.py") if not EXCLUDED_DIRS & set(p.parts))


def _check_report_matches_source(rel: str, line_count: int, lines: set[int]) -> str | None:
    if lines and max(lines) > line_count:
        return (
            f"{rel}: coverage report does not match the source (line {max(lines)} of {line_count})"
        )
    return None


def collect_scores(
    src_root: Path, repo_root: Path, coverage: dict[str, Any]
) -> dict[str, FunctionScore]:
    """Score every function under ``src_root``; keys are ``relpath::Qual.name``.

    Raises ``CoverageDataError`` when the report is missing evidence for any function.
    """
    files = coverage.get("files", {})
    scores: dict[str, FunctionScore] = {}
    problems: list[str] = []
    for path in _iter_sources(src_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        scopes = _Scopes()
        scopes.visit(ast.parse(text))
        functions = [(qual, node) for qual, node in scopes.found if not _is_overload(node)]
        if not functions:
            continue
        cov = files.get(rel)
        if cov is None:
            problems.append(f"{rel}: not in the coverage report")
            continue
        executed = set(cov.get("executed_lines", []))
        missing = set(cov.get("missing_lines", []))
        excluded = set(cov.get("excluded_lines", []))
        # excluded_lines can overshoot the end of file (a pragma on a trailing block).
        mismatch = _check_report_matches_source(rel, len(text.splitlines()), executed | missing)
        if mismatch:
            problems.append(mismatch)
            continue
        seen: dict[str, int] = {}
        for qual, node in sorted(functions, key=lambda item: item[1].lineno):
            seen[qual] = seen.get(qual, 0) + 1
            suffix = "" if seen[qual] == 1 else f"#{seen[qual]}"
            key = f"{rel}::{qual}{suffix}"
            fraction, measured = _body_coverage(node, executed, missing, excluded)
            if not measured:
                problems.append(f"{key}: body has statements but no coverage measurements")
                continue
            complexity = _complexity(node)
            scores[key] = FunctionScore(
                key=key,
                complexity=complexity,
                coverage=fraction,
                crap=crap(complexity, fraction),
                digest=_digest(node),
            )
    if problems:
        raise CoverageDataError(problems)
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


def waived_violations(
    scores: dict[str, FunctionScore],
    previous: dict[str, dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Violation]:
    """What recording ``scores`` over ``previous`` would let through (every rule violation)."""
    return compare(scores, previous, threshold, tolerance)


def build_baseline(
    scores: dict[str, FunctionScore],
    previous: dict[str, dict[str, Any]],
    allow_regress: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, dict[str, Any]]:
    """Baseline for ``scores``. Refuses to record any rule violation unless told to.

    With ``allow_regress`` the violations are waived and each affected entry carries
    ``"waived": <rule>`` until the next update rewrites it, so the exception shows up in
    the baseline diff.
    """
    violations = waived_violations(scores, previous, threshold, tolerance)
    if violations and not allow_regress:
        raise ValueError("; ".join(f"{v.key}: {v.message}" for v in violations))
    baseline = {key: _entry(score) for key, score in sorted(scores.items())}
    for v in violations:
        baseline[v.key]["waived"] = v.kind
    return baseline


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
        help="update: waive rule violations; each is listed as WAIVED and marked in the baseline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.coverage.exists():
        print(f"coverage file not found: {args.coverage}", file=sys.stderr)
        return 2
    coverage = json.loads(args.coverage.read_text())
    try:
        scores = collect_scores(args.src.resolve(), args.repo_root.resolve(), coverage)
    except CoverageDataError as exc:
        print("cannot score: the coverage report is missing evidence", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
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
        waived = [(key, entry["waived"]) for key, entry in new.items() if "waived" in entry]
        for key, kind in waived:
            print(f"WAIVED {kind}: {key}")
        args.baseline.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {len(new)} functions, {len(waived)} waived")
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
