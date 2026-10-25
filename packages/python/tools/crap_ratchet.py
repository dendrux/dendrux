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
    python tools/crap_ratchet.py stamp --coverage coverage.json   # right after the tests
    python tools/crap_ratchet.py check --coverage coverage.json
    python tools/crap_ratchet.py update --coverage coverage.json   # after improving scores
    python tools/crap_ratchet.py update --allow-regress ...        # waive; listed as WAIVED
"""

from __future__ import annotations

import argparse
import ast
import datetime
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


def _needs_measurement(stmt: ast.stmt) -> bool:
    """False for statements coverage.py never reports: bare constants (docstrings, ``...``),
    ``global``/``nonlocal`` and annotations without a value."""
    if isinstance(stmt, ast.Expr):
        return not isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, ast.Global | ast.Nonlocal):
        return False
    return not (isinstance(stmt, ast.AnnAssign) and stmt.value is None)


def _unaccounted_lines(node: _FunctionNode, known: set[int]) -> list[int]:
    """Body statements of ``node`` (not of nested functions) the report says nothing about.

    A stale or partial report, for example one taken before the function grew, leaves
    the added statements unmentioned; scoring only the surviving lines would hide them.
    """
    own = _own_lines(node)
    return sorted(
        {
            stmt.lineno
            for stmt in ast.walk(node)
            if isinstance(stmt, ast.stmt)
            and stmt.lineno in own
            and _needs_measurement(stmt)
            and stmt.lineno not in known
        }
    )


def _body_coverage(
    node: _FunctionNode, executed: set[int], missing: set[int], excluded: set[int]
) -> tuple[float, list[int]]:
    """(coverage fraction, body statement lines the report does not mention)."""
    if not _has_statements(node):
        return 1.0, []
    if node.body[0].lineno <= node.lineno:
        # The body shares the def line, which runs at import: execution cannot be seen.
        return 0.0, []
    unaccounted = _unaccounted_lines(node, executed | missing | excluded)
    if unaccounted:
        return 0.0, unaccounted
    own = _own_lines(node)
    ran = len(own & executed)
    unrun = len(own & (missing | excluded))
    return (ran / (ran + unrun) if ran + unrun else 1.0), []


def _iter_sources(src_root: Path) -> list[Path]:
    return sorted(p for p in src_root.rglob("*.py") if not EXCLUDED_DIRS & set(p.parts))


def _check_report_matches_source(rel: str, line_count: int, lines: set[int]) -> str | None:
    if lines and max(lines) > line_count:
        return (
            f"{rel}: coverage report does not match the source (line {max(lines)} of {line_count})"
        )
    return None


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar_path(coverage_path: Path) -> Path:
    """Where the source digests for ``coverage_path`` live."""
    return coverage_path.with_name(coverage_path.name + ".sources.json")


def write_source_digests(coverage_path: Path, repo_root: Path) -> dict[str, str]:
    """Record a SHA-256 of every source file the report covers, as it is right now.

    Run straight after the tests (``make test`` and CI do), so the digests describe the
    source that was measured. ``check`` then refuses a report whose source has changed.
    """
    files = json.loads(coverage_path.read_text()).get("files", {})
    digests = {
        rel: _digest_file(repo_root / rel) for rel in sorted(files) if (repo_root / rel).is_file()
    }
    sidecar_path(coverage_path).write_text(json.dumps({"files": digests}, indent=1) + "\n")
    return digests


def load_source_digests(coverage_path: Path) -> dict[str, str]:
    data = json.loads(sidecar_path(coverage_path).read_text())
    digests: dict[str, str] = data["files"]
    return digests


def collect_scores(
    src_root: Path,
    repo_root: Path,
    coverage: dict[str, Any],
    digests: dict[str, str] | None = None,
) -> dict[str, FunctionScore]:
    """Score every function under ``src_root``; keys are ``relpath::Qual.name``.

    Raises ``CoverageDataError`` when the report is missing evidence for any function.
    With ``digests`` (relative path to SHA-256 of the source when it was measured) it also
    refuses a file edited since: line numbers alone cannot show that a report is stale.
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
        if digests is not None:
            recorded = digests.get(rel)
            if recorded is None:
                problems.append(f"{rel}: no recorded source digest")
                continue
            if recorded != _digest_file(path):
                problems.append(f"{rel}: source changed since the coverage report was measured")
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
            fraction, unaccounted = _body_coverage(node, executed, missing, excluded)
            if unaccounted:
                lines = ", ".join(str(n) for n in unaccounted)
                problems.append(
                    f"{key}: statements at line(s) {lines} are not in the coverage report"
                )
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
    reason: str | None = None,
    today: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Baseline for ``scores``. Refuses to record any rule violation unless told to.

    With ``allow_regress`` (which needs a ``reason``) the violations are waived. Each
    affected entry records ``"waived": {"rule", "reason", "date"}``. Provenance of an
    earlier waiver is carried forward by later updates, so a routine refresh cannot erase
    who accepted what and why.
    """
    violations = waived_violations(scores, previous, threshold, tolerance)
    if violations and not allow_regress:
        raise ValueError("; ".join(f"{v.key}: {v.message}" for v in violations))
    if allow_regress and not (reason and reason.strip()):
        raise ValueError("--allow-regress needs a reason (--reason TEXT) that is recorded")
    date = today or datetime.date.today().isoformat()
    baseline = {key: _entry(score) for key, score in sorted(scores.items())}
    for key, entry in baseline.items():
        earlier = previous.get(key, {}).get("waived")
        if earlier:
            entry["waived"] = earlier
    for v in violations:
        baseline[v.key]["waived"] = {"rule": v.kind, "reason": (reason or "").strip(), "date": date}
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
    parser.add_argument("command", choices=["check", "update", "report", "stamp"])
    parser.add_argument("--coverage", type=Path, default=Path("coverage.json"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--reason",
        help="update --allow-regress: why the violations are accepted; recorded with the date",
    )
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
    if args.command == "stamp":
        stamped = write_source_digests(args.coverage, args.repo_root.resolve())
        print(f"recorded source digests for {len(stamped)} files: {sidecar_path(args.coverage)}")
        return 0
    coverage = json.loads(args.coverage.read_text())
    if not sidecar_path(args.coverage).exists():
        print(
            f"no source digests next to {args.coverage}; run "
            f"'python tools/crap_ratchet.py stamp --coverage {args.coverage}' right after the "
            "tests (make test does this)",
            file=sys.stderr,
        )
        return 2
    try:
        scores = collect_scores(
            args.src.resolve(),
            args.repo_root.resolve(),
            coverage,
            load_source_digests(args.coverage),
        )
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
        if args.allow_regress and not (args.reason and args.reason.strip()):
            print(
                "refusing to update the baseline: --allow-regress needs --reason", file=sys.stderr
            )
            return 2
        previous = baseline or {}
        try:
            new = build_baseline(
                scores,
                previous,
                args.allow_regress,
                args.threshold,
                args.tolerance,
                reason=args.reason,
            )
        except ValueError as exc:
            print(f"refusing to update the baseline: {exc}", file=sys.stderr)
            return 1
        waived = [key for key, entry in new.items() if "waived" in entry]
        fresh = [k for k in waived if new[k]["waived"] != previous.get(k, {}).get("waived")]
        for key in fresh:
            info = new[key]["waived"]
            print(f"WAIVED {info['rule']}: {key} ({info['reason']}, {info['date']})")
        args.baseline.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        print(
            f"baseline written: {len(new)} functions, {len(fresh)} newly waived, "
            f"{len(waived)} carrying a waiver"
        )
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
