"""Shrink-only per-file/rule counts for local imports, complexity and broad catches.

Use the project's Ruff config plus RULES; all other diagnostics are left to lint.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

RULES = ("PLC0415", "PLR0911", "PLR0912", "PLR0915", "BLE001")
BASELINE = Path(__file__).resolve().parent / "ruff_baseline.json"
CONFIG = r.PACKAGE_ROOT / "pyproject.toml"
NOQA = re.compile(r"#\s*(?:(?:ruff|flake8):\s*)?noqa\b", re.IGNORECASE)
NOQA_CODES = re.compile(r"\s*:\s*([A-Z]+\d+(?:[,\s]+[A-Z]+\d+)*)", re.IGNORECASE)
NOQA_REASON_PREFIX = "noqa-reason|"


@dataclass(frozen=True)
class NoqaViolation:
    """Location of a ratcheted suppression without an explanatory comment."""

    path: Path
    line: int
    code: str

    @property
    def key(self) -> str:
        """Return a stable reason-debt key distinct from ordinary Ruff counts."""
        return f"{NOQA_REASON_PREFIX}{r.module_name(self.path)}|{self.code}"


def _has_reason(text: str) -> bool:
    if NOQA.search(text):
        return False
    # Prose is required; a bare code, empty comment or punctuation is not a reason.
    return any(word.replace("_", "").isalpha() for word in re.findall(r"\w+", text))


def unreasoned_noqas(src: Path = r.SRC) -> list[NoqaViolation]:
    """Inspect actual comments, including blanket/file-wide noqa, never string literals.

    A reason is prose after the code list or a standalone comment immediately above.
    Human review decides whether that explanation justifies the suppression.
    """
    violations: list[NoqaViolation] = []
    for path in r.iter_sources(src):
        source = path.read_text(encoding="utf-8")
        lines = source.split("\n")
        comments = [
            token
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
        standalone = {
            token.start[0]: token.string
            for token in comments
            if not lines[token.start[0] - 1][: token.start[1]].strip()
        }
        for token in comments:
            marker = NOQA.search(token.string)
            if marker is None:
                continue
            tail = token.string[marker.end() :]
            listed = NOQA_CODES.match(tail)
            codes = set(re.findall(r"[A-Z]+\d+", listed[1].upper())) if listed else set(RULES)
            reason = tail[listed.end() :] if listed else tail
            if _has_reason(reason) or _has_reason(standalone.get(token.start[0] - 1, "")):
                continue
            violations.extend(
                NoqaViolation(path, token.start[0], code) for code in sorted(codes & set(RULES))
            )
    return violations


def noqa_counts(violations: Iterable[NoqaViolation]) -> dict[str, int]:
    """Record separate shrink-only reason debt per module/rule, without line-number keys."""
    counts: dict[str, int] = {}
    for violation in violations:
        counts[violation.key] = counts.get(violation.key, 0) + 1
    return counts


class RuffFinding(TypedDict):
    """Fields consumed from Ruff's JSON output; other diagnostic fields are ignored."""

    filename: str
    code: str


def collect_scores(src: Path = r.SRC) -> dict[str, int]:
    """Run pinned Ruff with project settings, retaining only ratcheted rules."""
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        str(src.resolve()),
        "--config",
        str(CONFIG),
        "--extend-select",
        ",".join(RULES),
        "--force-exclude",
        "--extend-exclude",
        "**/migrations/**",
        "--output-format",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Ruff returns 1 for findings, 2 for an execution/configuration error.
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    # Ruff's JSON schema is the tool boundary; only filename/code are consumed.
    report: list[RuffFinding] = json.loads(result.stdout)
    return scores_from_report(report, src)


def scores_from_report(report: Iterable[RuffFinding], src: Path = r.SRC) -> dict[str, int]:
    """Group in-scope findings as module:RULE -> count, without line-number keys."""
    scores: dict[str, int] = {}
    for item in report:
        if item["code"] not in RULES:
            continue
        path = Path(item["filename"]).resolve()
        if not path.is_relative_to(src.resolve()):
            continue
        if r.EXCLUDED_DIRS & set(path.relative_to(src.resolve()).parts):
            continue
        key = f"{r.module_name(path)}:{item['code']}"
        scores[key] = scores.get(key, 0) + 1
    return scores


def compare(scores: Mapping[str, int], baseline: Mapping[str, int]) -> r.CountComparison:
    """Every new file/rule pair and every increase fails."""
    return r.compare_counts(scores, baseline)


def moved_baseline(
    scores: Mapping[str, int],
    baseline: Mapping[str, int],
    pairs: Sequence[tuple[str, str]],
    present_modules: Collection[str],
) -> dict[str, int]:
    """Transfer file/rule allowances, conserving moved plus retained counts per rule.

    Returns a new mapping after validating the entire batch. Unlisted entries stay
    unchanged; callers write only after this succeeds.
    """
    for old, new in pairs:
        _, _, rule = old.rpartition(":")
        if rule not in RULES or new.rpartition(":")[2] != rule:
            raise ValueError(f"{old} -> {new}: moves require the same ratcheted rule")
    present = {old for old, _ in pairs if old.rpartition(":")[0] in present_modules}
    return r.transfer_counts(
        scores,
        baseline,
        pairs,
        present=present,
        retained={key: scores.get(key, 0) for key in present},
        minimum=lambda _key: 0,
    )


def _summary(scores: Mapping[str, int]) -> None:
    print(f"ruff: {sum(scores.values())} findings in {len(scores)} file/rule pairs")
    for rule in RULES:
        print(f"{rule}: {sum(value for key, value in scores.items() if key.endswith(':' + rule))}")


def main(argv: list[str] | None = None) -> int:
    """Check, stamp, report, or explicitly transfer allowances for coordinator-reviewed moves."""
    args = r.transfer_args(argv, description=__doc__, baseline=BASELINE)
    scores = collect_scores(args.src.resolve())
    _summary(scores)
    violations = unreasoned_noqas(args.src.resolve())
    print(f"noqa: {len(violations)} unreasoned suppression(s)")
    for violation in violations:
        print(f"{violation.path}:{violation.line}: {violation.code} noqa lacks reason")
    scores.update(noqa_counts(violations))
    if args.command == "report":
        for key, value in sorted(scores.items()):
            print(f"{key}: {value}")
        return 0
    old = r.baseline_or_none(args.baseline)
    if args.command == "move":
        return r.finish_transfer(
            args.keys,
            label="ruff",
            scores=scores,
            baseline=old,
            path=args.baseline,
            prepare=lambda old, pairs: moved_baseline(
                scores,
                old,
                pairs,
                {r.module_name(path) for path in r.iter_sources(args.src.resolve())},
            ),
        )
    comparison = None if old is None else compare(scores, old)
    if args.command == "stamp" and comparison is not None and comparison.failed:
        print("ruff: refusing to stamp new or worse entries; fix regressions (tools/QUALITY.md)")
        return 1
    return r.finish(
        args.command,
        label="ruff",
        comparison=comparison,
        new_baseline=dict(sorted(scores.items())),
        path=args.baseline,
        describe=lambda key: f"{key} has {scores[key]} findings",
    )


if __name__ == "__main__":
    raise SystemExit(main())
