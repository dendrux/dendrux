"""Cognitive-complexity gate (complexipy): a new function above the limit fails, and an
existing function above it may not get worse.

Baseline: ``tools/cognitive_baseline.json``, ``{"module:Class.func": score}`` for every
function above the limit (functions at or under it are not recorded). Functions that
share a name in one module (overloads, property setters) are keyed once at their worst
score.

    python tools/cognitive_ratchet.py check
    python tools/cognitive_ratchet.py stamp      # after improving; refuses to add entries
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

LIMIT = 15
BASELINE = Path(__file__).resolve().parent / "cognitive_baseline.json"


def scores_from_report(report: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for item in report:
        if "migrations" in Path(item["path"]).parts:
            continue
        qual = str(item["function_name"]).replace("::", ".")
        key = f"{r.module_name(item['path'])}:{qual}"
        scores[key] = max(scores.get(key, 0), int(item["complexity"]))
    return scores


def baseline_from_scores(scores: Mapping[str, int], limit: int = LIMIT) -> dict[str, int]:
    return {key: value for key, value in sorted(scores.items()) if value > limit}


def compare(
    scores: Mapping[str, int], baseline: Mapping[str, int], limit: int = LIMIT
) -> r.CountComparison:
    return r.compare_counts(scores, baseline, limit=limit)


def collect_scores(src: Path = r.SRC) -> dict[str, int]:
    exe = shutil.which("complexipy") or str(Path(sys.executable).parent / "complexipy")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "complexity.json"
        # The tool's own threshold is irrelevant here (the ratchet decides); set it high.
        cmd = [exe, str(src), "--max-complexity-allowed", "1000", "--quiet"]
        cmd += ["--output-format", "json", "--output", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return scores_from_report(json.loads(out.read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser)
    parser.add_argument("--src", type=Path, default=r.SRC)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args(argv)
    scores = collect_scores(args.src.resolve())
    old = r.baseline_or_none(args.baseline)
    comparison = None if old is None else compare(scores, old, args.limit)
    if args.command == "stamp" and old is None:
        comparison = None
    return r.finish(
        args.command,
        label="cognitive",
        comparison=comparison,
        new_baseline=baseline_from_scores(scores, args.limit),
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda key: f"{key} scores {scores[key]} (limit {args.limit})",
    )


if __name__ == "__main__":
    raise SystemExit(main())
