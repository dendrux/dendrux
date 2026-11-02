"""Debt reporting is independent of the hard CRAP admission rules."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

import pytest

from tools import crap_ratchet as cr

if TYPE_CHECKING:
    from pathlib import Path


def _fixture(tmp_path: Path) -> tuple[list[str], Path, Path, dict[str, cr.FunctionScore]]:
    source = (
        "\n\n".join(
            f"def f{i}(x):\n    if x or x or x or x or x:\n        return 1\n    return 0"
            for i in range(13)
        )
        + "\n\ndef small():\n    return 1\n"
    )
    src = tmp_path / "src/dendrux"
    src.mkdir(parents=True)
    (src / "debt.py").write_text(source)
    tree = ast.parse(source)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "files": {
                    "src/dendrux/debt.py": {
                        "executed_lines": [node.lineno for node in tree.body],
                        "missing_lines": [
                            node.lineno
                            for node in ast.walk(tree)
                            if isinstance(node, ast.stmt) and not isinstance(node, ast.FunctionDef)
                        ],
                    }
                }
            }
        )
    )
    cr.write_source_digests(coverage, tmp_path)
    scores = cr.collect_scores(src, tmp_path, json.loads(coverage.read_text()))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({key: cr._entry(score) for key, score in scores.items()}))
    return (
        [
            "--src",
            str(src),
            "--repo-root",
            str(tmp_path),
            "--coverage",
            str(coverage),
            "--baseline",
            str(baseline),
        ],
        coverage,
        baseline,
        scores,
    )


def test_debt_summary_counts_excess_and_top_ten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args, coverage, _, scores = _fixture(tmp_path)
    assert cr.main(["report", *args]) == 0
    output = capsys.readouterr().out
    assert "CRAP debt: 13 function(s) above 30" in output
    assert "total excess 156.00" in output
    summary = json.loads(coverage.with_name("coverage.json.crap.json").read_text())
    assert summary["functions_above_threshold"] == 13
    assert summary["total_excess"] == pytest.approx(156.0)
    assert len(summary["top"]) == 10
    expected = sorted(scores.values(), key=lambda score: (-score.crap, score.key))[:10]
    assert [entry["key"] for entry in summary["top"]] == [score.key for score in expected]


def test_touched_but_not_reduced_is_reported_without_blocking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args, coverage, baseline, scores = _fixture(tmp_path)
    previous = json.loads(baseline.read_text())
    held = "src/dendrux/debt.py::f0"
    improved = "src/dendrux/debt.py::f1"
    small = "src/dendrux/debt.py::small"
    previous[held]["digest"] = "edited"
    previous[improved].update(digest="edited", crap=43.0)
    previous[small]["digest"] = "edited"
    baseline.write_text(json.dumps(previous))
    assert cr.main(["check", *args]) == 0
    output = capsys.readouterr().out
    assert "CRAP warning: touched but not reduced" in output and held in output
    summary = json.loads(coverage.with_name("coverage.json.crap.json").read_text())
    assert [entry["key"] for entry in summary["touched_but_not_reduced"]] == [held]
    assert summary["touched_but_not_reduced"][0]["crap"] == scores[held].crap


def test_a_failing_check_still_writes_debt_evidence(tmp_path: Path) -> None:
    args, coverage, baseline, _ = _fixture(tmp_path)
    baseline.write_text("{}")
    assert cr.main(["check", *args]) == 1
    summary = json.loads(coverage.with_name("coverage.json.crap.json").read_text())
    assert summary["functions_above_threshold"] == 13
    assert summary["touched_but_not_reduced"] == []
