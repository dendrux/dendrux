"""tools/cognitive_ratchet.py: cognitive-complexity gate (new function > 15 fails)."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from tools import cognitive_ratchet as cog

if TYPE_CHECKING:
    from pathlib import Path

REPORT = [
    {"path": "src/dendrux/a.py", "function_name": "Box::m", "complexity": 20},
    {"path": "src/dendrux/a.py", "function_name": "f", "complexity": 4},
    {"path": "src/dendrux/a.py", "function_name": "f", "complexity": 9},
    {"path": "src/dendrux/pkg/__init__.py", "function_name": "g", "complexity": 16},
]


def test_keys_are_module_and_qualified_name_and_duplicates_keep_the_worst() -> None:
    scores = cog.scores_from_report(REPORT)
    assert scores == {
        "dendrux.a:Box.m": 20,
        "dendrux.a:f": 9,
        "dendrux.pkg:g": 16,
    }


def test_baseline_keeps_only_functions_over_the_limit() -> None:
    scores = cog.scores_from_report(REPORT)
    assert cog.baseline_from_scores(scores, limit=15) == {
        "dendrux.a:Box.m": 20,
        "dendrux.pkg:g": 16,
    }


def test_a_new_function_over_the_limit_fails() -> None:
    result = cog.compare({"dendrux.a:new": 16}, {}, limit=15)
    assert result.new == ["dendrux.a:new"]


def test_a_new_function_at_the_limit_passes() -> None:
    assert not cog.compare({"dendrux.a:new": 15}, {}, limit=15).failed


def test_an_existing_function_may_not_increase() -> None:
    result = cog.compare({"dendrux.a:old": 21}, {"dendrux.a:old": 20}, limit=15)
    assert result.worse == ["dendrux.a:old"]


def test_an_existing_function_may_shrink_and_is_reported_as_improved() -> None:
    result = cog.compare({"dendrux.a:old": 18}, {"dendrux.a:old": 20}, limit=15)
    assert not result.failed
    assert result.improved == ["dendrux.a:old"]


@pytest.mark.skipif(shutil.which("complexipy") is None, reason="complexipy not installed")
def test_end_to_end_on_a_real_tree(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    body = "\n".join(
        f"    if x == {i}:\n        for _ in x:\n            if y:\n                pass"
        for i in range(5)
    )
    (src / "mod.py").write_text(f"def tangled(x, y):\n{body}\n\ndef tiny():\n    return 1\n")
    scores = cog.collect_scores(src)
    assert scores["dendrux.mod:tangled"] > 15
    assert scores["dendrux.mod:tiny"] == 0
    baseline = tmp_path / "base.json"
    assert cog.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 2  # no baseline
    assert cog.main(["stamp", "--src", str(src), "--baseline", str(baseline)]) == 0
    assert cog.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 0
    (src / "mod2.py").write_text((src / "mod.py").read_text().replace("tangled", "tangled_two"))
    assert cog.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 1
    # stamping refuses to record the new violation without a waiver
    assert cog.main(["stamp", "--src", str(src), "--baseline", str(baseline)]) == 1


def test_migrations_are_not_scored() -> None:
    report = [
        {"path": "src/dendrux/db/migrations/versions/x.py", "function_name": "up", "complexity": 40}
    ]
    assert cog.scores_from_report(report) == {}
