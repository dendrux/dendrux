"""CRAP ratchet: a score rise on an unchanged function warns; a big rise or any rise on a
changed function fails (run-to-run coverage variance must not fail the build)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tools import crap_ratchet as cr

if TYPE_CHECKING:
    from pathlib import Path

KEY = "src/dendrux/a.py::f"


def _score(crap: float, digest: str = "d") -> cr.FunctionScore:
    return cr.FunctionScore(KEY, 10, 0.5, crap, digest)


def _baseline(crap: float, digest: str = "d") -> dict[str, dict[str, object]]:
    return {KEY: {"complexity": 10, "coverage": 0.5, "crap": crap, "digest": digest}}


def test_same_digest_rise_of_9_percent_warns_and_does_not_fail() -> None:
    failures, warnings = cr.compare_split({KEY: _score(21.8)}, _baseline(20.0))
    assert failures == []
    assert [w.kind for w in warnings] == ["worse-unchanged"]


def test_same_digest_rise_of_30_percent_fails() -> None:
    failures, warnings = cr.compare_split({KEY: _score(26.0)}, _baseline(20.0))
    assert [f.kind for f in failures] == ["worse"]
    assert warnings == []


def test_exactly_25_percent_warns_and_just_over_fails() -> None:
    assert cr.compare_split({KEY: _score(25.0)}, _baseline(20.0))[0] == []
    assert [f.kind for f in cr.compare_split({KEY: _score(25.1)}, _baseline(20.0))[0]] == ["worse"]


def test_changed_digest_rise_of_0_6_fails() -> None:
    failures, _ = cr.compare_split({KEY: _score(20.6, "new")}, _baseline(20.0, "old"))
    assert [f.kind for f in failures] == ["worse"]


def test_rise_inside_the_tolerance_is_silent() -> None:
    assert cr.compare_split({KEY: _score(20.4)}, _baseline(20.0)) == ([], [])


def test_the_strict_compare_used_by_update_still_reports_the_small_rise() -> None:
    assert [v.kind for v in cr.compare({KEY: _score(21.8)}, _baseline(20.0))] == ["worse"]


def _coverage_and_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path
    src = root / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    code = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    (src / "a.py").write_text(code)
    report = {
        "files": {
            "src/dendrux/__init__.py": {
                "executed_lines": [],
                "missing_lines": [],
                "excluded_lines": [],
            },
            "src/dendrux/a.py": {
                "executed_lines": [2, 3],
                "missing_lines": [4],
                "excluded_lines": [],
            },
        }
    }
    cov = root / "coverage.json"
    cov.write_text(json.dumps(report))
    cr.write_source_digests(cov, root)
    return root, src, cov


def test_end_to_end_check_exits_zero_with_a_warning(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, src, cov = _coverage_and_tree(tmp_path)
    scores = cr.collect_scores(src, root, json.loads(cov.read_text()), cr.load_source_digests(cov))
    score = scores["src/dendrux/a.py::f"]
    # an old baseline entry of the same function whose score was 9 percent lower
    old = {**cr._entry(score), "crap": round(score.crap / 1.09, 2)}
    base = tmp_path / "base.json"
    base.write_text(json.dumps({score.key: old}))
    args = ["check", "--coverage", str(cov), "--baseline", str(base), "--src", str(src)]
    args += ["--tolerance", "0"]  # this toy function scores about 2, below the usual noise band
    assert cr.main([*args, "--repo-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "CRAP warning" in out and "CRAP ratchet ok" in out
    old["crap"] = round(score.crap / 1.4, 2)  # a 40 percent rise now
    base.write_text(json.dumps({score.key: old}))
    assert cr.main([*args, "--repo-root", str(root)]) == 1
