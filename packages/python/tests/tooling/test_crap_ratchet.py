"""Tests for tools/crap_ratchet.py: the CRAP ratchet gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import crap_ratchet as cr

SRC = '''\
"""module doc"""


def simple(x):
    """doc"""
    return x


def branchy(x):
    if x > 1:
        return 1
    if x > 2:
        return 2
    for _ in range(x):
        pass
    return 0


class Box:
    def method(self, z):
        if z:
            return 1
        return 2

    @property
    def prop(self):
        return 1

    @prop.setter
    def prop(self, v):
        pass


async def runner():
    def inner(y):
        if y:
            return 1
        return 0

    return inner
'''


def _write_pkg(tmp_path: Path, source: str = SRC) -> tuple[Path, Path]:
    root = tmp_path
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text(source)
    return pkg, root


def _cov(executed: list[int], missing: list[int], path: str = "src/pkg/mod.py") -> dict[str, Any]:
    return {"files": {path: {"executed_lines": executed, "missing_lines": missing}}}


def _score(key: str, complexity: int, coverage: float, digest: str = "d") -> cr.FunctionScore:
    return cr.FunctionScore(key, complexity, coverage, cr.crap(complexity, coverage), digest)


def _base(*scores: cr.FunctionScore) -> dict[str, dict[str, Any]]:
    return {
        s.key: {
            "complexity": s.complexity,
            "coverage": s.coverage,
            "crap": s.crap,
            "digest": s.digest,
        }
        for s in scores
    }


class TestFormula:
    @pytest.mark.parametrize(
        ("complexity", "coverage", "expected"),
        [(1, 1.0, 1.0), (10, 0.0, 110.0), (5, 0.5, 8.125), (3, 1.0, 3.0)],
    )
    def test_crap(self, complexity: int, coverage: float, expected: float) -> None:
        assert cr.crap(complexity, coverage) == pytest.approx(expected)


class TestCollect:
    def test_keys_are_qualified_and_stable_across_line_moves(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        scores = cr.collect_scores(pkg, root, _cov([], []))
        assert set(scores) == {
            "src/pkg/mod.py::simple",
            "src/pkg/mod.py::branchy",
            "src/pkg/mod.py::Box.method",
            "src/pkg/mod.py::Box.prop",
            "src/pkg/mod.py::Box.prop#2",
            "src/pkg/mod.py::runner",
            "src/pkg/mod.py::runner.<locals>.inner",
        }
        shifted = cr.collect_scores(pkg, root, _cov([], []))
        assert set(shifted) == set(scores)

    def test_complexity_comes_from_radon(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        scores = cr.collect_scores(pkg, root, _cov([], []))
        assert scores["src/pkg/mod.py::simple"].complexity == 1
        assert scores["src/pkg/mod.py::branchy"].complexity == 4
        assert scores["src/pkg/mod.py::Box.method"].complexity == 2

    def test_coverage_is_the_executed_share_of_statement_lines(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        # branchy spans lines 9-16: 9 def, 10 if, 11 ret, 12 if, 13 ret, 14 for, 15 pass, 16 ret
        executed = [9, 10, 11, 12]
        missing = [13, 14, 15, 16]
        scores = cr.collect_scores(pkg, root, _cov(executed, missing))
        got = scores["src/pkg/mod.py::branchy"]
        assert got.coverage == pytest.approx(0.5)
        assert got.crap == pytest.approx(cr.crap(4, 0.5))

    def test_typing_overloads_are_not_scored(self, tmp_path: Path) -> None:
        source = (
            "from typing import overload\n\n\n"
            "@overload\ndef run(x: int) -> int: ...\n\n\n"
            "@overload\ndef run(x: str) -> str: ...\n\n\n"
            "def run(x):\n    if x:\n        return x\n    return 0\n"
        )
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _cov([], []))
        assert set(scores) == {"src/pkg/mod.py::run"}
        assert scores["src/pkg/mod.py::run"].complexity == 2

    def test_function_without_measured_lines_counts_as_covered(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        scores = cr.collect_scores(pkg, root, {"files": {}})
        assert scores["src/pkg/mod.py::simple"].coverage == 1.0

    def test_digest_ignores_docstrings_and_line_moves_but_not_logic(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        before = cr.collect_scores(pkg, root, _cov([], []))["src/pkg/mod.py::simple"].digest
        (pkg / "mod.py").write_text(
            SRC.replace('"""doc"""', '"""other doc"""').replace("def simple", "\n\ndef simple")
        )
        same = cr.collect_scores(pkg, root, _cov([], []))["src/pkg/mod.py::simple"].digest
        assert same == before
        (pkg / "mod.py").write_text(SRC.replace("return x", "return x + 1"))
        changed = cr.collect_scores(pkg, root, _cov([], []))["src/pkg/mod.py::simple"].digest
        assert changed != before


class TestCompare:
    def test_new_function_over_threshold_fails(self) -> None:
        new = _score("a.py::f", 10, 0.0)  # 110
        violations = cr.compare({new.key: new}, {})
        assert [(v.key, v.kind) for v in violations] == [("a.py::f", "new-over-threshold")]

    def test_new_function_under_threshold_passes(self) -> None:
        new = _score("a.py::f", 5, 1.0)
        assert cr.compare({new.key: new}, {}) == []

    def test_existing_function_getting_worse_fails(self) -> None:
        old = _score("a.py::f", 10, 0.5)
        worse = _score("a.py::f", 10, 0.1)
        violations = cr.compare({worse.key: worse}, _base(old))
        assert [v.kind for v in violations] == ["worse"]

    def test_noise_inside_tolerance_passes(self) -> None:
        old = _score("a.py::f", 10, 0.50)
        near = cr.FunctionScore("a.py::f", 10, 0.499, old.crap + 0.1, "d")
        assert cr.compare({near.key: near}, _base(old), tolerance=0.5) == []

    def test_existing_unchanged_hotspot_may_stay_as_is(self) -> None:
        hot = _score("a.py::f", 20, 0.2)
        assert hot.crap > 30
        assert cr.compare({hot.key: hot}, _base(hot)) == []

    def test_improvement_passes(self) -> None:
        old = _score("a.py::f", 20, 0.2)
        better = _score("a.py::f", 20, 0.9)
        assert cr.compare({better.key: better}, _base(old)) == []

    def test_changed_hotspot_that_stays_over_threshold_fails(self) -> None:
        old = _score("a.py::f", 20, 0.2, digest="old")
        edited = _score("a.py::f", 20, 0.2, digest="new")
        violations = cr.compare({edited.key: edited}, _base(old))
        assert [v.kind for v in violations] == ["changed-over-threshold"]

    def test_changed_function_brought_under_threshold_passes(self) -> None:
        old = _score("a.py::f", 20, 0.2, digest="old")
        fixed = _score("a.py::f", 3, 1.0, digest="new")
        assert cr.compare({fixed.key: fixed}, _base(old)) == []

    def test_removed_functions_are_ignored(self) -> None:
        old = _score("a.py::gone", 20, 0.2)
        assert cr.compare({}, _base(old)) == []


class TestBaseline:
    def test_update_records_improvements_and_drops_removed(self) -> None:
        old = _score("a.py::f", 20, 0.2)
        gone = _score("a.py::gone", 4, 1.0)
        better = _score("a.py::f", 20, 0.9)
        out = cr.build_baseline({better.key: better}, _base(old, gone))
        assert set(out) == {"a.py::f"}
        assert out["a.py::f"]["crap"] == pytest.approx(round(better.crap, 2))

    def test_update_refuses_regressions_unless_allowed(self) -> None:
        old = _score("a.py::f", 10, 0.9)
        worse = _score("a.py::f", 10, 0.1)
        with pytest.raises(ValueError, match=r"a\.py::f"):
            cr.build_baseline({worse.key: worse}, _base(old))
        out = cr.build_baseline({worse.key: worse}, _base(old), allow_regress=True)
        assert out["a.py::f"]["coverage"] == pytest.approx(0.1)

    def test_update_refuses_new_function_over_threshold(self) -> None:
        new = _score("a.py::f", 10, 0.0)
        with pytest.raises(ValueError, match=r"a\.py::f"):
            cr.build_baseline({new.key: new}, {})

    def test_committed_baseline_is_well_formed(self) -> None:
        path = Path(__file__).resolve().parents[2] / "tools" / "crap_baseline.json"
        data = json.loads(path.read_text())
        assert data, "baseline must not be empty"
        for key, entry in data.items():
            assert "::" in key
            assert set(entry) == {"complexity", "coverage", "crap", "digest"}


class TestReport:
    def test_top_n_sorted_worst_first(self) -> None:
        scores = {s.key: s for s in [_score(f"a.py::f{i}", i + 1, 0.0) for i in range(30)]}
        text = cr.format_report(scores, top=5)
        lines = [ln for ln in text.splitlines() if "a.py::f" in ln]
        assert len(lines) == 5
        assert "a.py::f29" in lines[0]
        assert "a.py::f25" in lines[-1]


class TestCli:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        pkg, root = _write_pkg(tmp_path)
        cov = root / "coverage.json"
        cov.write_text(json.dumps(_cov(list(range(1, 60)), [])))
        return pkg, root, cov

    def test_update_then_check_passes_and_regression_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pkg, root, cov = self._setup(tmp_path)
        base = root / "baseline.json"
        common = ["--src", str(pkg), "--repo-root", str(root), "--coverage", str(cov)]
        assert cr.main(["update", *common, "--baseline", str(base)]) == 0
        assert cr.main(["check", *common, "--baseline", str(base)]) == 0
        # lose all coverage: branchy's score rises above its baseline
        cov.write_text(json.dumps(_cov([], list(range(1, 60)))))
        assert cr.main(["check", *common, "--baseline", str(base)]) == 1
        out = capsys.readouterr().out
        assert "branchy" in out
        assert "worse" in out

    def test_check_without_baseline_is_an_error(self, tmp_path: Path) -> None:
        pkg, root, cov = self._setup(tmp_path)
        rc = cr.main(
            [
                "check",
                "--src",
                str(pkg),
                "--repo-root",
                str(root),
                "--coverage",
                str(cov),
                "--baseline",
                str(root / "missing.json"),
            ]
        )
        assert rc == 2
