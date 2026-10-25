"""Tests for tools/crap_ratchet.py: the CRAP ratchet gate."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
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


def _cov(
    executed: list[int],
    missing: list[int],
    path: str = "src/pkg/mod.py",
    excluded: list[int] | None = None,
) -> dict[str, Any]:
    entry = {"executed_lines": executed, "missing_lines": missing, "excluded_lines": excluded or []}
    return {"files": {path: entry}}


def _all_executed(source: str, path: str = "src/pkg/mod.py") -> dict[str, Any]:
    """Coverage in which every statement line ran: for tests about keys and complexity."""
    lines = sorted(
        {
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.stmt) and not _is_constant_expr(node)
        }
    )
    return _cov(lines, [], path)


def _is_constant_expr(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


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
        scores = cr.collect_scores(pkg, root, _all_executed(SRC))
        assert set(scores) == {
            "src/pkg/mod.py::simple",
            "src/pkg/mod.py::branchy",
            "src/pkg/mod.py::Box.method",
            "src/pkg/mod.py::Box.prop",
            "src/pkg/mod.py::Box.prop#2",
            "src/pkg/mod.py::runner",
            "src/pkg/mod.py::runner.<locals>.inner",
        }
        moved = "\n\n" + SRC
        (pkg / "mod.py").write_text(moved)
        shifted = cr.collect_scores(pkg, root, _all_executed(moved))
        assert set(shifted) == set(scores)

    def test_complexity_comes_from_radon(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        scores = cr.collect_scores(pkg, root, _all_executed(SRC))
        assert scores["src/pkg/mod.py::simple"].complexity == 1
        assert scores["src/pkg/mod.py::branchy"].complexity == 4
        assert scores["src/pkg/mod.py::Box.method"].complexity == 2

    def test_coverage_is_the_executed_share_of_the_functions_own_body(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        # branchy: 9 def, 10 if, 11 ret, 12 if, 13 ret, 14 for, 15 pass, 16 ret.
        # The def line runs at import and is not part of the body.
        others = [
            n
            for n in _all_executed(SRC)["files"]["src/pkg/mod.py"]["executed_lines"]
            if not 9 <= n <= 16
        ]
        scores = cr.collect_scores(pkg, root, _cov([*others, 9, 10, 11, 12], [13, 14, 15, 16]))
        got = scores["src/pkg/mod.py::branchy"]
        assert got.coverage == pytest.approx(3 / 7)
        assert got.crap == pytest.approx(cr.crap(4, 3 / 7))

    def test_typing_overloads_are_not_scored(self, tmp_path: Path) -> None:
        source = (
            "from typing import overload\n\n\n"
            "@overload\ndef run(x: int) -> int: ...\n\n\n"
            "@overload\ndef run(x: str) -> str: ...\n\n\n"
            "def run(x):\n    if x:\n        return x\n    return 0\n"
        )
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _all_executed(source))
        assert set(scores) == {"src/pkg/mod.py::run"}
        assert scores["src/pkg/mod.py::run"].complexity == 2

    def test_digest_ignores_docstrings_and_line_moves_but_not_logic(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)

        def digest(source: str) -> str:
            (pkg / "mod.py").write_text(source)
            scores = cr.collect_scores(pkg, root, _all_executed(source))
            return scores["src/pkg/mod.py::simple"].digest

        before = digest(SRC)
        same = digest(
            SRC.replace('"""doc"""', '"""other doc"""').replace("def simple", "\n\ndef simple")
        )
        assert same == before
        assert digest(SRC.replace("return x", "return x + 1")) != before


NESTED = """\
def outer(x):
    def inner(y):
        if y:
            return 1
        return 2

    return inner(x)
"""


class TestBodyAttribution:
    def test_nested_function_bodies_belong_to_the_nested_function(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path, NESTED)
        # outer runs its `def inner` (2) and `return inner(x)` (7); inner's body never ran.
        scores = cr.collect_scores(pkg, root, _cov([1, 2, 7], [3, 4, 5]))
        assert scores["src/pkg/mod.py::outer"].coverage == pytest.approx(1.0)
        assert scores["src/pkg/mod.py::outer.<locals>.inner"].coverage == pytest.approx(0.0)

    def test_excluded_lines_count_as_uncovered(self, tmp_path: Path) -> None:
        source = "def f(x):\n    if x:\n        return 1  # pragma: no cover\n    return 2\n"
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _cov([1, 2, 4], [], excluded=[3]))
        assert scores["src/pkg/mod.py::f"].coverage == pytest.approx(2 / 3)

    def test_an_inline_body_cannot_be_credited_by_its_own_def_line(self, tmp_path: Path) -> None:
        source = "def f(x): return x + 1\n"
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _cov([1], []))
        assert scores["src/pkg/mod.py::f"].coverage == 0.0

    def test_empty_and_abstract_bodies_need_no_measurement(self, tmp_path: Path) -> None:
        source = (
            "class P:\n"
            "    def stub(self): ...\n\n"
            "    def documented(self):\n"
            '        """Only a docstring."""\n\n'
            "    def spread(self):\n"
            "        ...\n"
        )
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _cov([1, 2, 4, 7], []))
        assert {k: v.coverage for k, v in scores.items()} == {
            "src/pkg/mod.py::P.stub": 1.0,
            "src/pkg/mod.py::P.documented": 1.0,
            "src/pkg/mod.py::P.spread": 1.0,
        }


class TestNestedClasses:
    FACTORY = (
        "def factory():\n"
        "    class Local:\n"
        "        def method(self, x):\n"
        + "".join(f"            if x == {i}:\n                return {i}\n" for i in range(35))
        + "            return None\n\n"
        "    return Local\n"
        "\n\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        def deep(self, y):\n"
        "            if y:\n"
        "                return 1\n"
        "            return 2\n"
    )

    def test_methods_of_classes_nested_in_functions_and_classes_are_scored(
        self, tmp_path: Path
    ) -> None:
        pkg, root = _write_pkg(tmp_path, self.FACTORY)
        scores = cr.collect_scores(pkg, root, _all_executed(self.FACTORY))
        assert set(scores) == {
            "src/pkg/mod.py::factory",
            "src/pkg/mod.py::factory.<locals>.Local.method",
            "src/pkg/mod.py::Outer.Inner.deep",
        }
        assert scores["src/pkg/mod.py::factory"].complexity == 1
        assert scores["src/pkg/mod.py::factory.<locals>.Local.method"].complexity == 36
        assert scores["src/pkg/mod.py::Outer.Inner.deep"].complexity == 2

    def test_a_35_branch_nested_method_fails_the_check(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path, "def factory():\n    return 1\n")
        cov = root / "coverage.json"
        base = root / "baseline.json"
        common = ["--src", str(pkg), "--repo-root", str(root), "--coverage", str(cov)]
        cov.write_text(json.dumps(_all_executed("def factory():\n    return 1\n")))
        assert cr.main(["update", *common, "--baseline", str(base)]) == 0
        (pkg / "mod.py").write_text(self.FACTORY)
        cov.write_text(json.dumps(_all_executed(self.FACTORY)))
        assert cr.main(["check", *common, "--baseline", str(base)]) == 1


class TestMissingCoverageEvidence:
    def test_a_report_without_the_file_fails(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        with pytest.raises(cr.CoverageDataError, match=r"src/pkg/mod\.py"):
            cr.collect_scores(pkg, root, {"files": {}})

    def test_a_new_module_left_out_of_the_report_fails(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        (pkg / "fresh.py").write_text("def risky(x):\n    if x:\n        return 1\n    return 2\n")
        with pytest.raises(cr.CoverageDataError, match=r"src/pkg/fresh\.py") as exc:
            cr.collect_scores(pkg, root, _all_executed(SRC))
        assert "mod.py" not in str(exc.value)

    def test_a_path_mismatch_fails(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path)
        cov = _all_executed(SRC, path="elsewhere/src/pkg/mod.py")
        with pytest.raises(cr.CoverageDataError, match=r"src/pkg/mod\.py"):
            cr.collect_scores(pkg, root, cov)

    def test_a_body_with_no_measurements_fails(self, tmp_path: Path) -> None:
        source = "def branchy(x):\n    if x > 1:\n        return 1\n    return 0\n"
        pkg, root = _write_pkg(tmp_path, source)
        # the file is reported and its def line ran, but nothing about the body lines
        with pytest.raises(cr.CoverageDataError, match=r"branchy"):
            cr.collect_scores(pkg, root, _cov([1], []))

    def test_a_report_longer_than_the_source_fails(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path, "def f():\n    return 1\n")
        with pytest.raises(cr.CoverageDataError, match="does not match"):
            cr.collect_scores(pkg, root, _cov([1, 2, 99], []))

    def test_the_cli_exits_with_an_error_and_names_the_problem(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pkg, root = _write_pkg(tmp_path)
        cov = root / "coverage.json"
        cov.write_text(json.dumps({"files": {}}))
        rc = cr.main(["check", "--src", str(pkg), "--repo-root", str(root), "--coverage", str(cov)])
        assert rc == 2
        assert "mod.py" in capsys.readouterr().err


class TestStaleOrPartialCoverage:
    def test_statements_the_report_does_not_mention_fail(self, tmp_path: Path) -> None:
        source = "def f(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c + x\n"
        pkg, root = _write_pkg(tmp_path, source)
        # only the first body statement was measured; lines 3-5 are new or unreported
        with pytest.raises(
            cr.CoverageDataError,
            match=r"f: statements at line\(s\) 3, 4, 5 are not in the coverage report",
        ):
            cr.collect_scores(pkg, root, _cov([1, 2], []))

    def test_lines_that_have_no_statement_need_no_data(self, tmp_path: Path) -> None:
        source = (
            "counter = 0\n\n\n"
            "def f(x):\n"
            '    """doc"""\n'
            "    global counter\n"
            "    y: int\n"
            "    counter = x\n"
            "    return counter\n"
        )
        pkg, root = _write_pkg(tmp_path, source)
        scores = cr.collect_scores(pkg, root, _cov([1, 4, 8, 9], []))
        assert scores["src/pkg/mod.py::f"].coverage == 1.0

    def test_a_stale_real_report_for_an_expanded_function_fails(self, tmp_path: Path) -> None:
        small = "def risky(a, b, c, d, e, f):\n    return f\n"
        pkg, root = _write_pkg(tmp_path, small)
        (pkg / "__init__.py").write_text("")
        report = _run_under_coverage(root, "import pkg.mod\npkg.mod.risky(1, 2, 3, 4, 5, 6)\n")
        assert cr.collect_scores(pkg, root, report)["src/pkg/mod.py::risky"].coverage == 1.0
        (pkg / "mod.py").write_text(
            "def risky(a, b, c, d, e, f):\n"
            "    if a or b or c or d or e:\n"
            "        return 1\n"
            "    return 0\n"
        )
        with pytest.raises(cr.CoverageDataError, match="risky"):
            cr.collect_scores(pkg, root, report)


def _run_under_coverage(root: Path, driver: str) -> dict[str, Any]:
    """Run ``driver`` under a real coverage.py and return its JSON report."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_", "COVERAGE_"))}
    data = root / ".coverage-data"
    script = root / "driver.py"
    script.write_text(driver)
    run = [sys.executable, "-m", "coverage"]
    subprocess.run(
        [*run, "run", f"--data-file={data}", "--source=src/pkg", str(script)],
        cwd=root,
        env=env | {"PYTHONPATH": str(root / "src")},
        check=True,
        capture_output=True,
    )
    out = root / "coverage.json"
    subprocess.run(
        [*run, "json", f"--data-file={data}", "-o", str(out)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
    )
    report: dict[str, Any] = json.loads(out.read_text())
    return report


class TestWithRealCoverage:
    NEVER_CALLED = textwrap.dedent(
        """\
        def risky(a, b, c, d, e):
            if a or b or c or d or e:
                return 1
            return 0
        """
    )

    def test_a_function_that_is_imported_but_never_called_scores_as_uncovered(
        self, tmp_path: Path
    ) -> None:
        pkg, root = _write_pkg(tmp_path, self.NEVER_CALLED)
        (pkg / "__init__.py").write_text("")
        report = _run_under_coverage(root, "import pkg.mod\n")
        score = cr.collect_scores(pkg, root, report)["src/pkg/mod.py::risky"]
        assert score.complexity == 6
        assert score.coverage == 0.0
        assert score.crap == pytest.approx(cr.crap(6, 0.0))
        assert score.crap > cr.DEFAULT_THRESHOLD
        assert [v.kind for v in cr.compare({score.key: score}, {})] == ["new-over-threshold"]

    def test_the_same_function_once_called_is_covered(self, tmp_path: Path) -> None:
        pkg, root = _write_pkg(tmp_path, self.NEVER_CALLED)
        (pkg / "__init__.py").write_text("")
        report = _run_under_coverage(
            root, "import pkg.mod\npkg.mod.risky(0, 0, 0, 0, 1)\npkg.mod.risky(0, 0, 0, 0, 0)\n"
        )
        score = cr.collect_scores(pkg, root, report)["src/pkg/mod.py::risky"]
        assert score.coverage == pytest.approx(1.0)


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
        out = cr.build_baseline(
            {worse.key: worse}, _base(old), allow_regress=True, reason="measurement change"
        )
        assert out["a.py::f"]["coverage"] == pytest.approx(0.1)

    def test_update_refuses_new_function_over_threshold(self) -> None:
        new = _score("a.py::f", 10, 0.0)
        with pytest.raises(ValueError, match=r"a\.py::f"):
            cr.build_baseline({new.key: new}, {})

    def test_update_refuses_a_changed_function_that_stays_over_threshold(self) -> None:
        old = _score("a.py::f", 40, 1.0, digest="old")
        edited = _score("a.py::f", 40, 1.0, digest="new")
        assert [v.kind for v in cr.compare({edited.key: edited}, _base(old))] == [
            "changed-over-threshold"
        ]
        with pytest.raises(ValueError, match=r"a\.py::f"):
            cr.build_baseline({edited.key: edited}, _base(old))

    def test_allow_regress_records_rule_reason_and_date(self) -> None:
        old = _score("a.py::f", 40, 1.0, digest="old")
        edited = _score("a.py::f", 40, 1.0, digest="new")
        out = cr.build_baseline(
            {edited.key: edited},
            _base(old),
            allow_regress=True,
            reason="split planned in AGT-1",
            today="2026-10-25",
        )
        assert out["a.py::f"]["waived"] == {
            "rule": "changed-over-threshold",
            "reason": "split planned in AGT-1",
            "date": "2026-10-25",
        }

    def test_allow_regress_needs_a_reason(self) -> None:
        old = _score("a.py::f", 40, 1.0, digest="old")
        edited = _score("a.py::f", 40, 1.0, digest="new")
        for reason in (None, "", "   "):
            with pytest.raises(ValueError, match="reason"):
                cr.build_baseline(
                    {edited.key: edited}, _base(old), allow_regress=True, reason=reason
                )

    def test_a_plain_update_keeps_the_waiver_provenance(self) -> None:
        waived = {"rule": "worse", "reason": "rebaseline", "date": "2026-10-25"}
        old = _score("a.py::f", 10, 0.5)
        previous = _base(old)
        previous["a.py::f"]["waived"] = waived
        better = _score("a.py::f", 10, 0.9)
        out = cr.build_baseline({better.key: better}, previous)
        assert out["a.py::f"]["waived"] == waived
        assert out["a.py::f"]["coverage"] == pytest.approx(0.9)

    def test_a_new_waiver_replaces_the_old_provenance(self) -> None:
        previous = _base(_score("a.py::f", 10, 0.9))
        previous["a.py::f"]["waived"] = {"rule": "worse", "reason": "old", "date": "2026-01-01"}
        worse = _score("a.py::f", 10, 0.1)
        out = cr.build_baseline(
            {worse.key: worse}, previous, allow_regress=True, reason="new", today="2026-10-25"
        )
        assert out["a.py::f"]["waived"]["reason"] == "new"

    def test_the_cli_lists_every_waived_violation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pkg, root = _write_pkg(tmp_path)
        cov = root / "coverage.json"
        cov.write_text(json.dumps(_all_executed(SRC)))
        base = root / "baseline.json"
        common = ["--src", str(pkg), "--repo-root", str(root), "--coverage", str(cov)]
        assert cr.main(["update", *common, "--baseline", str(base)]) == 0
        lost = _all_executed(SRC)["files"]["src/pkg/mod.py"]["executed_lines"]
        cov.write_text(json.dumps(_cov([], lost)))
        capsys.readouterr()
        assert cr.main(["update", *common, "--baseline", str(base)]) == 1
        assert cr.main(["update", *common, "--baseline", str(base), "--allow-regress"]) == 2
        assert "reason" in capsys.readouterr().err
        argv = ["update", *common, "--baseline", str(base), "--allow-regress"]
        assert cr.main([*argv, "--reason", "coverage lost on purpose"]) == 0
        out = capsys.readouterr().out
        assert "WAIVED" in out and "branchy" in out and "coverage lost on purpose" in out
        entry = json.loads(base.read_text())["src/pkg/mod.py::branchy"]
        assert entry["waived"]["reason"] == "coverage lost on purpose"
        assert entry["waived"]["date"]

    def test_committed_baseline_is_well_formed(self) -> None:
        path = Path(__file__).resolve().parents[2] / "tools" / "crap_baseline.json"
        data = json.loads(path.read_text())
        assert data, "baseline must not be empty"
        for key, entry in data.items():
            assert "::" in key
            assert set(entry) - {"waived"} == {"complexity", "coverage", "crap", "digest"}
            if "waived" in entry:
                assert set(entry["waived"]) == {"rule", "reason", "date"}, key
                assert all(entry["waived"].values()), key


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
        cov.write_text(json.dumps(_all_executed(SRC)))
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
        lost = _all_executed(SRC)["files"]["src/pkg/mod.py"]["executed_lines"]
        cov.write_text(json.dumps(_cov([], lost)))
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
