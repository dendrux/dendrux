"""The quality gates are switched on in CI, `make ci` and the dev dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")


def _jobs() -> dict[str, dict]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _steps(job: str) -> dict[str, dict]:
    return {s.get("name", ""): s for s in _jobs()[job]["steps"]}


def test_dev_dependencies_include_the_gate_tools_with_upper_bounds() -> None:
    dev = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    for tool in ("radon", "griffe", "import-linter", "pytest-randomly", "pytest-xdist"):
        spec = next((d for d in dev if d.startswith(tool)), None)
        assert spec is not None, f"{tool} missing from the dev extra"
        assert "<" in spec, f"{tool} needs an upper bound: {spec}"


def test_the_python_job_runs_tests_in_parallel_and_writes_coverage_json() -> None:
    run = _steps("python")["Test"]["run"]
    assert "-n auto" in run
    assert "--cov-report=json:coverage.json" in run


def test_the_python_job_checks_architecture_and_crap() -> None:
    steps = _steps("python")
    assert "lint-imports" in steps["Architecture rules"]["run"]
    crap = steps["CRAP ratchet"]
    assert "tools/crap_ratchet.py check" in crap["run"]
    assert "coverage.json" in crap["run"]
    assert "if" in crap, "run the ratchet on one matrix leg only"


def test_coverage_is_stamped_with_source_digests_before_the_ratchet_runs() -> None:
    names = list(_steps("python"))
    stamp = _steps("python")["Record coverage source digests"]
    assert "tools/crap_ratchet.py stamp" in stamp["run"]
    assert names.index("Record coverage source digests") < names.index("CRAP ratchet")
    assert names.index("Test") < names.index("Record coverage source digests")
    makefile = (PKG / "Makefile").read_text()
    test_recipe = makefile.split("\ntest:", 1)[1].split("\n\n", 1)[0]
    assert "tools/crap_ratchet.py stamp" in test_recipe


def test_a_job_compares_the_public_api_with_the_last_release() -> None:
    job = _jobs()["api-compat"]
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["fetch-depth"] == 0, "the release tag must be in the clone"
    runs = [s["run"] for s in job["steps"] if "tools/api_breaks.py" in s.get("run", "")]
    assert runs
    assert not any("--allow-gaps" in run for run in runs), "CI must not tolerate unmapped breaks"


def test_make_api_check_does_not_tolerate_unmapped_breaks() -> None:
    makefile = (PKG / "Makefile").read_text()
    recipe = makefile.split("api-check:", 1)[1].split("\n\n", 1)[0]
    assert "tools/api_breaks.py" in recipe
    assert "--allow-gaps" not in recipe


def test_make_ci_runs_every_gate() -> None:
    makefile = (PKG / "Makefile").read_text()
    ci_line = next(line for line in makefile.splitlines() if line.startswith("ci:"))
    for target in ("lint", "typecheck", "arch", "test", "crap", "api-check"):
        assert target in ci_line.split(), f"make ci does not run {target}"
    assert "-n auto" in makefile


def test_load_sensitive_probes_run_serially_before_the_parallel_pass() -> None:
    """The HOT statistics probe and the 300k-row writer-block probe measure server behaviour
    that depends on how many sessions share the server; under `-n auto` they failed in
    some full runs and not alone."""
    pyproject = tomllib.loads((PKG / "pyproject.toml").read_text())
    markers = pyproject["tool"]["pytest"]["ini_options"].get("markers", [])
    assert any(m.startswith("serial_probe:") for m in markers)
    probe = (PKG / "tests/integration/test_050_t7b_fix2.py").read_text()
    assert "@pytest.mark.serial_probe\nasync def test_hot_updates_recover" in probe
    migration = (PKG / "tests/integration/test_050_t7b_fix3.py").read_text()
    assert (
        "@pytest.mark.serial_probe\nasync def test_online_clock_migration_writer_block" in migration
    )

    makefile = (PKG / "Makefile").read_text()
    recipe = makefile.split("\ntest:", 1)[1].split("\n\n", 1)[0]
    ci_test = _steps("python")["Test"]["run"]
    for text in (recipe, ci_test):
        assert '-m "not serial_probe"' in text
        assert "-m serial_probe" in text
        assert "--cov-append" in text
        # on a quiet server: after the parallel pass its checkpoints and autovacuum stall writers
        assert text.index("-m serial_probe") < text.index('-m "not serial_probe"')
        assert "--cov-fail-under=80" in text.split('-m "not serial_probe"')[1]
