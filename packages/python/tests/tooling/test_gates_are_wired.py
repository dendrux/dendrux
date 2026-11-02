"""The quality gates are switched on in CI, `make ci` and the dev dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")


def _jobs() -> dict[str, dict]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _steps(job: str) -> dict[str, dict]:
    return {s.get("name", ""): s for s in _jobs()[job]["steps"]}


def test_dev_dependencies_pin_the_tools_exactly() -> None:
    dev = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    requirements = {req.name: req for req in map(Requirement, dev)}
    tools = {
        "mypy",
        "ruff",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "coverage",
        "radon",
        "griffe",
        "import-linter",
        "pytest-randomly",
        "pytest-xdist",
    }
    assert tools <= requirements.keys(), f"tools missing from dev: {tools - requirements.keys()}"
    for name, requirement in requirements.items():
        if name == "dendrux":
            continue
        pins = list(requirement.specifier)
        assert len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version, (
            f"{name} needs an exact dev pin: {requirement}"
        )
    assert str(requirements["griffe"]) in _steps("api-compat")["Install griffe"]["run"]


def test_the_python_job_runs_tests_in_parallel_and_measures_coverage() -> None:
    run = _steps("python")["Test"]["run"]
    assert "-n auto" in run
    assert "--cov=src/dendrux" in run


def test_the_coverage_job_combines_the_shards_and_writes_coverage_json() -> None:
    run = _steps("python-coverage")["Combine coverage and apply the floor"]["run"]
    assert "coverage combine" in run
    assert "--fail-under=80" in run
    assert "coverage json -o coverage.json" in run
    assert set(_jobs()["python-coverage"]["needs"]) == {"python", "mysql-family"}


def test_the_coverage_job_requires_both_mysql_family_artifacts() -> None:
    steps = _jobs()["python-coverage"]["steps"]
    downloads = {
        step["with"].get("name"): step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    }
    for name in ("coverage-mysql", "coverage-mariadb"):
        step = downloads[name]
        assert "if" not in step
        assert not step.get("continue-on-error", False)
        assert step["with"]["path"] == "packages/python"
    assert "--ci" in _steps("python-coverage")["CRAP ratchet"]["run"]


def test_ci_preserves_machine_readable_debt_even_when_crap_fails() -> None:
    steps = _steps("python-coverage")
    assert steps["CRAP ratchet"]["id"] == "crap"
    upload = steps["Upload CRAP debt summary"]
    assert "always()" in upload["if"] and "steps.crap.outcome" in upload["if"]
    assert upload["with"]["path"] == "packages/python/coverage.json.crap.json"
    assert upload["with"]["if-no-files-found"] == "error"


def test_mysql_family_suites_measure_and_upload_coverage() -> None:
    job = _jobs()["mysql-family"]
    assert {cell["server"] for cell in job["strategy"]["matrix"]["include"]} == {
        "mysql",
        "mariadb",
    }
    steps = _steps("mysql-family")
    run = steps["Test on ${{ matrix.server }}"]["run"]
    assert "--cov=src/dendrux" in run and "--no-cov" not in run
    assert steps["Test on ${{ matrix.server }}"]["env"]["COVERAGE_FILE"] == (
        "cov-${{ matrix.server }}.data"
    )
    upload = steps["Upload coverage data"]
    assert str(upload["uses"]).startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "coverage-${{ matrix.server }}"
    assert upload["with"]["path"] == "packages/python/cov-${{ matrix.server }}.data"
    assert upload["with"]["if-no-files-found"] == "error"
    assert not upload.get("continue-on-error", False)


@pytest.mark.parametrize(
    "missing",
    [(), ("mysql",), ("mariadb",), ("mysql", "mariadb"), ("1",)],
)
def test_coverage_combine_requires_every_backend_file(
    tmp_path: Path, missing: tuple[str, ...]
) -> None:
    for name in ("1", "2", "3", "4", "mysql", "mariadb"):
        if name not in missing:
            (tmp_path / f"cov-{name}.data").write_text("coverage fixture")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "coverage"
    stub.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath('coverage-called').touch()\n"
    )
    stub.chmod(0o755)
    run = subprocess.run(
        [
            "bash",
            "-e",
            "-o",
            "pipefail",
            "-c",
            _steps("python-coverage")["Combine coverage and apply the floor"]["run"],
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (run.returncode == 0) == (not missing), run.stderr
    assert (tmp_path / "coverage-called").exists() == (not missing)


def test_the_python_job_checks_architecture_and_the_coverage_job_runs_crap() -> None:
    assert "lint-imports" in _steps("python")["Architecture rules"]["run"]
    crap = _steps("python-coverage")["CRAP ratchet"]
    assert "tools/crap_ratchet.py check" in crap["run"]
    assert "coverage.json" in crap["run"]
    assert "if" in crap, "run the ratchet on one matrix leg only"


def test_coverage_is_stamped_with_source_digests_before_the_ratchet_runs() -> None:
    names = list(_steps("python-coverage"))
    stamp = _steps("python-coverage")["Record coverage source digests"]
    assert "tools/crap_ratchet.py stamp" in stamp["run"]
    assert names.index("Record coverage source digests") < names.index("CRAP ratchet")
    assert names.index("Combine coverage and apply the floor") < names.index(
        "Record coverage source digests"
    )
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
    assert all("--against v0.5.0" in run for run in runs)
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
    assert "-n $(TEST_WORKERS)" in makefile  # local runs: cores capped at 6 (CI uses -n auto)


def test_environment_dependent_benchmarks_stay_out_of_the_default_gates() -> None:
    """The HOT statistics probe and the 300k-row writer-stall probe measure the server, not
    the code: they failed whenever other sessions shared it. The deterministic HOT-eligibility
    test and the migration correctness tests stay in CI; the benchmarks run on request
    (`make bench`, a release step on a quiet machine)."""
    pyproject = tomllib.loads((PKG / "pyproject.toml").read_text())
    markers = pyproject["tool"]["pytest"]["ini_options"].get("markers", [])
    assert any(m.startswith("benchmark:") for m in markers)
    assert not any(m.startswith("serial_probe:") for m in markers)

    probe = (PKG / "tests/integration/test_050_t7b_fix2.py").read_text()
    assert "@pytest.mark.benchmark\nasync def test_hot_updates_recover" in probe
    migration = (PKG / "tests/integration/test_050_t7b_fix3.py").read_text()
    assert "marks=pytest.mark.benchmark" in migration
    # opt-in, so precision wins over flake resistance: serial runs differ by about 1.2 points
    assert 'metrics["9391133f"]["hot_pct"] - 2,' in probe
    assert "async def test_progress_updates_are_hot_eligible" in probe, (
        "keep the deterministic test"
    )

    makefile = (PKG / "Makefile").read_text()
    test_recipe = makefile.split("\ntest:", 1)[1].split("\n\n", 1)[0]
    assert '-m "not benchmark"' in test_recipe
    assert "-m benchmark" not in test_recipe
    bench_recipe = makefile.split("\nbench:", 1)[1].split("\n\n", 1)[0]
    assert "-m benchmark" in bench_recipe and "-n 0" in bench_recipe
    ci_line = next(line for line in makefile.splitlines() if line.startswith("ci:"))
    assert "bench" not in ci_line.split()

    ci_test = _steps("python")["Test"]["run"]
    assert '-m "not benchmark"' in ci_test
    assert "-m benchmark" not in ci_test


def test_a_blocking_benchmark_job_runs_the_speed_tests_serially_on_postgres() -> None:
    """Speed tests are marked benchmark so the parallel shards skip them; CI must still run them."""
    job = _jobs()["benchmarks"]
    assert "continue-on-error" not in job and "continue-on-error" not in str(job["steps"])
    assert "postgres" in job["services"]
    assert "DENDRUX_TEST_PG_URL" in job["env"]
    run = _steps("benchmarks")["Benchmarks"]["run"]
    assert "-m benchmark" in run and "-n 0" in run
    assert "not benchmark" not in run
    assert "--cov" not in run.replace("--no-cov", "")
    for step in job["steps"]:
        if "uses" in step:
            assert len(step["uses"].split("@")[1].split()[0]) == 40, "pin actions by SHA"
    assert "benchmarks" not in _jobs()["python-coverage"].get("needs", [])
