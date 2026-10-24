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


def test_a_job_compares_the_public_api_with_the_last_release() -> None:
    job = _jobs()["api-compat"]
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["fetch-depth"] == 0, "the release tag must be in the clone"
    assert any("tools/api_breaks.py" in s.get("run", "") for s in job["steps"])


def test_make_ci_runs_every_gate() -> None:
    makefile = (PKG / "Makefile").read_text()
    ci_line = next(line for line in makefile.splitlines() if line.startswith("ci:"))
    for target in ("lint", "typecheck", "arch", "test", "crap", "api-check"):
        assert target in ci_line.split(), f"make ci does not run {target}"
    assert "-n auto" in makefile
