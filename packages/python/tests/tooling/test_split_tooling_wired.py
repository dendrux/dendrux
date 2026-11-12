"""The split tooling and the per-test timeout are switched on."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def _pytest_config() -> dict:
    return tomllib.loads((PKG / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]


def test_pytest_timeout_is_a_pinned_dev_dependency() -> None:
    dev = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert any(spec.startswith("pytest-timeout==") for spec in dev["dev"])


def test_each_test_is_stopped_after_300_seconds_after_the_stack_dump() -> None:
    config = _pytest_config()
    assert config["timeout"] == 300
    assert config["faulthandler_timeout"] < config["timeout"]


def test_the_make_targets_exist() -> None:
    makefile = (PKG / "Makefile").read_text()
    assert "\nmove-check:" in makefile
    assert "tools/move_check.py" in makefile
    assert "\nstale-patches:" in makefile
    assert "tools/stale_patch_lint.py check" in makefile
    assert "stale-patches" in makefile.split("\nquality:")[1].splitlines()[0]


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")
def test_the_quality_job_runs_the_stale_patch_guard() -> None:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["quality"]["steps"]
    runs = [step.get("run", "") for step in steps]
    assert "python tools/stale_patch_lint.py check" in runs
