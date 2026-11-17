"""The quality gates are wired into `make ci`, `make quality` and the CI workflow."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[2]
WORKFLOW = PKG.parents[1] / ".github" / "workflows" / "ci.yml"
MAKEFILE = (PKG / "Makefile").read_text()


def _targets(name: str) -> list[str]:
    line = next(line for line in MAKEFILE.splitlines() if line.startswith(f"{name}:"))
    return line.split(":", 1)[1].split()


def _recipe(name: str) -> str:
    return MAKEFILE.split(f"\n{name}:", 1)[1].split("\n\n", 1)[0]


def test_make_quality_runs_every_gate_including_crap() -> None:
    assert set(_targets("quality")) == {
        "cognitive",
        "duplicates",
        "deadcode",
        "cycles",
        "semgrep",
        "crap",
        "size",
        "ruff-ratchet",
    }


def test_make_ci_runs_quality() -> None:
    assert "quality" in _targets("ci")


def test_crap_is_enforcing() -> None:
    assert "--report-only" not in _recipe("crap")
    assert "tools/crap_ratchet.py check" in _recipe("crap")


def test_quality_stamp_rewrites_every_baseline() -> None:
    recipe = _recipe("quality-stamp")
    for script in (
        "cognitive_ratchet.py",
        "duplicate_ratchet.py",
        "vulture_ratchet.py",
        "import_cycles.py",
        "semgrep_ratchet.py",
        "crap_ratchet.py update",
        "size_ratchet.py",
        "ruff_ratchet.py",
    ):
        assert script in recipe


def test_gate_tools_are_pinned_exactly_in_the_dev_extra() -> None:
    dev = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    for tool in ("complexipy", "pylint", "vulture"):
        assert any(d.startswith(f"{tool}==") for d in dev), f"{tool} needs an exact pin"
    assert not any(d.startswith("semgrep") for d in dev), "semgrep stays out of the dev extra"


def test_semgrep_is_pinned_in_its_own_requirements_file() -> None:
    lines = (PKG / "tools" / "semgrep-requirements.txt").read_text().split()
    assert any(line.startswith("semgrep==") for line in lines)


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repository checkout only")
def test_ci_has_a_quality_job_and_an_enforcing_crap_step() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    quality_runs = "\n".join(s.get("run", "") for s in jobs["quality"]["steps"])
    for script in (
        "cognitive_ratchet.py check",
        "size_ratchet.py check",
        "ruff_ratchet.py check",
        "duplicate_ratchet.py check",
        "vulture_ratchet.py check",
        "import_cycles.py check",
        "semgrep_ratchet.py test",
        "semgrep_ratchet.py check",
        "semgrep-requirements.txt",
    ):
        assert script in quality_runs, script
    crap = next(s for s in jobs["python-coverage"]["steps"] if s.get("name") == "CRAP ratchet")
    assert "--report-only" not in crap["run"]


def test_the_quality_guide_exists_and_names_every_gate() -> None:
    text = (PKG / "tools" / "QUALITY.md").read_text()
    for word in ("cognitive", "duplicate", "dead code", "import cycle", "semgrep", "CRAP"):
        assert word.lower() in text.lower(), word
    assert "only shrink" in text
