"""MySQL-only coverage is advisory locally and required by the CI ratchet."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools import crap_ratchet as cr

if TYPE_CHECKING:
    from pathlib import Path

MYSQL_MODULES = (
    "src/dendrux/db/_dialects/mysql.py",
    "src/dendrux/runtime/stores/sql/mysql.py",
)
SOURCE = "def risky(x):\n    if x or x or x or x or x:\n        return 1\n    return 0\n"


def _evidence(tmp_path: Path, module: str) -> tuple[list[str], Path]:
    path = tmp_path / module
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({"files": {module: {"executed_lines": [1], "missing_lines": [2, 3, 4]}}})
    )
    cr.write_source_digests(coverage, tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    return [
        "--src",
        str(tmp_path / "src/dendrux"),
        "--repo-root",
        str(tmp_path),
        "--coverage",
        str(coverage),
        "--baseline",
        str(baseline),
    ], baseline


@pytest.fixture(autouse=True)
def _local_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


@pytest.mark.parametrize("module", MYSQL_MODULES)
def test_mysql_only_module_warns_locally(
    tmp_path: Path, module: str, capsys: pytest.CaptureFixture[str]
) -> None:
    args, _ = _evidence(tmp_path, module)
    assert cr.main(["check", *args]) == 0
    output = capsys.readouterr().out
    assert "CRAP warning" in output and module in output
    assert "MySQL/MariaDB coverage is enforced in CI" in output


@pytest.mark.parametrize("module", MYSQL_MODULES)
@pytest.mark.parametrize("mode", ["CI", "GITHUB_ACTIONS", "--ci"])
def test_mysql_only_module_is_enforced_in_ci(
    tmp_path: Path, module: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _evidence(tmp_path, module)
    if mode.startswith("--"):
        args.append(mode)
    else:
        monkeypatch.setenv(mode, "true")
    assert cr.main(["check", *args]) == 1


@pytest.mark.parametrize(
    "module",
    [
        "src/dendrux/runtime/_blob_catalog.py",
        "src/dendrux/runtime/stores/sql/base.py",
        "src/dendrux/db/_dialects/mysql_extra.py",
        "src/dendrux/runtime/stores/sql/mysql/helper.py",
    ],
)
def test_local_warning_policy_does_not_cover_other_modules(tmp_path: Path, module: str) -> None:
    args, _ = _evidence(tmp_path, module)
    assert cr.main(["check", *args]) == 1


@pytest.mark.parametrize("module", MYSQL_MODULES)
def test_local_warning_does_not_allow_a_baseline_increase(tmp_path: Path, module: str) -> None:
    args, baseline = _evidence(tmp_path, module)
    before = baseline.read_bytes()
    assert cr.main(["update", *args]) == 1
    assert baseline.read_bytes() == before


def test_local_warning_does_not_hide_stale_mysql_coverage(tmp_path: Path) -> None:
    module = MYSQL_MODULES[0]
    args, _ = _evidence(tmp_path, module)
    (tmp_path / module).write_text(SOURCE.replace("return 1", "return 2"))
    assert cr.main(["check", *args]) == 2


def test_matching_backend_provenance_enforces_locally(tmp_path: Path) -> None:
    args, _ = _evidence(tmp_path, MYSQL_MODULES[0])
    for name in ("mysql", "mariadb"):
        (tmp_path / f"cov-{name}.data").write_bytes(name.encode())
    assert cr.main(["stamp", *args, "--mysql-coverage"]) == 0
    coverage = tmp_path / "coverage.json"
    assert cr.has_mysql_coverage(coverage)
    assert cr.main(["check", *args]) == 1
    # A new base-suite report no longer represents that backend collection.
    coverage.write_text(coverage.read_text() + "\n")
    assert not cr.has_mysql_coverage(coverage)
    assert cr.main(["check", *args]) == 0


@pytest.mark.parametrize("evidence", ["bad json", "[]", "{}"])
def test_invalid_backend_provenance_is_not_treated_as_measured(
    tmp_path: Path, evidence: str
) -> None:
    args, _ = _evidence(tmp_path, MYSQL_MODULES[0])
    coverage = tmp_path / "coverage.json"
    cr.mysql_evidence_path(coverage).write_text(evidence)
    assert not cr.has_mysql_coverage(coverage)
    assert cr.main(["check", *args]) == 0
