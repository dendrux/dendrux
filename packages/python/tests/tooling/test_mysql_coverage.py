"""The opt-in backend coverage target never manages container lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools import crap_ratchet as cr
from tools import mysql_coverage as mc

SOURCE = (
    "def backend(name):\n    if name == 'mysql':\n        return 1\n"
    "    if name == 'mariadb':\n        return 2\n    return 0\n"
)
PKG = Path(__file__).resolve().parents[2]
RUN = subprocess.run


def _measure(root: Path, output: Path, backend: str) -> None:
    driver = root / "driver.py"
    driver.write_text(
        f"from dendrux.probe import backend\nassert backend('{backend}') in (0, 1, 2)\n"
    )
    RUN(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--data-file={output}",
            "--source=src/dendrux",
            str(driver),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=True,
        capture_output=True,
    )


def _base(root: Path) -> None:
    src = root / "src/dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "probe.py").write_text(SOURCE)
    _measure(root, root / ".coverage", "postgres")
    RUN(
        [sys.executable, "-m", "coverage", "json", "--data-file=.coverage", "-o", "coverage.json"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    cr.write_source_digests(root / "coverage.json", root)


def test_make_target_is_opt_in() -> None:
    makefile = (PKG / "Makefile").read_text()
    assert "cov-mysql:" in makefile
    for line in makefile.splitlines():
        if line.startswith(("ci:", "test-merge:", "test:")):
            assert "cov-mysql" not in line


@pytest.mark.parametrize("state", ["false", "missing-docker"])
def test_stopped_or_unavailable_container_fails_before_tests(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], state: str
) -> None:
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if state == "missing-docker":
            raise FileNotFoundError
        return subprocess.CompletedProcess(command, 0, stdout="false\n", stderr="")

    monkeypatch.setattr(mc, "PKG", Path.cwd())
    monkeypatch.setattr(subprocess, "run", run)
    assert mc.main() == 2
    assert "not running or Docker is unavailable" in capsys.readouterr().err
    assert commands and all(command[:2] == ["docker", "inspect"] for command in commands)


@pytest.mark.parametrize("fail_suite", [False, True])
def test_backend_reports_are_merged_only_after_both_suites_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_suite: bool
) -> None:
    _base(tmp_path)
    before = (tmp_path / "coverage.json").read_bytes()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="true\n", stderr="")
        if "pytest" in command:
            if fail_suite:
                return subprocess.CompletedProcess(command, 1)
            env = kwargs["env"]
            backend = "mysql" if env.get("DENDRUX_TEST_MYSQL_URL") else "mariadb"
            _measure(tmp_path, Path(env["COVERAGE_FILE"]), backend)
            return subprocess.CompletedProcess(command, 0)
        return RUN(command, **kwargs)

    monkeypatch.setattr(mc, "PKG", tmp_path)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert mc.main() == (1 if fail_suite else 0)
    assert not any("start" in command or "stop" in command for command, _ in calls)
    if fail_suite:
        assert (tmp_path / "coverage.json").read_bytes() == before
        assert not cr.mysql_evidence_path(tmp_path / "coverage.json").exists()
        return
    suites = [(command, kwargs["env"]) for command, kwargs in calls if "pytest" in command]
    assert len(suites) == 2
    for command, env in suites:
        assert "--cov=src/dendrux" in command and "--no-cov" not in command
        assert env.get("DENDRUX_TEST_PG_URL") is None
        urls = [
            value
            for key, value in env.items()
            if key in {"DENDRUX_TEST_MYSQL_URL", "DENDRUX_TEST_MARIADB_URL"}
        ]
        assert len(urls) == 1 and urls[0].endswith("_test")
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["files"]["src/dendrux/probe.py"]["missing_lines"] == []
    assert cr.has_mysql_coverage(tmp_path / "coverage.json")
    assert cr.load_source_digests(tmp_path / "coverage.json")


def test_ci_combiner_joins_six_real_coverage_files(tmp_path: Path) -> None:
    _base(tmp_path)
    data = (tmp_path / ".coverage").read_bytes()
    for shard in range(1, 5):
        (tmp_path / f"cov-{shard}.data").write_bytes(data)
    for name in ("mysql", "mariadb"):
        _measure(tmp_path, tmp_path / f"cov-{name}.data", name)
    workflow = yaml.safe_load((PKG.parents[1] / ".github/workflows/ci.yml").read_text())
    combine = next(
        step["run"]
        for step in workflow["jobs"]["python-coverage"]["steps"]
        if step.get("name") == "Combine coverage and apply the floor"
    )
    result = RUN(
        ["bash", "-e", "-o", "pipefail", "-c", combine],
        cwd=tmp_path,
        env={**os.environ, "COVERAGE_FILE": str(tmp_path / ".coverage")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["files"]["src/dendrux/probe.py"]["missing_lines"] == []
    assert all((tmp_path / f"cov-{name}.data").is_file() for name in ("mysql", "mariadb"))
