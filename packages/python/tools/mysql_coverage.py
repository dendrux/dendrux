"""Opt-in real backend coverage; container lifecycle remains developer-owned."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools import crap_ratchet as cr

PKG = Path(__file__).resolve().parents[1]
BACKENDS = {"mysql": 53306, "mariadb": 53307}


def _containers_running() -> bool:
    for backend in BACKENDS:
        name = f"dendrux-test-{backend}"
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            running = result.returncode == 0 and result.stdout.strip() == "true"
        except (OSError, subprocess.TimeoutExpired):
            running = False
        if not running:
            print(f"cov-mysql: {name} is not running or Docker is unavailable", file=sys.stderr)
            return False
    return True


def _run(command: list[str], env: dict[str, str]) -> bool:
    return subprocess.run(command, cwd=PKG, env=env, check=False).returncode == 0


def _merge(base: Path, env: dict[str, str], digests: dict[str, str]) -> bool:
    with tempfile.TemporaryDirectory(prefix=".mysql-coverage-", dir=PKG) as directory:
        scratch = Path(directory)
        shutil.copyfile(base, scratch / "base.data")
        merged = scratch / ".coverage"
        merge_env = {**env, "COVERAGE_FILE": str(merged)}
        coverage = [sys.executable, "-m", "coverage"]
        inputs = [scratch / "base.data", *(PKG / f"cov-{name}.data" for name in BACKENDS)]
        if not _run([*coverage, "combine", "--keep", *map(str, inputs)], merge_env):
            return False
        report = scratch / "coverage.json"
        if not _run([*coverage, "json", "-o", str(report)], merge_env):
            return False
        # Refuse source edits made during the backend suites, just like the normal gate.
        cr.collect_scores(PKG / "src/dendrux", PKG, json.loads(report.read_text()), digests)
        shutil.copyfile(merged, base)
        shutil.copyfile(report, PKG / "coverage.json")
        cr.write_source_digests(PKG / "coverage.json", PKG)
        cr.record_mysql_coverage(PKG / "coverage.json")
    return True


def main() -> int:
    """Measure both already-running backends and refresh the combined report."""
    if not _containers_running():
        return 2
    if importlib.util.find_spec("asyncmy") is None:
        print(
            "cov-mysql: install the mysql extra before collecting backend coverage", file=sys.stderr
        )
        return 2
    report = PKG / "coverage.json"
    base = PKG / ".coverage"
    try:
        digests = cr.load_source_digests(report)
        cr.collect_scores(PKG / "src/dendrux", PKG, json.loads(report.read_text()), digests)
        if not base.is_file():
            raise FileNotFoundError(base)
    except (OSError, ValueError, cr.CoverageDataError):
        print(
            "cov-mysql: run the base suite and record fresh coverage source digests first",
            file=sys.stderr,
        )
        return 2
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DENDRUX_", "COVERAGE_", "PYTEST_"))
    }
    env["PYTHON_DOTENV_DISABLED"] = "1"
    for backend, port in BACKENDS.items():
        backend_env = {
            **env,
            f"DENDRUX_TEST_{backend.upper()}_URL": (
                f"mysql+asyncmy://root:dendrux_test@127.0.0.1:{port}/"
                f"dendrux_p5_b_cigreen_cov_{backend}_test"
            ),
            "COVERAGE_FILE": str(PKG / f"cov-{backend}.data"),
        }
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/db",
            "-v",
            "-rs",
            "-n",
            "4",
            "-m",
            "not benchmark",
            "--cov=src/dendrux",
            "--cov-report=",
        ]
        if not _run(command, backend_env):
            print(
                f"cov-mysql: {backend} suite failed; combined coverage was not replaced",
                file=sys.stderr,
            )
            return 1
    try:
        if not _merge(base, env, digests):
            return 1
    except (OSError, ValueError, cr.CoverageDataError):
        print("cov-mysql: cannot combine fresh backend coverage", file=sys.stderr)
        return 2
    print("cov-mysql: combined base, MySQL and MariaDB coverage; local CRAP enforcement enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
