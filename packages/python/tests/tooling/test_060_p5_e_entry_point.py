"""The store kit loads as a pytest plugin from an installed ``dendrux``, and from ``-p``."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
MODULE = "dendrux.testing.store_kit"


def _entry_points() -> dict[str, str]:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]
    return dict(project["entry-points"]["pytest11"])


def test_pyproject_registers_the_kit_under_its_module_name() -> None:
    # pluggy refuses one module registered under two names, so an entry-point name that
    # differs from the module path breaks `pytest -p dendrux.testing.store_kit`.
    assert _entry_points() == {MODULE: MODULE}


def test_the_mysql_driver_is_an_opt_in_extra_outside_all() -> None:
    extras = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert "asyncmy" in extras["mysql"][0]
    assert "mysql" not in extras["all"][0]


def _run_pytest(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    metadata = tmp_path / "site" / "dendrux_kit_probe-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dendrux-kit-probe\nVersion: 1.0\n"
    )
    lines = "".join(f"{name} = {target}\n" for name, target in _entry_points().items())
    (metadata / "entry_points.txt").write_text(f"[pytest11]\n{lines}")
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_probe.py").write_text("def test_probe():\n    pass\n")
    env = {
        "PATH": "",
        "PYTHONPATH": f"{PACKAGE / 'src'}{os.pathsep}{tmp_path / 'site'}",
        "PYTHON_DOTENV_DISABLED": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args, "test_probe.py"],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_installed_plugin_loads_without_collecting_anything(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_dash_p_with_the_module_path_still_works_beside_the_entry_point(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path, "-p", MODULE)
    output = result.stdout + result.stderr
    assert "already registered" not in output
    assert "ValueError" not in output


RUNNER = """
import sys


# pytest-asyncio is not installed in this session. pytest's own import hook finds modules
# through PathFinder, so the block goes there.
import functools
import importlib.machinery

_find = importlib.machinery.PathFinder.find_spec


def _find_spec(name, path=None, target=None):
    if name == "pytest_asyncio" or name.startswith("pytest_asyncio."):
        return None
    return _find(name, path, target)


importlib.machinery.PathFinder.find_spec = functools.partial(_find_spec)
import pytest

code = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:asyncio", *sys.argv[1:]])
print("KIT_LOADED=%s" % ("dendrux.testing.store_kit._plugin" in sys.modules))
sys.exit(int(code))
"""


def _run_without_pytest_asyncio(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    metadata = tmp_path / "site" / "dendrux_kit_probe-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dendrux-kit-probe\nVersion: 1.0\n"
    )
    lines = "".join(f"{name} = {target}\n" for name, target in _entry_points().items())
    (metadata / "entry_points.txt").write_text(f"[pytest11]\n{lines}")
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_probe.py").write_text("def test_probe():\n    pass\n")
    (work / "run.py").write_text(RUNNER)
    env = {
        "PATH": "",
        "PYTHONPATH": f"{PACKAGE / 'src'}{os.pathsep}{tmp_path / 'site'}",
        "PYTHON_DOTENV_DISABLED": "1",
    }
    return subprocess.run(
        [sys.executable, "run.py", *args, "test_probe.py"],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_session_without_the_testkit_extra_starts_and_does_not_load_the_kit(
    tmp_path: Path,
) -> None:
    result = _run_without_pytest_asyncio(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "KIT_LOADED=False" in result.stdout


def test_switching_the_kit_on_without_the_extra_names_the_extra(tmp_path: Path) -> None:
    result = _run_without_pytest_asyncio(tmp_path, "--dendrux-store-kit")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "dendrux[testkit]" in output
    assert "Traceback" not in output


def test_switching_the_kit_on_loads_it_when_the_extra_is_present(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path, "--dendrux-store-kit")
    output = result.stdout + result.stderr
    assert "unrecognized arguments" not in output
    assert "ValueError" not in output
    assert "dendrux[testkit]" not in output
