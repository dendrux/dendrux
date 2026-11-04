"""The architecture rules in packages/python/.importlinter hold, and they can fail."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
CONFIG = PKG / ".importlinter"
EXIT_STATUS_SUCCESS, EXIT_STATUS_ERROR = 0, 1

# import-linter calls logging.config.fileConfig, which disables every existing logger in its
# process. It runs in its own interpreter so it cannot silence the test process's loggers.
_RUNNER = (
    "import sys\n"
    "from importlinter.cli import lint_imports\n"
    "sys.exit(lint_imports(config_filename=sys.argv[1], is_debug_mode=False))\n"
)


def _run(config: Path) -> int:
    done = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(config)],
        cwd=PKG,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode not in (EXIT_STATUS_SUCCESS, EXIT_STATUS_ERROR):
        raise AssertionError(done.stderr[-1500:])
    return done.returncode


def test_the_package_follows_its_import_rules() -> None:
    assert _run(CONFIG) == EXIT_STATUS_SUCCESS


def test_the_layer_contracts_excuse_no_import() -> None:
    """No upward or persistence import is excused: those contracts carry no ignore list.

    (The last one, retry telemetry reaching into the loop helpers, went in 0.6.0 H8.)
    """
    text = CONFIG.read_text()
    for contract in ("layers", "no-persistence-below-the-runtime"):
        section = text.split(f"[importlinter:contract:{contract}]")[1].split("[importlinter:")[0]
        assert "ignore_imports" not in section, contract


@pytest.mark.parametrize(
    "package",
    sorted(
        p.name
        for p in (Path(__file__).resolve().parents[2] / "src" / "dendrux").iterdir()
        if (p / "__init__.py").is_file() and not p.name.startswith(("_", "."))
    ),
)
def test_every_subpackage_has_a_place_in_the_layers(package: str) -> None:
    assert re.search(rf"\b{package}\b", CONFIG.read_text()), f"{package} is not in the layers"
