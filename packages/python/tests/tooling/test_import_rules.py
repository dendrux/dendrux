"""The architecture rules in packages/python/.importlinter hold, and they can fail."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from importlinter.cli import EXIT_STATUS_ERROR, EXIT_STATUS_SUCCESS, lint_imports

CONFIG = Path(__file__).resolve().parents[2] / ".importlinter"


def _run(config: Path) -> int:
    return lint_imports(config_filename=str(config), is_debug_mode=False)


def test_the_package_follows_its_import_rules() -> None:
    assert _run(CONFIG) == EXIT_STATUS_SUCCESS


def test_the_known_upward_import_is_what_the_ignore_entry_covers(tmp_path: Path) -> None:
    """Dropping the documented ignore makes the layer contract fail: the rule really bites."""
    text = CONFIG.read_text()
    stripped = re.sub(
        r"^    dendrux\.llm\._retry_telemetry -> dendrux\.loops\._helpers\n", "", text, flags=re.M
    )
    assert stripped != text
    config = tmp_path / ".importlinter"
    config.write_text(stripped)
    assert _run(config) == EXIT_STATUS_ERROR


@pytest.mark.parametrize(
    "package",
    sorted(
        p.name
        for p in (Path(__file__).resolve().parents[2] / "src" / "dendrux").iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    ),
)
def test_every_subpackage_has_a_place_in_the_layers(package: str) -> None:
    assert re.search(rf"\b{package}\b", CONFIG.read_text()), f"{package} is not in the layers"
