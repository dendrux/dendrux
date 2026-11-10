"""Published source, tests and docs describe behavior without private tracking labels."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
AREAS = ("packages/python/src", "packages/python/tests", "docs")
PRIVATE_LABEL = re.compile(
    r"\bBL-\d+|\bR\d-P\d|\bP[012]-\d|" + "E3" + r"h-Q|\b(?:R0|S[12]|E[123])h\b"
)


def _private_labels(root: Path) -> list[str]:
    found = []
    for area in AREAS:
        for path in sorted((root / area).rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or "__pycache__" in path.parts
                or any(part.startswith(".env") for part in path.parts)
            ):
                continue
            lines = path.read_bytes().decode(errors="replace").splitlines()
            for number, line in enumerate(lines, 1):
                if PRIVATE_LABEL.search(line):
                    found.append(f"{path.relative_to(root)}:{number}")
    return found


def test_repository_text_has_no_private_labels():
    assert _private_labels(REPO) == []


@pytest.mark.parametrize(
    "label",
    ["BL" + "-9", "R1" + "-P2", "E3" + "h-Q"]
    + [f"P{rank}-1" for rank in range(3)]
    + [f"{prefix}h" for prefix in ("R0", "S1", "E1", "E2", "S2", "E3")],
)
def test_private_label_check_reads_every_area(tmp_path: Path, label: str):
    for area in AREAS:
        path = tmp_path / area / "example.txt"
        path.parent.mkdir(parents=True)
        path.write_text(f"ordinary behavior\n{label}: behavior\n")
    assert _private_labels(tmp_path) == [f"{area}/example.txt:2" for area in AREAS]


def test_behavior_and_public_requirement_labels_are_allowed(tmp_path: Path):
    path = tmp_path / AREAS[0] / "example.txt"
    path.parent.mkdir(parents=True)
    path.write_text("R1.10: blocked continuation\nP1 priority\nBL word\n")
    assert _private_labels(tmp_path) == []
