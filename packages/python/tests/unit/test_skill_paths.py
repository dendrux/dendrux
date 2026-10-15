"""Skill file path rules (spec §2.4) and script classification (D4)."""

from __future__ import annotations

import pytest

from dendrux.errors import SkillValidationError
from dendrux.skills._paths import is_script, validate_file_path


@pytest.mark.parametrize(
    "path",
    [
        "a",
        "references/policy.md",
        "scripts/forecast.py",
        "deep/er/nested/file.txt",
        "sub/SKILL.md",
        "unicode-é/ü.md",
        "with space.md",
        "x" * 512,
        ".config",
    ],
)
def test_valid_paths(path: str) -> None:
    validate_file_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/abs/path",
        "back\\slash",
        "a/./b",
        "./a",
        "a/../b",
        "..",
        "a//b",
        "trailing/",
        "SKILL.md",
        "x" * 513,
        "a<b",
        "a>b",
        'a"b',
        "nul\x00byte",
        "tab\tchar",
        "newline\nchar",
        "del\x7fchar",
        "esc\x1bchar",
    ],
)
def test_invalid_paths(path: str) -> None:
    with pytest.raises(SkillValidationError) as info:
        validate_file_path(path)
    assert info.value.field == "files"


@pytest.mark.parametrize("value", [None, 3, b"bytes/path"])
def test_non_string_rejected(value: object) -> None:
    with pytest.raises(SkillValidationError):
        validate_file_path(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("scripts/run.py", True),
        ("scripts/nested/run.sh", True),
        ("scripts", False),
        ("references/scripts/x.py", False),
        ("Scripts/run.py", False),
        ("scriptsx/run.py", False),
    ],
)
def test_is_script(path: str, expected: bool) -> None:
    assert is_script(path) is expected
