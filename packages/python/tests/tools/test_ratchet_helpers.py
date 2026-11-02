"""Shared ratchet helpers: stable baseline files and the grow/shrink comparison."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools import _ratchet as r

if TYPE_CHECKING:
    from pathlib import Path


def test_baseline_is_written_sorted_with_a_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    r.write_baseline(path, {"b": 2, "a": 1})
    text = path.read_text()
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b"]
    assert r.read_baseline(path) == {"a": 1, "b": 2}


def test_missing_baseline_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(r.BaselineMissingError, match="make quality-stamp"):
        r.read_baseline(tmp_path / "nope.json")


def test_compare_counts_flags_new_and_worse_and_reports_improved() -> None:
    result = r.compare_counts(
        {"new": 1, "same": 2, "worse": 5, "better": 1},
        {
            "same": 2,
            "worse": 3,
            "better": 4,
            "gone": 1,
        },
    )
    assert result.new == ["new"]
    assert result.worse == ["worse"]
    assert sorted(result.improved) == ["better", "gone"]
    assert result.failed


def test_compare_counts_passes_when_nothing_grew() -> None:
    result = r.compare_counts({"a": 1}, {"a": 2, "b": 1})
    assert not result.failed
    assert result.improved


def test_compare_counts_limit_lets_a_new_key_in_when_under_the_limit() -> None:
    result = r.compare_counts({"small": 15, "big": 16}, {}, limit=15)
    assert result.new == ["big"]
