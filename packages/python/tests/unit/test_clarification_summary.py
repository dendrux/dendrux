"""Display summaries have an explicit byte bound even for escaped and multibyte text."""

import json

import pytest

from dendrux.runtime._clarification import clarification_summary
from dendrux.types import Limits


@pytest.mark.parametrize("value", ["😀", '"', "\\", "\n"])
def test_encoded_summary_bound(value):
    summary = clarification_summary(value * 10000, [value] * 10000, max_bytes=128)
    assert len(json.dumps(summary).encode()) <= 128
    assert summary["question"].endswith("[truncated]")


def test_empty_options_still_consume_space():
    summary = clarification_summary("Which?", [""] * 10000, max_bytes=128)
    assert len(json.dumps(summary).encode()) <= 128
    assert summary["question"] == "Which?"
    assert len(summary["options"]) < 10000


@pytest.mark.parametrize("limit", [True, 0, 127, 524289])
def test_invalid_summary_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="max_pause_summary_bytes"):
        Limits(max_pause_summary_bytes=limit)
