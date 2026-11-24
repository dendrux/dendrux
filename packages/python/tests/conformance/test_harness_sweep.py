"""Golden sweep comparisons tolerate only empty additive adoption results."""

from copy import deepcopy

import pytest

from dendrux.runtime.state import SweepResult
from tests.conformance import harness as h


@pytest.mark.parametrize("adoptable", [[], ["root"]], ids=["empty", "nonempty"])
def test_golden_comparison_preserves_nonempty_adoptable_roots(adoptable):
    expected = h.read_json(h.golden_path("sweep__nothing_due"))
    actual = deepcopy(expected)
    actual["steps"][-1]["value"] = h.raw(SweepResult([], adoptable=adoptable))
    assert (h.diff(expected, h._norm(actual)) is None) is (adoptable == [])
