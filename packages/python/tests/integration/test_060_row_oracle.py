"""The historical oracle projects only the explicitly approved 0.6 columns."""

from copy import deepcopy

import pytest

from tests._helpers import v030_oracle as oracle

_COLUMNS = {
    "agent_runs": (
        "storage_format",
        "reclaim_attempts",
        "reclaim_limit",
        "lease_expired_at",
        "last_reclaimed_at",
        "loop_state",
        "trace_link",
    ),
    "react_traces": ("content_digest", "trace_kind"),
    "tool_calls": ("outcome", "idempotent", "replayed_at", "content_digest"),
    "run_events": ("content_digest",),
}


@pytest.mark.parametrize(("table", "columns"), _COLUMNS.items())
def test_060_oracle_projection_is_closed(table, columns):
    expected = {name: [] for name in oracle.ROW_TABLES}
    expected[table] = [{"id": "stable"}]
    actual = deepcopy(expected)
    actual[table][0].update(dict.fromkeys(columns, "approved-additive-value"))
    assert oracle.compare_rows(expected, actual) is None
    actual[table][0]["unapproved_future_column"] = "must-not-be-ignored"
    assert "unapproved_future_column" in oracle.compare_rows(expected, actual)
