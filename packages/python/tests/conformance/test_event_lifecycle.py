import pytest

from tests.conformance.format2 import _assert_tool_lifecycle


def event(kind, call="a"):
    return {"event_type": "tool." + kind, "correlation_id": call}


@pytest.mark.parametrize(
    "events",
    [
        [event("completed"), event("started")],
        [event("started")],
        [event("started"), event("completed"), event("completed")],
    ],
)
def test_raw_tool_lifecycle_refuses_invalid_order(events):
    with pytest.raises(AssertionError):
        _assert_tool_lifecycle(events)


def test_raw_tool_lifecycle_allows_interleaving_and_validation_results():
    _assert_tool_lifecycle(
        [
            event("completed", "invalid"),
            event("started", "a"),
            event("started", "b"),
            event("completed", "b"),
            event("completed", "a"),
        ]
    )
