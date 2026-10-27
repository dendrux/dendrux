"""The legacy text oracle validates and discounts only the new declaration event."""

import pytest

from tests._helpers.v030_oracle import ROW_TABLES, normalise_rows


def rows(declaration, *, completed_sequence=2):
    tables = {name: [] for name in ROW_TABLES}
    tables["run_events"] = [
        dict(
            id="first",
            agent_run_id="r",
            sequence_index=0,
            iteration_index=0,
            event_type="run.started",
            data={},
        ),
        dict(
            id="declared",
            agent_run_id="r",
            sequence_index=1,
            iteration_index=0,
            event_type="tools.declared",
            data=declaration,
        ),
        dict(
            id="last",
            agent_run_id="r",
            sequence_index=completed_sequence,
            iteration_index=1,
            event_type="run.completed",
            data={},
        ),
    ]
    return tables


def test_legacy_oracle_removes_validated_declarations_and_offsets_only_their_sequences():
    normalized = normalise_rows(rows({"local_tool_names": ["a_client", "z_server"]}))
    assert [event["event_type"] for event in normalized["run_events"]] == [
        "run.started",
        "run.completed",
    ]
    assert [event["sequence_index"] for event in normalized["run_events"]] == [0, 1]
    gap = normalise_rows(rows({"local_tool_names": []}, completed_sequence=3))
    assert gap["run_events"][-1]["sequence_index"] == 2, (
        "unrelated sequence gaps must remain visible"
    )


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"local_tool_names": "PRIVATE_VALUE"},
        {"local_tool_names": ["z", "a"]},
        {"local_tool_names": ["a", "a"]},
        {"local_tool_names": ["a" * 65]},
        {"local_tool_names": ["bad\x00"]},
        {"local_tool_names": [17]},
        {"local_tool_names": [], "extra": "PRIVATE_VALUE"},
        {"local_tool_names": [f"a{i}" for i in range(1001)]},
    ],
)
def test_legacy_oracle_does_not_silently_strip_malformed_declaration_data(data):
    with pytest.raises(AssertionError, match="invalid tools.declared") as caught:
        normalise_rows(rows(data))
    assert "PRIVATE_VALUE" not in str(caught.value)
