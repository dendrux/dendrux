"""Pure recognition and resolved-prefix rules for a v4 fan-out adoption (0.8 spec §7.3)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dendrux.errors import ManagedStateError
from dendrux.runtime._fanout_adoption_state import (
    FANOUT_ADOPTION_KEY,
    FANOUT_ADOPTION_VERSION,
    ROW_COMPLETED,
    ROW_FAILED,
    ROW_STARTED,
    FanoutBinding,
    IntentRow,
    adoption_marker,
    adoption_payload,
    fanout_adoption,
    resolved_prefix,
)

PARENT = "01JCPARENT0000000000000000"
BINDINGS = tuple(FanoutBinding(f"c{i}", f"child{i}") for i in range(1, 5))


def _row(index: int, outcome: str, *, digest: str | None = "d", child: str | None = None) -> Any:
    return IntentRow(
        tool_call_id=f"c{index}",
        outcome=outcome,
        content_digest=digest,
        child_run_id=child if child is not None else f"child{index}",
    )


def _rows(*outcomes: str) -> list[IntentRow]:
    return [_row(i, outcome) for i, outcome in enumerate(outcomes, start=1)]


def _claimed(call_ids: list[str] | None = None) -> dict[str, Any]:
    """A claimed checkpoint: v4 binding map plus the adoption marker (spec §4.1, §7.3)."""
    ids = ["c1", "c2"] if call_ids is None else call_ids
    return {
        "managed_wait": {"v": 1, "pause_id": "p1", "bindings": {"c1": "child1", "c2": "child2"}},
        "submission": {
            "submission_id": "s1",
            "kind": "tool_results",
            "pause_id": "p1",
            "payload_ref": {"pause_data_key": "submission"},
            "call_ids": ids,
            "reserved_order_index": 7,
            "managed_fanout_adoption": {"v": 1},
        },
    }


# --- marker constants ------------------------------------------------------


def test_marker_and_payload_match_the_spec_bytes() -> None:
    assert FANOUT_ADOPTION_KEY == "managed_fanout_adoption"
    assert FANOUT_ADOPTION_VERSION == 1
    assert adoption_marker() == {"v": 1}
    assert adoption_payload(["c1", "c2"]) == {
        "managed_fanout_adoption": {"v": 1, "calls": ["c1", "c2"]}
    }


def test_marker_and_payload_are_fresh_objects() -> None:
    first = adoption_marker()
    first["v"] = 99
    assert adoption_marker() == {"v": 1}
    ids = ["c1", "c2"]
    payload = adoption_payload(ids)
    ids.append("c3")
    assert payload["managed_fanout_adoption"]["calls"] == ["c1", "c2"]


def test_row_outcome_constants_are_the_stored_words() -> None:
    assert (ROW_STARTED, ROW_COMPLETED, ROW_FAILED) == ("started", "completed", "failed")


# --- fanout_adoption: recognition ------------------------------------------


def test_recognizes_a_claimed_v4_adoption_and_returns_call_ids_in_order() -> None:
    assert fanout_adoption(_claimed()) == ("c1", "c2")


def test_call_order_comes_from_the_submission_not_the_map() -> None:
    state = _claimed(["c2", "c1"])
    assert fanout_adoption(state) == ("c2", "c1")


def _without_marker() -> dict[str, Any]:
    state = _claimed()
    del state["submission"]["managed_fanout_adoption"]
    return state


def _with(path: tuple[str, ...], value: object) -> dict[str, Any]:
    state = _claimed()
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return state


NOT_RECOGNIZED = {
    "none": None,
    "empty": {},
    "not a mapping": ["managed_wait"],
    "parked v4, no submission (pre-claim)": {
        "managed_wait": {"v": 1, "pause_id": "p1", "bindings": {"c1": "a", "c2": "b"}}
    },
    "v4 wait, 0.7-shaped submission (no marker)": _without_marker(),
    "v3 wait with a submission": {
        "managed_wait": {"tool_call_id": "c1", "child_run_id": "a", "pause_id": "p1"},
        "submission": {"kind": "tool_results", "call_ids": ["c1"]},
    },
    "v3 wait carrying the marker": _with(
        ("managed_wait",), {"tool_call_id": "c1", "child_run_id": "a", "pause_id": "p1"}
    ),
    "marker without a wait": {"submission": _claimed()["submission"]},
    "marker version 2": _with(("submission", "managed_fanout_adoption"), {"v": 2}),
    "marker with extra keys": _with(
        ("submission", "managed_fanout_adoption"), {"v": 1, "calls": ["c1", "c2"]}
    ),
    "marker true": _with(("submission", "managed_fanout_adoption"), True),
    "marker v as bool": _with(("submission", "managed_fanout_adoption"), {"v": True}),
    "submission not a mapping": _with(("submission",), "x"),
    "wait version 2": _with(("managed_wait", "v"), 2),
    "wait bindings not a mapping": _with(("managed_wait", "bindings"), [["c1", "child1"]]),
    "wait bindings empty": _with(("managed_wait", "bindings"), {}),
}


@pytest.mark.parametrize("state", NOT_RECOGNIZED.values(), ids=NOT_RECOGNIZED.keys())
def test_everything_else_is_not_a_v4_adoption(state: Any) -> None:
    assert fanout_adoption(state) is None


CONFLICTING_IDS = {
    "call_ids missing": None,
    "call_ids not a list": "c1",
    "call_ids empty": [],
    "non-string id": ["c1", 2],
    "fewer ids than bindings": ["c1"],
    "id outside the bindings": ["c1", "c9"],
    "extra id": ["c1", "c2", "c3"],
    "duplicate id": ["c1", "c1"],
}


@pytest.mark.parametrize("ids", CONFLICTING_IDS.values(), ids=CONFLICTING_IDS.keys())
def test_a_marked_adoption_whose_ids_disagree_with_its_bindings_is_a_conflict(ids: Any) -> None:
    state = _claimed()
    if ids is None:
        del state["submission"]["call_ids"]
    else:
        state["submission"]["call_ids"] = ids
    with pytest.raises(ManagedStateError) as raised:
        fanout_adoption(state, run_id=PARENT)
    assert raised.value.reason == "binding_conflict"
    assert raised.value.run_id == PARENT


def test_a_marked_adoption_with_a_non_string_child_id_is_a_conflict() -> None:
    state = _with(("managed_wait", "bindings"), {"c1": "child1", "c2": 7})
    with pytest.raises(ManagedStateError) as raised:
        fanout_adoption(state)
    assert raised.value.reason == "binding_conflict"


def test_recognition_never_mutates_the_checkpoint() -> None:
    state = _claimed()
    before = copy.deepcopy(state)
    fanout_adoption(state)
    assert state == before


# --- resolved_prefix: resolved / started tables -----------------------------

PREFIXES = {
    "nothing resolved (cut C5)": (("started",) * 4, 0),
    "one completed": (("completed", "started", "started", "started"), 1),
    "one failed (C6b, R3-1)": (("failed", "started", "started", "started"), 1),
    "mixed prefix (C6d)": (("completed", "failed", "started", "started"), 2),
    "failed then completed": (("failed", "completed", "started", "started"), 2),
    "all but the last": (("completed", "completed", "failed", "started"), 3),
    "fully adopted": (("completed", "failed", "completed", "failed"), 4),
}


@pytest.mark.parametrize(("outcomes", "expected"), PREFIXES.values(), ids=PREFIXES.keys())
def test_resolved_prefix_counts_completed_and_failed_rows(
    outcomes: tuple[str, ...], expected: int
) -> None:
    assert resolved_prefix(_rows(*outcomes), BINDINGS) == expected


def test_rows_are_matched_by_call_id_not_by_position() -> None:
    rows = _rows("completed", "failed", "started", "started")
    assert resolved_prefix(list(reversed(rows)), BINDINGS) == 2


def test_rows_of_other_calls_in_the_iteration_are_ignored() -> None:
    server = IntentRow(tool_call_id="s1", outcome="unknown", content_digest=None, child_run_id=None)
    rows = [server, *_rows("completed", "started", "started", "started")]
    assert resolved_prefix(rows, BINDINGS) == 1


def test_plain_pairs_are_accepted_as_bindings() -> None:
    rows = _rows("completed", "started")
    assert resolved_prefix(rows, [("c1", "child1"), ("c2", "child2")]) == 1


# --- resolved_prefix: unknown / gap / identity refusals ---------------------

REFUSED = {
    "started then completed (C6d gap)": _rows("started", "completed", "started", "started"),
    "started then failed (gap)": _rows("started", "failed", "started", "started"),
    "resolved after a gap in the middle": _rows("completed", "started", "completed", "started"),
    "unknown first (C6d)": _rows("unknown", "started", "started", "started"),
    "unknown after a resolved row": _rows("completed", "unknown", "started", "started"),
    "unknown last": _rows("completed", "completed", "completed", "unknown"),
    "an outcome word this version does not know": _rows("skipped", "started", "started", "started"),
    "missing row": _rows("completed", "started", "started"),
    "no rows": [],
    "completed without a digest": [
        _row(1, "completed", digest=None),
        *_rows("x", "started", "started", "started")[1:],
    ],
    "failed with an empty digest": [
        _row(1, "failed", digest=""),
        *_rows("x", "started", "started", "started")[1:],
    ],
    "resolved row bound to another child": [
        _row(1, "completed", child="other"),
        *_rows("x", "started", "started", "started")[1:],
    ],
    "started row bound to another child": [
        *_rows("completed", "started", "started"),
        _row(4, "started", child="other"),
    ],
    "row without managed metadata": [
        *_rows("completed", "started", "started"),
        IntentRow(tool_call_id="c4", outcome="started", content_digest="d", child_run_id=None),
    ],
    "two rows for one call": [
        *_rows("completed", "started", "started", "started"),
        _row(2, "started"),
    ],
}


@pytest.mark.parametrize("rows", REFUSED.values(), ids=REFUSED.keys())
def test_anything_but_a_resolved_prefix_is_a_binding_conflict(rows: list[IntentRow]) -> None:
    with pytest.raises(ManagedStateError) as raised:
        resolved_prefix(rows, BINDINGS, run_id=PARENT)
    assert raised.value.reason == "binding_conflict"
    assert raised.value.run_id == PARENT


BAD_BINDINGS = {
    "no bindings": (),
    "duplicate call id": (FanoutBinding("c1", "child1"), FanoutBinding("c1", "child2")),
    "duplicate child id": (FanoutBinding("c1", "child1"), FanoutBinding("c2", "child1")),
}


@pytest.mark.parametrize("bindings", BAD_BINDINGS.values(), ids=BAD_BINDINGS.keys())
def test_malformed_bindings_are_refused(bindings: tuple[FanoutBinding, ...]) -> None:
    with pytest.raises(ManagedStateError) as raised:
        resolved_prefix(_rows("started", "started"), bindings)
    assert raised.value.reason == "binding_conflict"


def test_binding_order_decides_the_prefix() -> None:
    rows = _rows("started", "completed")
    swapped = (FanoutBinding("c2", "child2"), FanoutBinding("c1", "child1"))
    assert resolved_prefix(rows, swapped) == 1
    with pytest.raises(ManagedStateError):
        resolved_prefix(rows, BINDINGS[:2])


def test_the_error_message_carries_codes_only() -> None:
    rows = [_row(1, "completed", digest="SENTINEL-digest", child="SENTINEL-child")]
    with pytest.raises(ManagedStateError) as raised:
        resolved_prefix(rows, BINDINGS[:1], run_id=PARENT)
    assert "SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None
