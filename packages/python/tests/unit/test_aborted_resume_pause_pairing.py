"""Bounded pause reconstruction after aborted continuation attempts."""

from types import SimpleNamespace

import pytest

from dendrux.runtime._pending import _active_pause_from_events
from dendrux.store import _derive_pauses, _unresolved_pause_tail


def events(*types):
    return [
        SimpleNamespace(sequence_index=i, event_type=t, data={}, created_at=None)
        for i, t in enumerate(types)
    ]


@pytest.mark.parametrize(
    "types, expected",
    [
        ([], None),
        (["run.paused"], 0),
        (["run.paused", "run.resumed"], None),
        (["run.paused", "run.resumed", "run.resume_aborted"], 0),
        (
            [
                "run.paused",
                "run.resumed",
                "run.resume_aborted",
                "run.cancelled",
                "run.resume_aborted",
            ],
            None,
        ),
        (["run.resume_aborted"], None),
        (["run.paused"] * 1000, None),
    ],
)
def test_custom_pause_tail_fails_closed(types, expected):
    assert _active_pause_from_events(events(*types)) == expected


@pytest.mark.parametrize("tail", ["run.resumed", "run.resume_aborted"])
def test_truncated_abort_lookahead_cannot_close_or_expose_an_unresolved_pause(tail):
    rows = events("run.paused", "run.resumed", "run.resume_aborted", tail)
    pairs = _derive_pauses(rows)
    assert len(pairs) == 1
    assert _unresolved_pause_tail(pairs, rows, 4)
    assert not _unresolved_pause_tail(pairs, rows, 5)
    assert not _unresolved_pause_tail([], rows, 4)


def test_multiple_pauses_pair_only_non_aborted_resumes():
    rows = events(
        "run.resumed",
        "run.paused",
        "run.resumed",
        "run.resume_aborted",
        "run.resumed",
        "run.paused",
        "run.resumed",
        "run.resume_aborted",
    )
    pairs = _derive_pauses(rows)
    assert [(p.pause_sequence_index, p.resume_sequence_index) for p in pairs] == [(1, 4), (5, None)]
