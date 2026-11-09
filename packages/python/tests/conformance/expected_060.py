"""Explicit approved deltas from the immutable 0.5 recordings.

Rejected calls now carry ``dendrux.not_executed`` before persistence so they do not
consume call limits after resume. Only the listed rejected messages/traces gain
that marker; their golden digests are derived from the original recorded messages.
Successful retry requests omit the source's final assistant answer.
Actual output is never projected, and the 0.5 corpus and goldens remain untouched.
"""

from __future__ import annotations

from tests.conformance import harness as h

# (provider-message indexes in call 1, persisted trace indexes), from the 0.5 corpus.
REJECTED_POSITIONS = {
    "malformed_tool_args__run": ((3, 4), (2, 3)),
    "malformed_tool_args__stream": ((3,), (2,)),
    "malformed_tool_json__run": ((3,), (2,)),
}


def corpus(cell_id: str) -> dict:
    expected = h.read_json(h.corpus_path(cell_id))
    if cell_id == "guardrail_block__run":
        rows = expected["rows"]["agent_runs"]
        assert len(rows) == 1
        row = rows[0]
        assert row["failure_reason"] == "guardrail_blocked"
        assert "dendrux.blocked" not in row["meta"]
        row["meta"]["dendrux.blocked"] = True
    if cell_id == "retry__after_success":
        messages = expected["calls"][2]["messages"]
        final = messages[-1]
        assert final["role"] == "assistant" and not final["tool_calls"]
        assert final["meta"]["phase"] == "final_answer"
        messages.pop()
    if cell_id not in REJECTED_POSITIONS:
        return expected
    messages, traces = REJECTED_POSITIONS[cell_id]
    for index in messages:
        _mark_rejected(expected["calls"][1]["messages"][index])
    for index in traces:
        _mark_rejected(expected["rows"]["react_traces"][index])
    return expected


def _mark_rejected(item: dict) -> None:
    assert item["role"] == "tool" and item["meta"]["is_error"] is True
    assert "dendrux.not_executed" not in item["meta"]
    item["meta"]["dendrux.not_executed"] = True


def golden(cell_id: str, *, legacy: bool = False) -> dict:
    path = h.golden_path(cell_id)
    if legacy and (variant := h.golden_path(cell_id, "legacy")).exists():
        path = variant
    expected = h.read_json(path)
    if cell_id in REJECTED_POSITIONS:
        expected["provider_calls"][1]["messages_sha256"] = h.messages_digest(
            corpus(cell_id)["calls"][1]["messages"]
        )
    if cell_id == "retry__after_success":
        messages = corpus(cell_id)["calls"][2]["messages"]
        expected["provider_calls"][2]["messages_sha256"] = h.messages_digest(messages)
        expected["provider_calls"][2]["n_messages"] = len(messages)
    return expected
