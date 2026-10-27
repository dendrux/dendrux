"""Mirror safe in-process answer values and accept bounded plain client text."""

import pytest

from dendrux.types import RunResult, RunStatus, ToolResult


@pytest.mark.parametrize("answer", ["Hello alice@example.com", "Hello <<EMAIL_ADDRESS_1>>"])
def test_wire_answer_equals_in_process_answer(answer):
    result = RunResult(
        "R",
        RunStatus.SUCCESS,
        answer=answer,
        meta={"pii_mapping": {"<<EMAIL_ADDRESS_1>>": "alice@example.com"}},
    )
    assert result.to_dict()["answer"] == result.answer


@pytest.mark.parametrize("payload", ["Revenue 2024", "", "{unfinished", "Plain: café"])
def test_wire_parser_preserves_plain_text(payload):
    assert ToolResult.from_dict(dict(name="read", call_id="C", payload=payload)).payload == payload
