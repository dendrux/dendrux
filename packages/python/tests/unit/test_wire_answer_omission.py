"""Browser results explicitly identify answers omitted by the display bound."""

import pytest

from dendrux.types import RunResult, RunStatus


@pytest.mark.parametrize(
    "answer,omitted",
    [("x" * (1048576 + 1), True), ("done", False), ("", False)],
    ids=["oversized", "text", "empty"],
)
def test_wire_answer_has_omission_marker(answer, omitted, caplog):
    wire = RunResult(run_id="r", status=RunStatus.SUCCESS, answer=answer).to_dict()
    assert wire["answer_omitted"] is omitted
    assert wire["answer"] == (None if omitted else answer)
    assert wire["wire_version"] == 1
    if omitted:
        assert any(
            getattr(record, "code", None) == "wire_answer_omitted"
            and getattr(record, "run_id", None) == "r"
            for record in caplog.records
        )
