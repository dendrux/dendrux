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


@pytest.mark.parametrize(
    "target,approval,hidden",
    [("server", None, True), ("client", {"source": "static"}, True), ("client", None, False)],
)
def test_paused_wire_masks_raw_server_and_gated_arguments(target, approval, hidden):
    from dendrux.store import PendingPause, PendingToolCall

    pending = PendingPause(
        run_id="r",
        status="paused",
        kind="paused",
        pause_id="p",
        tool_calls=[
            PendingToolCall(
                id="c",
                name="send",
                target=target,
                params={"secret": "PRIVATE_ARGUMENT"},
                approval=approval,
            )
        ],
    )
    result = RunResult(run_id="r", status=RunStatus.PAUSED, pending=pending)
    projected = result.to_dict()["pending"]["pending_tool_calls"][0]
    assert projected["params"] == (None if hidden else {"secret": "PRIVATE_ARGUMENT"})
    assert projected.get("args_hidden", False) is hidden
    assert pending.tool_calls[0].params == {"secret": "PRIVATE_ARGUMENT"}
