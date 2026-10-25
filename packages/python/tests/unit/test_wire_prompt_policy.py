"""Approval/input prompts use the verified pending view within bounded scan work."""

import re
import time

import pytest

from dendrux import _wire
from dendrux.store import PendingPause, PendingToolCall
from dendrux.types import RunResult, RunStatus


@pytest.mark.parametrize(
    "status,kind",
    [(RunStatus.WAITING_APPROVAL, "approval"), (RunStatus.WAITING_HUMAN_INPUT, "input")],
)
def test_paused_prompt_never_duplicates_restored_pii_in_answer(status, kind):
    pending = PendingPause("r", status.value, kind, "p", question="Email <<EMAIL_ADDRESS_1>>?")
    wire = RunResult(
        "r", status, answer="Email alice.secret@example.com?", pending=pending
    ).to_dict()
    assert wire["answer"] is None, "approval and input prompts belong only in the placeholder view"
    assert wire["pending"]["question"] == pending.question


def test_many_pii_literals_cannot_amplify_a_large_wire_argument(monkeypatch, record_property):
    mapping = {f"<<EMAIL_{i}>>": f"a{i:03d}@ex.io" for i in range(990)}
    pending = PendingPause(
        "r",
        "waiting_approval",
        "approval",
        "p",
        tool_calls=[PendingToolCall("c", "approve", "server", {"blob": "a" * 840000})],
    )
    result = RunResult(
        "r", RunStatus.WAITING_APPROVAL, pending=pending, meta={"pii_mapping": mapping}
    )
    compile_pattern = re.compile

    class CheckedPattern:
        def __init__(self, pattern):
            self.pattern = pattern

        def sub(self, replacement, text):
            assert len(text) * len(mapping) <= 2_000_000, (
                "redaction work must be bounded before regex scanning"
            )
            return self.pattern.sub(replacement, text)

    monkeypatch.setattr(
        _wire.re, "compile", lambda pattern: CheckedPattern(compile_pattern(pattern))
    )
    start = time.perf_counter()
    wire = result.to_dict()
    record_property("wire_seconds", time.perf_counter() - start)
    assert wire["pending"]["pause_id"] == "p"
    assert wire["pending"]["pending_tool_calls"][0]["params"] is None
    assert wire["pending"]["pending_tool_calls"][0]["args_hidden"] is True
