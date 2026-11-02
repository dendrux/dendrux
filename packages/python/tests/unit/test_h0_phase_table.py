"""Persisted recovery vocabulary, inventoried from its writers and readers.

Loop phases: PersistenceRecorder._snapshot, checkpoint_outcome,
on_message_appended, consume_resume, on_tool_completed; SQLStateStore._cas_finalize;
state._apply_loop_patch and _reclaim.resolve. Call phases: on_message_appended,
checkpoint_outcome, consume_resume, on_tool_started and resolve_pending.
These are persisted phases, not message/answer phases (commentary/final_answer).
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.runtime._reclaim import _recovery_pause_result
from dendrux.runtime.persistence import ResolutionAction, resolve_pending
from dendrux.types import PauseState, RunStatus, UsageStats
from tests._helpers.h0_phases import EVIDENCE, LOOP_PHASES, PHASE_ACTIONS, source_phase_inventory


@pytest.mark.parametrize("call_phase", PHASE_ACTIONS)
@pytest.mark.parametrize("outcome", EVIDENCE)
@pytest.mark.parametrize("durable_idempotent", [False, True])
@pytest.mark.parametrize("current_idempotent", [None, False, True])
def test_persisted_phase_evidence_table(
    call_phase, outcome, durable_idempotent, current_idempotent
):
    state = {
        "phase": "dispatch",
        "pending": [
            {
                "call_id": "call",
                "tool": "effect",
                "phase": call_phase,
                "order_index": 17,
                "target": "client" if call_phase == "client" else "server",
            }
        ],
        "submission": None,
    }
    rows = (
        []
        if outcome == "absent"
        else [{"tool_call_id": "call", "outcome": outcome, "idempotent": durable_idempotent}]
    )
    current = None if current_idempotent is None else {"effect": current_idempotent}
    before = deepcopy((state, rows, current))
    safe = durable_idempotent and current_idempotent is not False
    column = {"absent": 0, "started": 2 if safe else 1}.get(outcome, 3)
    expected = PHASE_ACTIONS[call_phase][column]
    assert resolve_pending(state, rows, current_idempotent=current) == [
        ResolutionAction(expected, "call", 17)
    ]
    assert (state, rows, current) == before


@pytest.mark.parametrize("outcome", EVIDENCE)
def test_empty_pending_and_durable_submission_precedence(outcome):
    rows = [] if outcome == "absent" else [{"tool_call_id": "old", "outcome": outcome}]
    assert resolve_pending({"phase": "dispatch", "pending": []}, rows) == []
    for call_phase in PHASE_ACTIONS:
        state = {
            "phase": "dispatch",
            "pending": [{"phase": call_phase}],
            "submission": {"submission_id": "submitted"},
        }
        assert resolve_pending(state, rows) == [ResolutionAction("consume_submission")]


def test_unknown_phase_and_unrecognized_evidence_refuse():
    pending = {"call_id": "call", "phase": "future", "order_index": 17, "tool": "effect"}
    with pytest.raises(ValueError, match="Unsupported pending-call phase"):
        resolve_pending({"pending": [pending]}, [])
    pending["phase"] = "dispatchable"
    with pytest.raises(ValueError, match="Unsupported tool-call outcome"):
        resolve_pending({"pending": [pending]}, [{"tool_call_id": "call", "outcome": "future"}])


def test_source_phase_inventory_matches_the_evidence_table():
    inventory = source_phase_inventory()
    assert set(inventory) == set(LOOP_PHASES) | set(PHASE_ACTIONS), inventory


def test_call_resolution_is_independent_of_loop_phase():
    pending = [
        {"call_id": "gate", "phase": "gated", "order_index": 20, "tool": "effect"},
        {"call_id": "work", "phase": "approved", "order_index": 3, "tool": "effect"},
    ]
    for loop_phase in LOOP_PHASES:
        assert resolve_pending({"phase": loop_phase, "pending": pending}, []) == [
            ResolutionAction("approval", "gate", 20),
            ResolutionAction("dispatch", "work", 3),
        ]


@pytest.mark.parametrize(
    "phases,rows,expected",
    [
        (["client", "approved", "gated"], [], ["wait_client", "dispatch", "approval"]),
        (["gated", "invalid", "dispatchable"], [], ["approval", "resolve", "dispatch"]),
        (
            ["dispatchable", "rejected", "approved"],
            [
                {"tool_call_id": "c0", "outcome": "started", "idempotent": False},
                {"tool_call_id": "c1", "outcome": "completed"},
            ],
            ["approval_replay", "observe", "dispatch"],
        ),
    ],
)
def test_mixed_pending_calls_keep_declared_order_across_barriers(phases, rows, expected):
    pending = [
        {"call_id": f"c{i}", "phase": phase, "order_index": order, "tool": "effect"}
        for i, (phase, order) in enumerate(zip(phases, (20, 3, 11), strict=True))
    ]
    assert resolve_pending({"pending": pending}, rows) == [
        ResolutionAction(kind, f"c{i}", order)
        for i, (kind, order) in enumerate(zip(expected, (20, 3, 11), strict=True))
    ]


@pytest.mark.parametrize("loop_phase", LOOP_PHASES)
@pytest.mark.parametrize("submitted", [False, True])
async def test_only_unsubmitted_awaiting_input_recreates_a_human_pause(loop_phase, submitted):
    pause = PauseState(
        agent_name="phase-table",
        history=[],
        steps=[],
        pending_tool_calls=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
    )
    ctx = SimpleNamespace(pause_state=pause, history=[])
    read = AsyncMock(return_value=SimpleNamespace(content='{"question": "Choose a value"}'))
    result = await _recovery_pause_result(
        SimpleNamespace(get_trace=read),
        "run",
        ctx,
        {"phase": loop_phase, "clarification_ref": "question"},
        {"submission_id": "submitted"} if submitted else None,
        [],
        {},
    )
    if loop_phase == "awaiting_input" and not submitted:
        assert result.status == RunStatus.WAITING_HUMAN_INPUT
        assert result.run_id == "run" and result.answer == "Choose a value"
        read.assert_awaited_once_with("run", "question")
    else:
        assert result is None
        read.assert_not_awaited()


def test_inventory_includes_consume_resume_call_phase_assignments(monkeypatch):
    from pathlib import Path

    from dendrux.runtime import persistence

    path = Path(persistence.__file__).resolve()
    original = Path.read_text

    def source(candidate, *args, **kwargs):
        text = original(candidate, *args, **kwargs)
        if candidate == path:
            old = '"approved"\n                    if decisions.get'
            assert text.count(old) == 1
            return text.replace(old, '"new_approval_phase"\n                    if decisions.get')
        return text

    monkeypatch.setattr(Path, "read_text", source)
    assert "new_approval_phase" in source_phase_inventory()
