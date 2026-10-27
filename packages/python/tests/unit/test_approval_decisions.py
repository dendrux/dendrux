"""Unit tests for per-call approval helpers and types."""

from __future__ import annotations

import pytest

from dendrux import ApprovalDecision
from dendrux.runtime.submit import (
    approval_gates_of,
    build_rejection_results,
    deserialize_decisions,
    is_client_call,
    normalize_decisions,
    serialize_decisions,
    validate_approval_decisions,
)
from dendrux.types import PauseState, ToolCall, ToolResult, ToolTarget, UsageStats


def _pause(gates: dict | None = None, targets: dict | None = None) -> PauseState:
    calls = [
        ToolCall(name="refund", params={}, id="c1"),
        ToolCall(name="search", params={}, id="c2"),
        ToolCall(name="read_range", params={}, id="c3"),
    ]
    return PauseState(
        agent_name="A",
        pending_tool_calls=calls,
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
        pending_targets=targets
        if targets is not None
        else {"c1": "server", "c2": "server", "c3": "client"},
        approval_gates=gates,
    )


GATES = {"c1": {"source": "static", "reason": "requires_approval"}}


class TestTypes:
    def test_constructors(self) -> None:
        assert ApprovalDecision.approve() == ApprovalDecision(approved=True)
        assert ApprovalDecision.deny("x") == ApprovalDecision(approved=False, reason="x")
        assert ApprovalDecision.deny().reason is None
        with pytest.raises(ValueError):
            ApprovalDecision.deny(3)  # type: ignore[arg-type]

    def test_pause_state_round_trip(self) -> None:
        p = _pause(GATES)
        d = p.to_dict()
        assert d["approval_gates"] == GATES
        assert PauseState.from_dict(d).approval_gates == GATES

    def test_pause_state_without_gates_has_no_key(self) -> None:
        d = _pause(None).to_dict()
        assert "approval_gates" not in d
        assert PauseState.from_dict(d).approval_gates is None

    def test_tool_result_meta_not_compared(self) -> None:
        a = ToolResult(name="t", call_id="c", payload="1", meta={"substituted": True})
        assert a == ToolResult(name="t", call_id="c", payload="1")
        assert a.meta == {"substituted": True}


class TestRouting:
    def test_client_by_persisted_or_current_target(self) -> None:
        p = _pause(GATES)
        c1, c2, c3 = p.pending_tool_calls
        assert is_client_call(p, c3)
        assert not is_client_call(p, c1)
        # Fail-safe: current definition says client even if persisted says server.
        assert is_client_call(p, c2, {"search": ToolTarget.CLIENT})

    def test_legacy_gates_cover_all_server_calls(self) -> None:
        p = _pause(None, targets={})
        gates = approval_gates_of(p, {"read_range": ToolTarget.CLIENT})
        assert set(gates) == {"c1", "c2"}
        assert gates["c1"]["source"] == "legacy"

    def test_snapshot_used_verbatim(self) -> None:
        assert approval_gates_of(_pause(GATES), {"refund": ToolTarget.CLIENT}) == GATES


class TestValidation:
    def test_ok(self) -> None:
        got = validate_approval_decisions("r", _pause(GATES), [("c1", ApprovalDecision.approve())])
        assert got == {"c1": ApprovalDecision.approve()}

    @pytest.mark.parametrize(
        ("pairs", "match"),
        [
            ([], "missing call_ids=\\['c1'\\]"),
            ([("c1", ApprovalDecision.approve()), ("zz", ApprovalDecision.approve())], "unknown"),
            ([("c1", ApprovalDecision.approve()), ("c2", ApprovalDecision.approve())], "unknown"),
            ([("c1", ApprovalDecision.approve()), ("c1", ApprovalDecision.deny())], "duplicate"),
        ],
    )
    def test_rejects(self, pairs, match) -> None:
        with pytest.raises(ValueError, match=match):
            validate_approval_decisions("r", _pause(GATES), pairs)

    def test_normalize(self) -> None:
        assert normalize_decisions({"c1": ApprovalDecision.approve()}) == [
            ("c1", ApprovalDecision.approve())
        ]
        with pytest.raises(ValueError):
            normalize_decisions({1: ApprovalDecision.approve()})
        with pytest.raises(ValueError):
            normalize_decisions({"c1": True})
        with pytest.raises(ValueError):
            normalize_decisions([("c1", ApprovalDecision.approve())])

    def test_serialize_round_trip_in_pending_order(self) -> None:
        p = _pause({**GATES, "c2": {"source": "hook", "reason": None}})
        wire = serialize_decisions(
            p, {"c2": ApprovalDecision.deny("no"), "c1": ApprovalDecision.approve()}
        )
        assert [w["call_id"] for w in wire] == ["c1", "c2"]
        assert deserialize_decisions(wire) == [
            ("c1", ApprovalDecision.approve()),
            ("c2", ApprovalDecision.deny("no")),
        ]

    def test_deserialize_fails_closed(self) -> None:
        assert deserialize_decisions([{"call_id": "c1", "approved": "yes"}]) == [
            ("c1", ApprovalDecision(approved=False))
        ]
        with pytest.raises(ValueError):
            deserialize_decisions({"c1": True})
        with pytest.raises(ValueError):
            deserialize_decisions([{"approved": True}])

    def test_legacy_rejection_payload(self) -> None:
        results = build_rejection_results(_pause(GATES))
        assert {r.payload for r in results} == {'"User declined to run this tool."'}
