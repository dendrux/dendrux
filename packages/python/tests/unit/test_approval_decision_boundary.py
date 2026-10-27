"""Per-call decisions use strict booleans and bounded identifiers."""

import pytest

from dendrux.runtime.submit import normalize_decisions
from dendrux.types import ApprovalDecision


@pytest.mark.parametrize("value", ["false", "yes", 1, [], None])
def test_per_call_decision_rejects_truthy_or_nonboolean_flags(value):
    with pytest.raises(ValueError):
        normalize_decisions({"c": ApprovalDecision(value)})


def test_per_call_decision_rejects_oversized_identifiers_without_echo():
    private = "PRIVATE_ID" * 10000
    with pytest.raises(ValueError) as caught:
        normalize_decisions({private: ApprovalDecision.approve()})
    assert len(str(caught.value)) < 512
    assert "PRIVATE_ID" not in str(caught.value)


@pytest.mark.parametrize("token", ["x\x00", "x\ud800", "x" * 100000])
def test_pause_generation_tokens_are_bounded_without_reflecting_values(token):
    from dendrux.runtime.submit import _require_approval_token
    from dendrux.types import InputValidationError

    with pytest.raises(InputValidationError) as caught:
        _require_approval_token(True, token)
    assert caught.value.field == "expected_pause_id"
    assert len(str(caught.value)) < 128
