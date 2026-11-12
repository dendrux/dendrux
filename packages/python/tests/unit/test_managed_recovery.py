"""Managed subagents (0.7, T4): pure recovery and sweep decisions, no database."""

from __future__ import annotations

import logging
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux.errors import ManagedStateError, ManagedUnsupportedError
from dendrux.runtime._reclaim import _ManagedPending, _refuse_managed_approval
from dendrux.runtime.stores.sql.base import _SweepScan
from dendrux.types import ToolCall, generate_ulid

CALL = ToolCall(name="reviewer", params={"task": "t"}, id="m1")
RUN = generate_ulid()


def _action(kind: str, call_id: str = "s1") -> Any:
    return SimpleNamespace(kind=kind, call_id=call_id)


@pytest.mark.parametrize("kind", ["dispatch", "observe", "replay", "resolve"])
def test_a_managed_batch_without_an_approval_decision_is_completed(kind: str) -> None:
    _refuse_managed_approval(_ManagedPending(CALL, False), RUN, [_action(kind)], {}, None)
    _refuse_managed_approval(_ManagedPending(CALL, True), RUN, [], {}, {})


@pytest.mark.parametrize("kind", ["approval", "approval_replay"])
def test_a_recorded_approval_decision_keeps_the_refusal(kind: str) -> None:
    with pytest.raises(ManagedUnsupportedError) as info:
        _refuse_managed_approval(_ManagedPending(CALL, False), RUN, [_action(kind)], {}, None)
    assert info.value.operation == "recover_managed_approval"


def test_a_gate_raised_during_recovery_keeps_the_refusal() -> None:
    with pytest.raises(ManagedUnsupportedError):
        _refuse_managed_approval(_ManagedPending(CALL, False), RUN, (), {}, {"s1": {}})


def test_a_bound_intent_for_a_pending_call_is_invalid_state() -> None:
    with pytest.raises(ManagedStateError) as info:
        _refuse_managed_approval(_ManagedPending(CALL, True), RUN, [], {"m1": object()}, None)
    assert info.value.reason == "binding_outstanding"


def test_no_pending_managed_call_is_never_refused() -> None:
    _refuse_managed_approval(None, RUN, [_action("approval")], {"m1": object()}, {"x": {}})


class _Store:
    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        return None


def _row(version: Any, reason: Any, child: Any) -> Any:
    return SimpleNamespace(
        id=generate_ulid(), wait_version=version, wait_reason=reason, wait_child=child
    )


@pytest.mark.parametrize(
    ("version", "reason", "child", "skipped", "reported"),
    [
        (2, None, None, False, False),
        (None, "requested", None, False, False),
        (None, None, None, False, False),
        (3, "managed_wait", "c1", True, False),
        (3, "managed_wait", None, True, True),
        (2, "managed_wait", "c1", True, True),
        (3, "requested", "c1", True, True),
    ],
)
def test_waiting_rows_are_classified_from_their_projected_marks(
    caplog: Any, version: Any, reason: Any, child: Any, skipped: bool, reported: bool
) -> None:
    scan = _SweepScan(_Store(), waiting=True, older_than=timedelta(hours=1))  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        assert scan.skipped(_row(version, reason, child)) is skipped
    codes = [getattr(r, "code", None) for r in caplog.records]
    assert ("managed_state_invalid" in codes) is reported
