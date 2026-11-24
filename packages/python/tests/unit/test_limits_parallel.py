"""Parallel-call public types, construction bounds and safe diagnostic contracts."""

import inspect
import logging

import pytest

from dendrux import _codes, types
from dendrux.errors import ManagedStateError
from dendrux.runtime.submit import _governed_limits
from dendrux.types import InputValidationError, Limits, RunEvent, RunEventType, RunResult, RunStatus


def test_parallel_default_ceiling_and_keyword_only():
    assert getattr(Limits(), "max_parallel_subagents", None) == 4
    assert getattr(types, "MAX_PARALLEL_SUBAGENTS", None) == 32
    parameter = inspect.signature(Limits).parameters["max_parallel_subagents"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("value", [1, 4, 30, 32])
def test_parallel_limit_accepts_inclusive_range(value):
    assert Limits(**{"max_parallel_subagents": value}).max_parallel_subagents == value


@pytest.mark.parametrize("value", [0, -1, 33, True, False, "4", "SENTINEL", None, 4.0])
def test_parallel_limit_refuses_invalid_values_safely(value, caplog):
    try:
        Limits(**{"max_parallel_subagents": value})
    except TypeError:
        pytest.fail("parallel limit field is missing")
    except InputValidationError as error:
        assert error.field == "max_parallel_subagents"
        assert error.code == "invalid_limit"
        assert "SENTINEL" not in str(error)
    else:
        pytest.fail("invalid parallel limit accepted")
    assert any(
        r.code == "invalid_limit" and getattr(r, "run_id", None) is None for r in caplog.records
    )
    assert "SENTINEL" not in caplog.text


def test_pending_types_are_additive_and_pause_event_is_nonterminal():
    result = RunResult(run_id="root", status=RunStatus.SUCCESS)
    assert getattr(result, "pendings", None) == ()
    assert getattr(RunEventType, "SUBAGENT_PAUSED", None) == "subagent_paused"
    event = RunEvent(type=RunEventType.SUBAGENT_PAUSED)
    assert "pending" not in inspect.signature(RunEvent).parameters
    assert event.pause_id is None and event.call_ids == ()
    signal = RunEvent(type=event.type, run_id="child", pause_id="pause", call_ids=("a", "b"))
    assert (signal.pause_id, signal.call_ids) == ("pause", ("a", "b"))
    assert event.type not in types._TERMINAL_RUN_EVENTS


@pytest.mark.parametrize("reason", ["fanout_limit", "fanout_capability", "fanout_invalid"])
def test_fanout_state_reasons_are_preserved_and_logged(reason, caplog):
    error = ManagedStateError(reason=reason, run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert error.reason == reason
    assert caplog.records[-1].managed_reason == reason


@pytest.mark.parametrize(
    "name,fields",
    [
        ("managed_fanout_limit", {"requested": 33, "limit": 32, "store_capability": True}),
        ("managed_fanout_branch_error", {"error_type": "RuntimeError"}),
    ],
)
def test_fanout_diagnostic_codes_accept_only_safe_fields(name, fields, caplog):
    code = getattr(_codes, name.upper(), None)
    assert code is not None
    assert _codes.REGISTRY[name] == code
    with caplog.at_level(logging.WARNING):
        _codes.log_code(
            logging.getLogger("dendrux.test"), code, run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", **fields
        )
    record = caplog.records[-1]
    assert record.code == name
    for key, value in fields.items():
        assert getattr(record, key) == value


@pytest.mark.parametrize("reason", ["client_call", "fanout_limit", "exclusive"])
def test_batch_diagnostic_accepts_refusal_reason_and_counts(reason, caplog):
    _codes.log_code(
        logging.getLogger("dendrux.test"),
        _codes.MANAGED_BATCH_UNSUPPORTED,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        reason=reason,
        requested=5,
        limit=4,
        count=5,
    )
    record = caplog.records[-1]
    assert record.reason == reason
    assert (record.requested, record.limit, record.count) == (5, 4, 5)


def test_fanout_diagnostics_refuse_arbitrary_text():
    code = getattr(_codes, "MANAGED_FANOUT_LIMIT", None)
    assert code is not None
    with pytest.raises(ValueError) as exc:
        _codes.log_code(
            logging.getLogger("dendrux.test"),
            code,
            requested=33,
            limit=32,
            store_capability="SENTINEL",
        )
    assert "SENTINEL" not in str(exc.value)


def test_default_parallel_limit_is_not_governed():
    assert "max_parallel_subagents" not in _governed_limits(Limits())


def test_parallel_limit_override_is_governed():
    assert _governed_limits(Limits(max_parallel_subagents=8))["max_parallel_subagents"] == 8
    assert _governed_limits(Limits(max_parallel_subagents=16))["max_parallel_subagents"] == 16


def test_fanout_diagnostic_refuses_legacy_count_field():
    with pytest.raises(ValueError, match="Invalid structured log fields"):
        _codes.log_code(logging.getLogger("dendrux.test"), _codes.MANAGED_FANOUT_LIMIT, count=2)
