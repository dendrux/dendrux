"""The live-update seam is bounded, lossy and independent of store results."""

import logging
import pickle

import pytest

from dendrux import _codes, errors

NEW_CODES = (
    "live_updates_degraded",
    "live_updates_recovered",
    "live_updates_unavailable",
    "live_updates_missed_hint",
    "live_updates_send_failed",
    "live_updates_send_recovered",
    "live_updates_overflow",
    "live_updates_wrong_loop",
    "live_updates_bad_message",
    "live_updates_close_timeout",
    "live_updates_tls_unknown",
    "live_updates_many_stores",
    "follow_read_failed",
    "follow_coordinator_restart",
)


@pytest.mark.parametrize("name", NEW_CODES)
def test_live_codes_registered(name, caplog):
    assert name in _codes.REGISTRY
    code = _codes.REGISTRY[name]
    with caplog.at_level(logging.INFO):
        _codes.log_code(logging.getLogger(__name__), code, run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert caplog.records[-1].code == name


def test_live_code_fields_are_validated(caplog):
    assert "live_updates_send_failed" in _codes.REGISTRY
    with caplog.at_level(logging.INFO):
        _codes.log_code(
            logging.getLogger(__name__),
            _codes.LIVE_UPDATES_SEND_FAILED,
            error_kind="timeout",
            count=10,
        )
        _codes.log_code(
            logging.getLogger(__name__),
            _codes.LIVE_UPDATES_MANY_STORES,
            count=17,
            database_key="a" * 64,
        )
    assert caplog.records[0].error_kind == "timeout"
    assert caplog.records[0].kind == "live_updates"
    assert caplog.records[1].database_key == "a" * 64
    with pytest.raises(ValueError):
        _codes.log_code(
            logging.getLogger(__name__), _codes.LIVE_UPDATES_SEND_FAILED, error_kind="VENDOR_SECRET"
        )


def test_error_move_preserves_public_aliases_and_pickle():
    from dendrux.store import FollowCapacityError
    from dendrux.store._follow import FollowCapacityError as OldCapacityError

    assert getattr(errors, "FollowCapacityError", None) is FollowCapacityError is OldCapacityError
    original = FollowCapacityError(100, per_tenant=True)
    copy = pickle.loads(pickle.dumps(original))
    assert type(copy) is FollowCapacityError
    assert (copy.limit, copy.per_tenant, str(copy)) == (100, True, str(original))


def test_unavailable_error_sanitizes_reason(caplog):
    cls = getattr(errors, "LiveUpdatesUnavailableError", None)
    assert cls is not None
    assert issubclass(cls, RuntimeError)
    with caplog.at_level(logging.WARNING):
        error = cls("VENDOR_SECRET")
    assert error.reason == "other"
    assert "VENDOR_SECRET" not in str(error) + repr(vars(error)) + caplog.text
    assert caplog.records[-1].code == "live_updates_unavailable"
    assert pickle.loads(pickle.dumps(error)).reason == "other"
