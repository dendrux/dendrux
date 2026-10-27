"""Invalid read filters do not reflect arbitrary caller objects."""

import datetime as dt

import pytest

from dendrux.runtime.state import _check_scope, _require_aware_utc


class PrivateValue:
    def __repr__(self):
        return "PRIVATE_READ_VALUE"


class PrivateDate(dt.datetime):
    def __repr__(self):
        return "PRIVATE_DATE_VALUE"


def test_scope_error_omits_the_invalid_value():
    with pytest.raises(ValueError) as caught:
        _check_scope(PrivateValue())
    assert "PRIVATE_READ_VALUE" not in str(caught.value)


def test_date_error_omits_the_invalid_value():
    with pytest.raises(ValueError) as caught:
        _require_aware_utc("before", PrivateDate(2026, 1, 1))
    assert "PRIVATE_DATE_VALUE" not in str(caught.value)


@pytest.mark.parametrize("value", [42, "2026-01-01", PrivateValue()])
def test_date_filter_rejects_non_dates_with_fixed_value_error(value):
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        _require_aware_utc("before", value)
