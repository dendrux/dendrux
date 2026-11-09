"""Warning-filter controls; diagnostics-site coverage belongs to the next task."""

import warnings

import pytest

from tests._helpers.warnings import warnings_of


def test_061_warning_filter_ignores_unrelated_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.warn("intended warning", FutureWarning, stacklevel=1)
        warnings.warn("unrelated resource", ResourceWarning, stacklevel=1)
        warnings.warn("other future change", FutureWarning, stacklevel=1)
        warnings.warn("intended warning", UserWarning, stacklevel=1)
    matching = warnings_of(captured, FutureWarning, r"^intended warning$")
    assert len(matching) == 1
    assert matching[0] is captured[0]


@pytest.mark.parametrize("intended", [False, True])
def test_061_warning_filter_absence_still_detects_intended_warning(intended):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.warn("unrelated resource", ResourceWarning, stacklevel=1)
        warnings.warn("different message", FutureWarning, stacklevel=1)
        warnings.warn("intended warning", UserWarning, stacklevel=1)
        if intended:
            warnings.warn("intended warning", FutureWarning, stacklevel=1)
    if intended:
        with pytest.raises(AssertionError):
            assert not warnings_of(captured, FutureWarning, "intended warning")
    else:
        assert not warnings_of(captured, FutureWarning, "intended warning")


def test_061_warning_filter_accepts_subclasses_and_optional_message():
    class SpecializedWarning(FutureWarning):
        pass

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.warn("first 123", SpecializedWarning, stacklevel=1)
        warnings.warn("second", FutureWarning, stacklevel=1)
    assert warnings_of(captured, FutureWarning) == captured
    assert warnings_of(captured, FutureWarning, r"first \d+$") == [captured[0]]
    assert not warnings_of(captured, DeprecationWarning)
    assert not warnings_of([], FutureWarning)
