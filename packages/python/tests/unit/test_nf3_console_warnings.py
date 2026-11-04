"""Unused ConsoleNotifier options warn at their caller without breaking construction."""

import inspect
import warnings

import pytest

from dendrux.notifiers import ConsoleNotifier


@pytest.mark.parametrize(
    "kwargs, names",
    [
        ({}, []),
        ({"show_llm_text": False, "max_text_length": 120}, []),
        ({"show_params": False}, []),
        ({"show_llm_text": True}, ["show_llm_text"]),
        ({"max_text_length": 0}, ["max_text_length"]),
        ({"max_text_length": 200}, ["max_text_length"]),
        ({"show_llm_text": True, "max_text_length": 10}, ["show_llm_text", "max_text_length"]),
    ],
)
def test_unused_options_warn_at_caller(kwargs, names):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        notifier = ConsoleNotifier(**kwargs)
    assert len(caught) == len(names)
    for warning, name in zip(caught, names, strict=True):
        assert warning.category is DeprecationWarning
        assert str(warning.message) == (
            f"ConsoleNotifier.{name} has no effect and will be removed or implemented "
            "in a later release"
        )
        assert warning.filename == __file__
        assert warning.lineno == line
    assert notifier._show_llm_text == kwargs.get("show_llm_text", False)
    assert notifier._max_text_length == kwargs.get("max_text_length", 120)
    assert notifier._show_params == kwargs.get("show_params", True)
