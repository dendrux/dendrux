"""In-process pytest runs must preserve Dendrux module identity."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from dendrux.runtime.stores import sql

if TYPE_CHECKING:
    import pytest

pytest_plugins = ["pytester"]


def test_lazy_module_keeps_its_identity_after_an_inprocess_run(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_import_lazy_registry():
            from dendrux.runtime.stores.sql import _live_registry
            assert callable(_live_registry.register)
        """
    )
    result = pytester.runpytest_inprocess(
        "-q", "--no-cov", "-p", "no:cacheprovider", "-p", "no:randomly",
        "-o", "asyncio_default_fixture_loop_scope=function",
    )  # fmt: skip
    result.assert_outcomes(passed=1)

    original = sql._live_registry
    reimported = importlib.import_module("dendrux.runtime.stores.sql._live_registry")
    assert original is reimported, "pytester left two copies of the lazy registry"
    assert original.register is reimported.register
