"""0.5.0 contracts: the new public names are exported from ``dendrux``."""

from __future__ import annotations

import pytest

import dendrux
from dendrux import errors
from dendrux.runtime.state import UNSCOPED
from dendrux.store import UNSCOPED as STORE_UNSCOPED
from dendrux.types import ErrorInfo, Limits

_ERRORS = [
    "BudgetConfigurationError",
    "ContentFilteredError",
    "GovernanceMismatchError",
    "ModelRefusalError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderStreamIncompleteError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "TenantScopeRequiredError",
]


@pytest.mark.parametrize("name", _ERRORS)
def test_public_error_exports(name: str) -> None:
    assert name in errors.__all__
    assert name in dendrux.__all__
    assert getattr(dendrux, name) is getattr(errors, name)


def test_type_exports() -> None:
    assert dendrux.Limits is Limits
    assert dendrux.ErrorInfo is ErrorInfo
    assert "Limits" in dendrux.__all__
    assert "ErrorInfo" in dendrux.__all__


def test_unscoped_exports() -> None:
    assert dendrux.UNSCOPED is UNSCOPED is STORE_UNSCOPED
    assert "UNSCOPED" in dendrux.__all__


def test_internal_names_not_exported() -> None:
    assert "RunFencedError" not in dendrux.__all__
    assert not hasattr(dendrux, "RunFencedError")


def test_every_export_resolves() -> None:
    for name in dendrux.__all__:
        getattr(dendrux, name)
