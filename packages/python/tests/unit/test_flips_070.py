"""Release defaults promised by 0.6 and flipped in 0.7.0, with the explicit old choices."""

from __future__ import annotations

import inspect
import warnings
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from dendrux.runtime.sweep import sweep
from tests._helpers.warnings import warnings_of


def legacy_store(stale):
    """A store without the recovery capability, on the public fallback path."""
    store = AsyncMock()
    store.sweep_stale_runs.return_value = stale
    store.__class__ = object
    return store


def test_sweep_signature_defaults_to_reclaim():
    assert inspect.signature(sweep).parameters["on_stale"].default == "reclaim"


@pytest.mark.parametrize("kwargs", [{}, {"on_stale": None}])
async def test_omitted_policy_reclaims_and_needs_recovery(kwargs):
    store = legacy_store(["old-run"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match=r'recovery.*on_stale="error"'):
            await sweep(state_store=store, stale_running=timedelta(seconds=1), **kwargs)
    store.sweep_stale_runs.assert_not_awaited()
    assert not warnings_of(caught, FutureWarning)


@pytest.mark.parametrize("stale", [[], ["old-run"]])
async def test_explicit_error_policy_keeps_finalising_without_warning(stale):
    store = legacy_store(stale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await sweep(
            state_store=store, stale_running=timedelta(seconds=1), on_stale="error"
        )
    assert result.stale_running == stale
    store.sweep_stale_runs.assert_awaited_once()
    assert not warnings_of(caught, FutureWarning)
