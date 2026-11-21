"""Optional filter discovery validates declarations and preserves legacy signatures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from dendrux.errors import StoreCapabilityError
from dendrux.runtime._capabilities import validate_capabilities, waiting_filter_options
from dendrux.runtime.state import StoreCapabilities
from dendrux.store import RunStore
from dendrux.testing.store_kit.contracts.test_waiting_label import _LegacyStore


class _FilterStore:
    capabilities = StoreCapabilities(waiting_filter=True, deployment="single_process")

    async def list_runs(self, *, include_waiting: bool = True, **kwargs: Any) -> list[Any]:
        assert include_waiting is False
        return []

    async def count_runs(self, *, include_waiting: bool = True, **kwargs: Any) -> int:
        assert include_waiting is False
        return 0


def test_discovery_requires_explicit_flag_and_keyword_signatures() -> None:
    assert StoreCapabilities().waiting_filter is False
    assert waiting_filter_options(object(), include_waiting=True) == {}
    with pytest.raises(StoreCapabilityError):
        waiting_filter_options(object(), include_waiting=False)

    class FalseDeclaration(_FilterStore):
        capabilities = replace(_FilterStore.capabilities, waiting_filter=False)

    with pytest.raises(StoreCapabilityError):
        waiting_filter_options(FalseDeclaration(), include_waiting=False)

    class BadSignature(_FilterStore):
        async def count_runs(self, **kwargs: Any) -> int:
            return 0

    with pytest.raises(StoreCapabilityError):
        validate_capabilities(BadSignature())


async def test_capable_filter_forwards_false_to_declared_store() -> None:
    reader = RunStore(_FilterStore())
    assert await reader.list_runs(include_waiting=False) == []
    assert await reader.count_runs(include_waiting=False) == 0


@pytest.mark.parametrize("method", ["list_runs", "count_runs"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", []])
async def test_waiting_option_rejects_nonboolean_values(method, value) -> None:
    with pytest.raises(TypeError, match="include_waiting"):
        await getattr(RunStore(_FilterStore()), method)(include_waiting=value)


async def test_legacy_facade_keeps_default_labels_and_refuses_filter() -> None:
    reader = RunStore(_LegacyStore())
    assert (await reader.list_runs())[0].waiting_for is None
    assert (await reader.get_run("legacy")).waiting_for is None
    assert await reader.count_runs(include_waiting=True) == 1
    assert len(await reader.list_runs(include_waiting=True)) == 1
    for method in (reader.list_runs, reader.count_runs):
        with pytest.raises(StoreCapabilityError):
            await method(include_waiting=False)
