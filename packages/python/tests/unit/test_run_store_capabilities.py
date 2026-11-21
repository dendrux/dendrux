"""Capability inspection keeps the URL reader's lazy execution boundary intact."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.errors import StoreCapabilityError
from dendrux.runtime._blob_catalog import as_blob_catalog
from dendrux.runtime._capabilities import (
    capabilities,
    fanout_support,
    require_managed_fanout,
    require_managed_subagents,
    validate_capabilities,
)
from dendrux.runtime.state import StoreCapabilities
from dendrux.store import SQLiteStateStore, _InitOnFirstUse


@pytest.mark.parametrize(
    "discover",
    [
        capabilities,
        validate_capabilities,
        require_managed_subagents,
        require_managed_fanout,
        fanout_support,
    ],
)
async def test_discovery_inspects_inner_store_but_calls_remain_guarded(discover):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    inner = SQLiteStateStore(engine)
    inner.get_run = AsyncMock(return_value=None)
    inner.get_current = AsyncMock(return_value={})
    init = AsyncMock()
    wrapped = _InitOnFirstUse(inner, init, engine.dispose)
    try:
        found = discover(wrapped)
        assert found.capable if discover is fanout_support else found is inner.capabilities
        init.assert_not_awaited()
        assert await wrapped.get_run("missing") is None
        init.assert_awaited_once()
        catalog = as_blob_catalog(wrapped)
        assert catalog is wrapped
        assert await catalog.get_current("-", []) == {}
        init.assert_awaited_once()
        wrapped.closed = True
        with pytest.raises(InvalidRequestError):
            await wrapped.get_run("missing")
        with pytest.raises(InvalidRequestError):
            await catalog.get_current("-", [])
        inner.get_run.assert_awaited_once()
        inner.get_current.assert_awaited_once()
    finally:
        await engine.dispose()


def test_wrapper_does_not_hide_an_invalid_inner_declaration():
    class Broken:
        capabilities = StoreCapabilities(managed_fanout=True, deployment="single_process")

    wrapped = _InitOnFirstUse(Broken(), AsyncMock(), AsyncMock())
    with pytest.raises(StoreCapabilityError) as caught:
        validate_capabilities(wrapped)
    assert caught.value.capability == "managed_fanout"
    assert caught.value.method == "bind_managed_children"


def test_discovery_does_not_execute_dynamic_application_attributes():
    class Dynamic:
        def __getattr__(self, name):
            raise AssertionError(f"discovery executed {name}")

    store = Dynamic()
    assert capabilities(store) == validate_capabilities(store) == StoreCapabilities()
