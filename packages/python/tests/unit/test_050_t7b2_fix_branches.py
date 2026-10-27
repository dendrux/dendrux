"""Exercise cancellation-safe context keys and image replay failure boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dendrux.errors import BlobNotFoundError, ImagePersistenceUnsupportedError, InvalidImageError
from dendrux.media._catalog import BlobEdge
from dendrux.runtime.context import DelegationContext
from dendrux.runtime.runner import _idempotent_hit_after_image_error, _write_edges
from dendrux.types import IdempotencyConflictError


def test_context_remains_a_dictionary_key_after_warning():
    context = DelegationContext(run_id="parent", tenant_id="tenant")
    cache = {context: "parent resources"}
    context.warned_mismatches.add("store")
    assert cache[context] == "parent resources"
    assert cache[
        DelegationContext(run_id="parent", tenant_id="tenant", warned_mismatches={"store"})
    ]


@pytest.mark.parametrize("runtime", [None, SimpleNamespace(catalog=None)])
async def test_image_edges_fail_closed_without_catalog(runtime):
    await _write_edges(runtime, "r", [])
    edge = BlobEdge("t:A", "a" * 64, "input", "input", 0)
    with pytest.raises(ImagePersistenceUnsupportedError):
        await _write_edges(runtime, "r", [edge])


async def test_image_edges_keep_run_and_tenant_scope():
    catalog = SimpleNamespace(add_blob_refs=AsyncMock())
    edge = BlobEdge("t:A", "a" * 64, "input", "input", 0)
    await _write_edges(SimpleNamespace(catalog=catalog, tenant_scope="t:A"), "r", [edge])
    catalog.add_blob_refs.assert_awaited_once_with("r", "t:A", [edge])


async def replay(exc=None, store=None, key="k", items=("text",)):
    return await _idempotent_hit_after_image_error(
        exc or BlobNotFoundError(),
        store,
        key,
        tenant_id="A",
        agent_name="a",
        user_input="text",
        items=items,
        history=None,
        output_type=None,
        context=None,
        replay=None,
        skills=None,
    )


@pytest.mark.parametrize("reason", ["corrupt", "empty"])
async def test_unrelated_image_error_does_not_probe_idempotency(reason):
    check = AsyncMock()
    assert (
        await replay(InvalidImageError(reason), SimpleNamespace(_check_idempotency=check)) is None
    )
    check.assert_not_awaited()


async def test_image_replay_requires_key_store_items_and_lookup():
    check = AsyncMock()
    store = SimpleNamespace(_check_idempotency=check)
    assert await replay(store=store, key=None) is None
    assert await replay(store=None) is None
    assert await replay(store=store, items=()) is None
    assert await replay(store=SimpleNamespace()) is None
    check.assert_not_awaited()


@pytest.mark.parametrize("exc", [BlobNotFoundError(), InvalidImageError("purged")])
async def test_image_replay_returns_original_lookup_outcome(exc):
    check = AsyncMock(return_value="existing-outcome")
    assert await replay(exc, SimpleNamespace(_check_idempotency=check)) == "existing-outcome"
    check.assert_awaited_once_with("k", "", tenant_id="A")


@pytest.mark.parametrize(
    "run_id,record", [(None, None), ("r", None), ("r", SimpleNamespace(input_data={}))]
)
async def test_image_replay_conflict_never_discloses_or_invents_a_match(run_id, record):
    store = SimpleNamespace(
        _check_idempotency=AsyncMock(
            side_effect=IdempotencyConflictError(run_id=run_id, idempotency_key="k")
        ),
        get_run=AsyncMock(return_value=record),
    )
    with pytest.raises(IdempotencyConflictError) as caught:
        await replay(store=store)
    assert caught.value.run_id == run_id
    if run_id is None:
        store.get_run.assert_not_awaited()
    else:
        store.get_run.assert_awaited_once_with(run_id)
