"""``retry_ids``: a caller-owned identity makes a repeated write exactly-once.

The runtime mints an id before it writes. Repeating the write with that id and the same content
returns the first result and stores nothing more; the same id with different content, or another
id at the same natural key, is a ``write_conflict``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.errors import PersistenceError
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import EVENT, tool_call_kwargs, ulid, usage

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

retry = contract("retry_ids", needs=("retry_ids",))


async def _trace(store: Any, run_id: str, ident: str, changed: bool) -> Any:
    return await store.save_trace(
        run_id, "user", "different" if changed else "same", order_index=0, trace_id=ident
    )


async def _tool_call(store: Any, run_id: str, ident: str, changed: bool) -> Any:
    kwargs = tool_call_kwargs(
        tool_call_id="call-1", result_payload='"changed"' if changed else '"ok"'
    )
    return await store.save_tool_call(run_id, **kwargs, row_id=ident)


async def _event(store: Any, run_id: str, ident: str, changed: bool) -> Any:
    return await store.save_run_event(
        run_id, event_type=EVENT, data={"n": 2 if changed else 1}, event_id=ident
    )


async def _usage(store: Any, run_id: str, ident: str, changed: bool) -> Any:
    return await store.save_usage(
        run_id, iteration_index=0, usage=usage(2 if changed else 1, 1), usage_id=ident
    )


async def _interaction(store: Any, run_id: str, ident: str, changed: bool) -> Any:
    return await store.save_llm_interaction(
        run_id,
        iteration_index=0,
        usage=usage(),
        semantic_request={"messages": ["changed" if changed else "same"]},
        interaction_id=ident,
    )


WRITERS = {
    "save_trace": _trace,
    "save_tool_call": _tool_call,
    "save_run_event": _event,
    "save_usage": _usage,
    "save_llm_interaction": _interaction,
}


COUNTABLE = ["save_trace", "save_tool_call", "save_run_event", "save_llm_interaction"]


async def _stored(store: Any, run_id: str, writer: str) -> int:
    """How many rows a countable writer left, read back through the protocol."""
    reads = {
        "save_trace": store.get_traces,
        "save_tool_call": store.get_tool_calls,
        "save_run_event": store.get_run_events,
        "save_llm_interaction": store.get_llm_interactions,
    }
    return len(await reads[writer](run_id))


@retry
@pytest.mark.parametrize("writer", COUNTABLE)
async def test_a_repeated_write_with_the_same_identity_stores_once(
    dendrux_kit: Kit, writer: str
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ident = ulid()
    first = await WRITERS[writer](store, run_id, ident, False)
    second = await WRITERS[writer](store, run_id, ident, False)
    assert first == second
    assert await _stored(store, run_id, writer) == 1


@retry
@pytest.mark.parametrize("writer", COUNTABLE)
async def test_the_same_identity_with_different_content_conflicts(
    dendrux_kit: Kit, writer: str
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ident = ulid()
    await WRITERS[writer](store, run_id, ident, False)
    with pytest.raises(PersistenceError) as caught:
        await WRITERS[writer](store, run_id, ident, True)
    assert caught.value.code == "write_conflict"
    assert await _stored(store, run_id, writer) == 1, "the conflicting write must not add a row"


@retry
async def test_save_usage_repeats_without_error_and_conflicts_on_changed_content(
    dendrux_kit: Kit,
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ident = ulid()
    first = await _usage(store, run_id, ident, False)
    assert await _usage(store, run_id, ident, False) == first
    with pytest.raises(PersistenceError) as caught:
        await _usage(store, run_id, ident, True)
    assert caught.value.code == "write_conflict"


@retry
async def test_save_usage_stores_one_row_per_identity_is_unverified(dendrux_kit: Kit) -> None:
    """Usage rows have no read in the protocol, so a duplicate row cannot be observed."""
    pytest.skip(
        "unverified: save_usage retry identity: usage rows cannot be read through the "
        "protocol, so a duplicate row is not detected"
    )


@retry
async def test_another_identity_at_the_same_trace_position_conflicts(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.save_trace(run_id, "user", "ok", order_index=0, trace_id=ulid())
    with pytest.raises(PersistenceError) as caught:
        await store.save_trace(run_id, "user", "ok", order_index=0, trace_id=ulid())
    assert caught.value.code == "write_conflict"
    assert len(await store.get_traces(run_id)) == 1


@retry
async def test_another_identity_for_the_same_tool_call_conflicts(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    kwargs = tool_call_kwargs(tool_call_id="call-1")
    await store.save_tool_call(run_id, **kwargs, row_id=ulid())
    with pytest.raises(PersistenceError) as caught:
        await store.save_tool_call(run_id, **kwargs, row_id=ulid())
    assert caught.value.code == "write_conflict"
    assert len(await store.get_tool_calls(run_id)) == 1


@retry
async def test_a_retried_event_keeps_its_sequence_and_later_events_follow(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    ident = ulid()
    first = await _event(store, run_id, ident, False)
    other = await store.save_run_event(run_id, event_type=EVENT, event_id=ulid())
    again = await _event(store, run_id, ident, False)
    assert (first, other, again) == (0, 1, 0)
    assert await store.get_next_event_sequence(run_id) == 2


@retry
async def test_identities_are_per_run(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    one, two = await dendrux_kit.new_run(), await dendrux_kit.new_run()
    shared = ulid()
    await _event(store, one, shared, False)
    other_run_result = None
    # a store may keep identities globally unique; it must still not corrupt run one
    with contextlib.suppress(PersistenceError):
        other_run_result = await _event(store, two, shared, False)
    assert len(await store.get_run_events(one)) == 1
    if other_run_result is not None:
        assert len(await store.get_run_events(two)) == 1
