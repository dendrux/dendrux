"""``checkpointing``: execution state, conversation reconstruction and submission consumption.

Format-2 runs keep a bounded execution projection (``loop_state``) next to their traces so a
paused or crashed run can be rebuilt. These contracts run only when the store creates format-2
runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dendrux.errors import PersistenceError, RunFencedError
from dendrux.runtime.state import LoopStatePatch
from dendrux.testing.store_kit.contracts._marks import contract
from dendrux.testing.store_kit.contracts._support import trace_write, ulid
from dendrux.types import InputValidationError

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit

checkpointing = contract("checkpointing", needs=("checkpointing",), format2=True)
TRACEPARENT = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"


@checkpointing
async def test_seeds_rewrites_and_conversation_pages(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    seeds = [trace_write("system", "system", 0, "seed"), trace_write("user", "seed", 1, "seed")]
    run_id = await dendrux_kit.new_run(seed_traces=seeds)
    await store.save_trace(run_id, "assistant", "old", order_index=2)
    await store.save_trace(
        run_id, "state", '{"kind":"image_view","v":1}', order_index=3, trace_kind="state"
    )
    replacement = [
        {"role": "system", "content": "system", "meta": {}},
        {"role": "user", "content": "summary", "meta": {}},
    ]
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "model_next"},
        pii_mapping={"x": "y"},
        trace=trace_write("rewrite", json.dumps(replacement), 4, "rewrite"),
    )
    await store.save_trace(run_id, "assistant", "new", order_index=5)
    first = await store.conversation_traces(run_id, after_position=None, limit=1)
    assert [(e.position, e.content) for e in first.entries] == [(4001, "system")]
    second = await store.conversation_traces(run_id, after_position=first.next_position, limit=2)
    assert [(e.position, e.content) for e in second.entries] == [(4002, "summary"), (5000, "new")]
    assert second.next_position is None
    assert await store.get_loop_state(run_id) == {"v": 1, "phase": "model_next"}
    assert await store.get_pii_mapping(run_id) == {"x": "y"}


@checkpointing
async def test_raw_traces_page_by_order_index_and_kind(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    seed = trace_write("system", "s", 0, "seed")
    run_id = await dendrux_kit.new_run(seed_traces=[seed])
    await store.save_trace(run_id, "user", "u", order_index=1)
    await store.save_trace(
        run_id, "state", '{"kind":"image_view","v":1}', order_index=2, trace_kind="state"
    )
    await store.save_trace(run_id, "assistant", "a", order_index=3)
    after_one = await store.get_traces_after(run_id, after_order_index=1)
    assert [t.order_index for t in after_one] == [2, 3]
    assert [t.trace_kind for t in after_one] == ["state", "turn"]
    only_turns = await store.get_traces_after(run_id, after_order_index=None, kinds=("turn",))
    assert [t.order_index for t in only_turns] == [1, 3]
    assert [
        t.order_index for t in await store.get_traces_after(run_id, after_order_index=None, limit=2)
    ] == [0, 1]
    found = await store.get_trace(run_id, seed.trace_id)
    assert found is not None and found.content == "s"
    assert await store.get_trace(run_id, seed.trace_id, tenant_id="foreign") is None


@checkpointing
@pytest.mark.parametrize("payload", [{"provider_kwargs": {"x": "a" * 2048}}, {"x": "a" * 32768}])
async def test_an_oversized_checkpoint_is_refused_and_writes_nothing(
    dendrux_kit: Kit, payload: dict[str, object]
) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    with pytest.raises(PersistenceError) as caught:
        await store.checkpoint_step(
            run_id, claim_epoch=0, loop_state=payload, pii_mapping={"secret": "v"}, trace=None
        )
    assert caught.value.code == "checkpoint_too_large"
    assert await store.get_loop_state(run_id) is None
    assert await store.get_pii_mapping(run_id) is None


@checkpointing
async def test_an_unpersistable_checkpoint_is_refused_before_any_write(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    with pytest.raises(InputValidationError):
        await store.checkpoint_step(
            run_id, claim_epoch=0, loop_state={"v": 1}, pii_mapping={"t": "\x00"}, trace=None
        )
    assert await store.get_loop_state(run_id) is None
    assert await store.get_pii_mapping(run_id) is None


@checkpointing
async def test_a_checkpoint_and_its_trace_commit_together_or_not_at_all(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    clash = trace_write("user", "one", 0)
    await store.save_trace(run_id, "user", "one", order_index=0, trace_id=clash.trace_id)
    with pytest.raises(PersistenceError):
        await store.checkpoint_step(
            run_id,
            claim_epoch=0,
            loop_state={"v": 1, "phase": "dispatch"},
            pii_mapping=None,
            trace=trace_write("user", "two", 0),  # same position, another identity
        )
    assert await store.get_loop_state(run_id) is None, "the checkpoint must roll back with it"
    assert len(await store.get_traces(run_id)) == 1


@checkpointing
async def test_a_claim_carries_the_checkpointed_state(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    state = {"v": 1, "phase": "dispatch", "pending": [{"call_id": "a"}]}
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    await store.pause_run(
        run_id, status="waiting_client_tool", pause_data={"pause_id": "p"}, claim_epoch=0
    )
    claim = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert claim is not False
    assert claim.storage_format == 2
    assert claim.loop_state == state
    assert claim.claim_epoch == 1


@checkpointing
async def test_pii_mapping_is_stored_with_the_run(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run(pii_mapping={"<<EMAIL_1>>": "a@example.com"})
    assert await store.get_pii_mapping(run_id) == {"<<EMAIL_1>>": "a@example.com"}


@checkpointing
async def test_a_trace_link_is_validated_and_stored(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id = await dendrux_kit.new_run()
    await store.set_trace_link(run_id, claim_epoch=0, trace_link=TRACEPARENT)
    assert (await store.get_run(run_id)).trace_link == TRACEPARENT
    with pytest.raises(ValueError):
        await store.set_trace_link(run_id, claim_epoch=0, trace_link="not a traceparent")


@checkpointing
async def test_a_stale_epoch_cannot_checkpoint_or_link(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, claim = await dendrux_kit.claimed_run()
    stale = claim.claim_epoch - 1
    with pytest.raises(RunFencedError):
        await store.checkpoint_step(
            run_id,
            claim_epoch=stale,
            loop_state={"v": 1, "phase": "stale"},
            pii_mapping={"a": "b"},
            trace=trace_write("user", "stale", 0),
        )
    with pytest.raises(RunFencedError):
        await store.set_trace_link(run_id, claim_epoch=stale, trace_link=TRACEPARENT)
    assert (await store.get_loop_state(run_id) or {}).get("phase") != "stale"
    assert await store.get_pii_mapping(run_id) is None
    assert await store.get_traces(run_id) == []
    assert (await store.get_run(run_id)).trace_link is None


async def _paused_for_input(kit: Kit) -> tuple[str, dict[str, object], int]:
    store = kit.store
    run_id = await kit.new_run()
    descriptor = {
        "submission_id": ulid(),
        "kind": "input",
        "pause_id": "p",
        "call_ids": [],
        "reserved_order_index": 1,
    }
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"phase": "awaiting_input", "submission": descriptor},
        pii_mapping=None,
        trace=None,
    )
    await store.pause_run(
        run_id,
        status="waiting_human_input",
        pause_data={
            "schema_version": 2,
            "pause_id": "p",
            "submission": {**descriptor, "payload": "hello"},
        },
    )
    claim = await store.claim_paused_run(run_id, expected_status="waiting_human_input")
    assert claim is not False
    return run_id, descriptor, claim.claim_epoch


@checkpointing
async def test_a_submission_is_consumed_exactly_once(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, descriptor, epoch = await _paused_for_input(dendrux_kit)
    trace = trace_write("user", "hello", 1)
    patch = LoopStatePatch(set_phase="model_next")
    results = [
        await store.consume_submission(
            run_id,
            claim_epoch=epoch,
            submission_id=str(descriptor["submission_id"]),
            traces=[trace],
            patch=patch,
        )
        for _ in range(2)
    ]
    assert results == [True, False]
    assert await store.get_loop_state(run_id) == {"phase": "model_next", "submission": None}
    assert "submission" not in (await store.get_pause_state(run_id) or {})
    assert len(await store.get_traces(run_id)) == 1


@checkpointing
async def test_a_stale_epoch_cannot_consume_a_submission(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    run_id, descriptor, epoch = await _paused_for_input(dendrux_kit)
    with pytest.raises(RunFencedError):
        await store.consume_submission(
            run_id,
            claim_epoch=epoch - 1,
            submission_id=str(descriptor["submission_id"]),
            traces=[trace_write("user", "hello", 1)],
            patch=LoopStatePatch(set_phase="model_next"),
        )
    assert await store.get_traces(run_id) == []
    assert (await store.get_loop_state(run_id) or {}).get("submission") is not None
