"""Atomic claim epochs and stale-writer CAS protection on both database engines."""

import asyncio

import pytest

from dendrux.errors import RunFencedError
from dendrux.runtime.state import claimed_store


async def test_claim_race_returns_one_epoch(store):
    await store.create_run("claims", "integrity")
    await store.pause_run("claims", status="paused", pause_data={"pause_id": "one"})
    claims = await asyncio.gather(
        *[
            store.claim_paused_run("claims", expected_status="paused", expected_pause_id="one")
            for _ in range(20)
        ]
    )
    assert sum(bool(c) for c in claims) == 1
    assert [c.claim_epoch for c in claims if c] == [1]
    assert (await store.get_run_control("claims")).claim_epoch == 1


async def test_stale_epoch_cannot_finalize_pause_or_release_new_claim(store):
    await store.create_run("epochs", "integrity")
    await store.pause_run("epochs", status="paused", pause_data={"pause_id": "one"})
    old = (await store.claim_paused_run("epochs", expected_status="paused")).claim_epoch
    assert type(old) is int
    assert await store.try_pause_run(
        "epochs", status="paused", pause_data={"pause_id": "two"}, claim_epoch=old
    )
    new = (await store.claim_paused_run("epochs", expected_status="paused")).claim_epoch
    assert new == old + 1
    writes = await asyncio.gather(
        store.finalize_run(
            "epochs", status="error", expected_current_status="running", claim_epoch=old
        ),
        store.finalize_with_event(
            "epochs",
            status="error",
            allowed_current_statuses=["running"],
            event_type="run.error",
            claim_epoch=old,
        ),
        store.pause_with_event("epochs", status="paused", pause_data={}, claim_epoch=old),
        store.release_run_claim("epochs", status="paused", claim_epoch=old),
    )
    assert writes == [False, None, None, False]
    with pytest.raises(RunFencedError):
        await claimed_store(store, "epochs", old).get_run_control("epochs")
    assert (await store.get_run("epochs")).status == "running"
    assert await store.get_run_events("epochs") == []
    assert (
        await store.finalize_with_event(
            "epochs",
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            claim_epoch=new,
        )
        == 0
    )
