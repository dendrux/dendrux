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
    await store.create_run("00000000000000000000000100", "integrity")
    await store.pause_run(
        "00000000000000000000000100", status="paused", pause_data={"pause_id": "one"}
    )
    old = (
        await store.claim_paused_run("00000000000000000000000100", expected_status="paused")
    ).claim_epoch
    assert type(old) is int
    assert await store.try_pause_run(
        "00000000000000000000000100",
        status="paused",
        pause_data={"pause_id": "two"},
        claim_epoch=old,
    )
    new = (
        await store.claim_paused_run("00000000000000000000000100", expected_status="paused")
    ).claim_epoch
    assert new == old + 1
    writes = await asyncio.gather(
        store.finalize_run(
            "00000000000000000000000100",
            status="error",
            expected_current_status="running",
            claim_epoch=old,
        ),
        store.finalize_with_event(
            "00000000000000000000000100",
            status="error",
            allowed_current_statuses=["running"],
            event_type="run.error",
            claim_epoch=old,
        ),
        store.pause_with_event(
            "00000000000000000000000100", status="paused", pause_data={}, claim_epoch=old
        ),
        store.release_run_claim("00000000000000000000000100", status="paused", claim_epoch=old),
    )
    assert writes == [False, None, None, False]
    with pytest.raises(RunFencedError):
        await claimed_store(store, "00000000000000000000000100", old).get_run_control(
            "00000000000000000000000100"
        )
    assert (await store.get_run("00000000000000000000000100")).status == "running"
    assert await store.get_run_events("00000000000000000000000100") == []
    assert (
        await store.finalize_with_event(
            "00000000000000000000000100",
            status="success",
            allowed_current_statuses=["running"],
            event_type="run.completed",
            claim_epoch=new,
        )
        == 0
    )
