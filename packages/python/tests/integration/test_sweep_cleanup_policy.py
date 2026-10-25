"""Maintenance validates its age policy and clears terminal execution state."""

from datetime import timedelta

import pytest

from dendrux import sweep


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("entry", ["public", "store"])
async def test_negative_sweep_age_is_rejected_before_state_change(store, waiting, entry):
    await store.create_run("fresh", "agent")
    with pytest.raises(ValueError, match="non-negative"):
        if entry == "public":
            await sweep(
                state_store=store,
                **{"abandoned_waiting" if waiting else "stale_running": timedelta(seconds=-1)},
            )
        else:
            reader = store.sweep_abandoned_runs if waiting else store.sweep_stale_runs
            await reader(timedelta(seconds=-1))
    assert (await store.get_run("fresh")).status == "running"


async def test_abandoned_sweep_discards_private_pause_history(store):
    await store.create_run("abandoned", "agent")
    await store.pause_run(
        "abandoned", status="waiting_client_tool", pause_data={"history": ["PII_SENTINEL"]}
    )
    result = await store.sweep_abandoned_runs(timedelta(0))
    assert len(result) == 1
    assert await store.get_pause_state("abandoned") is None, (
        "terminal sweeps must discard pause history"
    )
