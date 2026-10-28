"""Coordinator amendments to usage equality, pending patches and reclaim policy."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, RunNotReclaimableError
from dendrux.runtime import state
from dendrux.types import UsageStats, generate_ulid


async def test_usage_equality_is_quantised_and_includes_metadata_columns(store):
    run_id, uid = generate_ulid(), generate_ulid()
    await store.create_run(run_id, "usage")
    for cost in (0.12345671, 0.12345672):
        await store.save_usage(
            run_id,
            usage_id=uid,
            iteration_index=1,
            model="m",
            provider="p",
            duration_ms=3,
            usage=UsageStats(cost_usd=cost),
        )
    for change in ({"model": "other"}, {"provider": "other"}, {"duration_ms": 4}):
        with pytest.raises(PersistenceError) as caught:
            await store.save_usage(
                run_id,
                usage_id=uid,
                iteration_index=1,
                usage=UsageStats(cost_usd=0.1234567),
                **{**dict(model="m", provider="p", duration_ms=3), **change},
            )
        assert caught.value.code == "write_conflict"


async def test_call_phase_patch_preserves_other_pending_fields(store):
    assert "set_call_phase" in state.LoopStatePatch.__dataclass_fields__
    run_id = generate_ulid()
    await store.create_run(run_id, "patch")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={
            "pending": [
                {"call_id": "a", "phase": "gated", "order_index": 5},
                {"call_id": "b", "phase": "client", "order_index": 6},
            ]
        },
        pii_mapping=None,
        trace=None,
    )
    # Exercise the same locked SQL expression used by completions.
    async with store._session_factory() as session:
        await store._patch_fence(
            session, run_id, 0, state.LoopStatePatch(set_call_phase={"a": "approved"})
        )
        await session.commit()
    assert (await store.get_loop_state(run_id))["pending"] == [
        {"call_id": "a", "phase": "approved", "order_index": 5},
        {"call_id": "b", "phase": "client", "order_index": 6},
    ]


@pytest.mark.parametrize("policy", ["warn", "raise"])
async def test_reclaim_policy_is_checked_under_lock(store, engine, policy):
    run_id = generate_ulid()
    await store.create_run(
        run_id,
        "policy",
        strategy="NativeToolCalling",
        meta={"dendrux.governance": {"digest": "old"}},
    )
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(hours=1))
        )
    result = await store.mark_lease_expired(
        cutoff=datetime.now(UTC) - timedelta(minutes=1),
        batch_size=10,
        max_runs=10,
        max_returned=10,
        reclaim_limit=3,
        cursor=None,
    )
    assert result.reclaimable, "legacy strategy values are read as react"
    import inspect

    assert "governance_policy" in inspect.signature(store.reclaim_run).parameters
    kwargs = dict(
        tenant_id=None,
        agent_name="policy",
        expected_claim_epoch=result.reclaimable[0].claim_epoch,
        governance={"digest": "new"},
        governance_policy=policy,
        skill_pins={},
        loop_type="react",
    )
    if policy == "raise":
        with pytest.raises(RunNotReclaimableError) as caught:
            await store.reclaim_run(run_id, **kwargs)
        assert caught.value.reason == "governance_changed"
        assert (await store.get_run(run_id)).reclaim_attempts == 0
    else:
        assert await store.reclaim_run(run_id, **kwargs)
        assert (await store.get_run_events(run_id))[-1].data["governance_changed"] is True
