"""Lossless budget checkpoints, independent of best-effort usage telemetry."""

import json
import random
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from dendrux.loops import _budget
from dendrux.types import Budget, UsageStats


def tracker(usage=None, **kwargs):
    return _budget.BudgetTracker(
        Budget(max_tokens=100_000, max_cost_usd=20, on_exceed="pause", count_cache_tokens=True),
        run_id="budget-checkpoint",
        usage=usage or UsageStats(),
        recorder=None,
        notifier=None,
        warnings=[],
        **kwargs,
    )


def state_type():
    result = getattr(_budget, "BudgetState", None)
    assert result is not None, "BudgetState must persist the complete enforcement state"
    return result


def test_snapshot_roundtrip_property():
    cls = state_type()
    rng = random.Random(42)
    for _ in range(100):
        state = cls(
            fired=(("tokens", rng.random()), ("cost_usd", 1.0)),
            unpriced_reported=rng.choice([True, False]),
            known_cost_usd=rng.random() * 100,
            unpriced_calls=rng.randrange(100),
            accepted_unpriced_baseline=rng.randrange(100),
            unpriced_warned=rng.choice([True, False]),
            gate=rng.choice([(), ("tokens",), ("cost_usd",)]),
            enforcement_floor={
                "tokens": rng.randrange(10000),
                "known_cost_usd": 7.5,
                "unpriced_calls": 8,
            },
            previous_local={
                "tokens": rng.randrange(10000),
                "known_cost_usd": 3.5,
                "unpriced_calls": 2,
            },
            price_source=rng.choice(["none", "table", "provider"]),
        )
        restored = tracker()
        restored.restore(cls(**json.loads(json.dumps(asdict(state)))))
        assert restored.snapshot() == state
        # The snapshot must own its nested containers.
        restored.snapshot().enforcement_floor["tokens"] = -1
        assert restored.snapshot() == state


async def test_accepted_unpriced_pause_never_refires_after_restore():
    state_type()
    usage = UsageStats(input_tokens=100, total_tokens=100, cost_unknown=True)
    accepted = tracker(
        usage,
        carry=_budget.BudgetCarry(known_cost_usd=0.5, unpriced_calls=3, gate=("cost_usd",)),
    )
    assert (await accepted.checkpoint(4))[1] == []
    restored = tracker(usage)
    restored.restore(accepted.snapshot())
    assert (await restored.checkpoint(4))[1] == []
    restored.add_response(UsageStats(cost_usd=None))
    assert [b.metric for b in await restored.evaluate(5)] == ["cost_usd"]


async def test_restore_preserves_floor_when_usage_read_fails():
    state_type()
    usage = UsageStats(input_tokens=100, total_tokens=100, cost_usd=1)

    async def read(_run_id):
        return SimpleNamespace(
            input_tokens=800,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            known_cost_usd=9.0,
            unpriced_calls=2,
        )

    original = tracker(usage)
    original._store_reader = read
    await original.evaluate(1)
    restored = tracker(usage)
    restored.restore(original.snapshot())
    usage.input_tokens += 30
    usage.total_tokens += 30
    restored.add_response(UsageStats(cost_usd=0.25))
    await restored.evaluate(2)
    state = restored.snapshot()
    assert state.enforcement_floor == {"tokens": 830, "known_cost_usd": 9.25, "unpriced_calls": 2}
    assert state.previous_local == {"tokens": 130, "known_cost_usd": 1.25, "unpriced_calls": 0}


async def test_restore_keeps_unknown_usage_fail_closed():
    state_type()
    usage = UsageStats(usage_reported=False, cost_unknown=True)
    original = tracker(usage)
    original.add_response(usage)
    restored = tracker(usage)
    restored.restore(original.snapshot())
    assert {b.metric for b in await restored.evaluate(1)} == {"tokens", "cost_usd"}


@pytest.mark.parametrize("source", [None, "table", "provider", "mixed"])
def test_snapshot_captures_price_source(source):
    state_type()
    original = tracker(UsageStats(cost_usd=1.5, cost_source=source))
    assert original.snapshot().price_source == (
        source if source in ("table", "provider") else "none"
    )


def test_nullable_known_cost_roundtrips_until_new_spend():
    state = replace(tracker().snapshot(), known_cost_usd=None)
    restored = tracker()
    restored.restore(state)
    assert restored.snapshot() == state
    restored.add_response(UsageStats(cost_usd=0.5))
    assert restored.snapshot().known_cost_usd == 0.5


def test_mixed_spend_keeps_attribution_in_usage_not_budget_source():
    usage = UsageStats(cost_usd=1.0, cost_source="table")
    original = tracker(usage)
    # The loop accumulates the response into shared UsageStats before add_response.
    usage.cost_source = "mixed"
    usage.cost_usd = 3.0
    original.add_response(UsageStats(cost_usd=2.0, cost_source="provider"))
    assert original.snapshot().price_source == "none"
    restored = tracker(usage)
    restored.restore(original.snapshot())
    restored.add_response(UsageStats(cost_usd=1.0, cost_source="table"))
    assert restored.snapshot().price_source == "none"
    assert usage.cost_source == "mixed"
