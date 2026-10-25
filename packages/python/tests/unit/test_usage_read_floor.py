"""Previously observed spend remains an enforcement floor during read failures."""

import pytest

from dendrux.loops._budget import BudgetTracker
from dendrux.loops.base import BaseRecorder
from dendrux.runtime.state import UsageRollup
from dendrux.types import Budget, UsageStats


@pytest.mark.parametrize("grow", [False, True])
@pytest.mark.parametrize("metric", ["tokens", "cost"])
async def test_observed_descendant_spend_survives_failed_live_reads(grow, metric):
    class Store:
        failed = False

        async def subtree_usage(self, run_id):
            if self.failed:
                raise ConnectionError("DB_SENTINEL")
            return UsageRollup(input_tokens=200, known_cost_usd=2)

        async def run_usage(self, run_id):
            return UsageRollup(input_tokens=1, known_cost_usd=0.01)

    store = Store()
    usage = UsageStats(input_tokens=1, total_tokens=1, cost_usd=0.01)
    cap = (205 if grow else 100) if metric == "tokens" else (2.05 if grow else 1)
    tracker = BudgetTracker(
        Budget(
            **{f"max_{metric if metric == 'tokens' else 'cost_usd'}": cap},
            on_exceed="cancel",
            count_cache_tokens=True,
        ),
        run_id="parent",
        usage=usage,
        recorder=BaseRecorder(),
        notifier=None,
        warnings=[],
        state_store=store,
    )
    first = await tracker.evaluate(1)
    assert bool(first) is not grow
    if grow:
        usage.input_tokens += 10
        usage.total_tokens += 10
        tracker.add_response(UsageStats(input_tokens=10, total_tokens=10, cost_usd=0.1))
    store.failed = True
    action, breaches = await tracker.checkpoint(2)
    assert action == "cancel"
    assert breaches, "observed descendant spend must remain enforced when a live read fails"
    assert breaches[0].used == pytest.approx(
        (210 if grow else 200) if metric == "tokens" else (2.1 if grow else 2)
    )
