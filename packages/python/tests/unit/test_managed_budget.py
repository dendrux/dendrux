"""Managed budget definition contracts; templates remain independently usable."""

import pytest

from dendrux import Agent
from dendrux.errors import BudgetConfigurationError
from dendrux.types import Budget


def child(**kwargs):
    return Agent(name="child", description="Worker", prompt="Work", **kwargs)


@pytest.mark.parametrize("nested", [False, True])
def test_registered_child_budget_refused(nested, caplog):
    template = child(budget=Budget(max_tokens=100))
    with pytest.raises(BudgetConfigurationError) as caught:
        branch = (
            Agent(name="branch", prompt="Branch", description="Branch", subagents=[template])
            if nested
            else template
        )
        Agent(prompt="Root", subagents=[branch])
    assert caught.value.reason == "child_budget"
    assert "child" in str(caught.value)
    assert any(getattr(r, "code", None) == "managed_budget_invalid" for r in caplog.records)
    assert template.budget.max_tokens == 100


def test_managed_root_requires_aggregate_budget():
    with pytest.raises(BudgetConfigurationError) as caught:
        Agent(
            prompt="Root",
            subagents=[child()],
            budget=Budget(max_tokens=100, include_children=False),
        )
    assert caught.value.reason == "root_scope"


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", -1),
        ("output_tokens", True),
        ("llm_calls", 1.5),
        ("known_cost_usd", float("inf")),
        ("known_cost_usd", -1),
        ("known_cost_usd", True),
    ],
)
def test_shared_usage_rejects_invalid_values(field, value):
    from dataclasses import replace

    from dendrux.runtime._managed_budget import valid_usage
    from dendrux.runtime.state import UsageRollup

    assert not valid_usage(replace(UsageRollup(), **{field: value}))


def test_shared_usage_accepts_known_unpriced_and_large_totals():
    from dendrux.runtime._managed_budget import valid_usage
    from dendrux.runtime.state import UsageRollup

    assert valid_usage(
        UsageRollup(input_tokens=10000000, known_cost_usd=900, unpriced_calls=10, llm_calls=20)
    )


async def test_failed_provider_recording_cannot_be_treated_as_zero_usage():
    from types import SimpleNamespace

    from dendrux.runtime._managed_budget import record_interaction

    recorder = SimpleNamespace(
        _run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        _loop_state={"managed": {}},
        _managed_policy=Budget(max_tokens=100),
    )

    async def failed():
        raise RuntimeError("SENTINEL-provider-write")

    with pytest.raises(RuntimeError):
        await record_interaction(recorder, failed)
    assert recorder._managed_usage_failed
