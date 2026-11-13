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
    from unittest.mock import AsyncMock

    from dendrux.runtime._managed_budget import record_interaction

    recorder = SimpleNamespace(
        _run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        _loop_state={"managed": {}},
        _managed_policy=Budget(max_tokens=100),
        _checkpoint=AsyncMock(),
    )

    async def failed():
        raise RuntimeError("SENTINEL-provider-write")

    with pytest.raises(RuntimeError):
        await record_interaction(recorder, failed)
    assert recorder._managed_usage_failed


@pytest.mark.parametrize(
    "raw",
    [
        True,
        {"origin_run_id": "SENTINEL-private", "text": "SENTINEL"},
        {"origin_run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"},
    ],
)
def test_fault_provenance_contains_only_safe_code_and_ids(raw):
    from dendrux.runtime._managed_budget import usage_fault

    marker = usage_fault(raw, "01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert marker == {
        "code": "managed_usage_unknown",
        "origin_run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    }


async def test_resume_ack_is_durable_but_does_not_accept_a_new_fault_twice():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.runtime._managed_budget import accept_paused_fault, hold_fault

    marker = {"code": "managed_usage_unknown", "origin_run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
    recorder = SimpleNamespace(
        _run_id=marker["origin_run_id"],
        _loop_state={"managed_usage_unknown": marker},
        _resume_pause_data={"pause_reason": "budget", "pause_message": "managed_usage_unknown"},
        _managed_budget_resuming=True,
        _checkpoint=AsyncMock(),
    )
    tracker = tracker_for(Budget(max_cost_usd=10))
    tracker._invalid_usage = True
    accept_paused_fault(recorder, tracker)
    assert not tracker._invalid_usage
    assert recorder._loop_state == {"managed_usage_accepted": marker}
    tracker._invalid_usage = True
    await hold_fault(recorder)
    accept_paused_fault(recorder, tracker)
    assert tracker._invalid_usage
    assert recorder._loop_state == {"managed_usage_unknown": marker}
    recorder._checkpoint.assert_awaited_once()


def test_reclaim_is_not_an_acknowledgement():
    from types import SimpleNamespace

    from dendrux.runtime._managed_budget import accept_paused_fault

    marker = {"code": "managed_usage_unknown", "origin_run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
    recorder = SimpleNamespace(
        _run_id=marker["origin_run_id"],
        _loop_state={"managed_usage_unknown": marker},
        _resume_pause_data={"pause_reason": "budget", "pause_message": "managed_usage_unknown"},
        _managed_budget_resuming=False,
    )
    tracker = tracker_for(Budget(max_cost_usd=10))
    tracker._invalid_usage = True
    accept_paused_fault(recorder, tracker)
    assert tracker._invalid_usage and recorder._loop_state["managed_usage_unknown"] == marker


@pytest.mark.parametrize("failure_kind", ["write", "fenced", "cancelled"])
@pytest.mark.parametrize("secondary", [RuntimeError, OSError])
async def test_record_failure_preserves_original_and_skips_checkpoint_on_cancellation(
    failure_kind, secondary
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.errors import PersistenceError, RunFencedError
    from dendrux.runtime._managed_budget import record_interaction

    failure = {
        "write": PersistenceError(),
        "fenced": RunFencedError("01ARZ3NDEKTSV4RRFFQ69G5FAV"),
        "cancelled": asyncio.CancelledError(),
    }[failure_kind]
    recorder = SimpleNamespace(
        _run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        _loop_state={"managed": {}},
        _managed_policy=Budget(max_tokens=100),
        _checkpoint=AsyncMock(side_effect=secondary("SENTINEL-secondary-checkpoint")),
    )

    async def failed():
        raise failure

    with pytest.raises(type(failure)) as caught:
        await record_interaction(recorder, failed)
    assert caught.value is failure
    if failure_kind != "write":
        recorder._checkpoint.assert_not_awaited()
    else:
        recorder._checkpoint.assert_awaited_once()
        assert recorder._managed_usage_failed


@pytest.mark.parametrize("phase", ["write", "checkpoint"])
async def test_task_cancellation_propagates_without_starting_another_write(phase):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.errors import PersistenceError
    from dendrux.runtime._managed_budget import record_interaction

    started = asyncio.Event()

    async def wait_for_cancellation():
        started.set()
        await asyncio.Event().wait()

    recorder = SimpleNamespace(
        _run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        _loop_state={"managed": {}},
        _managed_policy=Budget(max_tokens=100),
        _checkpoint=AsyncMock(
            side_effect=wait_for_cancellation
            if phase == "checkpoint"
            else RuntimeError("SENTINEL-cancelled-checkpoint")
        ),
    )

    async def write():
        if phase == "checkpoint":
            raise PersistenceError()
        await wait_for_cancellation()

    task = asyncio.create_task(record_interaction(recorder, write))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder._checkpoint.await_count == (phase == "checkpoint")


async def test_warn_fault_remains_enforced_after_binding_a_new_policy():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.loops._budget import BudgetTracker
    from dendrux.runtime._managed_budget import bind
    from dendrux.types import UsageStats

    run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    marker = {"code": "managed_usage_unknown", "origin_run_id": run_id}
    recorder = SimpleNamespace(
        _run_id=run_id,
        _loop_state={"managed": {"root_run_id": run_id}, "managed_usage_unknown": marker},
        _managed_policy=Budget(max_tokens=100, on_exceed="warn"),
        _checkpoint=AsyncMock(),
        _store=SimpleNamespace(subtree_usage=AsyncMock(side_effect=RuntimeError)),
    )
    for action in ("warn", "pause"):
        recorder._managed_policy = Budget(max_tokens=100, on_exceed=action)
        tracker = BudgetTracker(
            recorder._managed_policy,
            run_id=run_id,
            usage=UsageStats(),
            recorder=None,
            notifier=None,
            warnings=[],
        )
        bind(recorder, tracker)
        actual, breaches = await tracker.checkpoint(1)
        assert actual == action
        assert bool(breaches) == (action == "pause")
        assert recorder._loop_state["managed_usage_unknown"] == marker


async def test_fault_is_logged_even_when_its_checkpoint_fails(caplog):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from dendrux.errors import PersistenceError
    from dendrux.runtime._managed_budget import hold_fault

    run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    recorder = SimpleNamespace(
        _run_id=run_id, _loop_state={}, _checkpoint=AsyncMock(side_effect=PersistenceError())
    )
    with pytest.raises(PersistenceError):
        await hold_fault(recorder)
    records = [r for r in caplog.records if getattr(r, "code", None) == "managed_usage_unknown"]
    assert len(records) == 1 and records[0].run_id == run_id


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None, [3]])
def test_malformed_persisted_unpriced_baseline_falls_back_to_zero(value):
    from dendrux.runtime._managed_budget import accepted_count

    assert accepted_count(value) == 0
    assert accepted_count(7) == 7


def tracker_for(budget):
    from dendrux.loops._budget import BudgetTracker
    from dendrux.types import UsageStats

    return BudgetTracker(
        budget,
        run_id="run",
        usage=UsageStats(),
        recorder=None,
        notifier=None,
        warnings=[],
    )


@pytest.mark.parametrize("tracked,saved,expected", [(2, 4, 4), (5, 4, 5), (0, None, 0)])
def test_baseline_is_the_highest_of_tracker_and_durable_checkpoint(tracked, saved, expected):
    from types import SimpleNamespace

    from dendrux.runtime._managed_budget import UNPRICED_ACCEPTED, carry_unpriced_baseline

    tracker = tracker_for(Budget(max_cost_usd=10))
    tracker.accept_unpriced(tracked)
    recorder = SimpleNamespace(_loop_state={UNPRICED_ACCEPTED: saved})
    carry_unpriced_baseline(recorder, tracker)
    assert tracker.unpriced_baseline == expected
    assert recorder._loop_state[UNPRICED_ACCEPTED] == expected
    tracker.accept_unpriced(0)  # never lowers
    assert tracker.unpriced_baseline == expected


@pytest.mark.parametrize(
    "parent,saved,expected", [(5, 2, 5), (2, 5, 5), (True, -1, 0), ("5", 3, 3), (4, None, 4)]
)
def test_child_seed_copies_checkpoint_and_max_merges_valid_counts(parent, saved, expected):
    from dendrux.runtime._managed_budget import UNPRICED_ACCEPTED, child_state

    state = {"v": 1, "loop": "react", UNPRICED_ACCEPTED: saved}
    seeded = child_state(state, parent)
    assert seeded[UNPRICED_ACCEPTED] == expected
    assert seeded is not state and state[UNPRICED_ACCEPTED] == saved


@pytest.mark.parametrize("state", [None, {}])
def test_child_seed_preserves_default_loop_header(state):
    from dendrux.runtime._managed_budget import UNPRICED_ACCEPTED, child_state

    assert child_state(state, 3) == {"v": 1, "loop": "react", UNPRICED_ACCEPTED: 3}


@pytest.mark.parametrize("action,reason", [("pause", "unpriced_call"), ("warn", None)])
def test_unpriced_reason_applies_to_a_shared_reader_like_an_ordinary_run(action, reason):
    from dendrux.loops._budget import _Usage

    tracker = tracker_for(Budget(max_cost_usd=10, on_exceed=action))

    async def reader():
        return None

    tracker.bind_shared(reader)
    usage = _Usage(0, 0.0, 2)
    assert tracker._unknown_reason("cost_usd", usage) == reason
    tracker.accept_unpriced(2)
    assert tracker._unknown_reason("cost_usd", usage) is None
