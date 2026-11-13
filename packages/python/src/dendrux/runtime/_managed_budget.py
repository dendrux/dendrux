"""Bind managed execution meters and admission pauses to one authoritative root budget."""

from __future__ import annotations

import logging
import math
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import SQLAlchemyError

from dendrux import _codes
from dendrux._lifecycle import PauseReason
from dendrux._meta_keys import GOVERNANCE_KEY
from dendrux.errors import BudgetConfigurationError, GovernanceMismatchError, PersistenceError
from dendrux.loops._budget import BudgetTracker, breach_payload
from dendrux.loops.react import MANAGED_HANDOFF_KEY
from dendrux.runtime import _managed_cancel, _subagents
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.submit import _governance_change
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from dendrux.agent import Agent

logger = logging.getLogger(__name__)


class ManagedBudgetConfigurationError(BudgetConfigurationError):
    """A managed definition cannot provide one aggregate root allowance."""

    code = "managed_budget_invalid"
    _init_fields = ("model", "reason")

    def __init__(self, *, reason: str, model: str | None = None) -> None:
        self.model, self.reason = None, reason
        ValueError.__init__(
            self,
            "Registered child budget is not allowed; configure the root budget."
            if reason == "child_budget"
            else "Managed root budget must include all descendants.",
        )
        _codes.log_code(logger, _codes.MANAGED_BUDGET_INVALID, run_id=None)


def validate_definitions(root: Agent, *, agent_type: type[Agent]) -> None:
    """Validate budget placement after the ordinary definition tree checks."""
    _subagents.check_subagent_tree(root, agent_type=agent_type)
    if root.budget is not None and not root.budget.include_children:
        raise ManagedBudgetConfigurationError(reason="root_scope")
    stack = list(root.subagents)
    while stack:
        child = stack.pop()
        if child.budget is not None:
            raise ManagedBudgetConfigurationError(reason="child_budget")
        stack.extend(child.subagents)


async def route_checkpoint(agent: Agent, store: Any, record: Any) -> None:
    """Check the root's budget policy compatibility before admitting a routed leaf."""
    await _managed_cancel.route_checkpoint(store, record)
    if agent._on_governance_change != "raise":
        return
    change = _governance_change(agent, {GOVERNANCE_KEY: (record.meta or {}).get(GOVERNANCE_KEY)})
    if change is not None and "budget" in change["changed"]:
        _codes.log_code(logger, _codes.GOVERNANCE_MISMATCH, run_id=record.id)
        raise GovernanceMismatchError(record.id, **change)


def inherit(child: Agent, parent: Agent) -> Agent:
    """Borrow the root policy on an execution copy, preserving the registered template."""
    execution = copy(child)
    cast("Any", execution)._managed_budget_policy = getattr(
        parent, "_managed_budget_policy", parent.budget
    )
    return execution


def configure_recorder(recorder: Any, agent: Agent) -> None:
    """Pass execution policy through the recorder without changing Agent budget semantics."""
    recorder._history_limits = agent._limits
    recorder._managed_policy = getattr(agent, "_managed_budget_policy", agent.budget)


async def restore_recorder(recorder: Any, agent: Agent, history: Any) -> None:
    """Rebind the controlling root policy before restoring a continuation's recorder."""
    configure_recorder(recorder, agent)
    await recorder.restore_history(limits=agent._limits, records=history)


def bind(recorder: Any, tracker: BudgetTracker) -> None:
    """Connect a managed tracker to the persisted root header's authoritative rollup."""
    header = recorder._loop_state.get("managed")
    if header is None:
        return
    policy = getattr(recorder, "_managed_policy", tracker.budget)
    tracker.budget = policy
    tracker._count_cache = policy is not None and policy.count_cache_tokens is True
    root_id = header["root_run_id"]
    recorder._managed_budget_tracker = tracker

    async def measure() -> Any:
        if recorder._loop_state.get("managed_usage_unknown", False):
            return None
        try:
            usage = await recorder._store.subtree_usage(
                root_id, max_nodes=policy.max_subtree_nodes if policy is not None else 10000
            )
            return usage if valid_usage(usage) else None
        except (PersistenceError, SQLAlchemyError, ValueError, TypeError, RuntimeError):
            return None

    tracker.bind_shared(measure)


def valid_usage(usage: Any) -> bool:
    """Reject incomplete, non-finite or negative shared accounting."""
    if getattr(usage, "truncated", False):
        return False
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "unpriced_calls",
        "llm_calls",
    ):
        value = getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    cost = getattr(usage, "known_cost_usd", None)
    return (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    )


async def record_interaction(recorder: Any, write: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Finish recording before measurement and remember any ambiguous failed write."""
    completed = False
    try:
        await retry_transient_db(write, label="save_llm_interaction", run_id=recorder._run_id)
        completed = True
    finally:
        managed = "managed" in recorder._loop_state
        if not completed and managed and getattr(recorder, "_managed_policy", None) is not None:
            recorder._managed_usage_failed = True
            recorder._loop_state["managed_usage_unknown"] = True


async def admission(recorder: Any, result: RunResult) -> RunResult | None:
    """Measure once before binding, retaining the pending managed call across a pause."""
    tracker = getattr(recorder, "_managed_budget_tracker", None)
    if tracker is None:
        handoff = result.meta[MANAGED_HANDOFF_KEY]
        tracker = BudgetTracker(
            handoff.parent.budget,
            run_id=result.run_id,
            usage=result.usage,
            recorder=recorder,
            notifier=None,
            warnings=[],
            state_store=recorder._store,
        )
        bind(recorder, tracker)
    action, breaches = await tracker.checkpoint(result.iteration_count + 1)
    if not breaches or action == "warn":
        return None
    meta = {k: v for k, v in result.meta.items() if k != MANAGED_HANDOFF_KEY}
    meta["budget_exceeded"] = breach_payload(breaches)
    if action == "cancel":
        return replace(result, status=RunStatus.CANCELLED, meta=meta)
    pause = replace(
        meta["pause_state"],
        pause_reason=PauseReason.BUDGET.value,
        schema_version=2,
        budget_exceeded=breach_payload(breaches),
        **tracker.pause_fields(),
    )
    meta["pause_state"] = pause
    return replace(result, status=RunStatus.PAUSED, meta=meta)


async def finish(
    descend: RunResult,
    result: RunResult,
    recorder: Any,
    persist: Callable[..., Awaitable[RunResult]],
) -> RunResult:
    """Persist an ordinary admission budget stop or return the prepared invocation."""
    stopped = await admission(recorder, result)
    if stopped is not None:
        return await persist(
            state_store=recorder._store,
            run_id=result.run_id,
            result=stopped,
            recorder=recorder,
            provider_kwargs=result.meta["pause_state"].provider_kwargs,
        )
    return descend
