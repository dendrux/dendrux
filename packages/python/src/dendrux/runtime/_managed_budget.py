"""Bind managed execution meters and admission pauses to one authoritative root budget."""

from __future__ import annotations

import logging
import math
import re
import sys
from asyncio import CancelledError
from copy import copy
from dataclasses import dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import SQLAlchemyError

from dendrux import _codes
from dendrux._lifecycle import PauseReason
from dendrux._meta_keys import ERROR_INFO_META_KEY, GOVERNANCE_KEY
from dendrux.errors import (
    BudgetConfigurationError,
    GovernanceMismatchError,
    ManagedStateError,
    PersistenceError,
    RunFencedError,
)
from dendrux.loops._budget import BudgetTracker, breach_payload
from dendrux.loops.react import MANAGED_HANDOFF_KEY
from dendrux.runtime import _managed_cancel, _subagents
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import LoopStatePatch
from dendrux.runtime.submit import _governance_change
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from dendrux.agent import Agent

logger = logging.getLogger(__name__)
UNKNOWN = "managed_usage_unknown"
ACCEPTED = "managed_usage_accepted"
_RUN_ID = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")


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


async def restore_recorder(
    recorder: Any, agent: Agent, history: Any, recovering: bool = False
) -> None:
    """Rebind the controlling root policy before restoring a continuation's recorder."""
    configure_recorder(recorder, agent)
    recorder._managed_budget_resuming = not recovering
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

    accept_paused_fault(recorder, tracker)

    async def measure() -> Any:
        return await measure_shared(recorder, tracker, root_id, policy)

    tracker.bind_shared(measure)


def usage_fault(value: Any, run_id: str) -> dict[str, str]:
    """Normalize persisted provenance without copying untrusted checkpoint text."""
    origin = value.get("origin_run_id") if isinstance(value, dict) else None
    if not isinstance(origin, str) or _RUN_ID.fullmatch(origin) is None:
        origin = run_id
    return {"code": UNKNOWN, "origin_run_id": origin}


async def hold_fault(recorder: Any, fault: Any = None) -> None:
    """Fence the first observed fault before a terminal outcome can become adoptable."""
    state = recorder._loop_state
    marker = usage_fault(fault or state.get(UNKNOWN), recorder._run_id)
    _codes.log_code(logger, _codes.MANAGED_USAGE_UNKNOWN, run_id=recorder._run_id)
    if state.get(UNKNOWN) == marker:
        return
    state[UNKNOWN] = marker
    state.pop(ACCEPTED, None)
    await recorder._checkpoint()


async def measure_shared(recorder: Any, tracker: BudgetTracker, root_id: str, policy: Any) -> Any:
    """Read once, keeping every observed invalid/shared fault sticky until explicit resume."""
    if tracker._invalid_usage or recorder._loop_state.get(UNKNOWN):
        await hold_fault(recorder)
        return None
    try:
        usage = await recorder._store.subtree_usage(
            root_id, max_nodes=policy.max_subtree_nodes if policy is not None else 10000
        )
    except (PersistenceError, SQLAlchemyError, ValueError, TypeError, RuntimeError):
        usage = None
    if not valid_usage(usage):
        await hold_fault(recorder)
        return None
    return usage


def accept_paused_fault(recorder: Any, tracker: BudgetTracker) -> None:
    """An explicit budget resume acknowledges that pause's fault; new faults still stop."""
    state = recorder._loop_state
    pause = getattr(recorder, "_resume_pause_data", None) or {}
    resuming = getattr(recorder, "_managed_budget_resuming", False)
    if not getattr(recorder, "_managed_budget_ack_checked", False):
        recorder._managed_budget_ack_checked = True
        if (
            resuming
            and pause.get("pause_reason") == PauseReason.BUDGET.value
            and pause.get("pause_message") == UNKNOWN
            and state.get(UNKNOWN)
        ):
            state[ACCEPTED] = usage_fault(state.pop(UNKNOWN), recorder._run_id)
    if state.get(ACCEPTED) and not state.get(UNKNOWN):
        tracker._invalid_usage = False


@dataclass(frozen=True)
class _AdoptionClaim:
    claim_epoch: int
    loop_state: dict[str, Any]


async def completed_fault(store: Any, run_id: str, child_id: str, report: Any) -> Any:
    """Recover provenance even when its checkpoint failed but the error outcome committed."""
    state = await store.get_loop_state(child_id)
    if state is None:
        raise ManagedStateError(reason="child_missing", run_id=run_id)
    if state.get(UNKNOWN) or report.success:
        return state.get(UNKNOWN)
    child = await store.get_run(child_id)
    if child is None:
        raise ManagedStateError(reason="child_missing", run_id=run_id)
    info = (child.meta or {}).get(ERROR_INFO_META_KEY) or {}
    return usage_fault(True, child_id) if info.get("category") == "persistence_error" else None


def propagate_fault(
    complete: Callable[..., Awaitable[dict[str, Any]]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Read child provenance once and carry it into live or recovered atomic adoption."""

    @wraps(complete)
    async def adopt(store: Any, run_id: str, **fields: Any) -> dict[str, Any]:
        child_id = fields["child_run_id"]
        fault = await completed_fault(store, run_id, child_id, fields["report"])
        claim = fields["claim"]
        state = dict(claim.loop_state or {})
        if fault:
            state[UNKNOWN] = usage_fault(fault, child_id)
            state.pop(ACCEPTED, None)
        fields["claim"] = _AdoptionClaim(claim.claim_epoch, state)
        return await complete(store, run_id, **fields)

    return adopt


def adoption_patch(state: dict[str, Any], call_id: str) -> LoopStatePatch:
    """Preserve provenance in the same patch that consumes the managed result."""
    return LoopStatePatch(
        remove_pending=(call_id,), set_phase="model_next", set_fields=adoption_fields(state)
    )


def adoption_fields(state: dict[str, Any]) -> dict[str, Any]:
    """Preserve fault provenance when clearing the consumed managed binding."""
    fields: dict[str, Any] = {"managed_wait": None, "submission": None, "managed_admitted": None}
    if state.get(UNKNOWN):
        fields.update({UNKNOWN: state[UNKNOWN], ACCEPTED: None})
    return fields


def adoption_event(report: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Include only fault IDs/code alongside the existing outcome code."""
    fields = {"code": report.meta["code"]} if "code" in report.meta else {}
    if state.get(UNKNOWN):
        fields[UNKNOWN] = state[UNKNOWN]
    return fields


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
        failure = sys.exception()
        managed = "managed" in recorder._loop_state
        if (
            not completed
            and not isinstance(failure, (CancelledError, RunFencedError))
            and managed
            and getattr(recorder, "_managed_policy", None) is not None
        ):
            recorder._managed_usage_failed = True
            try:
                await hold_fault(recorder)
            except CancelledError:
                failure = None
                raise
            finally:
                # A secondary checkpoint error must not replace the failed evidence write.
                if failure is not None:
                    raise failure


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
