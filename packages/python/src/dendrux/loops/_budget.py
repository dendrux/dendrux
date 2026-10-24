"""Budget accounting and enforcement shared by the ReAct and SingleCall loops.

Evaluation happens after every LLM response (events only) and at the
top-of-iteration checkpoint (events plus the ``on_exceed`` action). Acting
only at the checkpoint means the response that crossed a limit has had its
tools run and their results appended, so a pause never strands an assistant
tool call without results; the cost is an overshoot of at most one iteration.

Usage source:

- With a recording state store, usage is read live from ``llm_interactions``:
  the root-inclusive subtree rollup when ``include_children`` is set, the
  run's own rollup otherwise. Tokens are input + output, plus cache read and
  cache write tokens when ``Budget(count_cache_tokens=True)``; the cost limit
  uses ``known_cost_usd`` (priced calls only).
- Without one (or a store predating the rollup reads), usage is the loop's
  in-memory total; ``include_children`` cannot be honoured and is ignored
  with a one-time warning.

Token formula (``max_tokens``)::

    input + output                                  count_cache_tokens False / None (0.4.1)
    input + output + cache_read + cache_write       count_cache_tokens True

``input`` is fresh input only (providers report cached tokens separately), so
the default undercounts what a cache-heavy run really sends. Choosing
``count_cache_tokens`` explicitly is recommended in 0.5.0; ``True`` becomes the
default in 0.6, and a token budget without a choice emits a one-time
``DeprecationWarning``.

Unpriced calls (no provider cost, no ``PriceTable`` entry) under
``max_cost_usd``: with ``on_exceed`` ``"pause"`` or ``"cancel"`` they count as
the limit being reached (``budget.exceeded`` with ``reason="unpriced_call"``),
because the spend is unknown; with ``"warn"`` they only emit
``budget.unpriced``. Resuming a pause that an unpriced call caused does not need
a policy change: the unpriced calls counted at the pause are accepted, and only a
new one stops the run again. With ``include_children`` the count covers delegated
descendants too, so an unpriced child call stops the parent (child agents are not
checked by the parent's ``check_budget_config``). Misconfigured agents are
normally caught earlier by ``pricing.check_budget_config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux.loops._helpers import notify_governance, record_governance
from dendrux.types import GovernanceEventType

if TYPE_CHECKING:
    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import StateStore
    from dendrux.types import Budget, PauseState, UsageStats

logger = logging.getLogger(__name__)

TOKENS = "tokens"
COST_USD = "cost_usd"
# Fired-key threshold for "limit reached". warn_at fractions are in (0, 1).
EXCEEDED = 1.0
# ``failure_reason`` and ``run.cancelled`` reason of a budget cancel.
BUDGET_EXCEEDED = "budget_exceeded"
# Persisted cost precision (llm_interactions.cost_usd is Numeric(10, 6)).
_COST_DECIMALS = 6
# ``budget.exceeded`` reason when an unpriced call stands in for the cost limit.
UNPRICED_CALL = "unpriced_call"


@dataclass(frozen=True)
class BudgetBreach:
    """One limit that live usage has reached."""

    metric: str
    used: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "used": self.used, "max": self.max}


@dataclass(frozen=True)
class BudgetCarry:
    """Budget bookkeeping a pause hands to the loop that resumes it."""

    fired: tuple[tuple[str, float], ...] = ()
    unpriced_reported: bool = False
    # Own in-memory spend when the run total is unknown (mixed priced/unpriced).
    known_cost_usd: float | None = None
    unpriced_calls: int = 0
    # Metrics a budget pause hit: the resume gate checks exactly these.
    gate: tuple[str, ...] = ()

    @classmethod
    def from_pause(cls, pause: PauseState) -> BudgetCarry:
        gate = (
            tuple(str(b.get("metric")) for b in pause.budget_exceeded or ())
            if pause.pause_reason == "budget"
            else ()
        )
        return cls(
            fired=tuple(pause.budget_fired or ()),
            unpriced_reported=pause.budget_unpriced_reported,
            known_cost_usd=pause.budget_known_cost_usd,
            unpriced_calls=pause.budget_unpriced_calls,
            gate=gate,
        )


@dataclass(frozen=True)
class _Usage:
    tokens: int
    known_cost_usd: float
    unpriced_calls: int


class BudgetTracker:
    """Per-loop budget state: fired keys, in-memory spend and the usage source.

    ``enforce=False`` (SingleCall) degrades ``pause``/``cancel`` to warn.
    """

    def __init__(
        self,
        budget: Budget | None,
        *,
        run_id: str,
        usage: UsageStats,
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
        warnings: list[str],
        state_store: StateStore | None = None,
        carry: BudgetCarry | None = None,
        enforce: bool = True,
        fresh: bool = False,
    ) -> None:
        self.budget = budget
        self._count_cache = budget is not None and budget.count_cache_tokens is True
        self._run_id = run_id
        self._usage = usage
        self._recorder = recorder
        self._notifier = notifier
        self._warnings = warnings
        self._enforce = enforce
        self.fired: list[tuple[str, float]] = list(carry.fired) if carry else []
        self.unpriced_reported = carry.unpriced_reported if carry else False
        self._gate = carry.gate if carry else ()
        # Unpriced calls a budget pause already stopped for: resuming accepts
        # them (the spend is unknown and stays so); a new unpriced call stops again.
        self._ack_unpriced = carry.unpriced_calls if carry and COST_USD in carry.gate else 0
        # Highest unpriced-call count measured (rollup includes children).
        self._seen_unpriced = 0
        self._fresh = fresh
        self._first_checkpoint = True
        # Breaches from the latest post-response evaluation (None: none yet).
        self._last: list[BudgetBreach] | None = None
        # In-memory priced spend: the carried subtotal when a pause persisted
        # one; else seeded from the restored total when it is fully priced
        # (an unknown restored total counts as one unpriced call).
        if carry is not None and carry.known_cost_usd is not None:
            self._known_cost = carry.known_cost_usd
            self._unpriced = carry.unpriced_calls
        elif usage.cost_usd is not None and not usage.cost_unknown:
            self._known_cost = usage.cost_usd
            self._unpriced = 0
        else:
            self._known_cost = 0.0
            self._unpriced = 1 if usage.cost_unknown or usage.total_tokens else 0
        self._store_reader = self._pick_reader(budget, state_store, recorder)

    @property
    def action(self) -> str:
        """The effective ``on_exceed`` action (``"warn"`` when not enforced)."""
        if self.budget is None or not self._enforce:
            return "warn"
        return self.budget.on_exceed

    def _pick_reader(
        self,
        budget: Budget | None,
        state_store: StateStore | None,
        recorder: LoopRecorder | None,
    ) -> Any:
        # Live reads need calls to be recorded to the same store.
        if budget is None or state_store is None or recorder is None:
            return None
        name = "subtree_usage" if budget.include_children else "run_usage"
        reader = getattr(state_store, name, None)
        if reader is None and budget.include_children:
            logger.warning(
                "Budget include_children ignored for run %s: the state store has no "
                "subtree_usage(); counting the run's own usage only.",
                self._run_id,
            )
        return reader

    def add_response(self, usage: UsageStats) -> None:
        """Account one LLM response in the in-memory spend."""
        if usage.cost_usd is None:
            self._unpriced += 1
        else:
            self._known_cost = round(self._known_cost + usage.cost_usd, _COST_DECIMALS)

    def pause_fields(self) -> dict[str, Any]:
        """``PauseState`` fields that carry budget bookkeeping across a pause.

        The in-memory spend is persisted only when the run total cannot
        reconstruct it (some call unpriced).
        """
        fields: dict[str, Any] = {
            "budget_fired": list(self.fired) or None,
            "budget_unpriced_reported": self.unpriced_reported,
        }
        if self._unpriced:
            fields["budget_known_cost_usd"] = self._known_cost
        unpriced = max(self._unpriced, self._seen_unpriced)
        if unpriced:
            fields["budget_unpriced_calls"] = unpriced
        return fields

    async def _emit(self, event_type: str, iteration: int, data: dict[str, Any]) -> None:
        await record_governance(self._recorder, self._run_id, event_type, iteration, data)
        await notify_governance(
            self._notifier, self._run_id, event_type, iteration, data, warnings=self._warnings
        )

    async def _measure(self) -> _Usage:
        if self._store_reader is not None:
            try:
                rollup = await self._store_reader(self._run_id)
            except Exception:
                logger.warning(
                    "Live budget usage read failed for run %s; using in-memory usage",
                    self._run_id,
                    exc_info=True,
                )
            else:
                tokens = rollup.input_tokens + rollup.output_tokens
                if self._count_cache:
                    tokens += rollup.cache_read_tokens + rollup.cache_write_tokens
                self._seen_unpriced = max(self._seen_unpriced, rollup.unpriced_calls)
                return _Usage(
                    tokens=tokens,
                    known_cost_usd=rollup.known_cost_usd,
                    unpriced_calls=rollup.unpriced_calls,
                )
        u = self._usage
        if self._count_cache:
            # Built from the components: ``total_tokens`` is supplied separately
            # by providers and may be absent.
            tokens = (
                u.input_tokens
                + u.output_tokens
                + (u.cache_read_input_tokens or 0)
                + (u.cache_creation_input_tokens or 0)
            )
        else:
            tokens = u.total_tokens
        self._seen_unpriced = max(self._seen_unpriced, self._unpriced)
        return _Usage(
            tokens=tokens,
            known_cost_usd=self._known_cost,
            unpriced_calls=self._unpriced,
        )

    async def evaluate(self, iteration: int) -> list[BudgetBreach]:
        """Measure usage, emit newly crossed budget events, return reached limits."""
        budget = self.budget
        if budget is None:
            return []
        usage = await self._measure()
        breaches: list[BudgetBreach] = []
        limits: list[tuple[str, float | None, float]] = [
            (TOKENS, budget.max_tokens, usage.tokens),
            (COST_USD, budget.max_cost_usd, usage.known_cost_usd),
        ]
        for metric, limit, used in limits:
            if limit is None:
                continue
            if metric == COST_USD and usage.unpriced_calls and not self.unpriced_reported:
                self.unpriced_reported = True
                await self._emit(
                    GovernanceEventType.BUDGET_UNPRICED,
                    iteration,
                    {
                        "metric": COST_USD,
                        "used": used,
                        "max": limit,
                        "unpriced_calls": usage.unpriced_calls,
                        "known_cost_usd": usage.known_cost_usd,
                    },
                )
            fraction = used / limit
            for threshold in budget.warn_at:
                if fraction >= threshold and (metric, threshold) not in self.fired:
                    self.fired.append((metric, threshold))
                    await self._emit(
                        GovernanceEventType.BUDGET_THRESHOLD,
                        iteration,
                        {
                            "metric": metric,
                            "fraction": threshold,
                            "used": used,
                            "max": limit,
                            "reason": "threshold_crossed",
                        },
                    )
            reached = used >= limit
            # An unpriced call hides spend: when the run is told to stop on
            # exceeding the cost limit it must stop here too (fail closed).
            unpriced = (
                metric == COST_USD
                and usage.unpriced_calls > self._ack_unpriced
                and self.action != "warn"
            )
            if reached or unpriced:
                breaches.append(BudgetBreach(metric=metric, used=used, max=limit))
                if (metric, EXCEEDED) not in self.fired:
                    self.fired.append((metric, EXCEEDED))
                    await self._emit(
                        GovernanceEventType.BUDGET_EXCEEDED,
                        iteration,
                        {
                            "metric": metric,
                            "used": used,
                            "max": limit,
                            "reason": BUDGET_EXCEEDED if reached else UNPRICED_CALL,
                            "action": self.action,
                        },
                    )
        self._last = breaches
        return breaches

    async def checkpoint(self, iteration: int) -> tuple[str, list[BudgetBreach]]:
        """Top-of-iteration check: ``(action, breaches)``; act when breaches is non-empty.

        The first checkpoint after resuming a budget pause is the resume
        gate: only the persisted exceeded metrics are checked against live
        usage and this Agent's limits; any not strictly above usage re-pauses
        whatever ``on_exceed`` now says, otherwise nothing is enforced here.
        Warn budgets never act at a checkpoint. Enforcing budgets reuse the
        post-response evaluation unless descendants can have spent since.
        """
        first, self._first_checkpoint = self._first_checkpoint, False
        if first and self._gate:
            return "pause", await self._resume_gate()
        budget = self.budget
        action = self.action
        if budget is None or action == "warn":
            return action, []
        if self._last is None:
            if first and self._fresh:
                return action, []  # nothing has been spent yet
            return action, await self.evaluate(iteration)
        if budget.include_children and self._store_reader is not None:
            return action, await self.evaluate(iteration)
        return action, self._last

    async def _resume_gate(self) -> list[BudgetBreach]:
        limits = {
            TOKENS: self.budget.max_tokens if self.budget else None,
            COST_USD: self.budget.max_cost_usd if self.budget else None,
        }
        gated = [m for m in self._gate if limits.get(m) is not None]
        if not gated:
            return []
        usage = await self._measure()
        used = {TOKENS: usage.tokens, COST_USD: usage.known_cost_usd}
        breaches = []
        for metric in gated:
            limit = limits[metric]
            assert limit is not None
            unpriced = (
                metric == COST_USD
                and usage.unpriced_calls > self._ack_unpriced
                and self.action != "warn"
            )
            if unpriced or not limit > used[metric]:
                breaches.append(BudgetBreach(metric=metric, used=used[metric], max=limit))
        return breaches


def breach_payload(breaches: list[BudgetBreach]) -> list[dict[str, Any]]:
    """JSON-safe list of reached limits for pause state, events and meta."""
    return [b.to_dict() for b in breaches]
