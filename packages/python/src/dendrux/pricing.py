"""Developer-owned model pricing.

Dendrux never ships vendor price lists: they change without notice, vary by
contract and tier, and a stale number presented as a dollar figure is worse
than no number at all. Instead the developer declares the rates they are
billed at, and Dendrux does the arithmetic on the normalized
:class:`~dendrux.types.UsageStats` every provider already reports.

Rates are USD per million tokens, the unit every vendor publishes. The formula
is vendor-agnostic because provider adapters normalize ``input_tokens`` to
*fresh* input (cached tokens are reported separately on both Anthropic and
OpenAI) and bill reasoning tokens inside ``output_tokens``::

    cost = input * input_rate
         + cache_read * cache_read_rate
         + cache_write * cache_write_rate
         + output * output_rate

A provider that reports its own cost (OpenRouter) always wins over the table.
A model the table does not cover is left unpriced (``cost_usd=None``) so a
missing entry never masquerades as a free call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dendrux.agent import Agent
    from dendrux.types import UsageStats

__all__ = ["CostSource", "ModelPricing", "PriceTable", "price_usage"]

CostSource = Literal["provider", "table", "mixed"]
"""Where a ``cost_usd`` came from.

``"provider"``: the vendor reported it on the response (OpenRouter).
``"table"``: computed from a developer-supplied :class:`PriceTable`.
``"mixed"``: a run total whose steps were priced by different sources.
"""

_PER_MILLION = 1_000_000.0


@dataclass(frozen=True)
class ModelPricing:
    """Billing rates for one model, in USD per million tokens.

    Args:
        input: Rate for fresh (uncached) input tokens.
        output: Rate for output tokens. Reasoning/thinking tokens are billed
            at this rate by every supported vendor.
        cache_read: Rate for input tokens served from the prompt cache.
            Defaults to ``input`` when omitted.
        cache_write: Rate for input tokens written to the prompt cache
            (Anthropic ``cache_creation_input_tokens``, OpenRouter
            ``cache_write_tokens``). Defaults to ``input`` when omitted.

    Raises:
        ValueError: If any rate is negative.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def __post_init__(self) -> None:
        for name in ("input", "output", "cache_read", "cache_write"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"ModelPricing.{name} must be >= 0, got {value!r}")

    def cost(self, usage: UsageStats) -> float:
        """Compute the USD cost of ``usage`` at these rates."""
        cache_read_rate = self.input if self.cache_read is None else self.cache_read
        cache_write_rate = self.input if self.cache_write is None else self.cache_write
        total = (
            usage.input_tokens * self.input
            + (usage.cache_read_input_tokens or 0) * cache_read_rate
            + (usage.cache_creation_input_tokens or 0) * cache_write_rate
            + usage.output_tokens * self.output
        )
        return total / _PER_MILLION


class PriceTable:
    """Maps model identifiers to :class:`ModelPricing`.

    Keys are exact model ids or ``fnmatch``-style globs (``"gpt-5*"``,
    ``"claude-haiku-4-5-*"``). Lookup prefers an exact match, then the
    longest matching glob, so a specific entry always beats a broad one.
    Matching is case-sensitive, like model ids themselves.

    Example::

        pricing = PriceTable({
            "claude-sonnet-4-6": ModelPricing(input=3.0, output=15.0,
                                              cache_read=0.30, cache_write=3.75),
            "gpt-5*": ModelPricing(input=1.25, output=10.0, cache_read=0.125),
        })
        agent = Agent(provider="anthropic:claude-sonnet-4-6", pricing=pricing)
    """

    def __init__(self, rates: Mapping[str, ModelPricing]) -> None:
        if not rates:
            raise ValueError("PriceTable requires at least one entry.")
        for key, value in rates.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"PriceTable keys must be non-empty model ids, got {key!r}")
            if not isinstance(value, ModelPricing):
                raise TypeError(
                    f"PriceTable values must be ModelPricing, got {type(value).__name__} "
                    f"for {key!r}"
                )
        self._exact: dict[str, ModelPricing] = {k: v for k, v in rates.items() if not _is_glob(k)}
        # Longest pattern first so the most specific glob wins on ties.
        self._globs: list[tuple[str, ModelPricing]] = sorted(
            ((k, v) for k, v in rates.items() if _is_glob(k)),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    def lookup(self, model: str | None) -> ModelPricing | None:
        """Return the rates for ``model``, or ``None`` if the table has no entry."""
        return self.lookup_exact(model) or self.lookup_glob(model)

    def lookup_exact(self, model: str | None) -> ModelPricing | None:
        """Rates for an entry spelled exactly ``model`` (no globs)."""
        return self._exact.get(model) if model else None

    def lookup_glob(self, model: str | None) -> ModelPricing | None:
        """Rates of the longest glob entry matching ``model``."""
        if not model:
            return None
        for pattern, pricing in self._globs:
            if fnmatchcase(model, pattern):
                return pricing
        return None

    def cost_for(self, model: str | None, usage: UsageStats) -> float | None:
        """Price ``usage`` for ``model``, or ``None`` if the model is not in the table."""
        pricing = self.lookup(model)
        return None if pricing is None else pricing.cost(usage)

    def __len__(self) -> int:
        return len(self._exact) + len(self._globs)

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and self.lookup(model) is not None

    def __repr__(self) -> str:
        return f"PriceTable({len(self)} entries)"


def price_usage(
    usage: UsageStats,
    *,
    model: str | None,
    pricing: PriceTable | None,
    fallback_models: tuple[str | None, ...] = (),
) -> UsageStats:
    """Return ``usage`` with ``cost_usd`` and ``cost_source`` resolved.

    Resolution order:

    1. A provider-reported cost is kept and tagged ``"provider"``. A call the
       provider flagged ``cost_unknown`` (a charge no table can price) stays
       unpriced.
    2. Otherwise, if ``pricing`` covers ``model`` (or, failing that, the first
       of ``fallback_models`` it covers, e.g. the requested model when the
       vendor returned a dated snapshot id), the cost is computed and tagged
       ``"table"``.
    3. Otherwise the usage is returned unpriced (``cost_usd=None``,
       ``cost_source=None``).

    Pure: never mutates the input.
    """
    if usage.cost_usd is not None:
        if usage.cost_source is not None:
            return usage
        return replace(usage, cost_source="provider")
    if usage.cost_unknown or pricing is None or not usage.usage_reported:
        return usage
    cost = pricing.cost_for(resolve_priced_model(pricing, model, *fallback_models), usage)
    if cost is None:
        return usage
    return replace(usage, cost_usd=cost, cost_source="table")


def resolve_priced_model(pricing: PriceTable | None, *models: str | None) -> str | None:
    """Pick which of ``models`` to price a call under.

    Callers pass ``(response.model, requested_model)``: vendors return dated
    snapshot ids (``gpt-5-2025-08-07``) for a requested alias (``gpt-5``), and
    a table written for the alias must still price the call. Order, first hit
    wins:

    1. an exact entry for ``response.model``;
    2. an exact entry for the requested model;
    3. a glob entry for ``response.model``;
    4. a glob entry for the requested model.

    So a specific entry for what was requested is never outranked by a
    catch-all glob that happens to match the vendor's longer id. With no hit,
    the first non-empty model is returned.
    """
    candidates = [m for m in models if m]
    if pricing is not None:
        for model in candidates:
            if pricing.lookup_exact(model) is not None:
                return model
        for model in candidates:
            if pricing.lookup_glob(model) is not None:
                return model
    return candidates[0] if candidates else None


def check_budget_config(agent: Agent, *, model: str | None = None) -> None:
    """Fail fast when a dollar budget cannot be priced (B4).

    A ``Budget(max_cost_usd=...)`` needs a pricing source for the model the
    run will request: a provider that reports cost (``reports_cost``) or a
    :class:`PriceTable` entry (exact id or glob, same lookup as pricing the
    calls). Otherwise the limit would silently never trip.

    Called from ``Agent.__init__`` (``model=None``: the provider's configured
    model) and at every ``run`` / ``stream`` / ``retry`` entry with the
    per-call ``model=`` override when one was passed.

    Raises:
        BudgetConfigurationError: ``reason="unpriced_model"``.
    """
    budget = agent.budget
    if budget is None:
        return
    if budget.max_cost_usd is None:
        return
    provider = agent.provider
    if provider is None:
        return
    if provider.reports_cost:
        return
    requested = model or provider.model
    if agent.pricing is not None and agent.pricing.lookup(requested) is not None:
        return
    from dendrux.errors import BudgetConfigurationError

    raise BudgetConfigurationError(model=requested or None, reason="unpriced_model")


def _is_glob(key: str) -> bool:
    return any(ch in key for ch in "*?[")
