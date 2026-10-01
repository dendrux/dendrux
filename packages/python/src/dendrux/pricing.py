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
        if not model:
            return None
        exact = self._exact.get(model)
        if exact is not None:
            return exact
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
) -> UsageStats:
    """Return ``usage`` with ``cost_usd`` and ``cost_source`` resolved.

    Resolution order:

    1. A provider-reported cost is kept and tagged ``"provider"``.
    2. Otherwise, if ``pricing`` covers ``model``, the cost is computed and
       tagged ``"table"``.
    3. Otherwise the usage is returned unpriced (``cost_usd=None``,
       ``cost_source=None``).

    Pure: never mutates the input.
    """
    if usage.cost_usd is not None:
        if usage.cost_source is not None:
            return usage
        return replace(usage, cost_source="provider")
    if pricing is None:
        return usage
    cost = pricing.cost_for(model, usage)
    if cost is None:
        return usage
    return replace(usage, cost_usd=cost, cost_source="table")


def _is_glob(key: str) -> bool:
    return any(ch in key for ch in "*?[")
