"""Vendor token fields are validated before arithmetic or integer coercion."""

from types import SimpleNamespace

import pytest

from dendrux.llm._usage import openai_wire_usage
from dendrux.llm.anthropic import AnthropicProvider


@pytest.mark.parametrize("value", [-1, 2**40, float("inf"), float("nan"), "9", True, 1.5])
@pytest.mark.parametrize("vendor", ["openai", "anthropic"])
def test_vendor_usage_rejects_invalid_raw_counts_before_normalization(value, vendor):
    if vendor == "openai":
        usage = openai_wire_usage(
            {"prompt_tokens": value, "completion_tokens": 1},
            input_field="prompt_tokens",
            output_field="completion_tokens",
            details_field="prompt_tokens_details",
            output_details_field="completion_tokens_details",
        )
    else:
        usage = AnthropicProvider._usage_from_response(
            SimpleNamespace(input_tokens=value, output_tokens=1)
        )
    assert usage.usage_reported is False
    assert usage.cost_unknown is True
    assert usage.total_tokens == 0


@pytest.mark.parametrize("cost", [-1, 1e12, 9999.9999999, float("nan"), float("inf"), "0.1", True])
def test_router_invalid_cost_is_not_silently_repriced(cost):
    from dendrux.llm.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(model="test", api_key="test")
    usage = provider._normalize_usage({"prompt_tokens": 1, "completion_tokens": 1, "cost": cost})
    assert usage.usage_reported is False
    assert usage.cost_unknown is True
    assert usage.cost_usd is None
