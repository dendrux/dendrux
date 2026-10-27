"""Public execution/read paths share tenant, pricing and metadata policies."""

from dataclasses import asdict

import pytest

import dendrux
from dendrux import Agent
from dendrux.errors import BudgetConfigurationError, ReplayRunError, TenantScopeRequiredError
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.pricing import ModelPricing, PriceTable
from dendrux.types import Budget, LLMResponse
from tests.integration.test_tenant_run_control import _pause


async def test_module_run_requires_an_explicit_tenant_under_strict_mode(store):
    llm = MockLLM([LLMResponse(text="ok")])
    a = Agent(name="strict", prompt="p", provider=llm, state_store=store, strict_tenancy=True)
    try:
        with pytest.raises(TenantScopeRequiredError):
            await dendrux.run(a, provider=llm, user_input="go", state_store=store)
        assert await store.count_runs() == 0
        result = await dendrux.run(
            a, provider=llm, user_input="go", state_store=store, tenant_id=None
        )
        assert result.answer == "ok", "an explicitly untenanted run is still valid"
    finally:
        await a.close()


async def test_module_run_checks_the_actual_provider_pricing_before_io(store):
    priced = MockLLM([LLMResponse(text="priced")], model="priced")
    unpriced = MockLLM([LLMResponse(text="unpriced")], model="unpriced")
    a = Agent(
        name="budget",
        prompt="p",
        strict_tenancy=False,
        provider=priced,
        budget=Budget(max_cost_usd=1),
        pricing=PriceTable({"priced": ModelPricing(input=1, output=1)}),
    )
    try:
        with pytest.raises(BudgetConfigurationError):
            await dendrux.run(
                a, provider=unpriced, user_input="go", state_store=store, tenant_id="A"
            )
        assert await store.count_runs() == 0
    finally:
        await a.close()
        await unpriced.close()


async def test_foreign_and_missing_replay_sources_are_indistinguishable(store):
    await store.create_run("foreign", "a", tenant_id="A")
    await store.finalize_run("foreign", status="success", answer="PRIVATE_SENTINEL")
    a = Agent(name="a", prompt="p", provider=MockLLM([]), state_store=store, strict_tenancy=False)
    errors = []
    try:
        for source in ("foreign", "missing"):
            with pytest.raises(ReplayRunError) as caught:
                await a.run("go", runs=[source], tenant_id="B")
            errors.append(
                (type(caught.value), caught.value.reason, str(caught.value).replace(source, "ID"))
            )
        assert errors[0] == errors[1], "foreign and absent replay IDs must have the same failure"
    finally:
        await a.close()


@pytest.mark.parametrize("persisted", [False, True])
async def test_live_pause_does_not_return_private_execution_state(store, persisted):
    if persisted:
        paused = await _pause("client", store, "A")
        a = paused.agent
        # Reconstruct the fresh path with the same scripted setup via its run.
        from tests._helpers.scripted_provider import ScriptedLLM, tool_call

        a._provider = ScriptedLLM([tool_call("read_range", {"sheet": "S1"}, pid="c")])
    else:
        from tests._helpers.scripted_provider import ScriptedLLM, tool_call
        from tests.integration.test_tenant_run_control import read_range

        a = Agent(
            prompt="p",
            provider=ScriptedLLM([tool_call("read_range", {"sheet": "S1"}, pid="c")]),
            tools=[read_range],
            strict_tenancy=False,
        )
    try:
        result = await a.run("PRIVATE_INPUT_SENTINEL", tenant_id="A")
        assert "pause_state" not in result.meta, (
            "internal execution history must not be public metadata"
        )
        assert "pause_state" not in asdict(result)["meta"]
        assert result.pending is not None
    finally:
        await a.close()


async def test_fetched_mapping_is_opt_in_without_changing_answer(store):
    a = Agent(
        name="privacy",
        prompt="p",
        provider=MockLLM([LLMResponse(text="Hello <<EMAIL_ADDRESS_1>>")]),
        state_store=store,
        strict_tenancy=False,
        guardrails=[PII()],
    )
    try:
        live = await a.run("alice@example.com", tenant_id="A")
        fetched = await a.get_result(live.run_id, tenant_id="A")
        assert "pii_mapping" not in fetched.meta, (
            "fetched results must omit the real-value map by default"
        )
        explicit = await a.get_result(live.run_id, tenant_id="A", include_pii=True)
        assert explicit.meta["pii_mapping"] == live.meta["pii_mapping"]
        assert fetched.answer == explicit.answer == live.answer
    finally:
        await a.close()
