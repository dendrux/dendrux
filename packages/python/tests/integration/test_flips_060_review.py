"""Permanent regressions from the phase 3c review probes (contracts §12.51)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from dendrux import Agent, Budget, ModelPricing, PriceTable, tool
from dendrux._sentinel import _OMITTED
from dendrux.db.models import Blob
from dendrux.errors import PauseStatusMismatchError, TenantScopeRequiredError
from dendrux.llm.mock import MockLLM
from dendrux.runtime import runner
from dendrux.runtime._tenant_checks import resolve_tenant
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import InputValidationError, LLMResponse, RunStatus, ToolCall, ToolResult


@pytest.fixture
async def store(engine):
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, 1)
    return store_for_engine(engine)


@pytest.mark.parametrize("policy", [{}, {"strict_tenancy": True}])
@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
async def test_runner_store_argument_requires_scope(store, policy, entry):
    a = Agent(provider=MockLLM([LLMResponse(text="ok")]), prompt="p", **policy)
    create = AsyncMock(wraps=store.create_run)
    store.create_run = create
    with pytest.raises(TenantScopeRequiredError):
        if entry == "run":
            await runner.run(a, provider=a.provider, user_input="hi", state_store=store)
        elif entry == "stream":
            events = runner.run_stream(a, provider=a.provider, user_input="hi", state_store=store)
            async for _ in events:
                pass
        else:
            await runner.retry(
                "00000000000000000000000001", agent=a, provider=a.provider, state_store=store
            )
    create.assert_not_awaited()


@pytest.mark.parametrize("entry", ["run", "stream"])
async def test_runner_no_store_remains_silent(entry):
    a = Agent(provider=MockLLM([LLMResponse(text="ok")]), prompt="p")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        if entry == "run":
            result = await runner.run(a, provider=a.provider, user_input="hi")
        else:
            events = [e async for e in runner.run_stream(a, provider=a.provider, user_input="hi")]
            result = events[-1].run_result
    assert result.status == RunStatus.SUCCESS


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("scope", ["omitted", None, "tenant"])
async def test_store_child_under_no_store_parent_requires_chosen_outer_scope(
    store, scope, streaming
):
    child = Agent(
        name="Child", provider=MockLLM([LLMResponse(text="c")]), prompt="c", state_store=store
    )
    seen = {}

    @tool()
    async def delegate() -> str:
        try:
            result = await child.run("sub")
        except TenantScopeRequiredError as exc:
            seen["error"] = exc
            raise
        seen["rid"] = result.run_id
        return result.answer or ""

    parent = Agent(
        name="Parent",
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="delegate", params={}, id="d")]),
                LLMResponse(text="p"),
            ]
        ),
        prompt="p",
        tools=[delegate],
    )
    kwargs = {} if scope == "omitted" else {"tenant_id": scope}
    if streaming:
        events = [e async for e in parent.stream("go", **kwargs)]
        assert events[-1].run_result.status == RunStatus.SUCCESS
    else:
        assert (await parent.run("go", **kwargs)).status == RunStatus.SUCCESS
    if scope == "omitted":
        assert isinstance(seen.get("error"), TenantScopeRequiredError)
        assert "outer run" in str(seen["error"])
        assert "rid" not in seen
    else:
        assert "error" not in seen
        assert (await store.get_run(seen["rid"])).tenant_id == scope


@tool(target="client")
async def client() -> str:
    return ""


async def test_legacy_pause_without_pause_id_can_resume_under_strict_tenancy(store):
    a = Agent(
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="client", params={}, id="p")]),
                LLMResponse(text="done"),
            ]
        ),
        prompt="p",
        tools=[client],
        state_store=store,
    )
    paused = await a.run("go", tenant_id=None)
    raw = await store.get_pause_state(paused.run_id)
    raw.pop("pause_id", None)
    await store.pause_run(paused.run_id, status="waiting_client_tool", pause_data=raw)
    result = await a.submit_tool_results(
        paused.run_id,
        [ToolResult(name="client", call_id=paused.pending.tool_calls[0].id, payload="x")],
        tenant_id=None,
    )
    assert result.status == RunStatus.SUCCESS
    with pytest.raises(TenantScopeRequiredError):
        await a.run("go")


@pytest.mark.parametrize("tokens", [None, 500])
def test_only_token_budget_cache_choice_changes_governance(tokens):
    from dendrux.runtime.submit import GOVERNANCE_KEY, _governance_change, governance_snapshot

    pricing = PriceTable({"mock": ModelPricing(input=1.0, output=1.0)})
    old = Agent(
        provider=MockLLM([]),
        prompt="p",
        pricing=pricing,
        budget=Budget(max_tokens=tokens, max_cost_usd=1.0, count_cache_tokens=None),
    )
    new = Agent(
        provider=MockLLM([]),
        prompt="p",
        pricing=pricing,
        budget=Budget(max_tokens=tokens, max_cost_usd=1.0),
    )
    change = _governance_change(new, {GOVERNANCE_KEY: governance_snapshot(old)})
    if tokens is None:
        assert change is None
    else:
        assert change is not None and change["changed"] == ["budget"]


async def test_env_url_is_ignored_with_one_redacted_process_log(monkeypatch, caplog):
    import dendrux.agent as agent_module

    monkeypatch.setattr(agent_module, "_env_url_ignored", False, raising=False)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", "postgresql+asyncpg://u:SECRETPW@invalid/db")
    for streaming in [False, True, False]:
        a = Agent(provider=MockLLM([LLMResponse(text="ok")]), prompt="p")
        if streaming:
            events = [e async for e in a.stream("hi")]
            assert events[-1].run_result.status == RunStatus.SUCCESS
        else:
            assert (await a.run("hi")).status == RunStatus.SUCCESS
    records = [r for r in caplog.records if getattr(r, "code", None) == "database_url_env_ignored"]
    assert len(records) == 1
    assert "SECRETPW" not in caplog.text


def test_missing_trust_attribute_fails_closed():
    a = SimpleNamespace(_state_store=object())
    with pytest.raises(TenantScopeRequiredError):
        resolve_tenant(a, "run", _OMITTED, default=None)


@pytest.mark.parametrize("entry", ["input", "resume", "resume_stream"])
@pytest.mark.parametrize("token", [None, "stale"])
async def test_missing_pause_token_precedes_image_writes(store, monkeypatch, entry, token):
    from dendrux.media import Image, InMemoryBlobStore
    from tests.integration.test_image_lifecycle import PNG

    blobs = InMemoryBlobStore()
    a = Agent(
        provider=MockLLM(
            [LLMResponse(tool_calls=[ToolCall(name="client", params={}, id="p")])],
            image_support="yes",
        ),
        prompt="p",
        tools=[client],
        state_store=store,
        blob_store=blobs,
    )
    paused = await a.run("go", tenant_id=None)
    raw = await store.get_pause_state(paused.run_id)
    await store.pause_run(paused.run_id, status="waiting_human_input", pause_data=raw)
    before = (await store.get_run_control(paused.run_id)).claim_epoch
    with pytest.raises((InputValidationError, PauseStatusMismatchError)) as caught:
        if entry == "input":
            await a.submit_input(
                paused.run_id,
                ["text", Image.from_bytes(PNG)],
                tenant_id=None,
                expected_pause_id=token,
            )
        elif entry == "resume":
            await a.resume(
                paused.run_id,
                user_input=["text", Image.from_bytes(PNG)],
                tenant_id=None,
                expected_pause_id=token,
            )
        else:
            async for _ in a.resume_stream(
                paused.run_id,
                user_input=["text", Image.from_bytes(PNG)],
                tenant_id=None,
                expected_pause_id=token,
            ):
                pass
    if token is None:
        assert caught.value.code == "pause_id_required"
    else:
        assert caught.value.reason == "pause_id"
    assert not [c for c in blobs.calls if c[0] == "put"]
    async with store._session_factory() as session:
        assert (await session.execute(select(func.count()).select_from(Blob))).scalar_one() == 0
    assert (await store.get_run_control(paused.run_id)).claim_epoch == before


async def test_stream_resolver_store_also_requires_scope(store):
    a = Agent(provider=MockLLM([LLMResponse(text="ok")]), prompt="p")
    create = AsyncMock(wraps=store.create_run)
    store.create_run = create
    events = [
        e
        async for e in runner.run_stream(
            a,
            provider=a.provider,
            user_input="go",
            state_store_resolver=AsyncMock(return_value=store),
        )
    ]
    assert events[-1].run_result.status == RunStatus.ERROR
    assert "explicit tenant_id" in events[-1].run_result.error
    create.assert_not_awaited()


async def test_resumed_parent_with_explicit_scope_can_delegate_to_strict_child(store):
    child = Agent(
        name="Child", provider=MockLLM([LLMResponse(text="c")]), prompt="c", state_store=store
    )
    seen = {}

    @tool()
    async def delegate() -> str:
        try:
            result = await child.run("sub")
        except TenantScopeRequiredError as exc:
            seen["error"] = exc
            raise
        seen["rid"] = result.run_id
        return "c"

    parent = Agent(
        name="Parent",
        strict_tenancy=False,
        prompt="p",
        tools=[client, delegate],
        state_store=store,
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="client", params={}, id="p")]),
                LLMResponse(tool_calls=[ToolCall(name="delegate", params={}, id="d")]),
                LLMResponse(text="done"),
            ]
        ),
    )
    paused = await parent.run("go", tenant_id=None)
    await parent.resume(
        paused.run_id,
        tool_results=[
            ToolResult(name="client", call_id=paused.pending.tool_calls[0].id, payload="ok")
        ],
        tenant_id=None,
        expected_pause_id=paused.pause_id,
    )
    assert "error" not in seen
    assert "rid" in seen


async def test_legacy_approval_without_generation_can_drain(store):
    @tool()
    async def server() -> str:
        return "done"

    a = Agent(
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="server", params={}, id="s")]),
                LLMResponse(text="done"),
            ]
        ),
        prompt="p",
        tools=[server],
        require_approval=["server"],
        state_store=store,
    )
    paused = await a.run("go", tenant_id=None)
    raw = await store.get_pause_state(paused.run_id)
    raw.pop("pause_id", None)
    await store.pause_run(paused.run_id, status="waiting_approval", pause_data=raw)
    result = await a.submit_approval(paused.run_id, approved=True, tenant_id=None)
    assert result.status == RunStatus.SUCCESS
