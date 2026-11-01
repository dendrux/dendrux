"""Release defaults and explicit compatibility choices for 0.6."""

from __future__ import annotations

import warnings
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, Budget, Limits
from dendrux._sentinel import _OMITTED
from dendrux.errors import TenantScopeRequiredError
from dendrux.llm.mock import MockLLM
from dendrux.loops._budget import BudgetTracker
from dendrux.runtime._tenant_checks import resolve_tenant
from dendrux.runtime.submit import GOVERNANCE_KEY, require_pause_id, submission_validator
from dendrux.runtime.sweep import sweep
from dendrux.types import InputValidationError, UsageStats


def agent(**kwargs):
    return Agent(provider=MockLLM([]), prompt="flip", **kwargs)


@pytest.mark.parametrize(
    "entry",
    [
        "run",
        "stream",
        "retry",
        "resume",
        "resume_stream",
        "submit_input",
        "submit_tool_results",
        "submit_resume",
        "submit_approval",
        "cancel_run",
        "submit_note",
        "request_pause",
    ],
)
@pytest.mark.parametrize("persistence", ["store", "url"])
def test_tenancy_default_requires_scope(entry, persistence):
    with pytest.raises(TenantScopeRequiredError):
        resolve_tenant(
            agent(
                **(
                    {"state_store": AsyncMock()}
                    if persistence == "store"
                    else {"database_url": "sqlite+aiosqlite://"}
                )
            ),
            entry,
            _OMITTED,
            default=None,
        )


def test_explicit_old_tenancy_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            resolve_tenant(
                agent(strict_tenancy=False, state_store=AsyncMock()), "run", _OMITTED, default=None
            )
            is None
        )


@pytest.mark.parametrize("pause_id", ["generation", None])
def test_pause_token_is_required_only_when_stored(pause_id):
    if pause_id is None:
        require_pause_id(agent(state_store=AsyncMock()), None, pause_id=None)
    else:
        with pytest.raises(InputValidationError) as caught:
            require_pause_id(agent(state_store=AsyncMock()), None, pause_id=pause_id)
        assert caught.value.field == "expected_pause_id"
        assert caught.value.code == "pause_id_required"


def test_old_nonstrict_submission_and_explicit_token_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        require_pause_id(agent(strict_tenancy=False), None, pause_id="generation")
        require_pause_id(agent(), "generation", pause_id="generation")


def test_governance_default_raises_and_old_warn_value_works():
    from dendrux.errors import GovernanceMismatchError
    from dendrux.runtime.submit import governance_snapshot

    original = agent(limits=Limits(max_tool_result_bytes=100))
    raw = {GOVERNANCE_KEY: governance_snapshot(original)}
    with pytest.raises(GovernanceMismatchError):
        submission_validator("00000000000000000000000001", raw, agent=agent())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        submission_validator(
            "00000000000000000000000001", raw, agent=agent(on_governance_change="warn")
        )


@pytest.mark.parametrize(("kwargs", "tokens"), [({}, 650), ({"count_cache_tokens": False}, 150)])
async def test_budget_default_counts_cache_and_old_false_works(kwargs, tokens):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        budget = Budget(max_tokens=500, **kwargs)
        assert budget.count_cache_tokens is kwargs.get("count_cache_tokens", True)
        agent(budget=budget)
        tracker = BudgetTracker(
            budget,
            run_id="run",
            usage=UsageStats(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=400,
                cache_creation_input_tokens=100,
            ),
            recorder=None,
            notifier=None,
            warnings=[],
        )
        assert (await tracker._measure()).tokens == tokens


async def test_implicit_database_url_is_ignored_without_engine(monkeypatch):
    monkeypatch.setenv("DENDRUX_DATABASE_URL", "sqlite+aiosqlite:///SECRET")
    implicit_engine = AsyncMock(side_effect=AssertionError("implicit engine requested"))
    monkeypatch.setattr("dendrux.db.session.get_engine", implicit_engine)
    assert await agent()._resolve_state_store() is None
    implicit_engine.assert_not_awaited()


async def test_explicit_store_overrides_env_without_warning(monkeypatch):
    monkeypatch.setenv("DENDRUX_DATABASE_URL", "unused")
    explicit = AsyncMock()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert await agent(state_store=explicit)._resolve_state_store() is explicit


@pytest.mark.parametrize("policy", [None, "error"])
@pytest.mark.parametrize("stale", [[], ["old-run"]])
async def test_sweep_keeps_07_warning_only_for_omitted_policy_with_stale_rows(policy, stale):
    store = AsyncMock()
    store.sweep_stale_runs.return_value = stale
    # A legacy store keeps this test on the public fallback path.
    store.__class__ = object
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await sweep(state_store=store, stale_running=timedelta(seconds=1), on_stale=policy)
    assert result.stale_running == stale
    assert len([w for w in caught if w.category is FutureWarning]) == int(
        policy is None and bool(stale)
    )


async def test_explicit_database_url_overrides_env_silently(monkeypatch):
    monkeypatch.setenv("DENDRUX_DATABASE_URL", "unused")
    implicit = AsyncMock(side_effect=AssertionError("implicit engine requested"))
    monkeypatch.setattr("dendrux.db.session.get_engine", implicit)
    explicit = agent(database_url="sqlite+aiosqlite://")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert await explicit._resolve_state_store() is not None
        implicit.assert_not_awaited()
    finally:
        await explicit.close()


@pytest.mark.parametrize("entry", ["tool", "input", "paused", "resume", "resume_stream"])
async def test_public_strict_submissions_refuse_missing_token_before_claim(entry):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux import tool
    from dendrux.db.models import Base
    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult

    @tool(target="client")
    async def client() -> str:
        return ""

    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        store = store_for_engine(engine)
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
        payload = [
            ToolResult(name="client", call_id=paused.pending.tool_calls[0].id, payload="rows")
        ]
        status = (
            "waiting_human_input"
            if entry == "input"
            else "paused"
            if entry == "paused"
            else "waiting_client_tool"
        )
        if entry in {"input", "paused"}:
            await store.pause_run(
                paused.run_id, status=status, pause_data=await store.get_pause_state(paused.run_id)
            )
        before = (await store.get_run_control(paused.run_id)).claim_epoch
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if entry == "resume_stream":
                events = [
                    e
                    async for e in a.resume_stream(
                        paused.run_id, tool_results=payload, tenant_id=None
                    )
                ]
                assert events[-1].run_result.status == RunStatus.ERROR
                assert "pause_id_required" in events[-1].run_result.error
            else:
                with pytest.raises(InputValidationError) as error:
                    if entry == "tool":
                        await a.submit_tool_results(paused.run_id, payload, tenant_id=None)
                    elif entry == "input":
                        await a.submit_input(paused.run_id, "answer", tenant_id=None)
                    elif entry == "paused":
                        await a.submit_resume(paused.run_id, tenant_id=None)
                    else:
                        await a.resume(paused.run_id, tool_results=payload, tenant_id=None)
                assert error.value.code == "pause_id_required"
        assert not caught
        assert (await store.get_run(paused.run_id)).status == status
        assert (await store.get_run_control(paused.run_id)).claim_epoch == before
    finally:
        await engine.dispose()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("policy", [{}, {"strict_tenancy": True}])
async def test_no_store_run_and_stream_allow_omitted_tenant_silently(streaming, policy):
    from dendrux.types import LLMResponse, RunStatus

    a = Agent(provider=MockLLM([LLMResponse(text="done")]), prompt="p", **policy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        if streaming:
            events = [event async for event in a.stream("go")]
            assert events[-1].run_result.status == RunStatus.SUCCESS
            assert events[-1].run_result.answer == "done"
        else:
            result = await a.run("go")
            assert result.status == RunStatus.SUCCESS
            assert result.answer == "done"


@pytest.mark.parametrize("persistence", ["store", "url"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_persistent_default_refuses_scope_before_io(persistence, streaming):
    config = (
        {"state_store": AsyncMock()}
        if persistence == "store"
        else {"database_url": "sqlite+aiosqlite://"}
    )
    a = agent(**config)
    if streaming:
        with pytest.raises(TenantScopeRequiredError):
            a.stream("go")
    else:
        with pytest.raises(TenantScopeRequiredError):
            await a.run("go")
    assert a._lazy_store is None
    if persistence == "store":
        a._state_store.get_run.assert_not_awaited()


def test_no_store_missing_nonapproval_pause_token_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        require_pause_id(agent(), None, pause_id="generation")


def test_budget_explicit_none_keeps_legacy_formula_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent(budget=Budget(max_tokens=500, count_cache_tokens=None))


def test_pause_state_keeps_v050_positional_provider_kwargs():
    from dendrux.types import PauseState

    provider_kwargs = {"model": "legacy-model"}
    pause = PauseState(
        "legacy",
        [],
        [],
        [],
        1,
        0,
        UsageStats(),
        {},
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        0,
        None,
        None,
        1,
        provider_kwargs,
    )
    assert pause.provider_kwargs is provider_kwargs
    assert pause.image_view_ref is None
    assert pause.loop_state_ref is False
    assert PauseState.from_dict(pause.to_dict()).provider_kwargs == provider_kwargs
