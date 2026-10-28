"""The custom-store fixtures (``tests/_fixtures/stores.py``): their declared surface today.

Phase 1 wires (b), (c) and (e) into the capability validation; this pins what they are now,
so a change in the fixtures is deliberate. (That all five run flows on 0.5, which reads no
declaration, is checked in ``test_legacy_stores.py``.)
"""

from __future__ import annotations

import inspect

import pytest

from dendrux.runtime.state import SQLAlchemyStateStore
from tests._fixtures import stores

UNDECLARED = ["a_legacy_in_memory", "b_undeclared_same_methods", "d_kwargs_legacy"]


def build(name: str, engine):
    return stores.STORE_FIXTURES[name](SQLAlchemyStateStore(engine))


def test_protocol_list_matches_the_050_state_store() -> None:
    from dendrux.runtime.state import StateStore

    declared = {
        n for n, _ in inspect.getmembers(StateStore, inspect.isfunction) if not n.startswith("_")
    }
    assert set(stores.PROTOCOL_0_5) < declared
    assert declared - set(stores.PROTOCOL_0_5) == stores.PROTOCOL_0_6_ADDITIONS
    assert len(set(stores.PROTOCOL_0_5)) == len(stores.PROTOCOL_0_5)


@pytest.mark.parametrize("name", UNDECLARED)
async def test_undeclared_fixtures_have_no_capabilities_attribute(name, sqlite_engine) -> None:
    store = build(name, sqlite_engine)
    assert not hasattr(store, "capabilities")
    for method in stores.PROTOCOL_0_5:
        assert inspect.iscoroutinefunction(getattr(store, method)), method


async def test_kwargs_fixture_methods_accept_anything(sqlite_engine) -> None:
    store = build("d_kwargs_legacy", sqlite_engine)
    for method in stores.PROTOCOL_0_5:
        params = inspect.signature(getattr(store, method)).parameters
        assert {p.kind for p in params.values()} == {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }, method


async def test_same_methods_fixture_follows_the_real_store_signatures(sqlite_engine) -> None:
    store = build("b_undeclared_same_methods", sqlite_engine)
    for method in stores.PROTOCOL_0_5:
        assert str(inspect.signature(getattr(store, method))) == str(
            inspect.signature(getattr(SQLAlchemyStateStore(sqlite_engine), method))
        ), method


async def test_fenced_fixture_declares_fencing_but_rejects_claim_epoch(sqlite_engine) -> None:
    store = build("c_fenced_rejects_epoch", sqlite_engine)
    assert store.capabilities.fenced_writes is True
    assert store.capabilities.retry_ids is False
    for method in (
        "save_trace",
        "save_tool_call",
        "save_usage",
        "save_llm_interaction",
        "save_run_event",
        "add_note",
        "touch_progress",
    ):
        assert "claim_epoch" not in inspect.signature(getattr(store, method)).parameters, method


async def test_partial_fixture_declares_fenced_writes_and_retry_ids_only(sqlite_engine) -> None:
    store = build("e_partial_fenced_retry", sqlite_engine)
    caps = store.capabilities
    assert (caps.fenced_writes, caps.retry_ids) == (True, True)
    assert not any((caps.checkpointing, caps.atomic_completion, caps.recovery, caps.notify))
    for method, extra in (
        ("save_trace", "trace_id"),
        ("save_tool_call", "row_id"),
        ("save_run_event", "event_id"),
        ("save_usage", "usage_id"),
        ("save_llm_interaction", "interaction_id"),
    ):
        params = inspect.signature(getattr(store, method)).parameters
        assert "claim_epoch" in params and extra in params, method
    # every method and keyword contracts.md section 2 requires for both declared capabilities
    assert stores.capability_gaps(store) == []
    # claim_epoch is only accepted on the inserts that fence; the 0.5 finalize/pause keep theirs
    assert "claim_epoch" in inspect.signature(store.finalize_run).parameters


@pytest.mark.parametrize("name", sorted(stores.STORE_FIXTURES))
async def test_claims_win_with_the_declared_shape_and_lose_with_false(name, sqlite_engine) -> None:
    """A lost CAS is ``False`` for every fixture; a legacy fixture's win is the epoch ``int``."""
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.tool import tool
    from dendrux.types import LLMResponse, ToolCall

    @tool(target="client")
    async def read_range(sheet: str) -> str:
        """Read a range on the client."""
        return ""

    store = build(name, sqlite_engine)
    call = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
    if name == "c_fenced_rejects_epoch":
        from dendrux.errors import StoreCapabilityError

        with pytest.raises(StoreCapabilityError):
            Agent(provider=MockLLM([]), prompt="p", state_store=store)
        return
    agent = Agent(
        provider=MockLLM([LLMResponse(tool_calls=[call])]),
        prompt="p",
        tools=[read_range],
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    paused = await agent.run("read")
    won = await store.claim_paused_run(paused.run_id, expected_status="waiting_client_tool")
    again = await store.claim_paused_run(paused.run_id, expected_status="waiting_client_tool")
    assert again is False
    assert store.claim_shape in ("int", "real")
    if store.claim_shape == "int":
        assert type(won) is int and won >= 1
    else:
        assert won is not False


async def test_the_method_table_accepts_the_partial_store_and_rejects_the_fenced_one(
    sqlite_engine,
) -> None:
    """What phase 1's construction-time validator must conclude about the declared fixtures."""
    partial = build("e_partial_fenced_retry", sqlite_engine)
    assert stores.capability_gaps(partial) == []
    assert "claim_epoch" in inspect.signature(partial.update_run_meta).parameters
    fenced = build("c_fenced_rejects_epoch", sqlite_engine)
    gaps = stores.capability_gaps(fenced)
    assert {m for _, m, _ in gaps} == set(stores.CAPABILITY_REQUIREMENTS["fenced_writes"])
    assert all(cap == "fenced_writes" and kw == ("claim_epoch",) for cap, _, kw in gaps)
    for name in UNDECLARED:
        assert stores.capability_gaps(build(name, sqlite_engine)) == []


async def test_capability_gaps_notices_a_positional_or_missing_keyword(sqlite_engine) -> None:
    """The checker itself: a keyword that is not keyword-only does not count."""
    partial = build("e_partial_fenced_retry", sqlite_engine)

    async def update_run_meta(run_id, patch, claim_epoch=None):  # positional-or-keyword
        return None

    partial.update_run_meta = update_run_meta
    assert stores.capability_gaps(partial) == [
        ("fenced_writes", "update_run_meta", ("claim_epoch",))
    ]
    del partial.update_run_meta
    partial.__class__ = type("NoMeta", (type(partial),), {"update_run_meta": None})
    assert stores.capability_gaps(partial)[0][1] == "update_run_meta"
