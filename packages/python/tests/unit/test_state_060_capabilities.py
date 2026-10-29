"""Capability declarations and per-run storage-format pinning."""

from dataclasses import FrozenInstanceError

import pytest
import pytest_asyncio
from sqlalchemy import text

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime import state
from dendrux.types import generate_ulid
from tests._fixtures.stores import (
    FencedButRejectsEpochStore,
    KwargsLegacyStore,
    LegacyInMemoryStore,
    PartialFencedRetryStore,
    UndeclaredSameMethodsStore,
)


@pytest_asyncio.fixture
async def sqlite_engine():
    from tests._helpers.v030_oracle import create_sqlite_engine

    engine = await create_sqlite_engine()
    try:
        yield engine
    finally:
        await engine.dispose()


def test_a0_capability_type_is_frozen_and_conservative():
    caps_type = getattr(state, "StoreCapabilities", None)
    assert caps_type is not None, "StateStore needs an explicit capability declaration"
    caps = caps_type()
    assert caps.version == 1
    assert not any(
        (
            caps.fenced_writes,
            caps.retry_ids,
            caps.checkpointing,
            caps.atomic_completion,
            caps.recovery,
            caps.notify,
        )
    )
    with pytest.raises(FrozenInstanceError):
        caps.version = 2


@pytest.mark.parametrize(
    "wrapper",
    [LegacyInMemoryStore, UndeclaredSameMethodsStore, KwargsLegacyStore, PartialFencedRetryStore],
)
async def test_a0_legacy_and_partial_stores_construct(wrapper, sqlite_engine):
    Agent(
        prompt="p",
        provider=MockLLM([]),
        state_store=wrapper(state.SQLAlchemyStateStore(sqlite_engine)),
    )


async def test_a0_misdeclared_store_refused_at_construction(sqlite_engine):
    store = FencedButRejectsEpochStore(state.SQLAlchemyStateStore(sqlite_engine))
    with pytest.raises(TypeError) as caught:
        Agent(prompt="p", provider=MockLLM([]), state_store=store)
    assert type(caught.value).__name__ == "StoreCapabilityError"
    assert caught.value.capability == "fenced_writes"
    assert caught.value.method == "save_trace"
    assert caught.value.missing == ("claim_epoch",)


async def test_a17_run_format_pinned_across_activation_and_both_claims(sqlite_engine):
    store = state.SQLAlchemyStateStore(sqlite_engine)
    async with sqlite_engine.begin() as conn:
        await conn.execute(text("UPDATE dendrux_settings SET value='1' WHERE key='storage_format'"))
    run_id = generate_ulid()
    created = await store.create_run(run_id, "pin", strategy="react")
    assert getattr(created, "storage_format", None) == 1, "create must return its pinned format"
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"pause_id": "old"})
    async with sqlite_engine.begin() as conn:
        await conn.execute(text("UPDATE dendrux_settings SET value='2' WHERE key='storage_format'"))
    claim = await store.claim_paused_run(run_id, expected_status="waiting_client_tool")
    assert getattr(claim, "storage_format", None) == 1
    assert claim.claim_epoch == 1 and claim.loop_state is None and claim.submission is None
    assert await store.claim_paused_run(run_id, expected_status="waiting_client_tool") is False
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"pause_id": "old"})
    claim = await store.submit_and_claim(
        run_id, expected_status="waiting_client_tool", submitted_data={"results": []}
    )
    assert claim.storage_format == 1 and claim.claim_epoch == 2
    record = await store.get_run(run_id)
    projected = await store._get_result_record(run_id, tenant_id=state.UNSCOPED)
    assert record.storage_format == projected.storage_format == 1
    assert record.claim_epoch == projected.claim_epoch == 2
    assert record.status_changed_at == projected.status_changed_at
    fresh = await store.create_run(generate_ulid(), "pin", strategy="react")
    assert fresh.storage_format == 2


def test_a0_dynamic_attributes_do_not_declare_capabilities():
    from unittest.mock import AsyncMock

    store = AsyncMock()
    Agent(prompt="p", provider=MockLLM([]), state_store=store)


def test_protocol_contains_transactional_store_methods():
    for name in (
        "checkpoint_step",
        "conversation_traces",
        "get_traces_after",
        "get_trace",
        "get_loop_state",
        "start_tool_call",
        "complete_tool_call",
        "consume_submission",
        "mark_lease_expired",
        "get_lease_expired",
        "reclaim_run",
        "set_trace_link",
        "get_replayed_tool_calls",
        "mark_cancel_requested",
        "get_children_page",
    ):
        assert name in vars(state.StateStore), f"StateStore must type its {name} contract"


async def test_partial_sql_store_pins_format_one_even_on_active_database(sqlite_engine):
    class Partial(state.SQLAlchemyStateStore):
        capabilities = state.StoreCapabilities(fenced_writes=True, retry_ids=True)

    store = Partial(sqlite_engine)
    result = await store.create_run(generate_ulid(), "partial")
    assert result.storage_format == 1
    assert (await store.get_run(result.run_id)).storage_format == 1


@pytest.mark.parametrize(
    "method,keyword",
    [
        ("create_run", "seed_traces"),
        ("create_run", "pii_mapping"),
        ("save_tool_call", "outcome"),
        ("complete_tool_call", "pii_mapping"),
        ("consume_submission", "pii_mapping"),
    ],
)
@pytest.mark.parametrize("shape", ["absent", "positional", "kwargs"])
async def test_checkpoint_keywords_refused_before_run(sqlite_engine, method, keyword, shape):
    import inspect

    from dendrux.errors import StoreCapabilityError

    original = getattr(state.SQLAlchemyStateStore, method)
    signature = inspect.signature(original)
    params = [p for p in signature.parameters.values() if p.name != keyword]
    if shape == "positional":
        params.insert(1, inspect.Parameter(keyword, inspect.Parameter.POSITIONAL_ONLY))
        params[0] = params[0].replace(kind=inspect.Parameter.POSITIONAL_ONLY)
    elif shape == "kwargs":
        params.append(inspect.Parameter("extra", inspect.Parameter.VAR_KEYWORD))

    async def incompatible(*args, **kwargs):
        raise AssertionError("constructor must reject before invoking store")

    incompatible.__signature__ = signature.replace(parameters=params)
    store_type = type(
        "IncompleteCheckpointStore", (state.SQLAlchemyStateStore,), {method: incompatible}
    )
    with pytest.raises(StoreCapabilityError) as caught:
        Agent(prompt="p", provider=MockLLM([]), state_store=store_type(sqlite_engine))
    assert caught.value.capability == "checkpointing"
    assert caught.value.method == method
    assert keyword in caught.value.missing


def test_recovery_requires_finalise_unrecoverable_keyword():
    from dendrux.errors import StoreCapabilityError
    from dendrux.runtime._capabilities import validate_capabilities

    class OldRecovery:
        capabilities = state.StoreCapabilities(recovery=True)

        async def mark_lease_expired(self, **kwargs):
            pass

        async def get_lease_expired(self):
            pass

        async def reclaim_run(self):
            pass

    with pytest.raises(StoreCapabilityError) as exc:
        validate_capabilities(OldRecovery())
    assert exc.value.missing == ("finalise_unrecoverable",)
