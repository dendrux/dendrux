"""T4a preparation and claim custody, with a store-only minimal driver."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
from dataclasses import replace
from functools import wraps

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dendrux import Agent
from dendrux._meta_keys import BLOCKED_META_KEY, CANCEL_PERSISTED_META_KEY
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.runtime import _run_entry, durability
from dendrux.runtime._managed import ManagedInvocation
from dendrux.runtime._managed_fanout_store import (
    BindManagedChildrenResult,
    BoundChild,
)
from dendrux.runtime.state import ClaimResult, TraceWrite
from dendrux.runtime.tasks import _drain_cleanup
from dendrux.types import LLMResponse, ToolCall, UsageStats, generate_ulid
from tests._helpers.gates import wait_for_gate
from tests.integration.test_fanout_store_races import batch, fanout_parent

SECRET = "AKIAIOSFODNN7EXAMPLE"


def ledger_type():
    name = "dendrux.runtime._fanout_ledger"
    assert importlib.util.find_spec(name) is not None, "claim custody ledger is required"
    return importlib.import_module(name).FanoutLedger


def handoff():
    method = getattr(durability, "_fanout_ownership_handoff", None)
    assert callable(method), "N-claim ownership handoff is required"
    return method


def tokens():
    return BindManagedChildrenResult(
        "created",
        generate_ulid(),
        tuple(
            BoundChild(str(i), generate_ulid(), ClaimResult(i + 3, 2, {}, None)) for i in range(3)
        ),
    )


def test_ledger_state_table_and_synchronous_transfers():
    ledger = ledger_type()(tokens())
    assert ledger.states == ("untransferred",) * 3
    assert not inspect.iscoroutinefunction(ledger.accept)
    assert not inspect.iscoroutinefunction(ledger.close)
    assert ledger.accept(1) is True
    assert ledger.accept(1) is False
    assert ledger.states == ("untransferred", "accepted", "untransferred")
    assert ledger.close() == [0, 2]
    assert ledger.states == ("driver_owned", "accepted", "driver_owned")
    assert ledger.close() == []
    assert all(ledger.accept(i) is False for i in range(3))
    assert ledger.states == ("driver_owned", "accepted", "driver_owned")


def test_existing_readback_has_no_claim_custody():
    result = tokens()
    existing = replace(
        result,
        outcome="existing",
        children=tuple(replace(child, child_claim=None) for child in result.children),
    )
    ledger = ledger_type()(existing)
    assert ledger.close() == []
    assert ledger.accept(0) is False


async def test_x7_cancel_before_any_owner_starts(store):
    cls = ledger_type()
    root, calls = await fanout_parent(store, 3)
    bound = await store.bind_managed_children(batch(root, calls))
    ledger = cls(bound)
    touched = []

    async def owner(i):
        if not ledger.accept(i):
            return
        touched.append(i)

    async def driver():
        owners = [asyncio.create_task(owner(i)) for i in range(3)]
        try:
            raise asyncio.CancelledError("first")
        finally:
            driver_owned = ledger.close()
            for task in owners:
                task.cancel()

            async def cleanup():
                await ledger.finalize(store, driver_owned)
                await asyncio.wait(owners)

            await _drain_cleanup(cleanup(), run_id=root)

    with pytest.raises(asyncio.CancelledError):
        await driver()
    assert touched == []
    assert ledger.states == ("finalized",) * 3
    for child in bound.children:
        assert (await store.get_run(child.child_run_id)).status == "cancelled"
        events = await store.get_run_events(child.child_run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1


async def test_x11_queued_owner_refuses_late_accept(store):
    cls = ledger_type()
    root, calls = await fanout_parent(store, 3)
    bound = await store.bind_managed_children(batch(root, calls))
    ledger = cls(bound)
    touched = []

    async def owner(i):
        if not ledger.accept(i):
            return
        touched.append(i)  # Stands for every child/store/queue operation.

    owners = [asyncio.create_task(owner(i)) for i in range(3)]
    driver_owned = ledger.close()  # Startup fails before the scheduler can run owners.

    async def cleanup():
        await ledger.finalize(store, driver_owned)
        await asyncio.wait(owners)

    await _drain_cleanup(cleanup(), run_id=root)
    assert touched == []
    assert ledger.states == ("finalized",) * 3
    for child in bound.children:
        events = await store.get_run_events(child.child_run_id)
        assert sum(e.event_type == "run.cancelled" for e in events) == 1


async def test_ledger_failed_finalize_never_retries_or_refreshes_epoch(monkeypatch):
    cls = ledger_type()
    module = importlib.import_module("dendrux.runtime._fanout_ledger")
    bound, attempts = tokens(), []
    ledger = cls(bound)

    def claimed(store, run_id, claim):
        attempts.append((run_id, claim.claim_epoch))
        return store

    async def persist(store, run_id):
        from dendrux.types import RunResult, RunStatus

        return RunResult(
            run_id=run_id,
            status=RunStatus.CANCELLED,
            meta={
                CANCEL_PERSISTED_META_KEY: run_id != bound.children[1].child_run_id,
            },
        )

    monkeypatch.setattr(module, "claimed_store", claimed)
    monkeypatch.setattr(module, "_persist_task_cancellation", persist)
    owned = ledger.close()
    await ledger.finalize(object(), owned)
    assert ledger.states == ("finalized", "unresolved", "finalized")
    await ledger.finalize(object(), owned)
    assert attempts == [(c.child_run_id, c.child_claim.claim_epoch) for c in bound.children]


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_x10_cancel_during_bind_retains_every_claim(store, monkeypatch, lost_ack):
    method = handoff()
    root, calls = await fanout_parent(store, 3)
    request = batch(root, calls)
    committed, release = asyncio.Event(), asyncio.Event()
    original = AsyncSession.commit
    armed = True

    async def commit(session):
        nonlocal armed
        await original(session)
        if armed:
            armed = False
            committed.set()
            await release.wait()
            if lost_ack:
                raise OSError("SENTINEL-private-ack")

    monkeypatch.setattr(AsyncSession, "commit", commit)
    caller = asyncio.create_task(
        method(store, lambda: store.bind_managed_children(request), run_id=root)
    )
    try:
        await wait_for_gate(committed, task=caller)
        caller.cancel("first")
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(caller, 10)
        assert cancelled.value.args == ("first",)
        for child in request.children:
            assert (await store.get_run(child.child_run_id)).status == "cancelled"
        replay = await store.bind_managed_children(request)
        assert replay.outcome == "existing"
        assert all(c.child_claim is None for c in replay.children)
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


def invocation(store, root, call, child, wait):
    return ManagedInvocation(
        store,
        "t1",
        root,
        root,
        0,
        0,
        child,
        call,
        child,
        ("reviewer",),
        wait,
        1,
        UsageStats(),
        10,
        generate_ulid(),
    )


async def test_p20_preparation_blocks_one_child_without_binding(store):
    prepare = getattr(_run_entry, "prepare_child_creation", None)
    assert callable(prepare), "write-free child creation preparation is required"
    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    child = Agent(
        name="reviewer",
        prompt="Answer briefly.",
        provider=MockLLM([LLMResponse(text="ok")]),
        guardrails=[SecretDetection()],
    )
    try:
        specs = []
        for call_id, task in zip(calls, (SECRET, "clean"), strict=True):
            branch = invocation(
                store,
                root,
                ToolCall("reviewer", {"task": task}, id=call_id),
                child,
                request.parent_wait,
            )
            specs.append(await prepare(child, branch))
            assert await store.get_run(branch.child_run_id) is None
        assert specs[0].child_meta[BLOCKED_META_KEY] is True
        assert not specs[1].child_meta.get(BLOCKED_META_KEY)
        assert SECRET not in repr([(s.child_input_data, s.child_seed_traces) for s in specs])
        assert specs[1].child_input_data == {"input": "clean"}
    finally:
        await child.close()


async def test_p19_mutation_while_root_lock_is_pending(store, monkeypatch):
    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    request = replace(
        request,
        children=tuple(
            replace(
                c,
                child_seed_traces=(
                    TraceWrite(
                        generate_ulid(), "user", "seed", 0, trace_kind="seed", meta={"test": []}
                    ),
                ),
            )
            for c in request.children
        ),
    )
    before = request.detached()
    entered, release = asyncio.Event(), asyncio.Event()
    lock = store._lock_managed_root

    async def held(session, plan):
        entered.set()
        await release.wait()
        await lock(session, plan)

    monkeypatch.setattr(store, "_lock_managed_root", held)
    caller = asyncio.create_task(store.bind_managed_children(request))
    try:
        await wait_for_gate(entered, task=caller)
        for spec in request.children:
            spec.params["task"] = "mutated"
            spec.child_seed_traces[0].meta["test"] = ["mutated"]
            spec.child_loop_state["phase"] = "mutated"
        request.parent_patch.set_fields["mutation"] = True
        request.parent_wait.encoded["managed_fanout"]["bindings"].clear()
        release.set()
        result = await caller
        assert result.outcome == "created"
        for spec in before.children:
            row = await store.get_run(spec.child_run_id)
            assert row.input_data == spec.child_input_data
            assert (await store.get_loop_state(spec.child_run_id)).get("phase") != "mutated"
            traces = await store.get_traces(spec.child_run_id)
            assert traces[0].meta.get("test") != ["mutated"]
        assert "mutation" not in await store.get_loop_state(root)
        wait = await store.get_pause_state(root)
        assert wait["managed_fanout"] == before.parent_wait.to_dict()["managed_fanout"]
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.xfail(
    strict=True, reason="Contract gap: GuardrailEngine converts detector exceptions to UNSCANNABLE"
)
async def test_p20_detector_exception_binds_nothing_and_uses_parent_failure(store):
    prepare = getattr(_run_entry, "prepare_child_creation", None)
    assert callable(prepare), "write-free child creation preparation is required"
    from dendrux.runtime import _managed

    class BrokenDetector:
        action = "block"

        async def scan(self, text):
            raise ValueError("SENTINEL-detector-failure")

    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    child = Agent(name="reviewer", prompt="p", provider=MockLLM([]), guardrails=[BrokenDetector()])
    branch = invocation(
        store,
        root,
        ToolCall("reviewer", {"task": "clean"}, id=calls[0]),
        child,
        request.parent_wait,
    )
    try:
        with pytest.raises(ValueError):
            try:
                await prepare(child, branch)
            except Exception as error:
                await _managed._fail_parent(branch, error)
                raise
        assert await store.get_run(branch.child_run_id) is None
        assert (await store.get_run(root)).status == "error"
        assert await store.get_pause_state(root) is None
        assert await store.get_tool_calls(root) == []
    finally:
        await child.close()


async def test_x10_lost_ack_without_cancellation_hands_over_tokens(store, monkeypatch):
    method = handoff()
    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    original = AsyncSession.commit
    armed = True

    async def commit(session):
        nonlocal armed
        await original(session)
        if armed:
            armed = False
            raise OSError("SENTINEL-private-ack")

    monkeypatch.setattr(AsyncSession, "commit", commit)
    result = await method(store, lambda: store.bind_managed_children(request), run_id=root)
    assert result.outcome == "created"
    assert all(c.child_claim is not None for c in result.children)
    assert (await store.bind_managed_children(request)).outcome == "existing"


async def test_handoff_repeated_cancel_cannot_skip_finalization(store, monkeypatch):
    method = handoff()
    root, calls = await fanout_parent(store, 3)
    request = batch(root, calls)
    committed, release = asyncio.Event(), asyncio.Event()
    cleaning, finish = asyncio.Event(), asyncio.Event()
    original = store.finalize_with_event
    writes = []

    async def operation():
        result = await store.bind_managed_children(request)
        committed.set()
        await release.wait()
        return result

    @wraps(original)
    async def finalize(run_id, *, claim_epoch=None, **kwargs):
        writes.append((run_id, claim_epoch))
        cleaning.set()
        await finish.wait()
        return await original(run_id, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)
    caller = asyncio.create_task(method(store, operation, run_id=root))
    try:
        await wait_for_gate(committed, task=caller)
        caller.cancel("first")
        release.set()
        await wait_for_gate(cleaning, task=caller)
        for _ in range(3):
            caller.cancel("later")
            await asyncio.sleep(0)
        finish.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(caller, 10)
        assert error.value.args == ("first",)
        assert writes == [(c.child_run_id, 0) for c in request.children]
        for spec in request.children:
            assert (await store.get_run(spec.child_run_id)).status == "cancelled"
    finally:
        release.set()
        finish.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.xfail(
    strict=True, reason="Contract gap: native creation strips child_meta block evidence"
)
async def test_p20_prepared_block_evidence_survives_native_batch_bind(store):
    prepare = getattr(_run_entry, "prepare_child_creation", None)
    assert callable(prepare), "write-free child creation preparation is required"
    from tests.integration.test_fanout_store_races import fanout_wait

    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    child = Agent(name="reviewer", prompt="p", provider=MockLLM([]), guardrails=[SecretDetection()])
    try:
        specs = []
        for call_id, text in zip(calls, (SECRET, "clean"), strict=True):
            branch = invocation(
                store,
                root,
                ToolCall("reviewer", {"task": text}, id=call_id),
                child,
                request.parent_wait,
            )
            specs.append(await prepare(child, branch))
        request = replace(
            request,
            children=tuple(specs),
            parent_wait=fanout_wait([(s.tool_call_id, s.child_run_id) for s in specs]),
        )
        await store.bind_managed_children(request)
        assert (await store.get_run(specs[0].child_run_id)).meta[BLOCKED_META_KEY] is True
        assert not (await store.get_run(specs[1].child_run_id)).meta.get(BLOCKED_META_KEY)
    finally:
        await child.close()


async def test_preparation_plugin_error_leaves_no_bind_and_fails_parent(store, monkeypatch):
    from dendrux.runtime import _managed

    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    child = Agent(name="reviewer", prompt="p", provider=MockLLM([]))
    branch = invocation(
        store,
        root,
        ToolCall("reviewer", {"task": "clean"}, id=calls[0]),
        child,
        request.parent_wait,
    )

    def broken(loop):
        raise ValueError("SENTINEL-plugin-failure")

    monkeypatch.setattr(child, "get_system_prompt", broken)
    try:
        with pytest.raises(ValueError):
            try:
                await _run_entry.prepare_child_creation(child, branch)
            except Exception as error:
                await _managed._fail_parent(branch, error)
                raise
        assert await store.get_run(branch.child_run_id) is None
        assert (await store.get_run(root)).status == "error"
        assert await store.get_tool_calls(root) == []
    finally:
        await child.close()


@pytest.mark.parametrize("single,structured", [(False, False), (True, False), (True, True)])
async def test_preparation_matches_single_child_creation_arguments(
    store, monkeypatch, single, structured
):
    from pydantic import BaseModel

    from dendrux.loops.single import SingleCall
    from dendrux.runtime import _managed
    from dendrux.runtime.context import managed_invocation_scope
    from dendrux.runtime.runner import _run_once

    class Answer(BaseModel):
        answer: str

    class CreationCapturedError(Exception):
        pass

    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    child = Agent(
        name="reviewer",
        prompt="p",
        provider=MockLLM([]),
        loop=SingleCall() if single else None,
        max_delegation_depth=3,
        output_type=Answer if structured else None,
    )
    branch = invocation(
        store,
        root,
        ToolCall("reviewer", {"task": "clean"}, id=calls[0]),
        child,
        request.parent_wait,
    )
    captured = {}

    async def bind(invocation, store, run_id, agent_name, **kwargs):
        captured.update(kwargs)
        raise CreationCapturedError

    monkeypatch.setattr(_managed, "bind_child", bind)
    try:
        spec = await _run_entry.prepare_child_creation(child, branch)
        with managed_invocation_scope(branch), pytest.raises(CreationCapturedError):
            await _run_once(
                child,
                provider=child.provider,
                user_input="clean",
                state_store=store,
                tenant_id="t1",
                max_delegation_depth=3,
                output_type=child.output_type,
            )
        for field, old in (
            ("child_input_data", "input_data"),
            ("child_model", "model"),
            ("child_strategy", "strategy"),
            ("child_meta", "meta"),
            ("child_pii_mapping", "pii_mapping"),
        ):
            assert getattr(spec, field) == captured[old]
        assert [(s.role, s.content, s.meta) for s in spec.child_seed_traces] == [
            (s.role, s.content, s.meta) for s in captured["seed_traces"]
        ]
        assert spec.child_loop_state == {**captured["loop_state"], "managed_unpriced_accepted": 0}
    finally:
        await child.close()


async def test_finalization_uses_each_original_epoch_and_keeps_unresolved(
    store, engine, monkeypatch, caplog
):
    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.errors import PersistenceError

    cls = ledger_type()
    root, calls = await fanout_parent(store, 3)
    result = await store.bind_managed_children(batch(root, calls))
    children = []
    async with AsyncSession(engine) as session:
        for i, child in enumerate(result.children, 4):
            await session.execute(
                update(AgentRun).where(AgentRun.id == child.child_run_id).values(claim_epoch=i)
            )
            children.append(replace(child, child_claim=replace(child.child_claim, claim_epoch=i)))
        await session.commit()
    ledger = cls(replace(result, children=tuple(children)))
    original = store.finalize_with_event
    writes = []

    @wraps(original)
    async def finalize(run_id, *, claim_epoch=None, **kwargs):
        writes.append((run_id, claim_epoch))
        if run_id == children[1].child_run_id:
            raise PersistenceError()
        return await original(run_id, claim_epoch=claim_epoch, **kwargs)

    monkeypatch.setattr(store, "finalize_with_event", finalize)
    owned = ledger.close()
    await _drain_cleanup(ledger.finalize(store, owned), run_id=root)
    assert ledger.states == ("finalized", "unresolved", "finalized")
    await _drain_cleanup(ledger.finalize(store, owned), run_id=root)
    assert writes == [(c.child_run_id, c.child_claim.claim_epoch) for c in children]
    for i, child in enumerate(children):
        record = await store.get_run(child.child_run_id)
        assert record.claim_epoch == i + 4
        assert record.status == ("running" if i == 1 else "cancelled")
    assert "cancel_persist_failed" in caplog.text


async def test_precommit_bind_cancel_has_no_children(store, monkeypatch):
    method = handoff()
    root, calls = await fanout_parent(store, 2)
    request = batch(root, calls)
    entered = asyncio.Event()

    async def locked(session, plan):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(store, "_lock_managed_root", locked)
    caller = asyncio.create_task(
        method(store, lambda: store.bind_managed_children(request), run_id=root)
    )
    await wait_for_gate(entered, task=caller)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, 10)
    assert (await store.get_run(root)).status == "running"
    for child in request.children:
        assert await store.get_run(child.child_run_id) is None


async def test_raised_unknown_never_rebinds_or_claims(store, caplog):
    from dendrux.errors import StoreOutcomeUnknown

    method = handoff()
    attempts = []
    root = generate_ulid()
    error = StoreOutcomeUnknown()

    async def operation():
        attempts.append(1)
        raise error

    with pytest.raises(StoreOutcomeUnknown) as raised:
        await method(store, operation, run_id=root)
    assert raised.value is error
    assert attempts == [1]
    assert "cancel_persist_failed" in caplog.text
    assert any(getattr(record, "run_id", None) == root for record in caplog.records)


async def test_p19_postgres_mutation_during_real_root_lock_wait(engine):
    from contextlib import AsyncExitStack

    from sqlalchemy import select

    from dendrux.db.models import AgentRun
    from tests.integration.test_fanout_store_races import _blocked_on_lock, _pools

    if engine.dialect.name != "postgresql":
        pytest.skip("real root row lock wait requires Postgres")
    async with AsyncExitStack() as stack:
        [store] = await _pools(stack, engine, ["t4a-p19-binder"])
        root, calls = await fanout_parent(store, 2)
        request = batch(root, calls)
        request = replace(
            request,
            children=tuple(
                replace(
                    c,
                    child_seed_traces=(
                        TraceWrite(
                            generate_ulid(),
                            "user",
                            "seed",
                            0,
                            trace_kind="seed",
                            meta={"original": [1]},
                        ),
                    ),
                )
                for c in request.children
            ),
        )
        before = request.detached()
        async with AsyncSession(engine) as locker:
            await locker.execute(select(AgentRun.id).where(AgentRun.id == root).with_for_update())
            caller = asyncio.create_task(store.bind_managed_children(request))
            try:
                await _blocked_on_lock(engine, "t4a-p19-binder")
                for child in request.children:
                    child.params["task"] = "mutated"
                    child.child_seed_traces[0].meta["original"].append(2)
                request.parent_patch.set_fields["mutation"] = True
                request.parent_wait.encoded["managed_fanout"]["bindings"].clear()
            finally:
                await locker.rollback()
                await caller
        rows = await store.get_tool_calls(root)
        assert [row.params for row in rows] == [child.params for child in before.children]
        wait = await store.get_pause_state(root)
        assert wait["managed_fanout"] == before.parent_wait.to_dict()["managed_fanout"]
        assert "mutation" not in await store.get_loop_state(root)
        for child in before.children:
            traces = await store.get_traces(child.child_run_id)
            assert traces[0].meta["original"] == [1]
