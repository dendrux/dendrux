"""Runtime contracts shared by fresh runs, continuations, retries and recovery.

Intentional differences: retry starts with fresh configuration and usage; streams
report errors as events; human-approved calls skip ``before_tool``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from dendrux import Agent, tool
from dendrux.errors import ProviderTimeoutError
from dendrux.loops.base import BaseNotifier
from dendrux.runtime.context import (
    DelegationContext,
    DelegationDepthExceededError,
    get_delegation_context,
    reset_delegation_context,
    set_delegation_context,
)
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall, ToolDecision
from tests._helpers.recovery_checkpoint import checkpoint as checkpoint
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import (
    PATHS as PATHS,
)
from tests._helpers.runtime_paths import (
    SUBMIT_KINDS as SUBMIT_KINDS,
)
from tests._helpers.runtime_paths import (
    Entry as Entry,
)
from tests._helpers.runtime_paths import (
    ProbeProvider as ProbeProvider,
)
from tests._helpers.runtime_paths import (
    client_read as client_read,
)
from tests._helpers.runtime_paths import (
    client_result as client_result,
)
from tests._helpers.runtime_paths import (
    make_agent as make_agent,
)
from tests._helpers.runtime_paths import (
    path_case as path_case,
)
from tests._helpers.runtime_paths import (
    submission_case as submission_case,
)
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@pytest.mark.parametrize("origin", ["run", "run_stream", "retry", "recovery"])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
@pytest.mark.parametrize("limit", [10, None])
async def test_retry_depth_survives_pause_and_refuses_child_at_boundary(
    store, engine, entry, limit, origin
):
    original = await make_agent(store, [LLMResponse(text="prior")]).run("go")
    seen = []

    @tool()
    async def probe() -> str:
        """Attempt a child at the inherited depth boundary."""
        ctx = get_delegation_context()
        seen.append(ctx.max_delegation_depth)
        try:
            await make_agent(store, [LLMResponse(text="child")]).run("child")
        except DelegationDepthExceededError:
            seen.append("refused")
        else:
            seen.append("allowed")
        return "probed"

    a = make_agent(
        store,
        [tool_call("client_read", pid="c"), tool_call("probe", pid="p"), LLMResponse(text="done")],
        tools=[client_read, probe],
    )
    token = set_delegation_context(
        DelegationContext(run_id="parent", delegation_level=9, max_delegation_depth=limit)
    )
    try:
        if origin == "retry":
            paused = await a.retry(original.run_id)
        elif origin == "run_stream":
            async with a.stream("go") as stream:
                async for _ in stream:
                    pass
            paused = stream.result
        else:
            paused = await a.run("go")
            if origin == "recovery":
                from sqlalchemy import update

                from dendrux.db.models import AgentRun

                async with engine.begin() as conn:
                    await conn.execute(
                        update(AgentRun)
                        .where(AgentRun.id == paused.run_id)
                        .values(status="running")
                    )
                item = await expire(store, engine, paused.run_id)
                paused = await a.reclaim(paused.run_id, item=item, tenant_id=None)
    finally:
        reset_delegation_context(token)
    assert paused.status == RunStatus.WAITING_CLIENT_TOOL
    if entry == "resume":
        result = await a.resume(paused.run_id, tool_results=[client_result(paused)])
    elif entry == "resume_stream":
        async with a.resume_stream(paused.run_id, tool_results=[client_result(paused)]) as stream:
            async for _ in stream:
                pass
        result = stream.result
    else:
        result = await a.submit_tool_results(paused.run_id, [client_result(paused)])
    assert result.status == RunStatus.SUCCESS
    assert seen == [limit, "allowed" if limit is None else "refused"]
    assert (await store.get_run(paused.run_id)).meta.get("dendrux.max_delegation_depth") == limit


@pytest.mark.parametrize("limit", [0, None])
@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results"])
async def test_retry_uses_configured_depth_including_unbounded(store, limit, entry):
    prior = await make_agent(store, [LLMResponse(text="prior")]).run("go")
    seen = []

    @tool()
    async def child() -> str:
        """Try a child with the fresh retry configuration."""
        seen.append(get_delegation_context().max_delegation_depth)
        try:
            await make_agent(store, [LLMResponse(text="child")]).run("child")
        except DelegationDepthExceededError:
            seen.append("refused")
        else:
            seen.append("allowed")
        return "ok"

    a = make_agent(
        store,
        [tool_call("client_read", pid="c"), tool_call("child", pid="d"), LLMResponse(text="done")],
        max_delegation_depth=limit,
        tools=[client_read, child],
    )
    paused = await a.retry(prior.run_id)
    if entry == "resume":
        result = await a.resume(paused.run_id, tool_results=[client_result(paused)])
    elif entry == "resume_stream":
        async with a.resume_stream(paused.run_id, tool_results=[client_result(paused)]) as stream:
            async for _ in stream:
                pass
        result = stream.result
    else:
        result = await a.submit_tool_results(paused.run_id, [client_result(paused)])
    assert result.status == RunStatus.SUCCESS
    assert seen == [limit, "allowed" if limit is None else "refused"]
    assert (await store.get_run(paused.run_id)).meta.get("dendrux.max_delegation_depth") == limit


@pytest.mark.parametrize("entry", ["run", "recovery", "approved", "recovered_approved"])
@pytest.mark.parametrize("allow", [True, False])
async def test_tool_policy_has_tenant_and_run_identity(store, engine, entry, allow):
    seen, executed = [], []

    @tool(idempotent=True)
    async def gated() -> str:
        """Record an allowed execution."""
        executed.append(True)
        return "ok"

    def policy(ctx):
        seen.append((ctx.tenant_id, ctx.run_id))
        return (
            ToolDecision.allow()
            if ctx.tenant_id == "tenant" and allow
            else ToolDecision.deny("policy_denied")
        )

    approved = entry in {"approved", "recovered_approved"}
    a = make_agent(
        store,
        [LLMResponse(text="done")]
        if entry.startswith("recover")
        else [tool_call("gated", pid="g"), LLMResponse(text="done")],
        tools=[gated],
        before_tool=policy,
        require_approval=["gated"] if approved else [],
    )
    if entry.startswith("recover"):
        item = await checkpoint(
            a,
            store,
            engine,
            calls=[ToolCall(name="gated", params={})],
            phases=["approved" if approved else "dispatchable"],
        )
        result = await a.reclaim(item.run_id, tenant_id="tenant", item=item)
    else:
        result = await a.run("go", tenant_id="tenant")
        if result.status == RunStatus.WAITING_APPROVAL:
            seen.clear()
            result = await a.submit_approval(
                result.run_id, approved=True, tenant_id="tenant", expected_pause_id=result.pause_id
            )
    assert result.status == RunStatus.SUCCESS
    if approved and (allow or entry == "recovered_approved"):
        assert seen == [], "approved calls intentionally skip before_tool"
        assert executed == [True]
    else:
        assert seen == [("tenant", result.run_id)]
        assert executed == ([True] if allow else [])


@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("cancel", ["caller", "manager"])
async def test_cancellation_interrupts_tool_and_pairs_hooks_once(store, entry, cancel):
    original = await make_agent(store, [LLMResponse(text="prior")]).run("go")
    entered, stopped = asyncio.Event(), asyncio.Event()
    identity = []

    @tool()
    async def hanging() -> str:
        """Wait until interrupted."""
        identity.append(get_delegation_context().run_id)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return "unreachable"

    a = make_agent(store, [tool_call("hanging", pid="h")], tools=[hanging])
    notifier = BaseNotifier()
    notifier.on_run_started = AsyncMock()
    notifier.on_run_finished = AsyncMock()
    notifier.on_run_failed = AsyncMock()
    task = asyncio.create_task(
        a.retry(original.run_id, notifier=notifier)
        if entry == "retry"
        else a.run("go", notifier=notifier)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        run_id = identity[0]
        if cancel == "caller":
            task.cancel("caller_cancel")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("caller_cancel",)
        else:
            await a.cancel_run(run_id)
            done, _ = await asyncio.wait({task}, timeout=2)
            assert done, "cancel_run must interrupt a hanging retry tool"
            assert (await task).status == RunStatus.CANCELLED
        assert stopped.is_set()
        assert (await store.get_run(run_id)).status == "cancelled"
        assert sum(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id)) == 1
        notifier.on_run_started.assert_awaited_once()
        notifier.on_run_finished.assert_awaited_once()
        notifier.on_run_failed.assert_not_awaited()
        assert run_id not in a._task_manager._owned
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_startup_event_write_failure_ends_error(store, monkeypatch, entry):
    original = await make_agent(store, [LLMResponse(text="prior")]).run("go")
    a = make_agent(store, [LLMResponse(text="unused")])
    save = store.save_run_event
    failure = RuntimeError("startup_write_failed")

    async def fail_started(run_id, **kwargs):
        if kwargs.get("event_type") == "run.started":
            raise failure
        return await save(run_id, **kwargs)

    monkeypatch.setattr(store, "save_run_event", fail_started)
    with pytest.raises(RuntimeError) as caught:
        if entry == "retry":
            await a.retry(original.run_id)
        else:
            await a.run("go")
    assert caught.value is failure
    rows = [r for r in await store.list_runs() if r.id != original.run_id]
    assert len(rows) == 1
    assert rows[0].status == "error"
    assert a._provider.calls == 0


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize(
    "failure_site",
    ["get_run", "get_pause_state", "get_pii_mapping", "get_next_trace_index", "consume_submission"],
)
async def test_unconsumed_submission_failure_releases_claim(
    store, monkeypatch, kind, failure_site, storage_format
):
    from tests._fixtures.storage import pin_storage_format

    if failure_site == "consume_submission" and (kind == "submit_resume" or storage_format == 1):
        pytest.skip("This path has no checkpointed submission to consume")
    await pin_storage_format(store._engine, storage_format)
    a, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    fail = RuntimeError("pre_consumption_failure")
    calls_before = a._provider.calls
    getter = getattr(store, failure_site)
    armed = False

    async def fail_owned(*args, **kwargs):
        if armed:
            raise fail
        return await getter(*args, **kwargs)

    def arm_claim(method):
        async def claim(*args, **kwargs):
            nonlocal armed
            result = await method(*args, **kwargs)
            armed = bool(result)
            return result

        return claim

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "submit_and_claim", arm_claim(store.submit_and_claim))
        patcher.setattr(store, "claim_paused_run", arm_claim(store.claim_paused_run))
        patcher.setattr(store, failure_site, fail_owned)
        with pytest.raises(RuntimeError) as caught:
            await invoke()
    assert caught.value is fail
    assert fail.__dict__ == {}, "release must not annotate the original setup error"
    row = await store.get_run(run_id)
    assert row.status == before.status, "unconsumed submissions must return to their waiting state"
    assert row.claim_epoch == before.claim_epoch + 1, "release must keep the claimed epoch"
    assert a._provider.calls == calls_before
    assert (await invoke()).status == RunStatus.SUCCESS


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
@pytest.mark.parametrize("release_fails", ["raises", "lost_cas", "cancelled"])
async def test_release_failure_preserves_original_error_and_logs_code(
    store, monkeypatch, caplog, kind, release_fails
):
    _, run_id, invoke = await submission_case(store, kind)
    original = RuntimeError("original_private_sentinel")
    secondary = RuntimeError("release_private_sentinel")
    if release_fails == "cancelled":
        secondary = asyncio.CancelledError("release_private_sentinel")
    monkeypatch.setattr(store, "get_pii_mapping", AsyncMock(side_effect=original))
    monkeypatch.setattr(
        store,
        "release_run_claim",
        AsyncMock(side_effect=secondary)
        if release_fails in {"raises", "cancelled"}
        else AsyncMock(return_value=False),
    )
    with pytest.raises(RuntimeError) as caught:
        await invoke()
    assert caught.value is original
    assert (await store.get_run(run_id)).status == "running"
    assert any(
        r.run_id == run_id and getattr(r, "code", None) == "persistence_error"
        for r in caplog.records
        if hasattr(r, "run_id")
    )
    assert "private_sentinel" not in caplog.text


@pytest.mark.parametrize(
    "failure_site", ["rebuild_pause", "get_pii_mapping", "get_next_trace_index"]
)
async def test_reclaim_setup_failure_releases_same_epoch(store, engine, monkeypatch, failure_site):
    from dendrux.runtime import _reclaim

    a = make_agent(store, [LLMResponse(text="done")])
    item = await checkpoint(a, store, engine)
    fail = RuntimeError("reclaim_setup_failure")
    released = AsyncMock(wraps=store.release_run_claim)
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "release_run_claim", released)
        patcher.setattr(
            _reclaim if failure_site == "rebuild_pause" else store,
            failure_site,
            AsyncMock(side_effect=fail),
        )
        with pytest.raises(RuntimeError) as caught:
            await a.reclaim(item.run_id, item=item, tenant_id="tenant")
    assert caught.value is fail
    released.assert_awaited_once_with(
        item.run_id, status="running", claim_epoch=item.claim_epoch + 1
    )
    assert (await store.get_run(item.run_id)).status == "running"
    assert a._provider.calls == 0


async def test_reclaim_transient_setup_failure_preserves_failure_hooks(store, engine, monkeypatch):
    from dendrux.runtime import runner
    from dendrux.runtime.persistence import PersistenceRecorder

    agent = make_agent(store, [LLMResponse(text="done")])
    item = await checkpoint(agent, store, engine)
    original = RuntimeError("transient_reclaim_setup")
    failed = AsyncMock()
    notifier = BaseNotifier()
    notifier.on_run_failed = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", failed)
    monkeypatch.setattr(runner, "_start_trace", AsyncMock(side_effect=original))
    with pytest.raises(RuntimeError) as caught:
        await agent.reclaim(item.run_id, tenant_id="tenant", item=item, notifier=notifier)
    assert caught.value is original
    failed.assert_awaited_once()
    notifier.on_run_failed.assert_awaited_once()
    assert (await store.get_run(item.run_id)).status == "running"
    assert not any(
        event.event_type == "run.resume_aborted"
        for event in await store.get_run_events(item.run_id)
    )


@pytest.mark.parametrize("path", PATHS)
async def test_caller_cancellation_grid_persists_once(store, engine, path):
    provider = ProbeProvider(block=True)
    case = await path_case(store, engine, path, provider)
    task = asyncio.create_task(case.invoke())
    try:
        await asyncio.wait_for(provider.entered.wait(), 5)
        task.cancel("grid_cancellation")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await store.get_run(provider.run_id)).status == "cancelled"
        assert (
            sum(
                e.event_type == "run.cancelled" for e in await store.get_run_events(provider.run_id)
            )
            == 1
        )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("path", PATHS)
async def test_safe_provider_errors_grid(store, engine, caplog, path):
    from dendrux.runtime.runner import _build_cached_result
    from dendrux.store import RunStore

    error = ProviderTimeoutError(
        message="vendor_private_sentinel",
        provider="fixture",
        model="scripted",
        timeout_reason="call_deadline",
        retryable=False,
    )
    provider = ProbeProvider(error=error)
    case = await path_case(store, engine, path, provider)
    if path in {"run_stream", "resume_stream"}:
        result = await case.invoke()
        assert result.status == RunStatus.ERROR
        assert case.events[-1].type == RunEventType.RUN_ERROR
        assert result.error_info is not None
    else:
        with pytest.raises(ProviderTimeoutError) as caught:
            await case.invoke()
        assert caught.value is error
    row = await store.get_run(provider.run_id)
    assert row.status == "error"
    assert row.error == "fixture timeout (HTTP n/a, code n/a)"
    info = row.meta["dendrux.error_info"]
    assert info["timeout_reason"] == "call_deadline"
    cached = await _build_cached_result(store, provider.run_id)
    assert cached.error_info.to_dict() == info
    detail = await RunStore(store).get_run(provider.run_id, tenant_id="tenant")
    assert detail.error == row.error
    events = await store.get_run_events(provider.run_id)
    assert next(e for e in events if e.event_type == "run.error").data["error_info"] == info
    assert "vendor_private_sentinel" not in json.dumps(
        [row.error, row.meta, [e.data for e in events]]
    )
    assert "vendor_private_sentinel" not in caplog.text


@pytest.mark.parametrize("path", PATHS)
async def test_policy_context_grid(store, engine, path):
    provider = ProbeProvider()
    seen = []
    active = False

    def deny(ctx):
        if active:
            seen.append(ctx)
            return ToolDecision.deny("test_policy")
        return None

    case = await path_case(store, engine, path, provider, before_tool=deny)
    active = True
    name = case.agent.tools[0].__name__
    params = (
        {"order_id": 1} if name == "refund" else {"sheet": "S1"} if name == "read_range" else {}
    )
    provider._responses = [tool_call(name, params, pid="policy"), LLMResponse(text="done")]
    result = await case.invoke()
    assert result.status == RunStatus.SUCCESS
    assert len(seen) == 1
    ctx = seen[0]
    assert (ctx.tenant_id, ctx.run_id, ctx.tool_name, ctx.params, ctx.source) == (
        "tenant",
        result.run_id,
        name,
        params,
        "local",
    )
    assert ctx.target == ("client" if name in {"read_range", "client_read"} else "server")
    assert ctx.call_id and ctx.iteration >= 1


@pytest.mark.parametrize("path", PATHS)
async def test_startup_write_failure_grid(store, engine, monkeypatch, path):
    provider = ProbeProvider()
    case = await path_case(store, engine, path, provider)
    before = await store.list_runs()
    old_status = (await store.get_run(case.run_id)).status if case.run_id else None
    save = store.save_run_event
    original = RuntimeError("grid_startup_write_failed")
    startup = "run.started" if path in {"run", "run_stream", "retry"} else "run.resumed"

    async def fail_start(run_id, **kwargs):
        if kwargs.get("event_type") == startup:
            raise original
        return await save(run_id, **kwargs)

    monkeypatch.setattr(store, "save_run_event", fail_start)
    if path in {"run_stream", "resume_stream"}:
        result = await case.invoke()
        assert result.status == RunStatus.ERROR
        assert case.events[-1].type == RunEventType.RUN_ERROR
    else:
        with pytest.raises(RuntimeError) as caught:
            await case.invoke()
        assert caught.value is original
    if path in {"run", "run_stream", "retry"}:
        new = [row for row in await store.list_runs() if row.id not in {r.id for r in before}]
        assert len(new) == 1 and new[0].status == "error"
    else:
        # An unconsumed submit/reclaim startup is retryable by the explicit
        # claim-release contract; direct resume owns ERROR finalization.
        assert (await store.get_run(case.run_id)).status == (
            old_status if path in SUBMIT_KINDS or path == "recovery" else "error"
        )
    assert provider.calls == 0


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
async def test_concurrent_submit_failure_releases_only_winning_epoch(
    store, engine, monkeypatch, kind
):
    from dendrux.errors import RunAlreadyClaimedError
    from dendrux.runtime.stores._factory import store_for_engine

    if engine.dialect.name != "postgresql":
        pytest.skip("Independent concurrent submitters require PostgreSQL")
    first, run_id, invoke = await submission_case(store, kind)
    second_store = store_for_engine(engine)
    second = Agent(
        name=first.name,
        prompt=first.prompt,
        tools=first.tools,
        require_approval=first.require_approval,
        state_store=second_store,
        provider=ScriptedLLM([LLMResponse(text="done")]),
        strict_tenancy=False,
    )
    before = await store.get_run(run_id)
    barrier = asyncio.Barrier(2)
    entered, release = asyncio.Event(), asyncio.Event()
    claims = []
    failure = RuntimeError("concurrent_setup_failure")

    async def fail_after_claim(*args, **kwargs):
        entered.set()
        await release.wait()
        raise failure

    def synchronized_claim(method):
        async def claim(*args, **kwargs):
            await barrier.wait()
            result = await method(*args, **kwargs)
            if result:
                claims.append(result)
            return result

        return claim

    with monkeypatch.context() as patcher:
        for backend in [store, second_store]:
            method_name = "claim_paused_run" if kind == "submit_resume" else "submit_and_claim"
            patcher.setattr(backend, method_name, synchronized_claim(getattr(backend, method_name)))
            patcher.setattr(backend, "get_pii_mapping", fail_after_claim)
        tasks = [asyncio.create_task(invoke(agent=owner)) for owner in [first, second]]
        try:
            await asyncio.wait_for(entered.wait(), 5)
            done, _ = await asyncio.wait(tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED)
            assert len(done) == 1
            assert isinstance(next(iter(done)).exception(), RunAlreadyClaimedError)
            assert len(claims) == 1, "exactly one submitter owns the epoch"
            owned = await store.get_run(run_id)
            assert owned.status == "running" and owned.claim_epoch == before.claim_epoch + 1
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(result is failure for result in results) == 1
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    released = await store.get_run(run_id)
    assert released.status == before.status
    assert released.claim_epoch == before.claim_epoch + 1
    assert (
        sum(
            event.event_type == "run.resume_aborted" for event in await store.get_run_events(run_id)
        )
        == 1
    )
    assert (await invoke()).status == RunStatus.SUCCESS
    final = await store.get_run(run_id)
    assert final.status == "success" and final.claim_epoch == before.claim_epoch + 2


@pytest.mark.parametrize("failure_site", ["_budget_carry_kwargs", "_resume_image_view"])
async def test_unconsumed_resume_argument_failure_releases(store, monkeypatch, failure_site):
    from dendrux.runtime import runner

    _, run_id, invoke = await submission_case(store, "submit_resume")
    original = RuntimeError("resume_argument_failure")
    with monkeypatch.context() as patcher:

        def fail(*args, **kwargs):
            raise original

        patcher.setattr(runner, failure_site, fail)
        with pytest.raises(RuntimeError) as caught:
            await invoke()
    assert caught.value is original
    assert (await store.get_run(run_id)).status == "paused"
    assert (await invoke()).status == RunStatus.SUCCESS


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
async def test_consumed_submission_keeps_error_handling(store, monkeypatch, kind):
    from dendrux.runtime import runner

    _, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    replay = runner._replay_resume_injected_history
    original = RuntimeError("post_consumption_failure")

    async def after_replay(*args, **kwargs):
        await replay(*args, **kwargs)
        raise original

    monkeypatch.setattr(runner, "_replay_resume_injected_history", after_replay)
    with pytest.raises(RuntimeError) as caught:
        await invoke()
    assert caught.value is original
    assert (await store.get_run(run_id)).status == (
        before.status if kind == "submit_resume" else "error"
    )


@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize(
    "kind,boundary",
    [
        ("tool_results", "consumed"),
        ("input", "consumed"),
        ("approve", "started"),
        ("decisions", "started"),
        ("approve", "executed"),
        ("decisions", "executed"),
    ],
)
async def test_submission_failure_after_durable_work_never_releases(
    store, monkeypatch, storage_format, kind, boundary
):
    from dendrux.loops import _helpers
    from dendrux.runtime import runner
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, storage_format)
    agent, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    released = AsyncMock(wraps=store.release_run_claim)
    original = RuntimeError("post_work_failure")
    effects, starts = [], []

    async def fail(*args, **kwargs):
        raise original

    if boundary == "consumed":
        replay = runner._replay_resume_injected_history

        async def after_replay(*args, **kwargs):
            await replay(*args, **kwargs)
            raise original

        monkeypatch.setattr(runner, "_replay_resume_injected_history", after_replay)
    else:
        from dendrux.loops import react as execute

        actual_execute = execute._execute_tool
        record_started = _helpers.record_tool_started

        async def count_start(*args, **kwargs):
            await record_started(*args, **kwargs)
            starts.append(True)

        async def count_execution(*args, **kwargs):
            result = await actual_execute(*args, **kwargs)
            effects.append(result)
            return result

        monkeypatch.setattr(execute, "_execute_tool", count_execution)
        monkeypatch.setattr(_helpers, "record_tool_started", count_start)
        monkeypatch.setattr(
            _helpers, "notify_tool_started" if boundary == "started" else "notify_tool", fail
        )
    monkeypatch.setattr(store, "release_run_claim", released)
    calls_before = agent._provider.calls
    with pytest.raises(RuntimeError) as caught:
        await invoke()
    assert caught.value is original
    released.assert_not_awaited()
    row = await store.get_run(run_id)
    assert row.status == "error" and row.claim_epoch == before.claim_epoch + 1
    assert agent._provider.calls == calls_before
    if boundary == "consumed":
        traces = await store.conversation_traces(run_id, after_position=None, limit=1000)
        assert (
            sum(
                trace.role == ("user" if kind == "input" else "tool")
                and (trace.content == "answer" if kind == "input" else trace.content == '"ok"')
                for trace in traces.entries
            )
            == 1
        )
    else:
        assert starts == [True]
        rows = await store.get_tool_calls(run_id)
        if storage_format == 1 and boundary == "started":
            # Legacy storage has no durable execution intent; the runner has
            # nevertheless crossed its tool-start boundary before this failure.
            assert rows == []
        else:
            assert len(rows) == 1
            assert rows[0].outcome == ("started" if boundary == "started" else "completed")
        assert len(effects) == (0 if boundary == "started" else 1)


async def paused_recovery(store, engine, *, before_tool=None):
    from functools import wraps

    effects = []

    @tool(idempotent=True)
    async def recovered_work() -> str:
        """Record an idempotent recovery dispatch."""
        effects.append(True)
        return "ok"

    agent = make_agent(
        store, [LLMResponse(text="done")], tools=[recovered_work], before_tool=before_tool
    )
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[ToolCall("recovered_work")],
        phases=["dispatchable"],
        started=True,
        idempotent=True,
    )
    reclaim = store.reclaim_run

    @wraps(reclaim)
    async def claim_then_pause(*args, **kwargs):
        claim = await reclaim(*args, **kwargs)
        await store.request_pause(item.run_id)
        return claim

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(store, "reclaim_run", claim_then_pause)
        paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status == RunStatus.PAUSED and not effects
    return agent, paused, effects


async def test_recovery_dispatch_failure_after_work_never_releases(store, engine, monkeypatch):
    from dendrux.runtime import _reclaim

    agent, paused, effects = await paused_recovery(store, engine)
    before = await store.get_run(paused.run_id)
    released = AsyncMock(wraps=store.release_run_claim)
    original = RuntimeError("after_recovery_dispatch")
    resolve = _reclaim.resolve

    async def after_resolve(*args, **kwargs):
        result = await resolve(*args, **kwargs)
        assert result is None and effects == [True]
        raise original

    monkeypatch.setattr(_reclaim, "resolve", after_resolve)
    monkeypatch.setattr(store, "release_run_claim", released)
    with pytest.raises(RuntimeError) as caught:
        await agent.submit_resume(
            paused.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
        )
    assert caught.value is original
    released.assert_not_awaited()
    row = await store.get_run(paused.run_id)
    assert row.status == "error" and row.claim_epoch == before.claim_epoch + 1
    assert effects == [True] and agent._provider.calls == 0


@pytest.mark.parametrize("allow", [True, False])
async def test_streaming_recovery_policy_receives_authorized_tenant(store, engine, allow):
    seen = []

    def policy(ctx):
        seen.append((ctx.tenant_id, ctx.run_id))
        return (
            ToolDecision.allow()
            if ctx.tenant_id == "tenant" and allow
            else ToolDecision.deny("tenant_policy_denied")
        )

    agent, paused, effects = await paused_recovery(store, engine, before_tool=policy)
    assert seen == []
    async with agent.resume_stream(
        paused.run_id, tenant_id="tenant", expected_pause_id=paused.pause_id
    ) as stream:
        events = [event async for event in stream]
    assert stream.result.status == RunStatus.SUCCESS
    assert events[0].type == RunEventType.RUN_RESUMED
    assert seen == [("tenant", paused.run_id)]
    assert effects == ([True] if allow else [])


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
async def test_caller_cancellation_during_claim_release_propagates_after_cleanup(
    store, monkeypatch, kind
):
    _, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    original = RuntimeError("setup_before_cancellation")
    entered, finish = asyncio.Event(), asyncio.Event()
    release = store.release_run_claim

    async def delayed_release(*args, **kwargs):
        entered.set()
        await finish.wait()
        return await release(*args, **kwargs)

    monkeypatch.setattr(store, "get_pii_mapping", AsyncMock(side_effect=original))
    released = AsyncMock(side_effect=delayed_release)
    monkeypatch.setattr(store, "release_run_claim", released)
    task = asyncio.create_task(invoke())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "caller cancellation must drain the fenced release"
        finish.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 5)
        assert caught.value.__context__ is original
        released.assert_awaited_once()
        row = await store.get_run(run_id)
        assert row.status == before.status and row.claim_epoch == before.claim_epoch + 1
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
@pytest.mark.parametrize("storage_format", [1, 2])
@pytest.mark.parametrize("failure_site", ["get_pii_mapping", "_start_trace"])
async def test_released_submit_reports_abort_without_failure_hooks(
    store, engine, monkeypatch, kind, storage_format, failure_site
):
    from dendrux.runtime import runner
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.store import RunStore
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(engine, storage_format)
    agent, run_id, invoke = await submission_case(store, kind)
    before = await store.get_run(run_id)
    original = RuntimeError("private_abort_sentinel")
    recorder_failed = AsyncMock()
    notifier = BaseNotifier()
    notifier.on_run_failed = AsyncMock()
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", recorder_failed)
    target = store if failure_site == "get_pii_mapping" else runner
    with monkeypatch.context() as patcher:
        patcher.setattr(target, failure_site, AsyncMock(side_effect=original))
        with pytest.raises(RuntimeError) as caught:
            await invoke(notifier=notifier)
    assert caught.value is original and original.__dict__ == {}
    recorder_failed.assert_not_awaited()
    notifier.on_run_failed.assert_not_awaited()
    row = await store.get_run(run_id)
    assert row.status == before.status and row.claim_epoch == before.claim_epoch + 1
    events = await store.get_run_events(run_id)
    aborted = [event for event in events if event.event_type == "run.resume_aborted"]
    assert len(aborted) == 1
    assert aborted[0].data == {"failure_kind": "setup_error"}
    resumed = [event for event in events if event.event_type == "run.resumed"]
    if failure_site == "_start_trace":
        assert resumed and resumed[-1].sequence_index < aborted[0].sequence_index
    reader = RunStore.from_engine(engine)
    try:
        raw = next(
            event
            for event in await reader.get_events(run_id, tenant_id="tenant")
            if event.event_type == "run.resume_aborted"
        )
        assert (await reader.get_event_view(run_id, raw, tenant_id="tenant")).data == raw.data
        feed = reader.follow(run_id, tenant_id="tenant", until_terminal=False)
        try:
            shown = []
            async for event in feed:
                shown.append(event)
                if event.sequence_index == raw.sequence_index:
                    break
        finally:
            await feed.aclose()
        assert shown[-1].event_type == "run.resume_aborted"
        assert shown[-1].data == {"failure_kind": "setup_error"}
        assert "private_abort_sentinel" not in json.dumps([event.data for event in shown])
    finally:
        await reader.close()
    assert (await invoke()).status == RunStatus.SUCCESS
    assert (
        sum(
            event.event_type == "run.resume_aborted" for event in await store.get_run_events(run_id)
        )
        == 1
    )


async def test_recovery_approval_abort_follows_decision_and_skips_failure_hooks(
    store, engine, monkeypatch
):
    from dendrux.runtime.persistence import PersistenceRecorder

    effects = []

    @tool()
    async def recovery_effect() -> str:
        """Record a non-idempotent recovery execution."""
        effects.append(True)
        return "ok"

    agent = make_agent(store, [LLMResponse(text="done")], tools=[recovery_effect])
    item = await checkpoint(
        agent,
        store,
        engine,
        calls=[ToolCall("recovery_effect")],
        phases=["dispatchable"],
        started=True,
    )
    paused = await agent.reclaim(item.run_id, tenant_id="tenant", item=item)
    assert paused.status == RunStatus.WAITING_APPROVAL
    notifier = BaseNotifier()
    notifier.on_run_failed = AsyncMock()
    recorder_failed = AsyncMock()
    original = RuntimeError("private_recovery_abort_sentinel")
    monkeypatch.setattr(PersistenceRecorder, "on_run_failed", recorder_failed)
    monkeypatch.setattr(store, "consume_submission", AsyncMock(side_effect=original))
    with pytest.raises(RuntimeError) as caught:
        await agent.submit_approval(
            paused.run_id,
            approved=True,
            tenant_id="tenant",
            expected_pause_id=paused.pause_id,
            notifier=notifier,
        )
    assert caught.value is original
    recorder_failed.assert_not_awaited()
    notifier.on_run_failed.assert_not_awaited()
    events = await store.get_run_events(paused.run_id)
    decided = [event for event in events if event.event_type == "approval.decided"]
    aborted = [event for event in events if event.event_type == "run.resume_aborted"]
    assert len(aborted) == 1 and decided
    assert decided[-1].sequence_index < aborted[0].sequence_index
    assert (await store.get_run(paused.run_id)).status == "waiting_approval"
    assert not effects


async def test_resume_abort_event_and_release_roll_back_together(store, monkeypatch, caplog):
    from dendrux.runtime import runner
    from dendrux.runtime.stores.sql import base

    _, run_id, invoke = await submission_case(store, "input")
    original = RuntimeError("abort_transaction_original")
    write = base._write_event

    async def reject_abort(session, event_run_id, event):
        if event.event_type == "run.resume_aborted":
            raise RuntimeError("abort_transaction_write_failure")
        return await write(session, event_run_id, event)

    monkeypatch.setattr(runner, "_start_trace", AsyncMock(side_effect=original))
    monkeypatch.setattr(base, "_write_event", reject_abort)
    with pytest.raises(RuntimeError) as caught:
        await invoke()
    assert caught.value is original
    assert (await store.get_run(run_id)).status == "running"
    assert not any(
        event.event_type == "run.resume_aborted" for event in await store.get_run_events(run_id)
    )
    assert any(
        getattr(record, "code", None) == "persistence_error"
        and getattr(record, "run_id", None) == run_id
        for record in caplog.records
    )


async def test_stale_resume_release_cannot_publish_abort(store):
    from dendrux.runtime.state import EventWrite
    from dendrux.types import generate_ulid

    _, run_id, _ = await submission_case(store, "input")
    before = await store.get_run(run_id)
    claimed = await store.claim_paused_run(run_id, expected_status=before.status)
    assert claimed
    event = EventWrite(
        generate_ulid(), "run.resume_aborted", 0, {"failure_kind": "setup_error"}, None, None
    )
    assert not await store.release_run_claim(
        run_id, status=before.status, claim_epoch=before.claim_epoch, event=event
    )
    row = await store.get_run(run_id)
    assert row.status == "running" and row.claim_epoch == before.claim_epoch + 1
    assert not any(
        item.event_type == "run.resume_aborted" for item in await store.get_run_events(run_id)
    )
    assert await store.release_run_claim(
        run_id, status=before.status, claim_epoch=before.claim_epoch + 1
    )


@pytest.mark.parametrize("failure_kind", ["error", "cancellation"])
async def test_retry_failure_before_creation_finalizes_nothing(store, monkeypatch, failure_kind):
    from dendrux.runtime import runner

    a = make_agent(store, [LLMResponse(text="prior")])
    original = await a.run("go")
    failure = (
        RuntimeError("before_creation")
        if failure_kind == "error"
        else asyncio.CancelledError("before_creation")
    )
    finalize = AsyncMock()
    monkeypatch.setattr(store, "finalize_run", finalize)
    monkeypatch.setattr(runner, "_create_runtime_run", AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)) as caught:
        await a.retry(original.run_id)
    assert caught.value is failure
    finalize.assert_not_awaited()
    assert [row.id for row in await store.list_runs()] == [original.run_id]


async def test_unknown_consumption_outcome_is_not_released(store, monkeypatch):
    from dendrux.errors import StoreOutcomeUnknown

    _, run_id, invoke = await submission_case(store, "tool_results")
    original = StoreOutcomeUnknown()
    release = AsyncMock(wraps=store.release_run_claim)
    monkeypatch.setattr(store, "consume_submission", AsyncMock(side_effect=original))
    monkeypatch.setattr(store, "release_run_claim", release)
    with pytest.raises(StoreOutcomeUnknown) as caught:
        await invoke()
    assert caught.value is original
    release.assert_not_awaited()
    assert (await store.get_run(run_id)).status == "error"


async def test_retry_intentionally_starts_fresh_configuration_and_usage(store):
    from dendrux.llm.mock import MockLLM
    from dendrux.types import UsageStats

    prior = await make_agent(
        store,
        [LLMResponse(text="prior", usage=UsageStats(input_tokens=40, output_tokens=20))],
        max_iterations=1,
    ).run("go", temperature=0.9)
    provider = MockLLM(
        [LLMResponse(text="fresh", usage=UsageStats(input_tokens=7, output_tokens=3))],
        model="fresh",
    )
    a = Agent(
        name="cross-path",
        prompt="fresh prompt",
        provider=provider,
        state_store=store,
        max_iterations=3,
        strict_tenancy=False,
    )
    result = await a.retry(prior.run_id, temperature=0.2)
    assert result.status == RunStatus.SUCCESS
    assert (result.usage.input_tokens, result.usage.output_tokens) == (7, 3)
    assert provider.call_history[0]["kwargs"]["temperature"] == 0.2
    row = await store.get_run(result.run_id)
    assert row.model == "fresh" and row.retry_of_run_id == prior.run_id
    assert row.total_input_tokens == 7


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_completion_beats_caller_cancellation_during_finish(store, entry):
    original = await make_agent(store, [LLMResponse(text="prior")]).run("go")
    a = make_agent(store, [LLMResponse(text="done")])
    finishing = asyncio.Event()
    notifier = BaseNotifier()
    identities = []

    async def finish(run_id, result):
        identities.append((run_id, result.status, asyncio.current_task()))
        finishing.set()
        await asyncio.Event().wait()

    notifier.on_run_finished = AsyncMock(side_effect=finish)
    task = asyncio.create_task(
        a.retry(original.run_id, notifier=notifier)
        if entry == "retry"
        else a.run("go", notifier=notifier)
    )
    try:
        await asyncio.wait_for(finishing.wait(), 5)
        task.cancel("late_cancellation")
        with pytest.raises(asyncio.CancelledError):
            await task
        run_id, status, hook_task = identities[0]
        assert status == RunStatus.SUCCESS and hook_task is task
        assert (await store.get_run(run_id)).status == "success"
        notifier.on_run_finished.assert_awaited_once()
        assert not any(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id))
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind", SUBMIT_KINDS)
async def test_overlapping_same_agent_submit_preserves_new_owner_and_interrupts(
    store, monkeypatch, kind
):
    from functools import wraps

    from dendrux.runtime import runner

    a, run_id, invoke = await submission_case(store, kind)
    provider = a._provider
    provider.gate_at = provider.calls
    failed, finish_failure = asyncio.Event(), asyncio.Event()
    notifier = BaseNotifier()
    original = RuntimeError("overlapping_setup_failure")
    start_trace = runner._start_trace
    failed_once = False

    release_claim = store.release_run_claim

    @wraps(release_claim)
    async def delay_release(*args, **kwargs):
        result = await release_claim(*args, **kwargs)
        if result:
            failed.set()
            await finish_failure.wait()
        return result

    async def fail_first_trace(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise original
        return await start_trace(*args, **kwargs)

    notifier.on_run_failed = AsyncMock()
    monkeypatch.setattr(runner, "_start_trace", fail_first_trace)
    monkeypatch.setattr(store, "release_run_claim", delay_release)
    before = await store.get_run(run_id)
    first = asyncio.create_task(invoke(notifier=notifier))
    second = None
    try:
        await asyncio.wait_for(failed.wait(), 5)
        released = await store.get_run(run_id)
        assert released.status == before.status
        assert released.claim_epoch == before.claim_epoch + 1
        second = asyncio.create_task(invoke())
        await asyncio.wait_for(provider.entered.wait(), 5)
        new_owner = a._task_manager._owned[run_id]
        assert a._task_manager._tasks[run_id] is new_owner
        finish_failure.set()
        with pytest.raises(RuntimeError) as caught:
            await first
        assert caught.value is original
        notifier.on_run_failed.assert_not_awaited()
        assert a._task_manager._tasks.get(run_id) is new_owner
        assert a._task_manager._owned.get(run_id) is new_owner
        assert not new_owner.done()
        await a.cancel_run(run_id, tenant_id="tenant")
        done, _ = await asyncio.wait({second}, timeout=2)
        assert done, "local cancellation must interrupt the new owner"
        result = await second
        assert result.status == RunStatus.CANCELLED
        assert result.meta["interrupted"] is True
        assert provider.cancelled
        row = await store.get_run(run_id)
        assert row.status == "cancelled" and row.claim_epoch == before.claim_epoch + 2
        assert sum(e.event_type == "run.cancelled" for e in await store.get_run_events(run_id)) == 1
        assert run_id not in a._task_manager._tasks
        assert run_id not in a._task_manager._owned
    finally:
        finish_failure.set()
        provider.gate.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None), return_exceptions=True
        )
        await a.close()
