"""``RunStore.follow`` cost and fairness at scale (SQLite + Postgres).

- Replaying a run's history costs each follower a small, fixed number of
  statements, however many past pauses the run had: the active-pause lookup
  is shared by every follower of the run, not repeated per event.
- One tenant cannot take every follower slot: ``EventFeed`` has a
  per-tenant cap (a fifth of ``max_followers`` by default, overridable).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import contextvars
import functools
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event

from dendrux.agent import Agent
from dendrux.store import (
    UNSCOPED,
    EventFeed,
    FollowCapacityError,
    FollowEnd,
    FollowHeartbeat,
    RunStore,
    _follow,
)
from dendrux.tool import tool
from dendrux.types import LLMResponse, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.asyncio


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool()
async def gated(n: int) -> str:
    return "ok"


async def _run_with_pauses(store: Any, pauses: int) -> str:
    """A finished run with ``pauses`` client pauses and ``pauses`` approval pauses."""
    script = []
    for i in range(pauses):
        script.append(tool_call("read_range", {"sheet": f"S{i}"}, pid=f"c{i}"))
        script.append(tool_call("gated", {"n": i}, pid=f"g{i}"))
    script.append(LLMResponse(text="done"))
    agent = Agent(
        provider=ScriptedLLM(script),
        prompt="t",
        state_store=store,
        tools=[read_range, gated],
        require_approval=["gated"],
        strict_tenancy=False,
        max_iterations=100,
    )
    result = await agent.run("go", tenant_id="A")
    while result.status.value != "success":
        assert result.pending is not None
        if result.pending.kind == "client_tool":
            call = result.pending.tool_calls[0]
            result = await agent.submit_tool_results(
                result.run_id,
                [ToolResult(name="read_range", call_id=call.id, payload='"x"')],
                tenant_id="A",
                expected_pause_id=result.pause_id,
            )
        else:
            result = await agent.submit_approval(
                result.run_id, approved=True, tenant_id="A", expected_pause_id=result.pause_id
            )
    return result.run_id


# Reads the followers of one run share. Concurrent followers race to run each
# one first (a follower that arrives while another is still reading pays it
# too), so how many of these statements run depends on scheduling and load,
# not on the run. The counter keeps them in their own bucket, one per reader.
_SHARED_READERS = (
    "_current_head",
    "read_pause_view",
    "read_tool_evidence",
    "read_declared_tools",
    "read_unknown_calls",
)
_reader: contextvars.ContextVar[str] = contextvars.ContextVar("follow_scale_reader", default="")
_FOLLOWER = "follower"


def _tagged(name: str, read: Any) -> Any:
    @functools.wraps(read)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        token = _reader.set(name)
        try:
            return await read(*args, **kwargs)
        finally:
            _reader.reset(token)

    return wrapped


@contextlib.contextmanager
def _counted_statements(engine: Any) -> Iterator[collections.Counter[str]]:
    """Statements the followers issue, by bucket.

    ``follower`` holds each follower's own work (scope check, event pages,
    argument pages); every shared reader has its own bucket. The store's
    coordinator (a named task) is not counted: its work grows with elapsed time.
    """
    counts: collections.Counter[str] = collections.Counter()

    def on_execute(*_a: Any) -> None:
        task = asyncio.current_task()
        if task is not None and task.get_name().startswith("dendrux-follow-coordinator"):
            return
        counts[_reader.get() or _FOLLOWER] += 1

    with pytest.MonkeyPatch.context() as patch:
        for name in _SHARED_READERS:
            patch.setattr(_follow, name, _tagged(name, getattr(_follow, name)))
        event.listen(engine.sync_engine, "before_cursor_execute", on_execute)
        try:
            yield counts
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", on_execute)


async def _statements_for(
    engine: Any, run_store: RunStore, run_id: str, followers: int
) -> collections.Counter[str]:
    feed = EventFeed(run_store, poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=5.0)

    async def one() -> int:
        seen = 0
        async with feed.follow(run_id, tenant_id="A", max_duration_s=20.0) as f:
            async for item in f:
                if isinstance(item, FollowEnd):
                    assert item.reason == "terminal"
                    break
                if not isinstance(item, FollowHeartbeat):
                    seen += 1
        return seen

    with _counted_statements(engine) as counts:
        try:
            seen = await asyncio.gather(*[one() for _ in range(followers)])
        finally:
            await feed.aclose()
    assert len(set(seen)) == 1 and seen[0] > 0
    return counts


async def test_backfill_cost_per_follower_is_flat(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    small = await _run_with_pauses(store, 1)
    large = await _run_with_pauses(store, 6)
    followers = 20
    small_cost = await _statements_for(engine, run_store, small, followers)
    large_cost = await _statements_for(engine, run_store, large, followers)
    print(f"statements: {followers} followers, 2 vs 12 past pauses: {small_cost} vs {large_cost}")
    for cost in (small_cost, large_cost):
        # A follower's own work is fixed: scope check, drained check, and one
        # events batch (the history fits in two). The head, poller and
        # active-pause reads are shared by concurrent followers; a follower
        # that arrives after the feed closed pays each again, never more than
        # once. So no shared reader exceeds one statement per follower.
        assert cost[_FOLLOWER] <= 3 * followers, cost
        for name in _SHARED_READERS:
            assert cost[name] <= followers, (name, cost)
    # Twelve more past pauses add no statements per follower.
    # Per-pause reads would add at least 10 * followers = 200.
    assert large_cost[_FOLLOWER] == small_cost[_FOLLOWER], (small_cost, large_cost)


# ---------------------------------------------------------------------------
# Per-tenant follower cap
# ---------------------------------------------------------------------------


async def _paused_run(store: Any, tenant: str | None) -> str:
    agent = Agent(
        provider=ScriptedLLM(
            [tool_call("read_range", {"sheet": "S"}, pid="c"), LLMResponse(text="d")]
        ),
        prompt="t",
        state_store=store,
        tools=[read_range],
        strict_tenancy=False,
    )
    return (await agent.run("go", tenant_id=tenant)).run_id


async def test_one_tenant_cannot_take_every_slot(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    run_a = await _paused_run(store, "A")
    run_b = await _paused_run(store, "B")
    feed = EventFeed(run_store, max_followers=10, max_followers_per_tenant=4)
    held = []
    try:
        for _ in range(4):
            f = feed.follow(run_a, tenant_id="A")
            await f.start()
            held.append(f)
        hostile = feed.follow(run_a, tenant_id="A")
        with pytest.raises(FollowCapacityError) as info:
            await hostile.start()
        assert info.value.limit == 4 and info.value.per_tenant is True
        assert feed.followers == 4
        victim = feed.follow(run_b, tenant_id="B")
        await victim.start()
        held.append(victim)
        await held.pop(0).aclose()  # a slot freed by A is A's again
        again = feed.follow(run_a, tenant_id="A")
        await again.start()
        held.append(again)
    finally:
        for f in held:
            await f.aclose()
    assert feed.followers == 0


async def test_default_cap_is_a_fifth_of_the_feed() -> None:
    run_store = RunStore(None)  # type: ignore[arg-type]  # no reads needed
    assert EventFeed(run_store).max_followers_per_tenant == 100
    assert EventFeed(run_store, max_followers=3).max_followers_per_tenant == 1
    assert EventFeed(run_store, max_followers_per_tenant=None).max_followers_per_tenant is None


@pytest.mark.parametrize("scope", [None, UNSCOPED], ids=["untenanted", "unscoped"])
async def test_untenanted_and_unscoped_scopes_are_capped_like_a_tenant(
    run_store: RunStore, store: Any, engine: Any, scope: Any
) -> None:
    """The cap is per scope: ``None`` (every user of a single-tenant app) and
    ``UNSCOPED`` are each one scope, so they cannot take every slot either."""
    run_none = await _paused_run(store, None)
    run_b = await _paused_run(store, "B")
    feed = EventFeed(run_store, max_followers=10, max_followers_per_tenant=4)
    held = []
    try:
        for _ in range(4):
            f = feed.follow(run_none, tenant_id=scope)
            await f.start()
            held.append(f)
        with pytest.raises(FollowCapacityError) as info:
            await feed.follow(run_none, tenant_id=scope).start()
        assert info.value.per_tenant is True and info.value.limit == 4
        victim = feed.follow(run_b, tenant_id="B")
        await victim.start()
        held.append(victim)
    finally:
        for f in held:
            await f.aclose()
    assert feed.followers == 0 and feed._by_scope == {}


async def test_single_tenant_apps_keep_an_override(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    """``max_followers_per_tenant=None`` lets one scope use the whole feed again."""
    run_none = await _paused_run(store, None)
    feed = EventFeed(run_store, max_followers=6, max_followers_per_tenant=None)
    held = []
    try:
        for _ in range(6):
            f = feed.follow(run_none, tenant_id=None)
            await f.start()
            held.append(f)
        with pytest.raises(FollowCapacityError) as info:
            await feed.follow(run_none, tenant_id=None).start()
        assert info.value.per_tenant is False and info.value.limit == 6
    finally:
        for f in held:
            await f.aclose()


async def test_concurrent_untenanted_flood_leaves_room_for_tenants(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    """60 untenanted followers start at once: exactly the cap (20) get a slot,
    the rest fail with the per-scope error, and tenant B still gets in."""
    run_none = await _paused_run(store, None)
    run_b = await _paused_run(store, "B")
    feed = EventFeed(run_store, max_followers=100, max_followers_per_tenant=20)
    follows = [feed.follow(run_none, tenant_id=None) for _ in range(60)]
    results = await asyncio.gather(*(f.start() for f in follows), return_exceptions=True)
    started = [f for f, r in zip(follows, results, strict=True) if r is None]
    refused = [r for r in results if isinstance(r, FollowCapacityError)]
    assert len(started) == 20 and len(refused) == 40
    assert all(r.per_tenant for r in refused)
    victim = feed.follow(run_b, tenant_id="B")
    await victim.start()
    for f in [*started, victim]:
        await f.aclose()
    assert feed.followers == 0 and feed._by_scope == {}


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
async def test_per_tenant_cap_is_validated(engine: Any, bad: Any) -> None:
    with pytest.raises(ValueError):
        EventFeed(RunStore.from_engine(engine), max_followers_per_tenant=bad)


# ---------------------------------------------------------------------------
# Trusted (include_payloads) followers: past pause arguments are read in pages
# ---------------------------------------------------------------------------


async def _many_approval_pauses(store: Any, pauses: int) -> str:
    run_id = "rp".ljust(26, "0")
    await store.create_run(run_id, "a", tenant_id="A")
    for i in range(pauses):
        call = f"c{i}"
        await store.save_run_event(
            run_id,
            event_type="approval.requested",
            data={"tool_name": "t", "call_id": call, "params": {"n": i}},
        )
        await store.save_run_event(
            run_id,
            event_type="run.paused",
            data={"status": "waiting_approval", "pending_tool_calls": [{"id": call, "name": "t"}]},
        )
        await store.save_run_event(run_id, event_type="run.resumed", data={})
    await store.finalize_run(run_id, status="success")
    return run_id


async def test_trusted_backfill_reads_past_arguments_in_pages(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    pauses, followers = 300, 5
    run_id = await _many_approval_pauses(store, pauses)
    feed = EventFeed(run_store, poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=5.0)

    async def one() -> dict[int, Any]:
        shown: dict[int, Any] = {}
        async with feed.follow(
            run_id, tenant_id="A", include_payloads=True, max_duration_s=30.0
        ) as f:
            async for item in f:
                if isinstance(item, FollowEnd):
                    break
                if not isinstance(item, FollowHeartbeat) and item.event_type == "run.paused":
                    shown[item.sequence_index] = item.data["pending_tool_calls"][0]["params"]
        return shown

    with _counted_statements(engine) as count:
        try:
            results = await asyncio.gather(*[one() for _ in range(followers)])
        finally:
            await feed.aclose()
    for shown in results:
        assert len(shown) == pauses
        assert [p["n"] for _, p in sorted(shown.items())] == list(range(pauses))
    events = 3 * pauses + 1
    print(f"trusted backfill: {followers} followers, {pauses} pauses -> {dict(count)}")
    # Per follower: about one statement per 100 events (event pages) plus one per
    # 1000 pause-related events (argument pages), never one per pause.
    assert count[_FOLLOWER] <= followers * (events // 100 + events // 1000 + 8), count
    # Shared reads run at most once per follower, whatever the interleaving.
    for name in _SHARED_READERS:
        assert count[name] <= followers, (name, count)


# ---------------------------------------------------------------------------
# One Follow holds at most one slot, whatever start()/aclose() interleave
# ---------------------------------------------------------------------------


def _counts(feed: EventFeed) -> tuple[int, dict[Any, int]]:
    return feed.followers, dict(feed._by_scope)


async def test_concurrent_starts_on_one_follow_take_one_slot(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    run_none = await _paused_run(store, None)
    feed = EventFeed(run_store, max_followers=10, max_followers_per_tenant=5)
    follow = feed.follow(run_none, tenant_id=None)
    await asyncio.gather(*(follow.start() for _ in range(5)))
    assert _counts(feed) == (1, {("untenanted",): 1})
    await follow.aclose()
    assert _counts(feed) == (0, {})


async def test_close_during_an_in_flight_start_takes_no_slot(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    run_none = await _paused_run(store, None)
    feed = EventFeed(run_store, max_followers=10)
    gate, entered = asyncio.Event(), asyncio.Event()
    original = run_store._status

    async def slow_status(*args: Any) -> Any:
        entered.set()
        await gate.wait()
        return await original(*args)

    run_store._status = slow_status  # type: ignore[method-assign]
    follow = feed.follow(run_none, tenant_id=UNSCOPED)
    starting = asyncio.create_task(follow.start())
    await entered.wait()
    await follow.aclose()  # closes while start() waits on the store
    gate.set()
    with pytest.raises(RuntimeError, match="closed"):
        await starting
    await follow.aclose()
    assert _counts(feed) == (0, {})


async def test_close_after_start_releases_the_slot(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    run_t = await _paused_run(store, "T")
    feed = EventFeed(run_store, max_followers=10)
    follow = feed.follow(run_t, tenant_id="T")
    await follow.start()
    assert _counts(feed) == (1, {("tenant", "T"): 1})
    await asyncio.gather(follow.aclose(), follow.aclose())
    assert _counts(feed) == (0, {})


@pytest.mark.parametrize("followers", [1, 8])
async def test_reader_close_drains_read_after_last_follower_closes(
    run_store: RunStore, store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch, followers: int
) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    entered, cancelling, release, finished = (asyncio.Event() for _ in range(4))
    execute = AsyncSession.execute

    async def held_read(session: Any, *args: Any, **kwargs: Any) -> Any:
        result = await execute(session, *args, **kwargs)
        task = asyncio.current_task()
        if task is not None and task.get_name() == "dendrux-follow-coordinator":
            entered.set()  # a real query ran; the session still owns its connection
            try:
                await asyncio.Event().wait()
            finally:
                cancelling.set()
                await release.wait()
                finished.set()
        return result

    run_id = await _paused_run(store, "T")
    monkeypatch.setattr(AsyncSession, "execute", held_read)
    ready = asyncio.Event()
    main = run_store._follow_coordinator()._main

    async def after_starts() -> None:
        await ready.wait()
        await main()

    monkeypatch.setattr(run_store._follow_coordinator(), "_main", after_starts)
    feed = EventFeed(run_store, max_followers_per_tenant=None)
    follows = [feed.follow(run_id, tenant_id="T") for _ in range(followers)]
    closing = None
    try:
        # All starts precede the gated read, including on single-connection SQLite.
        await asyncio.gather(*(follow.start() for follow in follows))
        ready.set()
        await asyncio.wait_for(entered.wait(), 10)
        coordinator = run_store._follow_coordinator()._task
        assert coordinator is not None
        await asyncio.gather(*(follow.aclose() for follow in follows))
        assert feed.followers == 0
        closing = asyncio.create_task(run_store.close())
        await asyncio.wait_for(cancelling.wait(), 10)
        assert not closing.done() and not finished.is_set()
        release.set()
        await asyncio.wait_for(closing, 10)
        assert coordinator.done() and finished.is_set()
        # The caller-owned engine remains usable, with no coordinator holding it.
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT 1")) == 1
    finally:
        release.set()
        await asyncio.gather(*(follow.aclose() for follow in follows))
        await run_store.close()
        if closing is not None:
            await closing


async def test_random_start_close_interleavings_never_leak(
    run_store: RunStore, store: Any, engine: Any
) -> None:
    """200 random start/aclose interleavings over shared Follow objects: the
    global and per-scope counters never exceed their caps and return to 0."""
    import random

    run_none = await _paused_run(store, None)
    run_t = await _paused_run(store, "T")
    rng = random.Random(1234)
    feed = EventFeed(run_store, max_followers=8, max_followers_per_tenant=3)
    scopes: list[Any] = [None, UNSCOPED, "T"]
    follows = [
        feed.follow(run_t if scope == "T" else run_none, tenant_id=scope)
        for scope in (rng.choice(scopes) for _ in range(30))
    ]

    async def op(follow: Any) -> None:
        await asyncio.sleep(rng.random() / 200)
        if rng.random() < 0.6:
            with contextlib.suppress(FollowCapacityError, RuntimeError):
                await follow.start()
        else:
            await follow.aclose()
        total, buckets = _counts(feed)
        assert total <= 8 and all(n <= 3 for n in buckets.values())
        assert total == sum(buckets.values())

    await asyncio.gather(*(op(rng.choice(follows)) for _ in range(200)))
    await asyncio.gather(*(f.aclose() for f in follows))
    assert _counts(feed) == (0, {})
