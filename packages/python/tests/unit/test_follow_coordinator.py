"""Spec §4.2: one coordinator per RunStore reads every watched run in batches.

A fake state store (batched reads optional) and a fake loop clock make the
schedule deterministic: nothing here waits for wall time. ``FakeLive`` is a
real ``LiveUpdates`` whose transport is an in-memory queue, so the lease, hub
and health logic under test are the shipped ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux.db._live_updates import LiveUpdates
from dendrux.runtime.state import UNSCOPED
from dendrux.store import EventFeed, FollowEnd, RunStore, StoredEvent, _follow, _follow_recovery

TERMINAL = "run.completed"


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
async def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    loop = asyncio.get_running_loop()
    fake = Clock(1000.0)
    monkeypatch.setattr(loop, "time", lambda: fake.now)
    return fake


KEEP: list[Any] = []  # follows a test must keep alive (a dropped Follow detaches)


@pytest.fixture(autouse=True)
def fast_budget(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Tests that measure the budget set it themselves."""
    monkeypatch.setattr(_follow, "MAX_READS_PER_S", 1e9, raising=False)
    yield
    KEEP.clear()


async def ticks(count: int = 20) -> None:
    for _ in range(count):
        await asyncio.sleep(0)


class Crash(BaseException):
    """Escapes per-chunk failure handling, like a bug in the coordinator."""


class FakeState:
    """A per-run state store; ``BatchedState`` adds the §4.3 batched reads."""

    def __init__(self) -> None:
        self.events: dict[str, list[SimpleNamespace]] = {}
        self.statuses: dict[str, str] = {}
        self.tenants: dict[str, str | None] = {}
        self.event_reads: list[tuple[str, ...]] = []
        self.status_reads: list[tuple[str, ...]] = []
        self.status_times: list[float] = []
        self.per_run_status_reads: list[str] = []
        self.limits: list[int] = []
        self.gate: asyncio.Event | None = None
        self.reading = asyncio.Event()
        self.fail_runs: set[str] = set()
        self.crash = 0
        self.after_events_read: Any = None
        self.before_events_read: Any = None
        self.live_updates: LiveUpdates | None = None

    def create(self, run_id: str, *, tenant: str | None = None, status: str = "running") -> None:
        self.events[run_id] = []
        self.statuses[run_id] = status
        self.tenants[run_id] = tenant

    def write(self, run_id: str, event_type: str = "tick") -> int:
        seq = len(self.events[run_id])
        self.events[run_id].append(
            SimpleNamespace(
                sequence_index=seq,
                iteration_index=0,
                event_type=event_type,
                correlation_id=None,
                data={},
                created_at=None,
            )
        )
        return seq

    def _rows(self, run_id: str, after: int | None, limit: int | None) -> list[SimpleNamespace]:
        rows = [e for e in self.events.get(run_id, []) if after is None or e.sequence_index > after]
        return rows[:limit] if limit is not None else rows

    async def _events_gate(self, run_ids: tuple[str, ...]) -> None:
        if self.before_events_read is not None:
            self.before_events_read(run_ids)
        self.event_reads.append(run_ids)
        self.reading.set()
        if self.crash:
            self.crash -= 1
            raise Crash
        await asyncio.sleep(0)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_runs & set(run_ids):
            raise OSError("driver text that must never be logged")

    async def get_run_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
        event_types: Any = None,
        before: Any = None,
        tenant_id: Any = UNSCOPED,
    ) -> list[SimpleNamespace]:
        if not _in_scope(self, run_id, tenant_id):
            return []
        rows = self._rows(run_id, after_sequence_index, limit)  # read before any gate
        if limit != 1:  # the start check's one-row probe is not a coordinator read
            await self._events_gate((run_id,))
        if self.after_events_read is not None:
            self.after_events_read(run_id)
        return rows

    async def get_run_status(self, run_id: str, *, tenant_id: Any = UNSCOPED) -> str | None:
        if tenant_id is UNSCOPED:  # the coordinator's per-run status read
            self.per_run_status_reads.append(run_id)
        if not _in_scope(self, run_id, tenant_id):
            return None
        return self.statuses.get(run_id)

    async def get_next_event_sequence(self, run_id: str) -> int:
        return len(self.events.get(run_id, []))

    async def get_run(self, run_id: str) -> SimpleNamespace | None:
        if run_id not in self.statuses:
            return None
        return SimpleNamespace(tenant_id=self.tenants[run_id], status=self.statuses[run_id])


def _in_scope(state: FakeState, run_id: str, tenant_id: Any) -> bool:
    if run_id not in state.statuses:
        return False
    return tenant_id is UNSCOPED or state.tenants[run_id] == tenant_id


class BatchedState(FakeState):
    async def get_events_for_runs(
        self, heads: dict[str, int], *, limit_per_run: int
    ) -> dict[str, list[SimpleNamespace]]:
        assert 1 <= len(heads) <= 200 and 1 <= limit_per_run <= 100
        self.limits.append(limit_per_run)
        result = {  # the snapshot is taken before any gate, like a real query
            run_id: self._rows(run_id, None if head < 0 else head, limit_per_run)
            for run_id, head in heads.items()
            if run_id in self.statuses
        }
        await self._events_gate(tuple(heads))
        if self.after_events_read is not None:
            for run_id in heads:
                self.after_events_read(run_id)
        return result

    async def get_run_statuses(self, run_ids: Any) -> dict[str, str]:
        assert len(run_ids) <= 200
        self.status_reads.append(tuple(run_ids))
        self.status_times.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0)
        return {r: self.statuses[r] for r in run_ids if r in self.statuses}


class FakeLive(LiveUpdates):
    """An in-memory bus: ``hint`` and ``gap`` feed ``receive`` directly."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()

    async def send(self, messages: Any) -> None:
        for message in messages:
            self.inbox.put_nowait(message)

    async def receive(self) -> Any:
        yield None
        while True:
            yield await self.inbox.get()

    def hint(self, *run_ids: str) -> None:
        self.inbox.put_nowait(json.dumps(list(run_ids)))

    def gap(self) -> None:
        self.inbox.put_nowait(LiveUpdates.GAP)


def _store(state: FakeState) -> RunStore:
    return RunStore(state)  # type: ignore[arg-type]


def _feed(store: RunStore, **kw: Any) -> EventFeed:
    params: dict[str, Any] = {"max_followers": 5_000, "max_followers_per_tenant": None}
    params.update(kw)
    return EventFeed(store, **params)


async def attach(feed: EventFeed, run_id: str, **kw: Any) -> Any:
    """Start a follow; the coordinator watches the run from ``start()`` on."""
    follow = feed.follow(run_id, tenant_id=kw.pop("tenant_id", None), **kw)
    await follow.start()
    assert follow._follower is not None, "the follower must be attached at start"
    KEEP.append(follow)  # a dropped Follow detaches when collected
    return follow


def _seqs(follow: Any) -> list[int]:
    return [e.sequence_index for e in follow._follower.buffer]


def _coordinator_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith("dendrux-follow-coordinator") and not t.done()
    ]


async def _healthy(state: FakeState, store: RunStore, feed: EventFeed, run_id: str) -> Any:
    live = FakeLive()
    state.live_updates = live
    follow = await attach(feed, run_id)
    await ticks()
    assert live.healthy
    return live, follow


# --------------------------------------------------------------------------- batching


async def test_one_events_query_and_one_status_query_per_chunk(clock: Clock) -> None:
    state = BatchedState()
    runs = ["ra", "rb", "rc"]
    for run in runs:
        state.create(run)
        state.write(run)
    feed = _feed(_store(state))
    follows = [await attach(feed, run) for run in runs]
    await ticks()
    assert state.event_reads == [tuple(runs)]
    assert state.status_reads == [tuple(runs)]
    assert state.limits == [100]
    assert all(_seqs(f) == [] for f in follows)  # head came from the database at join
    state.write("rb")
    clock.advance(0.51)
    await ticks()
    assert state.event_reads[1] == tuple(runs)
    assert _seqs(follows[1]) == [1]


async def test_row_budget_caps_rows_per_query_at_two_thousand(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = BatchedState()
    runs = [f"r{i:04d}" for i in range(2_000)]  # ten times one full chunk
    for run in runs:
        state.create(run)
    feed = _feed(_store(state))
    for run in runs:
        (await attach(feed, run))
    await ticks(400)
    assert [len(chunk) for chunk in state.event_reads] == [200] * 10
    assert state.limits == [10] * 10
    pairs = zip(state.event_reads, state.limits, strict=True)
    assert all(len(chunk) * limit <= 2_000 for chunk, limit in pairs)
    assert all(len(chunk) <= 200 for chunk in state.status_reads)


async def test_full_page_is_requeued_and_read_again_at_once(clock: Clock) -> None:
    state = BatchedState()
    state.create("big")
    feed = _feed(_store(state))
    follow = await attach(feed, "big")
    await ticks()
    for _ in range(250):
        state.write("big")
    clock.advance(0.51)
    await ticks(60)
    assert len(state.event_reads) == 4  # join read, then 100 + 100 + 50 without waiting
    assert _seqs(follow) == list(range(250))


# --------------------------------------------------------------------------- terminal


async def test_status_only_hint_reaches_follow_end_within_one_cycle(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    state.write("r1")
    feed = _feed(_store(state))
    live, follow = await _healthy(state, follow_store := feed.store, feed, "r1")
    del follow_store
    state.statuses["r1"] = "success"  # finalized without an event
    live.hint("r1")
    await ticks()
    clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
    await ticks(40)
    assert follow._follower.end_reason == "terminal"


async def test_drain_reads_until_an_empty_page(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state), status_every=1)
    follow = await attach(feed, "r1")
    await ticks()
    reads = {"n": 0}

    def writer(run_id: str) -> None:
        reads["n"] += 1
        if reads["n"] == 1:  # finalize lands after the events read, before the status read
            state.write("r1", TERMINAL)
            state.statuses["r1"] = "success"
        elif reads["n"] == 2:  # a late event lands right after the first drain read
            state.write("r1")

    state.write("r1")
    state.after_events_read = writer
    clock.advance(0.51)
    await ticks(60)
    assert _seqs(follow) == [0, 1, 2]
    clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
    await ticks(40)
    assert follow._follower.end_reason == "terminal"
    assert reads["n"] == 5  # events, two tails, empty drain, grace drain


async def test_deleted_run_is_terminal_after_one_drain_read(clock: Clock) -> None:
    state = BatchedState()
    state.create("gone")
    feed = _feed(_store(state), status_every=1)
    follow = await attach(feed, "gone")
    await ticks()
    del state.statuses["gone"]
    clock.advance(0.51)
    await ticks(60)
    clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
    await ticks(40)
    assert follow._follower.end_reason == "terminal"


async def test_failed_status_query_is_never_treated_as_absence(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state), status_every=1)
    follow = await attach(feed, "r1")
    await ticks()

    async def failing(run_ids: Any) -> dict[str, str]:
        raise OSError("status read failed")

    state.get_run_statuses = failing  # type: ignore[method-assign]
    clock.advance(0.51)
    await ticks(60)
    assert follow._follower.end_reason is None


# --------------------------------------------------------------------------- schedule


async def test_status_age_is_bounded_under_constant_events(clock: Clock) -> None:
    state = BatchedState()
    state.create("hot")
    feed = _feed(_store(state), status_every=10_000)
    await attach(feed, "hot")
    await ticks()
    start = clock.now
    while clock.now - start < 75:
        state.write("hot")
        clock.advance(0.5)
        await ticks()
    gaps = [b - a for a, b in zip(state.status_times, state.status_times[1:], strict=False)]
    assert len(state.status_times) >= 3
    assert max(gaps) <= 30.5


async def test_degraded_polling_backs_off_and_reads_status_every_fourth_read(clock: Clock) -> None:
    state = BatchedState()
    state.create("idle")
    feed = _feed(_store(state))
    await attach(feed, "idle")
    await ticks()
    times = []
    for _ in range(80):
        clock.advance(0.25)
        before = len(state.event_reads)
        await ticks()
        if len(state.event_reads) > before:
            times.append(clock.now)
    gaps = [round(b - a, 2) for a, b in zip(times, times[1:], strict=False)]
    assert gaps[:4] == [1.0, 2.0, 2.0, 2.0]  # 0.5 doubling to 2.0 (first read at +0.5)
    assert len(state.status_reads) == 1 + len(times) // 4


async def test_healthy_safety_poll_is_ten_seconds_with_jitter(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    await _healthy(state, feed.store, feed, "r1")
    reads = len(state.event_reads)
    clock.advance(7.9)
    await ticks()
    assert len(state.event_reads) == reads
    clock.advance(4.2)
    await ticks()
    assert len(state.event_reads) == reads + 1
    assert len(state.status_reads) == 2  # with it


async def test_hint_makes_only_its_run_due(clock: Clock) -> None:
    state = BatchedState()
    for run in ("r1", "r2"):
        state.create(run)
    feed = _feed(_store(state))
    live, f1 = await _healthy(state, feed.store, feed, "r1")
    (await attach(feed, "r2"))
    await ticks()
    reads = len(state.event_reads)
    state.write("r1")
    live.hint("r1", "not-watched")
    await ticks()
    assert state.event_reads[reads:] == [("r1",)]
    assert _seqs(f1) == [0]


async def test_gap_makes_every_watched_run_due_once_with_jitter(clock: Clock) -> None:
    state = BatchedState()
    runs = [f"r{i}" for i in range(5)]
    for run in runs:
        state.create(run)
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, runs[0])
    for run in runs[1:]:
        (await attach(feed, run))
    await ticks()
    reads = len(state.event_reads)
    live.gap()
    await ticks()
    clock.advance(0.5)  # every jitter is within poll_interval_s
    await ticks(40)
    read_after = {r for chunk in state.event_reads[reads:] for r in chunk}
    assert read_after == set(runs)
    count = len(state.event_reads)
    clock.advance(1)
    await ticks()
    assert len(state.event_reads) == count  # once, then the healthy cadence


async def test_hint_during_read_makes_the_run_due_again(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    live, follow = await _healthy(state, feed.store, feed, "r1")
    state.gate = asyncio.Event()
    state.reading.clear()
    live.hint("r1")
    await ticks()
    assert state.reading.is_set()
    state.write("r1")
    live.hint("r1")  # arrives while the read is in flight
    await ticks()
    state.gate.set()
    await ticks(40)
    assert _seqs(follow) == [0]


async def test_oldest_due_first_hot_run_cannot_starve_quiet_runs(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_follow, "POLL_BATCH_RUNS", 1, raising=False)
    state = BatchedState()
    runs = ["hot"] + [f"q{i:03d}" for i in range(199)]
    for run in runs:
        state.create(run)
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, "hot")
    for run in runs[1:]:
        (await attach(feed, run))
    await ticks(2_000)
    clock.advance(13)  # every quiet run's safety poll is now overdue
    start = len(state.event_reads)
    state.before_events_read = lambda _ids: live.hint("hot")  # the hot run never rests
    await ticks(3_000)
    quiet_seen: set[str] = set()
    reads_needed = 0
    for chunk in state.event_reads[start:]:
        reads_needed += 1
        quiet_seen.update(r for r in chunk if r != "hot")
        if len(quiet_seen) == 199:
            break
    assert len(quiet_seen) == 199 and reads_needed <= 205


async def test_budget_is_shared_by_full_page_and_drain_reads(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_follow, "MAX_READS_PER_S", 2.0, raising=False)
    state = BatchedState()
    state.create("done", status="success")
    for _ in range(900):
        state.write("done")
    state.write("done", TERMINAL)
    feed = _feed(_store(state))
    starting = asyncio.create_task(attach(feed, "done", after_sequence_index=899))
    await ticks()
    clock.advance(_follow.TERMINAL_DRAIN_GRACE_S)
    await ticks()
    await starting
    start = clock.now
    for _ in range(20):
        await ticks()
        queries = len(state.event_reads) + len(state.status_reads)
        assert queries <= 2 + 2 * (clock.now - start) + 1e-9
        clock.advance(0.25)


async def test_custom_store_falls_back_to_per_run_reads_under_the_budget(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_follow, "MAX_READS_PER_S", 4.0, raising=False)
    state = FakeState()
    runs = [f"r{i}" for i in range(6)]
    for run in runs:
        state.create(run)
        state.write(run)
    feed = _feed(_store(state))
    follows = [await attach(feed, run) for run in runs]
    await ticks()
    assert len(state.event_reads) <= 4  # one token per per-run read
    start = clock.now
    state.write("r3")
    every_run = {(run,) for run in runs}
    while (not _seqs(follows[3]) or set(state.event_reads) != every_run) and (
        clock.now - start < 10
    ):
        clock.advance(0.25)
        await ticks()
        reads = len(state.event_reads) + len(state.per_run_status_reads)
        assert reads <= 4 + 4 * (clock.now - start) + 1e-9
    assert set(state.event_reads) == every_run
    assert _seqs(follows[3]) == [1]


class NoSequenceState(FakeState):
    """A custom store that cannot say where a run's log ends: only paged event reads."""

    get_next_event_sequence = None  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.page_limits: list[int | None] = []

    async def get_run_events(self, run_id: str, **kw: Any) -> list[SimpleNamespace]:
        self.page_limits.append(kw.get("limit"))
        return await super().get_run_events(run_id, **kw)


@pytest.mark.parametrize("count, pages", [(0, 1), (1, 2), (99, 2), (100, 2), (101, 3), (250, 4)])
async def test_head_of_a_store_without_next_sequence_is_the_last_index_across_pages(
    count: int, pages: int
) -> None:
    state = NoSequenceState()
    state.create("r1")
    for _ in range(count):
        state.write("r1")
    head = await _follow._current_head(_store(state), "r1")
    assert head == (count - 1 if count else None)
    assert state.page_limits == [_follow.DEFAULT_BATCH] * pages  # the last read is empty


async def test_head_fallback_passes_a_limit_so_dendrux_never_warns_itself() -> None:
    state = NoSequenceState()
    state.create("r1")
    for _ in range(150):
        state.write("r1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert await _follow._current_head(_store(state), "r1") == 149
    assert state.page_limits and None not in state.page_limits


async def test_new_watch_on_a_store_without_next_sequence_neither_misses_nor_repeats(
    clock: Clock,
) -> None:
    state = NoSequenceState()
    state.create("old")
    state.create("fresh")
    for _ in range(250):  # more than two pages already in the log
        state.write("old")
    feed = _feed(_store(state))
    old = await attach(feed, "old")
    fresh = await attach(feed, "fresh")
    await ticks()
    assert _seqs(old) == [] and _seqs(fresh) == []  # nothing replayed, nothing from before
    state.write("old")
    state.write("fresh")
    clock.advance(0.51)
    await ticks()
    assert _seqs(old) == [250]
    assert _seqs(fresh) == [0]  # an empty run's first event is not skipped
    clock.advance(0.51)
    await ticks()
    assert _seqs(old) == [250] and _seqs(fresh) == [0]


# --------------------------------------------------------------------------- failures


async def test_failed_chunk_is_isolated_logged_without_text_and_retried(
    clock: Clock, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(_follow, "POLL_BATCH_RUNS", 1, raising=False)
    state = BatchedState()
    for run in ("bad", "good"):
        state.create(run)
        state.write(run)
    feed = _feed(_store(state))
    bad = await attach(feed, "bad")
    good = await attach(feed, "good")
    state.fail_runs = {"bad"}
    await ticks()
    state.write("good")
    clock.advance(0.51)
    await ticks()
    assert _seqs(good) == [1]
    assert _seqs(bad) == []
    state.fail_runs = set()
    state.write("bad")
    clock.advance(_follow_recovery.SUSPECT_RETRY_MIN_S + 0.1)
    await ticks(40)
    assert _seqs(bad) == [1]
    failed = [r for r in caplog.records if getattr(r, "code", None) == "follow_read_failed"]
    assert failed and all(r.error_kind == "connection" and r.count == 1 for r in failed)
    assert "driver text" not in caplog.text


async def test_crashed_coordinator_is_restarted_by_an_existing_follower(caplog: Any) -> None:
    caplog.set_level(logging.WARNING)
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state), poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=0.05)
    async with feed.follow("r1", tenant_id=None, until_terminal=False) as follow:
        state.write("r1")
        assert (await asyncio.wait_for(anext(follow), 5)).sequence_index == 0
        state.crash = 1  # the next read escapes chunk isolation and kills the task
        state.write("r1")
        item = await asyncio.wait_for(_next_event(follow), 5)
        assert item.sequence_index == 1
    codes = [getattr(r, "code", None) for r in caplog.records]
    assert codes.count("follow_coordinator_restart") == 1


async def _next_event(follow: Any) -> StoredEvent:
    while True:
        item = await anext(follow)
        if isinstance(item, StoredEvent):
            return item


async def test_repeated_crashes_end_every_follower_with_error() -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state), poll_interval_s=0.01, poll_max_s=0.02, heartbeat_s=0.05)
    state.crash = 100
    async with feed.follow("r1", tenant_id=None) as follow:
        items = await asyncio.wait_for(_until_end(follow), 5)
    assert items[-1] == FollowEnd("error", None)
    assert state.crash == 100 - 4  # the first run and three restarts


async def test_revive_accounts_for_crash_before_its_done_callback() -> None:
    state = BatchedState()
    state.create("r1")
    async with _store(state) as store:
        follow = await attach(_feed(store), "r1")
        coordinator = store._follow_coordinator()
        task = coordinator._task
        assert task is not None
        task.remove_done_callback(coordinator._on_exit)
        state.crash = 1
        await asyncio.gather(task, return_exceptions=True)
        assert not coordinator._crashed

        coordinator.revive()
        assert len(coordinator._restarts) == 1
        replacement = coordinator._task
        assert replacement is not task
        coordinator._on_exit(task)  # the queued callback arrives after replacement
        assert coordinator._task is replacement and not coordinator._crashed
        await follow.aclose()


async def _until_end(follow: Any) -> list[Any]:
    items = []
    async for item in follow:
        items.append(item)
    return items


async def test_per_run_fallback_isolates_a_failing_run(clock: Clock, caplog: Any) -> None:
    caplog.set_level(logging.WARNING)
    state = FakeState()
    for run in ("bad", "good"):
        state.create(run)
    feed = _feed(_store(state))
    bad = await attach(feed, "bad")
    good = await attach(feed, "good")
    await ticks()
    state.fail_runs = {"bad"}
    state.write("bad")
    state.write("good")
    clock.advance(0.51)
    await ticks()
    assert _seqs(good) == [0] and _seqs(bad) == []
    state.fail_runs = set()
    for _ in range(24):
        clock.advance(0.25)
        await ticks()
    assert _seqs(bad) == [0]
    failed = [r for r in caplog.records if getattr(r, "code", None) == "follow_read_failed"]
    assert failed and failed[0].count == 1


async def test_status_failures_back_off_even_while_event_reads_succeed(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    attempts: list[float] = []

    async def failing(run_ids: Any) -> dict[str, str]:
        attempts.append(clock.now)
        raise OSError("status backend down")

    state.get_run_statuses = failing  # type: ignore[method-assign]
    feed = _feed(_store(state), status_every=1)
    await attach(feed, "r1")
    await ticks()
    start = clock.now
    while clock.now - start < 36:
        state.write("r1")  # every events read succeeds with new events
        clock.advance(0.25)
        await ticks()
    gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:], strict=False)]
    assert gaps[:3] == [5.0, 10.0, 20.0]


async def test_hints_never_shorten_status_failure_backoff(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    attempts: list[float] = []
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, "r1")

    async def failing(run_ids: Any) -> dict[str, str]:
        attempts.append(clock.now)
        raise OSError("status backend down")

    state.get_run_statuses = failing  # type: ignore[method-assign]
    start = clock.now
    while clock.now - start < 37:
        state.write("r1")
        live.hint("r1")  # a healthy bus keeps announcing new events
        await ticks()
        clock.advance(0.1)
    gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:], strict=False)]
    assert len(gaps) >= 3
    for gap, backoff in zip(gaps, [5.0, 10.0, 20.0], strict=False):
        assert backoff <= gap <= backoff + 1 / _follow_recovery.SUSPECT_READS_PER_S + 0.1


async def test_a_hint_during_a_failing_status_read_does_not_skip_the_backoff(
    clock: Clock,
) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, "r1")
    attempts: list[float] = []
    gate = asyncio.Event()

    async def failing(run_ids: Any) -> dict[str, str]:
        attempts.append(clock.now)
        await gate.wait()
        raise OSError("status backend down")

    state.get_run_statuses = failing  # type: ignore[method-assign]
    live.hint("r1")
    await ticks()
    assert len(attempts) == 1
    live.hint("r1")  # arrives while the status read is failing
    await ticks()
    gate.set()
    await ticks()
    clock.advance(0.4)
    live.hint("r1")
    await ticks()
    assert len(attempts) == 1  # still inside the suspect retry delay
    clock.advance(_follow_recovery.SUSPECT_RETRY_MIN_S)
    await ticks()
    assert len(attempts) == 2


# --------------------------------------------------------------------------- missed hints


async def _safety_poll_finds(state: BatchedState, clock: Clock, run: str) -> None:
    state.write(run)  # committed without any hint
    clock.advance(12.1)
    await ticks()


async def test_missed_hint_is_logged_once_at_first_discovery(clock: Clock, caplog: Any) -> None:
    caplog.set_level(logging.WARNING)
    state = BatchedState()
    state.create("01JBQX7Q6Z1K7Y4ZZZZZZZZZZ1")
    feed = _feed(_store(state))
    await _healthy(state, feed.store, feed, "01JBQX7Q6Z1K7Y4ZZZZZZZZZZ1")
    await _safety_poll_finds(state, clock, "01JBQX7Q6Z1K7Y4ZZZZZZZZZZ1")
    assert len(_missed(caplog)) == 1
    clock.advance(5.1)
    await ticks()
    assert len(_missed(caplog)) == 1
    assert _missed(caplog)[0].count == 1
    assert not hasattr(_missed(caplog)[0], "run_id")
    await _safety_poll_finds(state, clock, "01JBQX7Q6Z1K7Y4ZZZZZZZZZZ1")
    clock.advance(5.1)
    await ticks()
    assert len(_missed(caplog)) == 1


async def test_late_hint_does_not_repeat_the_discovery_diagnostic(
    clock: Clock, caplog: Any
) -> None:
    caplog.set_level(logging.WARNING)
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, "r1")
    await _safety_poll_finds(state, clock, "r1")
    clock.advance(2)
    live.hint("r1")
    await ticks()
    clock.advance(4)
    await ticks()
    assert len(_missed(caplog)) == 1


@pytest.mark.parametrize("cause", ["catch_up", "gap", "drain"])
async def test_no_missed_hint_on_first_read_gap_or_drain(
    clock: Clock, caplog: Any, cause: str
) -> None:
    caplog.set_level(logging.WARNING)
    state = BatchedState()
    state.create("r1")
    state.write("r1")
    feed = _feed(_store(state))
    live = FakeLive()
    state.live_updates = live
    if cause == "catch_up":
        state.write("r1")
        (await attach(feed, "r1", after_sequence_index=None))
        await ticks()
    else:
        await attach(feed, "r1")
        await ticks()
        state.write("r1")
        if cause == "gap":
            live.gap()
            clock.advance(12.1)
        else:  # a hinted read finds the terminal event; the drain read finds a late one
            state.statuses["r1"] = "success"
            state.write("r1", TERMINAL)
            state.after_events_read = lambda _run: (
                state.write("r1") if len(state.events["r1"]) == 3 else None
            )
            live.hint("r1")
        await ticks(60)
    clock.advance(6)
    await ticks()
    assert not _missed(caplog)


def _missed(caplog: Any) -> list[Any]:
    return [r for r in caplog.records if getattr(r, "code", None) == "live_updates_missed_hint"]


# --------------------------------------------------------------------------- lifecycle


async def test_many_event_feeds_share_one_coordinator(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    store = _store(state)
    first, second = _feed(store), _feed(store)
    a = await attach(first, "r1")
    b = await attach(second, "r1")
    await ticks()
    assert len(_coordinator_tasks()) == 1
    assert state.event_reads == [("r1",)]  # both followers, one read
    state.write("r1")
    clock.advance(0.51)
    await ticks()
    assert _seqs(a) == _seqs(b) == [0]
    assert first.watched_runs == second.watched_runs == 1


async def test_coordinator_exits_when_idle_and_restarts_on_the_next_follow(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    follow = await attach(feed, "r1")
    await ticks()
    assert len(_coordinator_tasks()) == 1
    await follow.aclose()
    await ticks()
    clock.advance(59)
    await ticks()
    assert len(_coordinator_tasks()) == 1
    clock.advance(1.1)
    await ticks()
    assert _coordinator_tasks() == []
    again = await attach(feed, "r1")
    await ticks()
    assert len(_coordinator_tasks()) == 1
    state.write("r1")
    clock.advance(0.51)
    await ticks()
    assert _seqs(again) == [0]


async def test_run_store_close_stops_its_coordinator_and_releases_only_its_lease(
    clock: Clock,
) -> None:
    state = BatchedState()
    state.create("r1")
    live = FakeLive()
    state.live_updates = live
    store, other = _store(state), _store(state)
    (await attach(_feed(store), "r1"))
    (await attach(_feed(other), "r1"))
    await ticks()
    assert len(live._leases) == 2 and len(_coordinator_tasks()) == 2
    await store.close()
    await ticks()
    assert len(live._leases) == 1 and len(_coordinator_tasks()) == 1
    assert live.healthy and live._state == "open"  # the store's sender is never closed
    await other.close()
    assert not live._leases


async def test_unavailable_live_updates_mean_polling_never_a_failed_follow(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    live = FakeLive()
    state.live_updates = live
    await live.close()  # _acquire_listener and _register now raise LiveUpdatesUnavailableError
    feed = _feed(_store(state))
    follow = await attach(feed, "r1")
    state.write("r1")
    clock.advance(0.51)
    await ticks()
    assert _seqs(follow) == [0]


async def test_from_engine_reader_never_leases(tmp_path: Any) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.db.models import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        reader = RunStore.from_engine(engine)
        assert reader._live_updates() is None
        assert not hasattr(RunStore.from_engine, "notifier")
        import inspect

        for factory in (RunStore.from_engine, RunStore.from_database_url):
            params = inspect.signature(factory).parameters
            assert "notifier" not in params and "listen_url" not in params
        assert "notifier" not in inspect.signature(EventFeed).parameters
        await reader.close()
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- stream_events


async def test_stream_events_is_woken_by_the_hub_and_unregisters(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    live = FakeLive()
    state.live_updates = live
    store = _store(state)
    stream = store.stream_events("r1", tenant_id=None, poll_interval_s=1_000)
    reader = asyncio.ensure_future(anext(stream))
    await ticks()
    assert live._stream_consumers == 1 and live.healthy
    state.write("r1")
    live.hint("r1")
    await ticks()
    assert not reader.done()  # Coalesced until poll_interval_s / 10.
    clock.advance(100)
    await ticks(40)
    assert reader.done() and reader.result().sequence_index == 0
    await stream.aclose()
    assert live._stream_consumers == 0
    await store.close()


async def test_stream_events_capacity_is_refused_before_the_first_yield(clock: Clock) -> None:
    from dendrux.store import FollowCapacityError

    state = BatchedState()
    state.create("r1")
    state.write("r1")
    live = FakeLive()
    live.max_stream_consumers = 1
    state.live_updates = live
    store = _store(state)
    first = store.stream_events("r1", tenant_id=None)
    assert (await anext(first)).sequence_index == 0
    with pytest.raises(FollowCapacityError):
        await anext(store.stream_events("r1", tenant_id=None))
    await first.aclose()
    await store.close()


async def test_stream_events_hint_during_a_read_is_not_lost(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    live = FakeLive()
    state.live_updates = live
    store = _store(state)
    stream = store.stream_events("r1", tenant_id=None, poll_interval_s=1_000)
    state.gate = asyncio.Event()
    reader = asyncio.ensure_future(anext(stream))
    await ticks()
    assert state.reading.is_set() and live.healthy
    state.write("r1")
    live.hint("r1")  # lands while the empty read is in flight
    await ticks()
    state.gate.set()
    await ticks(40)
    assert not reader.done()  # Coalesced until poll_interval_s / 10.
    clock.advance(100)
    await ticks(40)
    assert reader.done() and reader.result().sequence_index == 0
    await stream.aclose()
    await store.close()


async def test_a_thousand_concurrent_joins_coalesce_into_one_read(clock: Clock) -> None:
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    await attach(feed, "r1")
    await ticks()
    reads = len(state.event_reads)
    state.write("r1")
    follows = await asyncio.gather(*(attach(feed, "r1") for _ in range(1_000)))
    await ticks()
    assert len(state.event_reads) == reads + 1  # one read for the whole burst
    assert all(_seqs(f) == [0] for f in follows)  # each got the event once, from memory
    await ticks(20)
    assert len(state.event_reads) == reads + 1


async def test_hints_revive_cancelled_coordinator_but_gaps_do_not(clock: Clock) -> None:
    """A gap alone does not restart; a hint recovers an unexpected cancellation."""
    state = BatchedState()
    state.create("r1")
    feed = _feed(_store(state))
    live, _ = await _healthy(state, feed.store, feed, "r1")
    (task,) = _coordinator_tasks()
    task.cancel()  # what the event loop runner does to every task at shutdown
    await ticks()
    live.gap()
    await ticks()
    assert _coordinator_tasks() == []
    live.hint("r1")
    await ticks()
    assert len(_coordinator_tasks()) == 1
