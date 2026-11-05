"""Failed follow reads: getting over a database blip must not depend on followers leaving.

Fake clock throughout. ``down`` makes every read fail (an outage); ``fail_runs`` makes only the
reads that include those runs fail (a truly bad run).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from dendrux.store import RunStore, _follow, _follow_recovery
from tests.unit.test_follow_coordinator import BatchedState, FakeState, ticks
from tests.unit.test_follow_coordinator import clock as clock
from tests.unit.test_follow_coordinator import fast_budget as fast_budget


class Outage:
    """Fail every read of a fake store while ``down``, whatever the runs."""

    def __init__(self, state: FakeState) -> None:
        self.down = False
        state.before_events_read = self._check
        batched = getattr(state, "get_run_statuses", None)
        per_run = state.get_run_status

        async def statuses(run_ids: Any) -> Any:
            self._check(run_ids)
            return await batched(run_ids)

        async def status(run_id: str, **kwargs: Any) -> Any:
            self._check(run_id)
            return await per_run(run_id, **kwargs)

        if batched is not None:
            state.get_run_statuses = statuses  # type: ignore[method-assign]
        state.get_run_status = status  # type: ignore[method-assign]

    def _check(self, _: Any) -> None:
        if self.down:
            raise OSError("database unavailable")


def watch(reader: RunStore, state: FakeState, rid: str) -> _follow._WatchedRun:
    state.create(rid)
    coordinator = reader._follow_coordinator()
    run = _follow._WatchedRun(coordinator, rid, None, next(coordinator._order))
    run.followers.add(_follow._Follower(False, 10_000))
    run.events_due = run.status_due = coordinator._now()
    coordinator.runs[rid] = run
    return run


def many(reader: RunStore, state: FakeState, count: int) -> list[_follow._WatchedRun]:
    runs = [watch(reader, state, f"r{i}") for i in range(count)]
    reader._follow_coordinator().revive()
    return runs


def real_budget(monkeypatch: Any) -> float:
    monkeypatch.setattr(_follow, "MAX_READS_PER_S", 20.0)
    return 20.0


async def pass_time(clock: Any, seconds: float, step: float = 0.05) -> None:
    for _ in range(round(seconds / step)):
        clock.advance(step)
        await ticks(6)


def suspects(runs: list[_follow._WatchedRun]) -> int:
    return sum(run.suspect for run in runs)


def queued(coordinator: Any) -> int:
    return len(coordinator._isolations) + len(coordinator._suspects.jobs)


def reads(state: FakeState) -> int:
    return len(state.event_reads) + len(state.status_reads)


@pytest.mark.parametrize("count", [100, 1000])
@pytest.mark.parametrize("seconds", [15, 60])
async def test_every_run_recovers_soon_after_an_outage(clock, monkeypatch, count, seconds):
    rate = real_budget(monkeypatch)
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, count)
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, seconds)
        outage.down = False
        before, waited = reads(state), 0.0
        while suspects(runs) and waited < 30:
            await pass_time(clock, 0.5)
            waited += 0.5
        assert suspects(runs) == 0 and waited < 30
        assert reads(state) - before <= rate * (waited + 1)
        state.write("r7")
        reader._follow_coordinator().on_hints(("r7",))
        await pass_time(clock, 0.5)
        assert runs[7].head == 0
    finally:
        await reader.close()


async def test_one_group_read_clears_many_runs(clock):
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 150)
    coordinator = reader._follow_coordinator()
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        while suspects(runs) < 150:
            await pass_time(clock, 1)
        for jobs in (coordinator._suspects.jobs,):
            while jobs:  # the queue a long outage leaves: forget it, keep only the sweep
                for run in jobs.popleft().runs:
                    run.events_pending = run.status_pending = None
        outage.down = False
        state.event_reads.clear()
        state.status_reads.clear()
        await pass_time(clock, 11)
        assert suspects(runs) == 0
        suspect_sized = [g for g in state.event_reads + state.status_reads if len(g) >= 100]
        assert len(suspect_sized) >= 2  # one for each failing phase
    finally:
        await reader.close()


async def test_a_truly_bad_run_never_reads_with_healthy_runs_again(clock):
    state = BatchedState()
    reader = RunStore(state)
    runs = many(reader, state, 40)
    state.fail_runs.add("r3")
    state.write("r3")
    coordinator = reader._follow_coordinator()
    try:
        await pass_time(clock, 3)
        assert runs[3].suspect and suspects(runs) == 1
        state.event_reads.clear()
        for _ in range(20):
            state.write("r5")
            coordinator.on_hints(("r3", "r5"))
            await pass_time(clock, 5)
        assert suspects(runs) == 1
        with_bad = [group for group in state.event_reads if "r3" in group]
        assert with_bad and all(group == ("r3",) for group in with_bad)
        assert runs[5].head == 19
    finally:
        await reader.close()


async def test_permanently_bad_runs_cost_at_most_one_read_each_per_retry_ceiling(clock):
    state = BatchedState()
    reader = RunStore(state)
    many(reader, state, 25)
    state.fail_runs |= {f"r{i}" for i in range(0, 25, 5)}
    for run_id in state.fail_runs:
        state.write(run_id)
    try:
        await pass_time(clock, 120)  # both backoffs reach their ceiling
        state.event_reads.clear()
        await pass_time(clock, 60)
        bad_reads = [group for group in state.event_reads if set(group) & state.fail_runs]
        # Five runs, one read per 30 s each; a shared group read may add its bisection.
        assert len(bad_reads) <= 5 * 2 * 3
    finally:
        await reader.close()


async def test_runs_that_heal_one_at_a_time_each_recover(clock):
    state = BatchedState()
    reader = RunStore(state)
    runs = many(reader, state, 10)
    state.fail_runs |= {run.run_id for run in runs}
    for run in runs:
        state.write(run.run_id)
    try:
        await pass_time(clock, 40)
        assert suspects(runs) == 10
        for index, run in enumerate(runs):
            state.fail_runs.discard(run.run_id)
            await pass_time(clock, 35)
            assert suspects(runs) == 9 - index
            assert run.head == 0
    finally:
        await reader.close()


async def test_a_run_that_heals_and_fails_again_backs_off_longer_each_time(clock):
    state = BatchedState()
    reader = RunStore(state)
    runs = many(reader, state, 5)
    bad = runs[0]
    coordinator = reader._follow_coordinator()
    delays = []
    try:
        await pass_time(clock, 1)
        for _ in range(4):
            state.fail_runs.add(bad.run_id)
            state.write(bad.run_id)
            coordinator.on_hints((bad.run_id,))
            await pass_time(clock, 1)
            assert bad.suspect
            delays.append(bad.events_fail_delay)
            state.fail_runs.discard(bad.run_id)
            await pass_time(clock, bad.events_fail_delay + 6)
            assert not bad.suspect
        assert delays == [5.0, 10.0, 20.0, 30.0]
        await pass_time(clock, _follow_recovery.RELAPSE_WINDOW_S + 1)
        state.fail_runs.add(bad.run_id)
        state.write(bad.run_id)
        coordinator.on_hints((bad.run_id,))
        await pass_time(clock, 1)
        assert bad.events_fail_delay == 5.0  # forgiven after a quiet window
    finally:
        await reader.close()


async def test_failures_with_no_read_succeeding_do_not_lengthen_the_retry(clock):
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 2)
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, 60)
        assert all(run.events_fail_delay == 5.0 for run in runs)
    finally:
        await reader.close()


async def test_hints_do_not_pull_a_suspect_out_of_its_group_or_past_its_deadline(clock):
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 8)
    coordinator = reader._follow_coordinator()
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, 2.0)
        for _ in range(100):
            before = len(state.event_reads)
            coordinator.on_hints(tuple(run.run_id for run in runs))
            await ticks(10)
            assert len(state.event_reads) <= before + 1
            clock.advance(0.05)
    finally:
        await reader.close()


async def test_per_run_store_recovers_without_a_group_read(clock):
    state = FakeState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 100)
    try:
        await pass_time(clock, 12)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, 15)
        outage.down = False
        await pass_time(clock, 30)
        assert suspects(runs) == 0
    finally:
        await reader.close()


async def test_isolation_queues_hold_only_live_work(clock):
    state = BatchedState()
    reader = RunStore(state)
    runs = [watch(reader, state, f"r{i}") for i in range(4)]
    coordinator = reader._follow_coordinator()
    try:
        for _ in range(500):
            coordinator._queue_isolation(runs, False)
            coordinator.on_hints(tuple(run.run_id for run in runs))  # empties both jobs
        assert queued(coordinator) <= 2
    finally:
        await reader.close()


async def test_isolation_queues_stay_bounded_during_an_outage_with_hints(clock):
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 300)
    coordinator = reader._follow_coordinator()
    peak = 0
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        for _ in range(600):
            await pass_time(clock, 0.05)
            coordinator.on_hints(tuple(run.run_id for run in runs[:30]))
            peak = max(peak, queued(coordinator))
        assert peak <= 2 * len(runs)
    finally:
        await reader.close()


async def test_groups_left_waiting_by_an_outage_merge_once_the_store_is_back(clock, monkeypatch):
    rate = real_budget(monkeypatch)
    state = BatchedState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 1000)
    coordinator = reader._follow_coordinator()
    try:
        await pass_time(clock, 1)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, 30)
        waiting = queued(coordinator)
        assert waiting > 100  # Long enough for the groups to have become small.
        outage.down = False
        before, waited = reads(state), 0.0
        while queued(coordinator) and waited < 15:
            await pass_time(clock, 0.5)
            waited += 0.5
        assert queued(coordinator) == 0
        assert reads(state) - before <= min(waiting, rate * (waited + 1))
    finally:
        await reader.close()


async def test_healthy_hints_keep_their_share_while_groups_wait(clock, monkeypatch):
    real_budget(monkeypatch)
    state = BatchedState()
    reader = RunStore(state)
    runs = many(reader, state, 600)
    coordinator = reader._follow_coordinator()
    state.fail_runs |= {f"r{i}" for i in range(1, 600, 2)}  # Half are bad: groups keep failing.
    statuses = state.get_run_statuses

    async def failing_statuses(run_ids):
        if state.fail_runs & set(run_ids):
            raise OSError("bad run")
        return await statuses(run_ids)

    state.get_run_statuses = failing_statuses
    for run_id in state.fail_runs:
        state.write(run_id)
    try:
        await pass_time(clock, 1)
        for run in runs:
            run.events_due = run.status_due = clock.now
        await pass_time(clock, 1)
        assert coordinator._isolations
        healthy = runs[100]
        for index in range(15):
            state.write(healthy.run_id)
            coordinator.on_hints((healthy.run_id,))
            await pass_time(clock, 0.5)
            assert healthy.head == index
    finally:
        await reader.close()


async def test_a_failed_read_returns_suspects_to_the_small_share(clock, monkeypatch):
    real_budget(monkeypatch)
    state = BatchedState()
    reader = RunStore(state)
    runs = many(reader, state, 300)
    state.fail_runs |= {run.run_id for run in runs}
    for run in runs:
        state.write(run.run_id)
    try:
        await pass_time(clock, 60)
        assert suspects(runs) == 300
        state.fail_runs.discard("r0")  # One healed read speeds the share up, once.
        while runs[0].suspect:
            await pass_time(clock, 1)
        state.event_reads.clear()
        await pass_time(clock, 30)
        failing = [read for read in state.event_reads if set(read) & state.fail_runs]
        assert failing and len(failing) / 30 <= _follow_recovery.SUSPECT_READS_PER_S + 0.1
    finally:
        await reader.close()


def group(runs: list[_follow._WatchedRun], *, suspects: bool = True) -> _follow_recovery._Isolation:
    job = _follow_recovery._Isolation(list(runs), False, suspects)
    job.queue()
    return job


async def test_a_hint_leaves_a_suspect_in_its_group_and_frees_a_healthy_run(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    suspect, healthy = watch(reader, state, "s"), watch(reader, state, "h")
    coordinator._suspects.mark(suspect)
    suspect_job, healthy_job = group([suspect]), group([healthy], suspects=False)
    try:
        coordinator.on_hints(("s", "h"))
        assert suspect_job.holds(suspect) and not healthy_job.holds(healthy)
    finally:
        await reader.close()


async def test_a_group_only_reads_runs_of_its_own_kind(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    runs = [watch(reader, state, f"r{i}") for i in range(4)]
    job = group(runs, suspects=False)
    coordinator._suspects.mark(runs[0])
    try:
        assert job.take(coordinator.runs) == runs[1:]
        assert runs[0].events_pending is None
    finally:
        await reader.close()


async def test_a_failed_group_of_suspects_stays_on_the_suspect_queue(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    runs = [watch(reader, state, f"r{i}") for i in range(4)]
    for run in runs:
        coordinator._suspects.mark(run)
    try:
        coordinator._queue_isolation(runs, False)
        assert len(coordinator._suspects.jobs) == 2 and not coordinator._isolations
    finally:
        await reader.close()


async def test_groups_split_during_an_outage_merge_and_others_stay_apart(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    runs = [watch(reader, state, f"r{i}") for i in range(30)]
    jobs = deque(group(runs[start : start + 5], suspects=False) for start in range(0, 25, 5))
    for job in jobs:
        job.outage = True
    apart = group(runs[25:], suspects=False)
    jobs.append(apart)
    try:
        _follow_recovery.merge_outage_jobs(jobs, coordinator.runs, 10)
        assert [len(job.runs) for job in jobs] == [10, 10, 5, 5]
        assert jobs[-1] is apart
        assert all(job.holds(run) for job in jobs for run in job.runs)
    finally:
        await reader.close()


async def test_a_failed_group_is_only_an_outage_when_nothing_has_worked_for_a_while(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    runs = [watch(reader, state, f"r{i}") for i in range(4)]
    suspects_ = coordinator._suspects
    try:
        suspects_.note_read(True, clock.now)
        early = suspects_.split(runs, False, clock.now + _follow_recovery.OUTAGE_AFTER_S - 0.1)
        assert not any(job.outage for job in early)
        late = suspects_.split(runs, False, clock.now + _follow_recovery.OUTAGE_AFTER_S)
        assert all(job.outage for job in late)
        assert not suspects_.store_is_back(clock.now + _follow_recovery.OUTAGE_AFTER_S)
        suspects_.note_read(True, clock.now + 20)
        assert suspects_.store_is_back(clock.now + 20)
        assert not suspects_.store_is_back(clock.now + 20)  # Once per outage.
    finally:
        await reader.close()


async def test_a_suspect_that_stops_being_watched_is_forgotten(clock):
    state = BatchedState()
    reader = RunStore(state)
    coordinator = reader._follow_coordinator()
    run = watch(reader, state, "gone")
    coordinator._suspects.mark(run)
    try:
        coordinator.unwatch(run)
        assert not coordinator._suspects.runs
    finally:
        await reader.close()


async def test_per_run_store_keeps_recovering_under_the_real_read_budget(clock, monkeypatch):
    real_budget(monkeypatch)
    state = FakeState()
    outage = Outage(state)
    reader = RunStore(state)
    runs = many(reader, state, 200)
    coordinator = reader._follow_coordinator()
    try:
        await pass_time(clock, 30)
        outage.down = True
        for run in runs:
            run.events_due = run.status_due = clock.now
        while suspects(runs) < 200:
            await pass_time(clock, 1)
        outage.down = False
        waited = 0.0
        while suspects(runs) and waited < 60:  # Two reads per run, at most ten a second.
            await pass_time(clock, 1)
            waited += 1
        assert suspects(runs) == 0
        state.write("r3")
        coordinator.on_hints(("r3",))
        await pass_time(clock, 2)
        assert runs[3].head == 0
    finally:
        await reader.close()
