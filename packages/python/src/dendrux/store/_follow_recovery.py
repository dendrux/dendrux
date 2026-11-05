"""Suspect runs of a follow coordinator: kept out of healthy batches, re-checked in groups.

A run whose read fails alone becomes a suspect. Suspects are read only with other suspects, at
a small share of the read budget, so a truly bad run never holds healthy runs back. A database
blip makes every run a suspect, so suspects are re-checked as groups: one good read clears many
runs, and a failed group is halved like any failed batch.

Each failed phase waits out its own retry deadline, which hints never shorten. The deadline only
grows when some other read worked since this run last failed, or when the run keeps relapsing;
during an outage nothing works, so every run stays ready to be re-checked as soon as it ends.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dendrux.store._follow import _WatchedRun

SUSPECT_READS_PER_S = 2.0
SUSPECT_RECOVERY_READS_PER_S = 10.0
SUSPECT_RETRY_MIN_S = 5.0
SUSPECT_RETRY_MAX_S = 30.0
SUSPECT_RECHECK_S = 5.0
RELAPSE_WINDOW_S = 60.0


def _pending_field(status: bool) -> str:
    return "status_pending" if status else "events_pending"


@dataclass
class _Isolation:
    """Runs that failed together and wait to be read again without the others."""

    runs: list[_WatchedRun]
    status: bool
    suspects: bool = False
    token: object = field(default_factory=object)

    def queue(self) -> None:
        for run in self.runs:
            setattr(run, _pending_field(self.status), self.token)

    def holds(self, run: _WatchedRun) -> bool:
        return getattr(run, _pending_field(self.status)) is self.token

    def is_live(self, watched: dict[str, _WatchedRun]) -> bool:
        return any(
            watched.get(run.run_id) is run and run.active and self.holds(run) for run in self.runs
        )

    def take(self, watched: dict[str, _WatchedRun]) -> list[_WatchedRun]:
        """The members still waiting on this job; every token is given up either way."""
        members = []
        for run in self.runs:
            if self.holds(run):
                setattr(run, _pending_field(self.status), None)
                if watched.get(run.run_id) is run and run.active and run.suspect is self.suspects:
                    members.append(run)
        return members


def take_job(jobs: deque[_Isolation], watched: dict[str, _WatchedRun]) -> _Isolation | None:
    """The next job with live members, with ``runs`` narrowed to them."""
    while jobs:
        job = jobs.popleft()
        members = job.take(watched)
        if members:
            job.runs = members
            return job
    return None


def split_jobs(runs: list[_WatchedRun], status: bool, suspects: bool) -> list[_Isolation]:
    middle = len(runs) // 2
    return [_Isolation(group, status, suspects) for group in (runs[:middle], runs[middle:])]


def merge_jobs(jobs: deque[_Isolation], watched: dict[str, _WatchedRun], size: int) -> None:
    """Rebuild waiting jobs into groups of up to ``size`` runs, per phase, oldest first."""
    if len(jobs) < 2:
        return
    suspects = jobs[0].suspects
    waiting: dict[bool, list[_WatchedRun]] = {False: [], True: []}
    while jobs:
        job = jobs.popleft()
        waiting[job.status].extend(job.take(watched))
    for status, runs in waiting.items():
        for group in _batches(runs, size):
            job = _Isolation(group, status, suspects)
            job.queue()
            jobs.append(job)


def retry_backoff(delay: float) -> float:
    return min(max(delay * 2, SUSPECT_RETRY_MIN_S), SUSPECT_RETRY_MAX_S)


def _phase_ready(run: _WatchedRun, status: bool, now: float) -> bool:
    delay = run.status_fail_delay if status else run.events_fail_delay
    retry_at = run.status_retry_at if status else run.events_retry_at
    pending = run.status_pending if status else run.events_pending
    return delay > 0 and retry_at <= now and pending is None and run.active


def _batches(runs: list[_WatchedRun], size: int) -> Iterable[list[_WatchedRun]]:
    return (runs[start : start + size] for start in range(0, len(runs), size))


class _Suspects:
    """The suspects of one coordinator, their group re-checks and their read allowance."""

    def __init__(self) -> None:
        self.runs: set[_WatchedRun] = set()
        self.jobs: deque[_Isolation] = deque()
        self.probe_at = -math.inf
        self.read_ok_at = -math.inf
        self.good_reads = 0
        self._recheck_at = -math.inf
        self._interval = 1 / SUSPECT_READS_PER_S

    def mark(self, run: _WatchedRun) -> None:
        run.suspect = True
        self.runs.add(run)

    def forget(self, run: _WatchedRun) -> None:
        self.runs.discard(run)

    def clear(self) -> None:
        self.runs.clear()
        self.jobs.clear()

    def settle(self, run: _WatchedRun, healed: bool, now: float) -> None:
        """A read worked; ``healed`` when that phase was failing. No failing phase: healthy."""
        if healed:
            self._interval = 1 / SUSPECT_RECOVERY_READS_PER_S
        if run.suspect and not run.events_fail_delay and not run.status_fail_delay:
            run.suspect = False
            run.relapse_until = now + RELAPSE_WINDOW_S
            self.runs.discard(run)

    def next_delay(self, run: _WatchedRun, delay: float, retry_at: float, now: float) -> float:
        """The next retry delay of a phase whose read just failed alone."""
        if delay == 0:
            base = run.relapse_delay if now < run.relapse_until else 0.0
        elif self.read_ok_at > retry_at - delay:
            base = delay
        else:
            return delay  # Nothing worked since this run last failed: an outage, not this run.
        run.relapse_delay = retry_backoff(base)
        return run.relapse_delay

    def charge(self, now: float) -> None:
        self.probe_at = now + self._interval

    def note_read(self, worked: bool, now: float) -> None:
        if worked:
            self.read_ok_at = now
            self.good_reads += 1

    def note_failure(self, now: float) -> None:
        self._interval = 1 / SUSPECT_READS_PER_S
        self.probe_at = max(self.probe_at, now + self._interval)

    def queue_due(self, now: float, batch: int) -> None:
        """Group every suspect phase past its deadline, at most once per re-check period."""
        if not self.runs or now < self._recheck_at:
            return
        self._recheck_at = now + SUSPECT_RECHECK_S
        ordered = sorted(self.runs, key=lambda run: run.order)
        for status in (False, True):
            ready = [run for run in ordered if _phase_ready(run, status, now)]
            for group in _batches(ready, batch):
                job = _Isolation(group, status, suspects=True)
                job.queue()
                self.jobs.append(job)

    def wake_at(self) -> float:
        times = [self.probe_at] if self.jobs else []
        if self.runs:
            times.append(self._recheck_at)
        return min(times, default=math.inf)
