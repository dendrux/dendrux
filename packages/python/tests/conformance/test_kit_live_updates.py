"""The kit's live-update contract: the recorder it hands a store, and the contract itself.

The contracts run in inner pytest sessions (``pytester``) against a stand-in for track A2's
announcing store (``tests/conformance/_announcing_store.py``), correct and with each promise
broken in turn. Against the repo's own stores they run from ``tests/conformance/conftest.py``
once those stores take ``live_updates=``.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from dendrux.testing.store_kit._live import RecordedBatch, RecordingLiveUpdates, RunView
from tests.conformance._announcing_store import MODES

pytest_plugins = ["pytester"]


async def eventually(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


# -- the recorder ----------------------------------------------------------------------------


async def test_the_recorder_keeps_each_decoded_message_in_send_order() -> None:
    updates = RecordingLiveUpdates(max_batch=2)
    updates.enqueue(["a", "b", "c"])
    await updates.flush()
    assert [batch.ids for batch in updates.batches] == [("a", "b"), ("c",)]
    assert updates.ids == ["a", "b", "c"]
    assert updates.stats().sent == 3
    updates.clear()
    assert updates.batches == [] and updates.ids == []
    await updates.close()


async def test_the_recorder_asks_the_probe_what_a_second_connection_sees_at_send_time() -> None:
    calls: list[tuple[str, ...]] = []

    async def probe(ids: tuple[str, ...]) -> dict[str, RunView]:
        calls.append(ids)
        return {run_id: RunView(True, "running", 3) for run_id in ids}

    updates = RecordingLiveUpdates(probe=probe)
    updates.enqueue(["R1"])
    await updates.flush()
    (batch,) = updates.batches
    assert calls == [("R1",)]
    assert batch.seen == {"R1": RunView(True, "running", 3)} and batch.probe_error is None
    await updates.close()


async def test_a_failing_probe_is_recorded_and_the_send_still_succeeds() -> None:
    async def probe(ids: tuple[str, ...]) -> dict[str, RunView]:
        raise RuntimeError("probe secret text")

    updates = RecordingLiveUpdates(probe=probe)
    updates.enqueue(["R1"])
    await updates.flush()
    (batch,) = updates.batches
    assert batch.ids == ("R1",) and batch.seen == {}
    assert batch.probe_error == "RuntimeError"  # the kind only
    assert updates.stats().sent == 1 and updates.stats().dropped_failed == 0
    await updates.close()


async def test_the_recorder_has_roomy_default_timeouts_and_accepts_overrides() -> None:
    assert RecordingLiveUpdates().send_timeout_s == 5.0
    assert RecordingLiveUpdates().flush_deadline_s == 10.0
    assert RecordingLiveUpdates(send_timeout_s=1.0, flush_deadline_s=2.0).flush_deadline_s == 2.0


async def test_the_recorder_listens_but_never_delivers() -> None:
    updates = RecordingLiveUpdates(health_check_s=30.0)
    lease = await updates._acquire_listener()
    await eventually(lambda: updates.healthy)
    updates.enqueue(["R1"])
    await updates.flush()
    await asyncio.sleep(0.05)  # a negative claim: slowness can only make it pass vacuously
    gaps = updates.stats().gaps
    assert gaps == 1  # only the healthy transition
    await updates._release_listener(lease)
    await updates.close()


@pytest.mark.parametrize("backend_kind", ["sqlite-memory", "sqlite-format1"])
@pytest.mark.parametrize("kind,identity", [("error", True), ("error", False), ("cancel", True)])
async def test_failed_commit_is_checked_before_a_fallback_reader_can_flush(
    tmp_path: Any, backend_kind: str, kind: str, identity: bool
) -> None:
    from dendrux.testing.store_kit._declaration import (
        FAULT_INJECTOR,
        LIVE_FACTORY,
        SECOND_INSTANCE,
    )
    from dendrux.testing.store_kit._kit import Kit
    from dendrux.testing.store_kit.contracts.test_live_updates import (
        test_live_a_commit_that_was_issued_is_announced_even_when_the_call_failed as contract,
    )
    from dendrux.testing.store_kit.sql import SqlFaultInjector
    from tests.conformance._kit_backends import KitBackend

    backend = KitBackend(backend_kind, tmp_path)
    injector = SqlFaultInjector()
    kit = Kit(
        {
            LIVE_FACTORY: backend.fresh_live_store,
            SECOND_INSTANCE: backend.second,
            FAULT_INJECTOR: injector,
        },
        None,
        SimpleNamespace(needs_second_instance=False, has=lambda name: name == "retry_ids"),
        needs_format_2=False,
    )
    try:
        # The writer doubles as reader: a successful read is allowed to flush pending ids.
        # All assertions must pass before the kit reports independent visibility unverified.
        with pytest.raises(pytest.skip.Exception, match="unverified: live-update visibility"):
            await contract(kit, kind, identity)
    finally:
        injector.close()
        await backend.close()


def test_a_batch_is_plain_data() -> None:
    batch = RecordedBatch(("a",), {"a": RunView(True, "x", 0)})
    assert json.dumps(batch.ids) == '["a"]'
    with pytest.raises(AttributeError):
        batch.ids = ()  # type: ignore[misc]


# -- the contract, against a stand-in for an announcing store --------------------------------

CONFTEST = """
import pytest
import pytest_asyncio
from tests.conformance import _kit_backends as kb
from tests.conformance._announcing_store import AnnouncingSqliteStore
from dendrux.testing.store_kit.sql import SqlFaultInjector

@pytest_asyncio.fixture
async def kit_backend(tmp_path):
    backend = kb.KitBackend({kind!r}, tmp_path)
    yield backend
    await backend.close()

@pytest.fixture
def dendrux_store_factory(kit_backend): return kit_backend.fresh_store
@pytest.fixture
def dendrux_store_reopen(kit_backend): return kit_backend.reopen
@pytest.fixture
def dendrux_store_second_instance(kit_backend): return kit_backend.second
@pytest.fixture
def dendrux_clock(kit_backend): return kit_backend
"""
INJECTOR = """
@pytest.fixture
def dendrux_fault_injector():
    injector = SqlFaultInjector()
    yield injector
    injector.close()
"""
LIVE = """
@pytest.fixture
def dendrux_live_store_factory(kit_backend):
    def build(engine, bus):
        return AnnouncingSqliteStore(engine, live_updates=bus, mode={mode!r})
    return lambda updates: kit_backend.fresh_live_store(updates, build=build)
"""
INNER = (
    "-p", "dendrux.testing.store_kit", "-p", "no:randomly", "-p", "no:cacheprovider",
    "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
    "--no-cov", "-q", "-k", "test_live_", "-rfs",
)  # fmt: skip
CONTRACTS = 7  # test_live_* variants the kit ships


def run_live(
    pytester: pytest.Pytester,
    mode: str | None = "correct",
    *,
    kind: str = "sqlite-file",
    injector: bool = True,
) -> pytest.RunResult:
    text = CONFTEST.format(kind=kind)
    text += INJECTOR if injector else ""
    text += LIVE.format(mode=mode) if mode is not None else ""
    pytester.makeconftest(text)
    return pytester.runpytest(*INNER)


def failed_functions(result: pytest.RunResult) -> set[str]:
    return set(re.findall(r"^FAILED \S+::(\w+)", result.stdout.str(), flags=re.MULTILINE))


def test_a_store_that_announces_correctly_passes_every_live_contract(
    pytester: pytest.Pytester,
) -> None:
    result = run_live(pytester)
    result.assert_outcomes(passed=CONTRACTS)
    result.stdout.fnmatch_lines(["*live_updates*7 passed*"])


def test_a_store_with_no_independent_reader_is_reported_as_unverified_not_passed(
    pytester: pytest.Pytester,
) -> None:
    """In-memory SQLite has no second connection: visibility cannot be checked, and says so."""
    result = run_live(pytester, kind="sqlite-memory")
    result.assert_outcomes(passed=1, skipped=CONTRACTS - 1)  # only the not-applied case is whole
    result.stdout.fnmatch_lines(["*unverified: live-update visibility on a second connection*"])
    assert "failed" not in result.parseoutcomes()


T_CREATE = "test_live_a_created_run_is_announced_after_its_row_is_visible_elsewhere"
T_EVENT = "test_live_a_run_event_is_announced_after_the_event_is_visible_elsewhere"
T_STATUS = "test_live_a_status_change_is_announced_after_the_new_status_is_visible_elsewhere"
T_NOT_APPLIED = "test_live_a_write_proved_not_applied_announces_nothing"
T_ISSUED = "test_live_a_commit_that_was_issued_is_announced_even_when_the_call_failed"

BROKEN = {
    "never": {T_CREATE, T_EVENT, T_STATUS, T_ISSUED},
    "no_flush": {T_CREATE, T_EVENT, T_STATUS},
    "early": {T_CREATE, T_EVENT, T_STATUS, T_NOT_APPLIED, T_ISSUED},
    "on_not_applied": {T_NOT_APPLIED},
    "only_on_success": {T_ISSUED},
    "flush_on_failure": {T_ISSUED},
}


def test_every_broken_mode_has_a_case_and_the_correct_mode_has_none() -> None:
    assert set(BROKEN) == set(MODES) - {"correct"}


@pytest.mark.parametrize("mode", sorted(BROKEN))
def test_the_kit_fails_a_store_that_breaks_one_announcing_promise(
    pytester: pytest.Pytester, mode: str
) -> None:
    result = run_live(pytester, mode)
    assert failed_functions(result) == BROKEN[mode], result.stdout.str()[-3000:]


def test_without_the_live_fixture_the_contracts_skip_and_the_report_says_not_tested(
    pytester: pytest.Pytester,
) -> None:
    result = run_live(pytester, mode=None)
    result.assert_outcomes(skipped=CONTRACTS)
    result.stdout.fnmatch_lines(["*live updates: not tested*dendrux_live_store_factory*"])


def test_without_a_fault_injector_only_the_commit_cases_skip(pytester: pytest.Pytester) -> None:
    result = run_live(pytester, injector=False)
    result.assert_outcomes(passed=3, skipped=CONTRACTS - 3)
    result.stdout.fnmatch_lines(["*crash recovery: not certified*"])


# -- a store that publishes before its commit, but whose own reads see the staged state --------


class StagedStore:
    """Announces before COMMIT. Its own reads see staged rows; another connection sees only the
    committed ones. Nothing here talks to a database."""

    def __init__(self, updates: Any, *, early: bool) -> None:
        self.updates = updates
        self.early = early
        self.rows: dict[str, SimpleNamespace] = {}
        self.items: dict[str, list[dict[str, Any]]] = {}
        self.committed_rows: dict[str, SimpleNamespace] = {}
        self.committed_items: dict[str, list[dict[str, Any]]] = {}
        updates.bind(self)

    async def _commit(self, run_id: str) -> None:
        if self.early:
            self.updates.enqueue([run_id])
            await self.updates.flush()  # before the data is committed
        self.committed_rows = dict(self.rows)
        self.committed_items = {key: list(value) for key, value in self.items.items()}
        if not self.early:
            self.updates.enqueue([run_id])
            await self.updates.flush()

    async def create_run(self, run_id: str, name: str, **kwargs: Any) -> SimpleNamespace:
        self.rows[run_id] = SimpleNamespace(run_id=run_id, storage_format=1, status="running")
        self.items[run_id] = []
        await self._commit(run_id)
        return self.rows[run_id]

    async def get_run(self, run_id: str) -> SimpleNamespace | None:
        return self.rows.get(run_id)

    async def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        return self.items.get(run_id, [])

    async def save_run_event(self, run_id: str, **kwargs: Any) -> None:
        self.items[run_id].append(kwargs)
        await self._commit(run_id)

    async def finalize_run(self, run_id: str, status: str, **kwargs: Any) -> None:
        self.rows[run_id] = SimpleNamespace(run_id=run_id, storage_format=1, status=status)
        await self._commit(run_id)

    async def close(self) -> None:
        await self.updates.close()


class CommittedReader:
    def __init__(self, store: StagedStore) -> None:
        self.store = store

    async def get_run(self, run_id: str) -> SimpleNamespace | None:
        return self.store.committed_rows.get(run_id)

    async def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.committed_items.get(run_id, [])


VISIBILITY_CONTRACTS = (
    "test_live_a_created_run_is_announced_after_its_row_is_visible_elsewhere",
    "test_live_a_run_event_is_announced_after_the_event_is_visible_elsewhere",
    "test_live_a_status_change_is_announced_after_the_new_status_is_visible_elsewhere",
)


async def run_visibility_contracts(
    *, early: bool, reader: str, needs_second: bool
) -> dict[str, str]:
    """Outcome of each visibility contract: passed, failed or skipped."""
    from dendrux.testing.store_kit._declaration import LIVE_FACTORY, SECOND_INSTANCE
    from dendrux.testing.store_kit._kit import Kit
    from dendrux.testing.store_kit.contracts import test_live_updates as contracts

    def second(store: StagedStore) -> CommittedReader:
        if reader == "raises":
            raise RuntimeError("an in-memory database has no second connection")
        return CommittedReader(store)

    provided: dict[str, Any] = {LIVE_FACTORY: lambda updates: StagedStore(updates, early=early)}
    if reader != "none":
        provided[SECOND_INSTANCE] = second
    declaration = SimpleNamespace(needs_second_instance=needs_second, has=lambda name: False)
    outcomes: dict[str, str] = {}
    for name in VISIBILITY_CONTRACTS:
        kit = Kit(provided, None, declaration, needs_format_2=False)
        try:
            await getattr(contracts, name)(kit)
            outcomes[name] = "passed"
        except AssertionError:
            outcomes[name] = "failed"
        except pytest.skip.Exception as skipped:
            assert "unverified:" in str(skipped)
            outcomes[name] = "skipped"
    return outcomes


async def test_an_independent_reader_is_used_even_for_a_single_process_store() -> None:
    """The reviewer's case: staged reads hide the early announcement unless another reader looks."""
    outcomes = await run_visibility_contracts(early=True, reader="available", needs_second=False)
    assert set(outcomes.values()) == {"failed"}, outcomes


async def test_a_store_that_announces_after_commit_passes_with_an_independent_reader() -> None:
    outcomes = await run_visibility_contracts(early=False, reader="available", needs_second=False)
    assert set(outcomes.values()) == {"passed"}, outcomes


@pytest.mark.parametrize("reader", ["none", "raises"])
async def test_without_an_independent_reader_visibility_is_untested_not_passed(reader: str) -> None:
    outcomes = await run_visibility_contracts(early=True, reader=reader, needs_second=False)
    assert set(outcomes.values()) == {"skipped"}, outcomes
