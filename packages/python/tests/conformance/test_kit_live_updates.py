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
    lease = await updates.acquire_listener()
    await eventually(lambda: updates.healthy)
    updates.enqueue(["R1"])
    await updates.flush()
    await asyncio.sleep(0.05)
    assert updates.stats().gaps == 1  # only the healthy transition
    await updates.release_listener(lease)
    await updates.close()


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


def test_a_single_process_store_is_read_back_through_itself(pytester: pytest.Pytester) -> None:
    result = run_live(pytester, kind="sqlite-memory")
    result.assert_outcomes(passed=CONTRACTS)


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
