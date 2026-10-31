"""What the kit prints and returns: honest verdicts, a gating exit code, one setup message.

Each case runs the kit in an inner pytest session (``pytester``) against the real SQLite store,
the way a store author runs it. A declared capability is certified only when its contracts really
ran: contracts that the store itself caused to skip (format-1 runs from a store that declares the
four format-2 flags, no disconnect forms, no ``release_run_claim``) make that capability NOT
CERTIFIED, and any session that prints NOT CERTIFIED exits non-zero.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tests.conformance.test_kit_catches_broken_stores import (
    NATIVE_FORMS,
    NONE,
    PLAIN_INJECTOR,
    failed_functions,
    mutant,
)
from tests.conformance.test_kit_plugin import AUTHOR, XDIST_CAPS

if TYPE_CHECKING:
    from _pytest.pytester import Pytester, RunResult

pytest_plugins = ["pytester"]

INNER_ARGS = (
    "-p", "dendrux.testing.store_kit", "-p", "no:randomly", "-p", "no:cacheprovider",
    "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
    "--no-cov", "-q",
)  # fmt: skip

CONFTEST = """
import pytest
import pytest_asyncio
from datetime import timedelta
from dendrux.runtime.state import ClaimResult
from dendrux.testing.store_kit.sql import SqlFaultInjector
from tests.conformance import _kit_backends as kb

{mutation}

{injector}

@pytest_asyncio.fixture
async def kit_backend(tmp_path):
    backend = kb.KitBackend({kind!r}, tmp_path, wrap=mutate)
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
@pytest.fixture
def dendrux_fault_injector():
    injector = Injector()
    yield injector
    injector.close()
"""

#: The reviewer's store: it declares recovery and checkpointing, fakes the reclaim and does not
#: write the checkpoint. Run at format 1 the kit used to certify it (every format-2 contract
#: skipped); at format 2 it fails the contracts that guard both promises.
FAKE_RECLAIM_NO_OP_CHECKPOINT = mutant("""
    async def reclaim_run(self, run_id, **kw):
        return ClaimResult(1, 2, None, None)

    async def checkpoint_step(self, run_id, *, claim_epoch, **kw):
        return None
""")


def run_kit(
    pytester: Pytester,
    *,
    kind: str = "sqlite-file",
    mutation: str = NONE,
    injector: str | None = PLAIN_INJECTOR,
    k: str | None = None,
) -> RunResult:
    conftest = CONFTEST.format(kind=kind, mutation=mutation, injector=injector or "")
    if injector is None:  # an author who has no fault injector provides no fixture
        conftest = conftest.split("@pytest.fixture\ndef dendrux_fault_injector")[0]
    pytester.makeconftest(conftest)
    return pytester.runpytest(*INNER_ARGS, *(["-k", k] if k else []), "-rf")


# -- a format-1 store that declares the four format-2 flags ---------------------------------


def test_a_store_that_declares_format_2_but_creates_format_1_runs_is_not_certified(
    pytester: Pytester,
) -> None:
    result = run_kit(pytester, kind="sqlite-format1")
    text = result.stdout.str()
    assert result.parseoutcomes().get("failed", 0) == 0, text[-2500:]
    assert "NOT CERTIFIED" in text, text[-2500:]
    assert "skipped by the store" in text and "storage-format-1" in text
    for capability in ("checkpointing", "atomic_completion", "recovery"):
        assert re.search(rf"{capability}: \d+ contract", text), (capability, text[-2500:])
    assert "  CERTIFIED" not in text
    assert result.ret != 0


def test_the_reviewers_broken_store_is_not_certified_at_format_1(pytester: Pytester) -> None:
    """Fake reclaim and no-op checkpoint behind format 1: nothing fails, nothing certifies."""
    result = run_kit(pytester, kind="sqlite-format1", mutation=FAKE_RECLAIM_NO_OP_CHECKPOINT)
    text = result.stdout.str()
    assert failed_functions(result) == set(), text[-2500:]
    assert "NOT CERTIFIED" in text and "recovery:" in text and "checkpointing:" in text
    assert result.ret != 0


# -- store-controlled skips inside the crash contracts ---------------------------------------

NO_FORMS = """
class Injector(SqlFaultInjector):
    def disconnect_forms(self, store):
        return ()
"""
NO_RELEASE = mutant("""
    @property
    def release_run_claim(self):
        raise AttributeError("release_run_claim")
""")


def test_an_injector_with_no_disconnect_forms_leaves_the_claims_uncertified(
    pytester: Pytester,
) -> None:
    result = run_kit(pytester, injector=NO_FORMS, k="ambiguous_claim_commit")
    text = result.stdout.str()
    assert failed_functions(result) == set(), text[-2000:]
    assert "skipped by the store" in text and "disconnect_forms" in text, text[-2000:]
    assert re.search(r"checkpointing: \d+ contract", text) and "NOT CERTIFIED" in text
    assert result.ret != 0


def test_a_store_without_release_run_claim_leaves_recovery_uncertified(
    pytester: Pytester,
) -> None:
    result = run_kit(
        pytester, mutation=NO_RELEASE, injector=NATIVE_FORMS, k="ambiguous_claim_commit"
    )
    text = result.stdout.str()
    assert failed_functions(result) == set(), text[-2000:]
    assert "skipped by the store" in text and "release_run_claim" in text, text[-2000:]
    assert re.search(r"recovery: \d+ contract", text) and "NOT CERTIFIED" in text
    assert result.ret != 0


# -- exit code and the caveated verdict --------------------------------------------------------


def test_a_complete_run_with_caveats_prints_certified_with_caveats_and_exits_zero(
    pytester: Pytester,
) -> None:
    result = run_kit(pytester)
    text = result.stdout.str()
    assert "NOT CERTIFIED" not in text, text[-2500:]
    assert "CERTIFIED with caveats:" in text and "unverified: save_usage" in text, text[-2500:]
    assert result.ret == 0


def test_a_complete_run_with_no_fault_injector_is_a_caveat_not_a_failure(
    pytester: Pytester,
) -> None:
    result = run_kit(pytester, injector=None)
    assert "CERTIFIED with caveats:" in result.stdout.str()


def test_a_partial_run_prints_not_certified_and_exits_non_zero(pytester: Pytester) -> None:
    result = run_kit(pytester, k="test_create_run_round_trips_what_it_was_given")
    assert result.parseoutcomes().get("failed", 0) == 0
    assert "NOT CERTIFIED" in result.stdout.str() and "incomplete run" in result.stdout.str()
    assert result.ret != 0


def test_failing_contracts_still_exit_non_zero(pytester: Pytester) -> None:
    stale = mutant("""
    async def finalize_run(self, run_id, *, claim_epoch=None, **kw):
        return await super().finalize_run(run_id, **kw)
    """)
    result = run_kit(pytester, mutation=stale, k="stale_finalize_changes_nothing")
    assert result.ret != 0


# -- no fixtures at all ---------------------------------------------------------------------


def test_a_missing_factory_prints_one_setup_message_not_one_failure_per_contract(
    pytester: Pytester,
) -> None:
    pytester.makeconftest("")
    result = pytester.runpytest(*INNER_ARGS)
    text = result.stdout.str()
    outcomes = result.parseoutcomes()
    assert outcomes.get("failed") == 1 and "passed" not in outcomes, text[-2500:]
    assert text.count("no dendrux_store_factory fixture exists") <= 2, text[-2500:]
    assert "build-your-own-store" in text
    assert result.ret != 0


# -- a store that declares notify ------------------------------------------------------------

DECLARES_NOTIFY = """
def mutate(store):
    from dataclasses import replace
    store.capabilities = replace(store.capabilities, notify=True)
    return store
"""
IMPLEMENTS_NOTIFY = """
def mutate(store):
    from dataclasses import replace
    base = type(store)

    class Notifying(base):
        async def _after_event_insert(self, session, run_id):
            return None

    store.__class__ = Notifying
    store.capabilities = replace(store.capabilities, notify=True)
    return store
"""


def test_a_store_that_declares_notify_without_implementing_it_fails(pytester: Pytester) -> None:
    result = run_kit(pytester, mutation=DECLARES_NOTIFY, k="notify")
    text = result.stdout.str()
    assert failed_functions(result) == {"test_declared_notify_is_implemented"}, text[-2000:]
    assert "_after_event_insert" in text and "notify=False" in text


def test_a_store_that_implements_the_notify_hook_passes_that_check(pytester: Pytester) -> None:
    result = run_kit(pytester, mutation=IMPLEMENTS_NOTIFY, k="notify")
    assert failed_functions(result) == set(), result.stdout.str()[-2000:]
    assert result.parseoutcomes().get("passed", 0) == 1


# -- distinct implementations with one class name -------------------------------------------

TWO_STORES_ONE_NAME = """
import sys, types

_count = {"n": 0}


def mutate(store):
    _count["n"] += 1
    module_name = "vendor_a" if _count["n"] % 2 else "vendor_b"
    module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
    base = type(store)
    cls = type("SameName", (base,), {"__module__": module_name})
    setattr(module, "SameName", cls)
    store.__class__ = cls
    return store
"""


def test_distinct_stores_that_share_a_class_name_get_distinct_reports(
    pytester: Pytester,
) -> None:
    result = run_kit(
        pytester,
        mutation=TWO_STORES_ONE_NAME,
        k="test_deployment_is_declared or test_declared_methods_match",
    )
    text = result.stdout.str()
    summary = text.split("certification summary", 1)[1]
    assert "vendor_a.SameName" in summary and "vendor_b.SameName" in summary, summary
    assert summary.count("  declaration") == 2, summary


# -- the replayed-tool-call read --------------------------------------------------------------

NO_REPLAYS = mutant("""
    async def get_replayed_tool_calls(self, run_id, *, limit=100, tenant_id=None):
        return [], False
""")


def test_a_store_that_never_reports_replayed_calls_fails_the_replay_contract(
    pytester: Pytester,
) -> None:
    result = run_kit(pytester, mutation=NO_REPLAYS, k="replayed_tool_calls")
    text = result.stdout.str()
    assert failed_functions(result) == {"test_replayed_tool_calls_are_listed_bounded_and_scoped"}, (
        text[-2000:]
    )


def test_the_replay_contract_passes_on_the_unbroken_store(pytester: Pytester) -> None:
    result = run_kit(pytester, k="replayed_tool_calls")
    assert result.parseoutcomes().get("passed", 0) >= 1, result.stdout.str()[-1500:]
    assert failed_functions(result) == set()


# -- under xdist ----------------------------------------------------------------------------


def test_under_xdist_a_partial_run_exits_non_zero(pytester: Pytester) -> None:
    pytester.makeconftest(AUTHOR.format(caps=XDIST_CAPS, body=""))  # workers import no repo code
    result = pytester.runpytest(*INNER_ARGS, "-n", "2", "-k", "test_create_run_round_trips")
    assert "NOT CERTIFIED" in result.stdout.str() and result.ret != 0


def test_under_xdist_a_missing_factory_still_ends_the_session_with_an_error(
    pytester: Pytester,
) -> None:
    pytester.makeconftest("")
    result = pytester.runpytest(*INNER_ARGS, "-n", "2")
    assert result.ret != 0
    assert result.parseoutcomes().get("failed", 0) <= 2  # at most one per worker
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
