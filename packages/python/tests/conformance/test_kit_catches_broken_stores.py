"""The kit fails stores that are declared-but-broken (the revert check for each contract).

Each case runs a slice of the kit in an inner pytest session against the real file-SQLite store
with one promise deliberately broken. The same slice must pass on the unbroken store, so the
failure is the mutation's, and the failing contract must be the one that guards that promise.
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit  # noqa: F401

pytest_plugins = ["pytester"]

CONFTEST = """
import pytest
import pytest_asyncio
from datetime import timedelta
from dendrux.runtime.state import ClaimResult
from dendrux.errors import PersistenceError
from tests.conformance import _kit_backends as kb
from dendrux.testing.store_kit.sql import SqlFaultInjector

{mutation}

{injector}

@pytest_asyncio.fixture
async def kit_backend(tmp_path):
    backend = kb.KitBackend("sqlite-file", tmp_path, wrap=mutate)
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
NONE = "def mutate(store):\n    return store\n"
PLAIN_INJECTOR = "Injector = SqlFaultInjector"


def mutant(methods: str) -> str:
    return (
        "def mutate(store):\n    base = type(store)\n    class Mutant(base):\n"
        + textwrap.indent(textwrap.dedent(methods), "        ")
        + "\n    store.__class__ = Mutant\n    return store\n"
    )


MUTANTS = {
    "a stale epoch is accepted by save_run_event": (
        mutant("""
        async def save_run_event(self, run_id, *, claim_epoch=None, event_id=None, **kw):
            return await super().save_run_event(run_id, event_id=event_id, **kw)
        """),
        "stale_epoch_is_refused and save_run_event",
        {"test_a_stale_epoch_is_refused_by_every_fenced_writer"},
        PLAIN_INJECTOR,
    ),
    "a stale finalize is accepted": (
        mutant("""
        async def finalize_run(self, run_id, *, claim_epoch=None, **kw):
            return await super().finalize_run(run_id, **kw)
        """),
        "stale_finalize_changes_nothing",
        {"test_a_stale_finalize_changes_nothing"},
        PLAIN_INJECTOR,
    ),
    "a trace retry is not recognised by its identity": (
        mutant("""
        async def save_trace(self, run_id, role, content, *, order_index, trace_kind="turn",
                             claim_epoch=None, trace_id=None, **kw):
            return await super().save_trace(
                run_id, role, content, order_index=order_index, trace_kind=trace_kind,
                claim_epoch=claim_epoch, **kw,
            )
        """),
        "same_identity_stores_once and save_trace",
        {"test_a_repeated_write_with_the_same_identity_stores_once"},
        PLAIN_INJECTOR,
    ),
    "a changed trace under the same identity is swallowed": (
        mutant("""
        async def save_trace(self, run_id, role, content, *, order_index, trace_id=None,
                             claim_epoch=None, trace_kind="turn", **kw):
            try:
                return await super().save_trace(
                    run_id, role, content, order_index=order_index, trace_id=trace_id,
                    claim_epoch=claim_epoch, trace_kind=trace_kind, **kw,
                )
            except PersistenceError:
                return trace_id
        """),
        "different_content_conflicts and save_trace",
        {"test_the_same_identity_with_different_content_conflicts"},
        PLAIN_INJECTOR,
    ),
    "the note cap is ignored": (
        mutant("""
        async def add_note(self, run_id, *, note_id, text, max_notes=None, claim_epoch=None, **kw):
            return await super().add_note(
                run_id, note_id=note_id, text=text, claim_epoch=claim_epoch, **kw
            )
        """),
        "note_cap",
        {"test_the_note_cap_counts_exactly"},
        PLAIN_INJECTOR,
    ),
    "every claim wins": (
        mutant("""
        async def claim_paused_run(self, run_id, **kw):
            got = await super().claim_paused_run(run_id, **kw)
            return ClaimResult(1, 2, None, None) if got is False else got
        """),
        "claim_has_one_winner or parallel_claims_have",
        {
            "test_pause_state_is_stored_and_a_claim_has_one_winner",
            "test_parallel_claims_have_exactly_one_winner",
        },
        PLAIN_INJECTOR,
    ),
    "tenants are trimmed and case-folded": (
        mutant("""
        async def list_runs(self, *, tenant_id=None, **kw):
            if isinstance(tenant_id, str):
                tenant_id = tenant_id.strip().lower()
            return await super().list_runs(tenant_id=tenant_id, **kw)
        """),
        "tenant_scope_is_exact",
        {"test_tenant_scope_is_exact_not_padded_or_case_folded"},
        PLAIN_INJECTOR,
    ),
    "sweeps ignore their age threshold": (
        mutant("""
        async def sweep_stale_runs(self, older_than, **kw):
            return await super().sweep_stale_runs(timedelta(0), **kw)
        """),
        "sweeping_stale_running",
        {"test_sweeping_stale_running_runs_spares_fresh_ones"},
        PLAIN_INJECTOR,
    ),
    "finalize_with_event is two transactions": (
        mutant("""
        async def finalize_with_event(self, run_id, *, status, allowed_current_statuses,
                                      event_type, event_data=None, claim_epoch=None,
                                      trace=None, event=None, **kw):
            ok = await self.finalize_run_if_status_in(
                run_id, status=status, allowed_current_statuses=allowed_current_statuses,
                claim_epoch=claim_epoch, answer=kw.get("answer"),
            )
            if not ok:
                return None
            if trace is not None:
                await self.save_trace(
                    run_id, trace.role, trace.content, order_index=trace.order_index,
                    trace_id=trace.trace_id, claim_epoch=claim_epoch,
                )
            return await self.save_run_event(
                run_id, event_type=event_type, data=event_data, claim_epoch=claim_epoch
            )
        """),
        "finalize_with_event_is_all_or_nothing and before_commit and 1",
        {"test_finalize_with_event_is_all_or_nothing"},
        PLAIN_INJECTOR,
    ),
    "a lost acknowledgement duplicates an event": (
        mutant("""
        async def save_run_event(self, run_id, *, claim_epoch=None, event_id=None, **kw):
            return await super().save_run_event(run_id, claim_epoch=claim_epoch, **kw)
        """),
        "failed_write_is_repeated_safely and save_run_event and after_commit",
        {"test_a_failed_write_is_repeated_safely_with_the_same_identity"},
        PLAIN_INJECTOR,
    ),
    "pause_run is not fenced": (
        mutant("""
        async def pause_run(self, run_id, *, claim_epoch=None, **kw):
            return await super().pause_run(run_id, **kw)
        """),
        "stale_pause",
        {"test_a_stale_pause_run_changes_nothing"},
        PLAIN_INJECTOR,
    ),
    "pause_with_event is not fenced": (
        mutant("""
        async def pause_with_event(self, run_id, *, claim_epoch=None, **kw):
            return await super().pause_with_event(run_id, **kw)
        """),
        "stale_pause",
        {"test_a_stale_pause_with_event_changes_nothing"},
        PLAIN_INJECTOR,
    ),
    "mark_notes_delivered is not fenced": (
        mutant("""
        async def mark_notes_delivered(self, run_id, note_ids, *, iteration, claim_epoch=None):
            return await super().mark_notes_delivered(run_id, note_ids, iteration=iteration)
        """),
        "stale_epoch_is_refused and mark_notes_delivered",
        {"test_a_stale_epoch_is_refused_by_every_fenced_writer"},
        PLAIN_INJECTOR,
    ),
}


def run_slice(
    pytester: pytest.Pytester, mutation: str, k: str, injector: str = PLAIN_INJECTOR
) -> pytest.RunResult:
    pytester.makeconftest(CONFTEST.format(mutation=mutation, injector=injector))
    return pytester.runpytest(
        "-p", "dendrux.testing.store_kit", "-p", "no:randomly", "-p", "no:cacheprovider",
        "-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function",
        "--no-cov", "-q", "-k", k, "-rf",
    )  # fmt: skip


def failed_functions(result: pytest.RunResult) -> set[str]:
    """Names (without parameters) of the contracts that failed."""
    found = re.findall(r"^FAILED \S+::(\w+)", result.stdout.str(), flags=re.MULTILINE)
    return set(found)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_the_kit_fails_a_broken_store_on_exactly_the_contracts_that_guard_the_promise(
    pytester: pytest.Pytester, name: str
) -> None:
    mutation, k, expected, injector = MUTANTS[name]
    broken = run_slice(pytester, mutation, k, injector)
    text = broken.stdout.str()
    assert failed_functions(broken) == expected, text[-2500:]
    if expected:
        assert "NOT CERTIFIED" in text


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_the_same_slice_passes_on_the_unbroken_store(pytester: pytest.Pytester, name: str) -> None:
    _, k, _, _ = MUTANTS[name]
    intact = run_slice(pytester, NONE, k)
    outcomes = intact.parseoutcomes()
    assert outcomes.get("failed", 0) == 0 and outcomes.get("passed", 0) >= 1, intact.stdout.str()


DUPLICATE_USAGE = mutant("""
    async def save_usage(self, run_id, *, iteration_index, usage, usage_id=None,
                         claim_epoch=None, **kw):
        await super().save_usage(
            run_id, iteration_index=iteration_index, usage=usage, usage_id=usage_id,
            claim_epoch=claim_epoch, **kw,
        )
        await super().save_usage(
            run_id, iteration_index=iteration_index, usage=usage, claim_epoch=claim_epoch, **kw
        )  # a second row under a fresh identity
""")


def test_usage_rows_cannot_be_read_so_usage_retry_identity_is_reported_unverified(
    pytester: pytest.Pytester,
) -> None:
    """The protocol has no read of usage rows, so duplicates cannot be seen: say so, never pass."""
    broken = run_slice(pytester, DUPLICATE_USAGE, "save_usage")
    text = broken.stdout.str()
    assert failed_functions(broken) == set(), "no portable read can see the duplicate rows"
    assert "unverified: save_usage" in text, text[-1500:]
    assert "usage rows" in text
    # the caveat travels with the verdict line when the rest of the group is certified
    intact = run_slice(pytester, NONE, "save_usage")
    assert "unverified: save_usage" in intact.stdout.str()
