"""The conformance matrix through custom stores that predate 0.6 capability declarations.

Fixtures (a) ``LegacyInMemoryStore`` (exactly the 0.5 protocol signatures) and (d)
``KwargsLegacyStore`` (every method ``(*args, **kwargs)``) run every cell on SQLite. Both
must equal the base golden, except where a protocol-only store loses information 0.5 reads
through optional methods; those cells have a pinned ``golden/legacy`` file (approval pauses:
the stored arguments cannot be read, so ``args_hidden`` is true and ``params`` is None).
The run-meta ``dendrux.prefix`` entry is also absent on such stores (it is written through the
optional ``update_run_meta``); that is checked below rather than pinned per cell.

Cells tagged ``blob_catalog`` (images) need the built-in store's blob catalog and are skipped.
Phase 1 must keep these results: a store with no declaration is a legacy store.
"""

from __future__ import annotations

import pytest

from dendrux.runtime.stores._factory import store_for_engine
from tests._fixtures import stores
from tests._fixtures.stores import KwargsLegacyStore, LegacyInMemoryStore
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS

LEGACY_CELLS = [c for c in CELLS if "blob_catalog" not in c.tags]
CELL_IDS = [c.id for c in LEGACY_CELLS]
SMOKE = ["plain_answer__run", "client_tool__run_resume", "human_input__run_submit_input"]
STORES = {"a_legacy_in_memory": LegacyInMemoryStore, "d_kwargs_legacy": KwargsLegacyStore}


def expected_golden(cell_id: str):
    legacy = h.golden_path(cell_id, "legacy")
    return h.read_json(legacy if legacy.exists() else h.golden_path(cell_id))


@pytest.mark.parametrize("fixture", sorted(STORES))
@pytest.mark.parametrize("cell", LEGACY_CELLS, ids=CELL_IDS)
async def test_legacy_store_matches_golden(cell, fixture, sqlite_engine) -> None:
    store = STORES[fixture](store_for_engine(sqlite_engine))
    assert not hasattr(store, "capabilities")
    recording = await h.run_cell(cell, sqlite_engine, store)
    golden = expected_golden(cell.id)
    actual = h.golden_view(recording)
    assert h.diff(golden, actual) is None, f"{cell.id}: {h.diff(golden, actual)}"


#: Keyword names that do not exist on any 0.5 store method (contracts.md section 3).
NEW_IN_060 = {"trace_id", "row_id", "event_id", "trace_kind", "expected_claim_epoch"}
#: The only 0.5 methods that take ``claim_epoch``.
FENCED_050 = {
    "finalize_run",
    "finalize_run_if_status_in",
    "finalize_with_event",
    "try_pause_run",
    "pause_with_event",
    "pause_run",
}


@pytest.mark.parametrize("fixture", sorted(STORES))
@pytest.mark.parametrize("cell_id", ["client_tool__two_pauses", "approval__submit_approve"])
async def test_legacy_store_only_ever_receives_zero_five_keywords(
    fixture, cell_id, sqlite_engine
) -> None:
    """Across multi-path flows no keyword that is new in 0.6 reaches a legacy store."""
    cell = next(c for c in CELLS if c.id == cell_id)
    store = STORES[fixture](store_for_engine(sqlite_engine))
    await h.run_cell(cell, sqlite_engine, store)
    assert store.calls, "the fixture recorded no calls"
    for name, keys in store.calls:
        assert not set(keys) & NEW_IN_060, (name, set(keys) & NEW_IN_060)
        if "claim_epoch" in keys:
            assert name in FENCED_050, name
    called = {name for name, _ in store.calls}
    assert not called & {"update_run_meta", "release_run_claim"}, "optional methods are absent"


@pytest.mark.parametrize("name", sorted(stores.STORE_FIXTURES))
@pytest.mark.parametrize("cell_id", SMOKE)
async def test_every_fixture_runs_a_flow_on_050_which_ignores_declarations(
    name, cell_id, sqlite_engine
) -> None:
    """0.5 reads no declaration: all five behave as 0.5 legacy stores (statuses agree)."""
    cell = next(c for c in CELLS if c.id == cell_id)
    if name == "c_fenced_rejects_epoch":
        from dendrux.errors import StoreCapabilityError

        with pytest.raises(StoreCapabilityError):
            await h.run_cell(
                cell,
                sqlite_engine,
                stores.STORE_FIXTURES[name](store_for_engine(sqlite_engine)),
            )
        return
    recording = await h.run_cell(
        cell, sqlite_engine, stores.STORE_FIXTURES[name](store_for_engine(sqlite_engine))
    )
    expected = h.read_json(h.golden_path(cell_id))
    got = [s.get("result", {}).get("status") for s in recording["steps"]]
    want = [s.get("result", {}).get("status") for s in expected["steps"]]
    assert got == want
    golden = h.golden_view(recording)
    assert [c["messages_sha256"] for c in golden["provider_calls"]] == [
        c["messages_sha256"] for c in expected["provider_calls"]
    ]
