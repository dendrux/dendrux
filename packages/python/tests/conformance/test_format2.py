"""Fresh databases use format 2: pin its behavior on SQLite and serial Postgres."""

import pytest

from dendrux.runtime.stores._factory import store_for_engine
from tests._fixtures.storage import pin_storage_format
from tests.conformance import format2
from tests.conformance import harness as h
from tests.conformance.edge_scenarios import EDGE_CELLS, shared
from tests.conformance.scenarios import CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.id for c in CELLS])
async def test_format2_matches_golden_and_shared_contract(cell, backend_engine):
    await pin_storage_format(backend_engine, 2)
    store = store_for_engine(backend_engine)
    recording = await h.run_cell(cell, backend_engine, store)
    actual = await format2.view(recording, store)
    format2.assert_shared_contract(cell.id, actual)
    expected = h.read_json(h.golden_path(cell.id, "format2"))
    assert h.diff(expected, actual) is None, f"{cell.id}: {h.diff(expected, actual)}"
    assert h.o.dumps_canonical(expected) == h.o.dumps_canonical(actual)


@pytest.mark.parametrize("cell", EDGE_CELLS, ids=[c.id for c in EDGE_CELLS])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_extended_format_contract(cell, storage_format, backend_engine):
    await pin_storage_format(backend_engine, storage_format)
    store = store_for_engine(backend_engine)
    actual = await format2.view(await h.run_cell(cell, backend_engine, store), store)
    version = "format2" if storage_format == 2 else "format1_additions"
    assert h.diff(h.read_json(h.golden_path(cell.id, version)), actual) is None
    legacy = h.read_json(h.golden_path(cell.id, "format1_additions"))
    assert h._norm(shared(actual)) == h._norm(shared(legacy))
