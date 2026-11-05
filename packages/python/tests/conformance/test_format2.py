"""Fresh databases use format 2: pin its behavior on SQLite and serial Postgres."""

import pytest

from dendrux.runtime.stores._factory import store_for_engine
from tests._fixtures.storage import pin_storage_format
from tests.conformance import format2
from tests.conformance import harness as h
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
