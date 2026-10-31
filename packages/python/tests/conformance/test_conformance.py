"""Conformance: every scenario flow on every entry path equals the recorded 0.5.0 behaviour.

Each cell is run once and compared with two committed records: the golden and the corpus.

The golden files (``tests/conformance/golden``) pin, per cell: the ``RunResult`` of every
entry-path call (status, answer, ``error_info``, pause id presence, pending shape, usage,
``to_dict()`` wire shape), the stream event sequences, and per run the persisted event-type
sequence, the trace roles and the tool-call rows. Tests never regenerate them; see
``regen.py`` (refused unless ``src/dendrux`` equals tag ``v0.5.0``).

SQLite always; Postgres when ``DENDRUX_TEST_PG_URL`` is set::

    DENDRUX_TEST_PG_URL=postgresql+asyncpg://postgres:root@localhost:5432/dendrux_060_0c_test \\
        PYTHONPATH=src pytest tests/conformance -p no:randomly
"""

from __future__ import annotations

import pytest

from dendrux.runtime.stores._factory import store_for_engine
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS

CELL_IDS = [c.id for c in CELLS]


def test_dendrux_comes_from_this_checkout() -> None:
    h.assert_imports_this_checkout()


def test_cell_ids_are_unique_and_named_scenario_flow() -> None:
    assert len(set(CELL_IDS)) == len(CELL_IDS)
    for c in CELLS:
        assert c.id.startswith(f"{c.scenario}__"), c.id


@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
async def test_cell_matches_golden_and_corpus(cell, backend_engine) -> None:
    recording = await h.run_cell(cell, backend_engine, store_for_engine(backend_engine))
    golden = h.read_json(h.golden_path(cell.id))
    actual = h.golden_view(recording)
    assert h.diff(golden, actual) is None, f"{cell.id} golden: {h.diff(golden, actual)}"
    assert h.o.dumps_canonical(golden) == h.o.dumps_canonical(actual)
    # the recorded corpus (provider messages and persisted rows) is reproduced as well
    recorded = h.normalised_corpus(h.read_json(h.corpus_path(cell.id)))
    fresh = h.normalised_corpus(h.corpus_view(recording))
    assert h.diff(recorded, fresh) is None, f"{cell.id} corpus: {h.diff(recorded, fresh)}"
