"""v0.3.0 text-only parity for persisted rows.

Runs the same R-05 scenario matrix as ``tests/unit/test_v030_text_parity.py``
on the integration ``engine`` matrix (in-memory SQLite always; Postgres
``dendrux_test`` when ``DENDRUX_TEST_PG_URL`` is set) and compares:

- every row of ``agent_runs`` (incl. ``meta`` prefix/seed hashes,
  ``idempotency_fingerprint`` and ``pause_data``), ``react_traces``,
  ``tool_calls``, ``token_usage``, ``llm_interactions`` (semantic request/response,
  provider request/response) and ``run_events`` against the v0.3.0 goldens;
- the wire section again, because pause state and replayed history now round-trip
  through the database (a dialect-specific JSON difference would surface there);
- that tables which did not exist in v0.3.0 stay empty for text-only runs.

A new nullable column is tolerated only while it is NULL on every text-only row;
JSON columns must be identical (no added keys). Goldens are written by the unit
test's regeneration mode, never here.
"""

from __future__ import annotations

import pytest

from tests._helpers import v030_oracle as o

CELL_IDS = [o.cell_name(r, m) for r, m in o.CELLS]


@pytest.mark.parametrize(("rig", "mode"), o.CELLS, ids=CELL_IDS)
async def test_text_only_rows_match_v030(engine, rig, mode) -> None:
    if o.regen_enabled():
        pytest.skip("goldens are written by tests/unit/test_v030_text_parity.py")
    name = o.cell_name(rig, mode)
    result = await o.run_cell(rig, mode, engine)

    golden_rows = o.read_golden("rows", name)
    assert list(golden_rows) == list(result.rows), "scenario set/order changed"
    for scenario, expected in golden_rows.items():
        diff = o.compare_rows(expected, result.rows[scenario])
        assert diff is None, f"{engine.dialect.name} {name}/{scenario}: {diff}"
        assert not any(result.extra_tables[scenario].values()), (
            f"{name}/{scenario}: text-only run wrote to new tables {result.extra_tables[scenario]}"
        )

    golden_wire = o.read_golden("wire", name)
    for scenario, expected in golden_wire.items():
        diff = o.compare_wire(expected, result.wire[scenario])
        assert diff is None, f"{engine.dialect.name} {name}/{scenario}: {diff}"
