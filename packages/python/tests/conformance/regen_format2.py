"""Record format-2 goldens after checking the immutable format-1 contract.

Maintainer command from packages/python:
DENDRUX_REGEN_CONFORMANCE_FORMAT2=1 PYTHONPATH=src python -m tests.conformance.regen_format2

Writes golden/format2 and new edge-cell format1_additions only.
The v0.5 corpus and its source guard stay immutable.
Review every changed golden: the opt-in flag is not approval for new behavior.
"""

from __future__ import annotations

import asyncio
import os

from dendrux.runtime.stores._factory import store_for_engine
from tests._fixtures.storage import pin_storage_format
from tests.conformance import format2
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS


async def record() -> None:
    for cell in CELLS:
        engine = await h.o.create_sqlite_engine()
        try:
            await pin_storage_format(engine, 2)
            store = store_for_engine(engine)
            recording = await h.run_cell(cell, engine, store)
            actual = await format2.view(recording, store)
            format2.assert_shared_contract(cell.id, actual)
            h.write_json(h.golden_path(cell.id, "format2"), actual)
            print(f"recorded format2 {cell.id}")
        finally:
            await engine.dispose()


async def record_edges() -> None:
    from tests.conformance.edge_scenarios import EDGE_CELLS, shared

    for cell in EDGE_CELLS:
        legacy = None
        for version in (1, 2):
            engine = await h.o.create_sqlite_engine()
            try:
                await pin_storage_format(engine, version)
                store = store_for_engine(engine)
                actual = await format2.view(await h.run_cell(cell, engine, store), store)
                if version == 1:
                    legacy = actual
                else:
                    assert h._norm(shared(actual)) == h._norm(shared(legacy)), cell.id
                folder = "format2" if version == 2 else "format1_additions"
                h.write_json(h.golden_path(cell.id, folder), actual)
                print(f"recorded {folder} {cell.id}")
            finally:
                await engine.dispose()


def main() -> None:
    if os.environ.get("DENDRUX_REGEN_CONFORMANCE_FORMAT2") != "1":
        raise SystemExit("refusing: set DENDRUX_REGEN_CONFORMANCE_FORMAT2=1")
    h.assert_imports_this_checkout()
    asyncio.run(record())
    asyncio.run(record_edges())


if __name__ == "__main__":
    main()
