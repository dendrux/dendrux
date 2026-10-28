"""Record the conformance goldens and the 0.5 corpus (maintainer command, not a test).

Run from ``packages/python`` on a checkout whose ``src/dendrux`` equals tag ``v0.5.0``::

    DENDRUX_REGEN_CONFORMANCE_050=1 PYTHONPATH=src:. python -m tests.conformance.regen

Refused on any other tree: a candidate must never bless its own output. Writes one golden per
cell (``tests/conformance/golden``), one raw corpus file per cell and ``manifest.json``
(``tests/_fixtures/corpus_050``). Recording runs on in-memory SQLite.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys

from dendrux.runtime.state import SQLAlchemyStateStore
from tests._fixtures.stores import KwargsLegacyStore, LegacyInMemoryStore
from tests._helpers import v030_oracle as o
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS

LEGACY_VARIANTS = {"a_legacy_in_memory": LegacyInMemoryStore, "d_kwargs_legacy": KwargsLegacyStore}


async def record_legacy(base: dict[str, dict]) -> None:
    """Goldens of the legacy-store variants, stored once, only for cells that differ.

    Fixtures (a) and (d) must agree on every cell: one ``golden/legacy`` file serves both.
    """
    variant_dir = h.GOLDEN_DIR / "legacy"
    for stale in variant_dir.glob("*.json"):
        stale.unlink()
    for cell in CELLS:
        if "blob_catalog" in cell.tags:
            continue
        views = {}
        for name, store_cls in LEGACY_VARIANTS.items():
            engine = await o.create_sqlite_engine()
            try:
                store = store_cls(SQLAlchemyStateStore(engine))
                views[name] = h.golden_view(await h.run_cell(cell, engine, store))
            finally:
                await engine.dispose()
        canon = {n: h.o.dumps_canonical(v) for n, v in views.items()}
        if len(set(canon.values())) != 1:
            raise SystemExit(f"legacy fixtures (a) and (d) disagree on {cell.id}")
        if canon["a_legacy_in_memory"] != h.o.dumps_canonical(base[cell.id]):
            h.write_json(h.golden_path(cell.id, "legacy"), views["a_legacy_in_memory"])
            print(f"recorded legacy delta for {cell.id}")


async def record() -> None:
    manifest = []
    base: dict[str, dict] = {}
    for cell in CELLS:
        engine = await o.create_sqlite_engine()
        try:
            recording = await h.run_cell(cell, engine, SQLAlchemyStateStore(engine))
        finally:
            await engine.dispose()
        base[cell.id] = h.golden_view(recording)
        h.write_json(h.golden_path(cell.id), base[cell.id])
        corpus = h.corpus_view(recording)
        h.write_json(h.corpus_path(cell.id), corpus)
        body = h.corpus_path(cell.id).read_bytes()
        manifest.append(
            {
                "id": cell.id,
                "scenario": cell.scenario,
                "file": h.corpus_path(cell.id).name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "provider_calls": len(corpus["calls"]),
                "steps": len(corpus["steps"]),
                "runs": len(corpus["runs"]),
                "entry_paths": sorted({s["path"] for s in corpus["steps"]}),
                # the loop type is the existing `strategy` column (the class name in 0.5)
                "strategies": sorted(
                    {r["strategy"] for r in corpus["rows"]["agent_runs"] if r["strategy"]}
                ),
                "loops": sorted(
                    {
                        (r["meta"] or {}).get("dendrux.loop")
                        for r in corpus["rows"]["agent_runs"]
                        if (r["meta"] or {}).get("dendrux.loop")
                    }
                ),
                "on_governance_change": sorted(
                    {a["on_governance_change"] for a in corpus["agents"]}
                ),
            }
        )
        print(f"recorded {cell.id}")
    h.write_json(
        h.CORPUS_DIR / "manifest.json",
        {
            "dendrux_version": h.RECORDED_VERSION,
            "recorded_from": f"tag {h.RECORDED_TAG}",
            "cells": manifest,
        },
    )
    await record_legacy(base)


def main() -> int:
    if os.environ.get(h.REGEN_ENV) != "1":
        print(f"refusing: set {h.REGEN_ENV}=1 to record", file=sys.stderr)
        return 2
    refusal = h.regen_refusal()
    if refusal:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 3
    h.assert_imports_this_checkout()
    asyncio.run(record())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
