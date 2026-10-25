"""One run's unconvertible metadata never breaks a metadata_filter scan (SQLite + Postgres).

The ``json`` column on Postgres accepts ``\\u0000`` and unpaired surrogate
escapes that ``jsonb`` rejects. ``list_runs`` / ``count_runs`` with a
``metadata_filter`` convert ``meta`` to ``jsonb``; such a row must simply not
match, and every other row (emoji, escaped backslashes, plain text) must
match exactly as before, scoped or unscoped.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import text

from dendrux.store import UNSCOPED, RunStore

pytestmark = pytest.mark.asyncio

# Stored JSON text of meta -> whether {"thread_id": "T"} should match it.
ROWS = {
    "plain": ('{"thread_id": "T"}', True),
    "emoji": ('{"thread_id": "T", "note": "\\ud83d\\ude00"}', True),
    "backslash_u": ('{"thread_id": "T", "note": "\\\\u0000 literal"}', True),
    "backslash_ud800": ('{"thread_id": "T", "note": "\\\\ud800"}', True),
    "nul": ('{"thread_id": "T", "x": "evil\\u0000"}', False),
    "lone_high": ('{"thread_id": "T", "x": "\\ud800"}', False),
    "lone_low": ('{"thread_id": "T", "x": "\\udc00"}', False),
    "nul_after_backslash": ('{"thread_id": "T", "x": "\\\\\\u0000"}', False),
    "other": ('{"thread_id": "U"}', False),
}


async def _rows(engine: Any, store: Any) -> None:
    for i, (name, (raw, _)) in enumerate(ROWS.items()):
        run_id = f"r{i:02d}".ljust(26, "0")
        await store.create_run(run_id, "agent", tenant_id="X" if name != "plain" else "V")
        cast = "CAST(:m AS json)" if engine.dialect.name == "postgresql" else ":m"
        async with engine.begin() as conn:
            await conn.execute(
                text(f"UPDATE agent_runs SET meta = {cast} WHERE id = :i"), {"m": raw, "i": run_id}
            )


async def test_unconvertible_rows_never_break_a_filtered_scan(store: Any, engine: Any) -> None:
    await _rows(engine, store)
    run_store = RunStore.from_engine(engine)
    expected = sorted(
        f"r{i:02d}".ljust(26, "0") for i, (_, (_, hit)) in enumerate(ROWS.items()) if hit
    )
    if engine.dialect.name != "postgresql":
        # SQLite json_extract tolerates every row; the unconvertible ones match too.
        expected = sorted(
            f"r{i:02d}".ljust(26, "0") for i, (n, _) in enumerate(ROWS.items()) if n != "other"
        )
    found = await run_store.list_runs(tenant_id=UNSCOPED, metadata_filter={"thread_id": "T"})
    assert sorted(r.run_id for r in found) == expected
    assert await run_store.count_runs(
        tenant_id=UNSCOPED, metadata_filter={"thread_id": "T"}
    ) == len(expected)
    scoped = await run_store.list_runs(tenant_id="V", metadata_filter={"thread_id": "T"})
    assert [r.run_id for r in scoped] == ["r00".ljust(26, "0")]
    assert json.loads(ROWS["emoji"][0])["note"] == "\U0001f600"
