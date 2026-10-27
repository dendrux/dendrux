"""Listing index preference at representative PostgreSQL cardinality and statistics."""

import datetime as dt
import json

import pytest
from sqlalchemy import insert, text

from dendrux.db.models import AgentRun


@pytest.mark.benchmark
async def test_listing_prefers_ordered_tenant_index_with_realistic_statistics(
    engine, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL EXPLAIN benchmark")
    now = dt.datetime.now(dt.UTC)
    async with engine.begin() as conn:
        for start in range(0, 20000, 1000):
            await conn.execute(
                insert(AgentRun),
                [
                    dict(
                        id=f"{i:026d}",
                        agent_name="listing",
                        tenant_id="selected" if i % 100 == 0 else f"tenant-{i % 100}",
                        status=("running", "success", "waiting_client_tool")[i % 3],
                        status_changed_at=now,
                        created_at=now - dt.timedelta(seconds=i),
                        input_data={"input": "representative text " * 20},
                        meta={"thread": f"thread-{i % 50}"},
                    )
                    for i in range(start, start + 1000)
                ],
            )
        await conn.execute(text("ANALYZE agent_runs"))
        settings = {
            key: (await conn.execute(text("SHOW " + key))).scalar_one()
            for key in ("enable_seqscan", "enable_indexscan", "enable_bitmapscan", "enable_sort")
        }
        plan = (
            await conn.execute(
                text(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM agent_runs "
                    "WHERE tenant_id='selected' ORDER BY created_at DESC, id DESC LIMIT 50"
                )
            )
        ).scalar_one()[0]
    encoded = json.dumps(plan)
    record_property("listing_20k_plan", encoded)
    record_property("listing_planner_settings", json.dumps(settings))
    assert settings == {key: "on" for key in settings}, "benchmark must use normal planner choices"
    assert "ix_agent_runs_tenant_id_created_at" in encoded
    assert plan["Plan"]["Actual Rows"] == 50

    def node_types(node):
        return [
            node["Node Type"],
            *(kind for child in node.get("Plans", []) for kind in node_types(child)),
        ]

    # The existing index orders created_at; id is a deterministic tie breaker,
    # so PostgreSQL may perform a bounded incremental sort of equal timestamps.
    assert "Sort" not in node_types(plan["Plan"]), (
        "listing must not sort the full tenant population"
    )
    print(
        "listing_20k",
        json.dumps(
            {
                "execution_ms": plan["Execution Time"],
                "hit_blocks": plan["Plan"]["Shared Hit Blocks"],
                "read_blocks": plan["Plan"]["Shared Read Blocks"],
            }
        ),
    )
