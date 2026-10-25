"""Root-first ancestor traversal, including incomplete legacy/cyclic links."""

import pytest
from sqlalchemy import select, update

from dendrux.db.models import AgentRun


@pytest.mark.parametrize("shape", ["root", "chain", "missing", "cycle", "self"])
async def test_walk_ancestry_behaviors(store, shape):
    await store.create_run("root", "root-agent")
    await store.create_run("middle", "middle-agent", parent_run_id="root", delegation_level=1)
    await store.create_run("leaf", "leaf-agent", parent_run_id="middle", delegation_level=2)
    async with store._session_factory() as session:
        row = await session.scalar(select(AgentRun).where(AgentRun.id == "leaf"))
        if shape == "root":
            row = await session.scalar(select(AgentRun).where(AgentRun.id == "root"))
        elif shape == "missing":
            # A detached legacy/damaged record; do not bypass live FK rules.
            session.expunge(row)
            row.parent_run_id = "removed-parent"
        elif shape == "cycle":
            await session.execute(
                update(AgentRun).where(AgentRun.id == "root").values(parent_run_id="leaf")
            )
        elif shape == "self":
            session.expunge(row)
            row.parent_run_id = row.id
        ancestry, complete = await store._walk_ancestry(session, row)
    assert [a.run_id for a in ancestry] == (
        [] if shape in ("root", "missing", "self") else ["root", "middle"]
    )
    assert complete is (shape in ("root", "chain"))
    if ancestry:
        assert [a.agent_name for a in ancestry] == ["root-agent", "middle-agent"]
        assert [a.delegation_level for a in ancestry] == [0, 1]
