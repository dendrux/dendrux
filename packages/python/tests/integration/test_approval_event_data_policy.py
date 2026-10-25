"""Malformed legacy approval evidence is an absent argument view."""

import pytest
from sqlalchemy import update

from dendrux.db.models import RunEvent


@pytest.mark.parametrize(
    "data", [None, [], ["PRIVATE_APPROVAL_DATA"], "PRIVATE_APPROVAL_DATA", 42, True]
)
async def test_nondictionary_approval_evidence_is_ignored(store, engine, data):
    await store.create_run("r", "a")
    await store.save_run_event("r", event_type="approval.requested", data={})
    async with engine.begin() as conn:
        await conn.execute(update(RunEvent).where(RunEvent.agent_run_id == "r").values(data=data))
    assert await store.get_approval_params("r", 1) == {}
