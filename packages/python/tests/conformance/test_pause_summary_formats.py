"""The bounded pause projection works on every SQL dialect in the CI store kit."""

import pytest

from dendrux.store import RunStore
from dendrux.types import generate_ulid


@pytest.mark.parametrize("version", [1, 2])
async def test_clarification_projection_on_every_dialect(dendrux_store_factory, version):
    store = await dendrux_store_factory()
    run_id = generate_ulid()
    await store.create_run(run_id, "question", tenant_id="t")
    action = {"type": "clarification", "question": "Which file?", "options": ["one", "two"]}
    data = {"agent_name": "question", "schema_version": version, "pending_tool_calls": []}
    if version == 1:
        data["steps"] = [{"action": action}]
    else:
        data["clarification"] = {k: v for k, v in action.items() if k != "type"}
    await store.pause_run(run_id, status="waiting_human_input", pause_data=data)
    pending = await RunStore(store).get_pending(run_id, tenant_id="t")
    assert pending.question == "Which file?"
    assert pending.options == ["one", "two"]
    assert pending.to_dict()["options"] == ["one", "two"]
