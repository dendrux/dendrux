"""URL readers discover managed pauses through the lazy SQLite initializer."""

import pytest

from dendrux import Agent, tool
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


@tool()
def publish() -> str:
    """Publish after approval."""
    return "published"


@pytest.mark.parametrize("child_count", [1, 3])
@pytest.mark.parametrize("method", ["get_pending", "get_pendings"])
async def test_url_reader_returns_managed_child_pauses(engine, store, child_count, method):
    children = [
        Agent(
            name=f"worker{i}",
            description="Publish one item",
            prompt="Publish",
            provider=ScriptedLLM([LLMResponse(tool_calls=[ToolCall("publish", {})])]),
            tools=[publish],
            require_approval=["publish"],
        )
        for i in range(child_count)
    ]
    async with Agent(
        name="lead",
        prompt="Delegate",
        provider=ScriptedLLM(
            [LLMResponse(tool_calls=[ToolCall(c.name, {"task": "publish"}) for c in children])]
        ),
        subagents=children,
        state_store=store,
    ) as lead:
        result = await lead.run("go", tenant_id="reader-test")
    assert result.status == RunStatus.WAITING_APPROVAL
    assert len(result.pendings) == child_count
    async with RunStore.from_database_url(
        engine.url.render_as_string(hide_password=False)
    ) as reader:
        pending = await getattr(reader, method)(result.run_id, tenant_id="reader-test")
        actual = [pending] if method == "get_pending" else pending
        expected = result.pendings[:1] if method == "get_pending" else result.pendings
        assert [(p.origin_run_id, p.pause_id, p.agent_name, p.kind) for p in actual] == [
            (p.origin_run_id, p.pause_id, p.agent_name, "approval") for p in expected
        ]
