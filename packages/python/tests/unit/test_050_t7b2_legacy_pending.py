"""Legacy event readers must fail closed without unbounded event-history reads."""

from dendrux import Agent, tool
from dendrux.types import LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.unit.test_resume import RecordingStateStore


async def test_legacy_event_reader_returns_hidden_active_approval_arguments():
    @tool()
    def refund(email: str) -> str:
        return "must await approval"

    store = RecordingStateStore()
    a = Agent(
        name="legacy",
        prompt="p",
        tools=[refund],
        require_approval=["refund"],
        provider=ScriptedLLM(
            [tool_call("refund", {"email": "PII_SENTINEL"}, pid="C"), LLMResponse(text="done")]
        ),
        state_store=store,
        strict_tenancy=False,
    )
    try:
        result = await a.run("go", tenant_id=None)
        for outcome in (result, await a.get_result(result.run_id, tenant_id=None)):
            assert outcome.pending is not None
            call = outcome.pending.tool_calls[0]
            assert call.params is None and call.args_hidden
            assert "PII_SENTINEL" not in str(outcome.to_dict())
    finally:
        await a.close()
