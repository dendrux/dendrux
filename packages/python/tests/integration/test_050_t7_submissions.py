"""Generation checks on every submission shape and conservative result handling."""

import json

import pytest

from dendrux import Agent, ApprovalDecision, Limits, tool
from dendrux.errors import InvalidToolResultError, PauseStatusMismatchError
from dendrux.runtime.submit import GOVERNANCE_KEY
from dendrux.types import LLMResponse, Message, PauseState, Role, RunStatus, UsageStats
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import agent, client_read, consume, result_for


@tool()
async def effect() -> str:
    raise AssertionError("stale approval executed a tool")


@pytest.mark.parametrize(
    "entry", ["approve", "reject", "decisions", "input", "paused", "resume", "resume_stream"]
)
async def test_every_submit_validates_pause_generation(store, entry):
    if entry in {"approve", "reject", "decisions"}:
        a = Agent(
            name="integrity",
            prompt="p",
            tools=[effect],
            require_approval=["effect"],
            state_store=store,
            provider=ScriptedLLM([tool_call("effect", pid="a")]),
            strict_tenancy=False,
            on_governance_change="warn",
        )
        paused = await a.run("go")
        call = paused.pending.tool_calls[0]
        kwargs = (
            {"decisions": {call.id: ApprovalDecision.approve()}}
            if entry == "decisions"
            else {"approved": entry == "approve"}
        )
        action = a.submit_approval(paused.run_id, expected_pause_id="stale", **kwargs)
    else:
        a = agent(store, [LLMResponse(text="done")])
        await store.create_run("seed", "integrity")
        pause = PauseState(
            agent_name="integrity",
            pending_tool_calls=[],
            history=[Message(role=Role.USER, content="go")],
            steps=[],
            iteration=1,
            trace_order_offset=1,
            usage=UsageStats(),
            pause_id="current",
        )
        status = "waiting_human_input" if entry == "input" else "paused"
        await store.pause_run("seed", status=status, pause_data=pause.to_dict())
        paused = type("Paused", (), {"run_id": "seed", "status": RunStatus(status)})()
        if entry == "input":
            action = a.submit_input("seed", "answer", expected_pause_id="stale")
        elif entry == "paused":
            action = a.submit_resume("seed", expected_pause_id="stale")
        elif entry == "resume":
            action = a.resume("seed", expected_pause_id="stale")
        else:
            action = consume(a.resume_stream("seed", expected_pause_id="stale"))
    if entry == "resume_stream":
        result, _ = await action
        assert result.status == RunStatus.ERROR
    else:
        with pytest.raises(PauseStatusMismatchError) as caught:
            await action
        assert caught.value.reason == "pause_id"
    assert (await store.get_run(paused.run_id)).status == paused.status.value


async def test_plain_store_claim_checks_token_in_transaction(store):
    await store.create_run("seed", "integrity")
    await store.pause_run("seed", status="paused", pause_data={"pause_id": "current"})
    with pytest.raises(PauseStatusMismatchError):
        await store.claim_paused_run("seed", expected_status="paused", expected_pause_id="stale")
    assert (await store.get_run("seed")).status == "paused"
    assert (await store.get_run_control("seed")).claim_epoch == 0


async def test_strict_tenancy_warns_without_pause_token(store):
    a = Agent(
        name="integrity",
        prompt="p",
        tools=[client_read],
        state_store=store,
        provider=ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="done")]),
        strict_tenancy=True,
        on_governance_change="warn",
    )
    paused = await a.run("go", tenant_id=None)
    with pytest.warns(FutureWarning, match="expected_pause_id"):
        result = await a.submit_tool_results(paused.run_id, [result_for(paused)], tenant_id=None)
    assert result.status == RunStatus.SUCCESS


async def test_oversized_plain_text_is_capped_before_transform(store):
    seen = []
    a = agent(
        store,
        [tool_call("client_read", pid="a"), LLMResponse(text="done")],
        limits=Limits(max_tool_result_bytes=100),
        tool_result_transform=lambda ctx: seen.append(ctx),
    )
    paused = await a.run("go")
    result = await a.submit_tool_results(paused.run_id, [result_for(paused, "{" * 1000)])
    assert result.status == RunStatus.SUCCESS
    assert not seen
    messages = [m for m in a.provider.messages[-1] if m.role.value == "tool"]
    assert json.loads(messages[-1].content)["code"] == "size_limit"


async def test_failed_transform_does_not_claim_or_expose_payload(store):
    def failing(ctx):
        raise ValueError("private client content")

    a = agent(store, [tool_call("client_read", pid="a")], tool_result_transform=failing)
    paused = await a.run("go")
    with pytest.raises(InvalidToolResultError) as caught:
        await a.submit_tool_results(paused.run_id, [result_for(paused)])
    assert "private client content" not in str(caught.value)
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"


async def test_default_governance_change_emits_future_warning(store):
    a = agent(store, [tool_call("client_read", pid="a")], limits=Limits(max_tool_result_bytes=100))
    paused = await a.run("go")
    b = Agent(
        name="integrity",
        prompt="p",
        tools=[client_read],
        state_store=store,
        provider=ScriptedLLM([LLMResponse(text="done")]),
        strict_tenancy=False,
    )
    with pytest.warns(FutureWarning, match="0.6"):
        result = await b.submit_tool_results(paused.run_id, [result_for(paused)])
    assert result.status == RunStatus.SUCCESS
    assert GOVERNANCE_KEY in (await store.get_run(paused.run_id)).meta


async def test_transform_cannot_move_submission_to_a_new_pause_generation(store):
    run_id = None

    async def transform(ctx):
        raw = await store.get_pause_state(run_id)
        raw["pause_id"] = "new-generation"
        await store.pause_run(run_id, status="waiting_client_tool", pause_data=raw)
        return ctx.result

    a = agent(
        store,
        [tool_call("client_read", pid="a"), LLMResponse(text="done")],
        tool_result_transform=transform,
    )
    paused = await a.run("go")
    run_id = paused.run_id
    with pytest.raises(PauseStatusMismatchError):
        await a.submit_tool_results(run_id, [result_for(paused)])
    assert (await store.get_run(run_id)).status == "waiting_client_tool"
