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
        await store.create_run("00000000000000000000000106", "integrity")
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
        await store.pause_run(
            "00000000000000000000000106", status=status, pause_data=pause.to_dict()
        )
        paused = type(
            "Paused", (), {"run_id": "00000000000000000000000106", "status": RunStatus(status)}
        )()
        if entry == "input":
            action = a.submit_input(
                "00000000000000000000000106", "answer", expected_pause_id="stale"
            )
        elif entry == "paused":
            action = a.submit_resume("00000000000000000000000106", expected_pause_id="stale")
        elif entry == "resume":
            action = a.resume("00000000000000000000000106", expected_pause_id="stale")
        else:
            action = consume(
                a.resume_stream("00000000000000000000000106", expected_pause_id="stale")
            )
    if entry == "resume_stream":
        result, _ = await action
        assert result.status == RunStatus.ERROR
    else:
        with pytest.raises(PauseStatusMismatchError) as caught:
            await action
        assert caught.value.reason == "pause_id"
    assert (await store.get_run(paused.run_id)).status == paused.status.value


async def test_plain_store_claim_checks_token_in_transaction(store):
    await store.create_run("00000000000000000000000106", "integrity")
    await store.pause_run(
        "00000000000000000000000106", status="paused", pause_data={"pause_id": "current"}
    )
    with pytest.raises(PauseStatusMismatchError):
        await store.claim_paused_run(
            "00000000000000000000000106", expected_status="paused", expected_pause_id="stale"
        )
    assert (await store.get_run("00000000000000000000000106")).status == "paused"
    assert (await store.get_run_control("00000000000000000000000106")).claim_epoch == 0


async def test_strict_tenancy_requires_pause_token(store):
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
    from dendrux.types import InputValidationError

    with pytest.raises(InputValidationError) as caught:
        await a.submit_tool_results(paused.run_id, [result_for(paused)], tenant_id=None)
    assert caught.value.code == "pause_id_required"
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    result = await a.submit_tool_results(
        paused.run_id, [result_for(paused)], tenant_id=None, expected_pause_id=paused.pause_id
    )
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


async def test_default_governance_change_raises(store):
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
    from dendrux.errors import GovernanceMismatchError

    with pytest.raises(GovernanceMismatchError):
        await b.submit_tool_results(paused.run_id, [result_for(paused)])
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
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
