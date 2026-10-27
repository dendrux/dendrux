"""Unpersistable client data is rejected before it can claim waiting work."""

import pytest

from dendrux import Agent
from dendrux.errors import InvalidToolResultError
from dendrux.llm.mock import MockLLM
from dendrux.types import InputValidationError, LLMResponse, ToolResult
from tests.integration.test_tenant_run_control import _pause

BAD_RESULTS = [
    {"duration_ms": 2**40},
    {"duration_ms": -1},
    {"duration_ms": True},
    {"success": "false"},
    {"payload": "a\x00b"},
    {"error": "a\x00b"},
    {"payload": '"a\\u0000b"'},
    {"payload": "a\ud800b"},
    {"payload": '{"a": NaN}'},
    {"payload": "Infinity"},
    {"payload": "1e999"},
    {"payload": "[" * 1000 + "]" * 1000},
    {"error": "E" * 70000},
]


@pytest.mark.parametrize("patch", BAD_RESULTS)
@pytest.mark.parametrize("entry", ["submit", "resume"])
async def test_unpersistable_tool_result_leaves_the_pause_unclaimed(store, patch, entry):
    paused = await _pause("client", store, "A")
    raw = await store.get_pause_state(paused.run_id)
    call = raw["pending_tool_calls"][0]
    result = ToolResult(**{**dict(name=call["name"], call_id=call["id"], payload='"ok"'), **patch})
    before = await store.get_run_control(paused.run_id)
    try:
        with pytest.raises(InvalidToolResultError):
            if entry == "submit":
                await paused.agent.submit_tool_results(
                    paused.run_id, [result], tenant_id="A", expected_pause_id=raw["pause_id"]
                )
            else:
                await paused.agent.resume(
                    paused.run_id,
                    tool_results=[result],
                    tenant_id="A",
                    expected_pause_id=raw["pause_id"],
                )
        after = await store.get_run(paused.run_id)
        assert after.status == "waiting_client_tool"
        assert (await store.get_run_control(paused.run_id)).claim_epoch == before.claim_epoch
        assert await store.get_tool_calls(paused.run_id) == []
    finally:
        await paused.agent.close()


async def test_unknown_call_identifiers_are_never_reflected_unbounded(store):
    paused = await _pause("client", store, "A")
    results = [
        ToolResult(name="read_range", call_id=str(i) + "CLIENT_SECRET_ID" * 4000, payload='"ok"')
        for i in range(200)
    ]
    try:
        with pytest.raises(InvalidToolResultError) as caught:
            await paused.agent.submit_tool_results(paused.run_id, results, tenant_id="A")
        assert len(str(caught.value)) < 1024, "client ID errors must remain bounded"
        assert "CLIENT_SECRET_ID" not in str(caught.value)
        assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    finally:
        await paused.agent.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("input", "x\x00"),
        ("input", "x\ud800"),
        ("tenant", "x\x00"),
        ("tenant", "T" * 256),
        ("key", "k\x00"),
        ("key", "k" * 256),
        ("metadata", {"thread_id": "x\x00"}),
        ("metadata", {"x\x00": 1}),
    ],
)
async def test_invalid_new_run_values_are_typed_and_create_no_row(store, field, value):
    a = Agent(
        name="validation",
        prompt="p",
        provider=MockLLM([LLMResponse(text="ok")]),
        state_store=store,
        strict_tenancy=False,
    )
    kwargs = {"tenant_id": "A"}
    text = "go"
    if field == "input":
        text = value
    elif field == "tenant":
        kwargs["tenant_id"] = value
    elif field == "key":
        kwargs["idempotency_key"] = value
    else:
        kwargs["metadata"] = value
    try:
        with pytest.raises(InputValidationError):
            await a.run(text, **kwargs)
        assert await store.count_runs() == 0
    finally:
        await a.close()


@pytest.mark.parametrize("field", ["input", "rejection", "note", "reason"])
async def test_invalid_continuation_text_never_changes_state(store, field):
    paused = await _pause(
        "approval" if field == "rejection" else "input" if field == "input" else "client",
        store,
        "A",
    )
    before = await store.get_run(paused.run_id)
    before_epoch = (await store.get_run_control(paused.run_id)).claim_epoch
    try:
        with pytest.raises(InputValidationError):
            if field == "input":
                await paused.agent.submit_input(paused.run_id, "x\x00", tenant_id="A")
            elif field == "rejection":
                await paused.agent.submit_approval(
                    paused.run_id, approved=False, rejection_reason="x\x00", tenant_id="A"
                )
            elif field == "note":
                await paused.agent.submit_note(paused.run_id, "x\x00", tenant_id="A")
            else:
                await paused.agent.request_pause(paused.run_id, reason="x\x00", tenant_id="A")
        after = await store.get_run(paused.run_id)
        assert (
            after.status == before.status
            and (await store.get_run_control(paused.run_id)).claim_epoch == before_epoch
        )
    finally:
        await paused.agent.close()


async def test_direct_store_rejects_poisonous_metadata_before_insert(store):
    with pytest.raises(InputValidationError):
        await store.create_run("bad-meta", "agent", meta={"thread_id": "bad\x00"})
    assert await store.count_runs() == 0
