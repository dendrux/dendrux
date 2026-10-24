"""0.4.1 tool-result read shapes and raw client-text transforms."""

import json

import pytest

from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import agent, consume, result_for


@pytest.mark.parametrize("target", ["client", "server"])
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Revenue: $100M\nline2", {"raw": "Revenue: $100M\nline2"}),
        ('{"rows": [1, 2]}', {"rows": [1, 2]}),
        ('"NaN"', "NaN"),
        ("1.5", 1.5),
        ("NaN", {"raw": "NaN"}),
        ("Infinity", {"raw": "Infinity"}),
        ("-Infinity", {"raw": "-Infinity"}),
        (
            '{"values": [NaN, Infinity, -Infinity]}',
            {"raw": '{"values": [NaN, Infinity, -Infinity]}'},
        ),
        ("1e400", {"raw": "1e400"}),
        ('{"values": [1e400]}', {"raw": '{"values": [1e400]}'}),
    ],
)
async def test_tool_result_storage_and_public_reads_keep_041_shape(
    store, target, payload, expected
):
    await store.create_run("r", "integrity")
    await store.save_tool_call(
        "r",
        tool_call_id="call",
        provider_tool_call_id="provider-call",
        tool_name="read",
        target=target,
        params=None,
        result_payload=payload,
        success=True,
        duration_ms=0,
        iteration_index=1,
    )
    (record,) = await store.get_tool_calls("r")
    (invocation,) = await RunStore(store).get_tool_invocations("r", tenant_id=None)
    # c52fb9e save_tool_call: JSON values are parsed; text uses {"raw": payload}.
    # Non-finite numbers now take that same documented text fallback on both DBs.
    assert record.result == invocation.result == expected
    assert invocation.target == target and invocation.success
    json.dumps(invocation.result, allow_nan=False)


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize("mode", ["identity", "rewrite", "json_text", "mapping"])
async def test_plain_text_transform_receives_and_returns_raw_text(store, entry, mode):
    payload = 'Revenue: $100M\n"quoted"\té\\sheet'
    seen = []
    output = {
        "identity": payload,
        "rewrite": "Changed:\n" + payload,
        "json_text": ' {"changed": true} ',
        "mapping": {"received": payload},
    }[mode]

    async def transform(ctx):
        seen.append(ctx.result)
        return output

    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="done")])
    a = agent(store, llm, tool_result_transform=transform)
    paused = await a.run("go")
    results = [result_for(paused, payload)]
    if entry == "submit":
        result = await a.submit_tool_results(paused.run_id, results)
    elif entry == "resume":
        result = await a.resume(paused.run_id, tool_results=results)
    else:
        result, _ = await consume(a.resume_stream(paused.run_id, tool_results=results))
    assert result.status == RunStatus.SUCCESS and seen == [payload]
    message = next(m for m in llm.messages[-1] if m.role.value == "tool")
    expected = output if isinstance(output, str) else json.dumps(output)
    assert message.content.encode("utf-8") == expected.encode("utf-8")
    (invocation,) = await RunStore(store).get_tool_invocations(paused.run_id, tenant_id=None)
    try:
        stored = json.loads(expected)
    except ValueError:
        stored = {"raw": expected}
    assert invocation.result == stored
