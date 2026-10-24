"""Forward-compatible pause state and task/context ownership."""

import asyncio
import gc

import pytest

from dendrux import Agent
from dendrux.llm import _retry_telemetry as telemetry
from dendrux.loops.react import _accumulate_usage
from dendrux.loops.single import SingleCall, _call_usage
from dendrux.runtime.tasks import RunTaskManager
from dendrux.types import (
    LLMResponse,
    PauseState,
    RunEventType,
    RunStatus,
    StreamEvent,
    StreamEventType,
    UsageStats,
)
from tests._helpers.scripted_provider import ScriptedLLM


def test_pause_version_defaults_and_unknown_keys():
    pause = PauseState.from_dict({"agent_name": "a", "future": {"ignored": True}})
    assert pause.schema_version == 1
    assert pause.history == pause.steps == pause.pending_tool_calls == []
    assert pause.iteration == pause.trace_order_offset == 0
    assert pause.usage == UsageStats()
    assert pause.to_dict()["schema_version"] == 1


def test_provider_unknown_cost_survives_both_accumulators():
    usage = UsageStats(input_tokens=1, total_tokens=1, cost_usd=0.4, cost_unknown=True)
    single = _call_usage(usage)
    assert single.cost_unknown and single.cost_usd is None
    total = UsageStats()
    _accumulate_usage(total, usage)
    assert total.cost_unknown and total.cost_usd is None


@pytest.mark.parametrize("loop", [None, SingleCall()])
async def test_stream_telemetry_never_leaks_or_resets_foreign_tokens(loop):
    observed = []

    class Streaming(ScriptedLLM):
        async def complete_stream(self, *args, **kwargs):
            try:
                observed.append(telemetry._run_id.get())
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="first")
                observed.append(telemetry._run_id.get())
                yield StreamEvent(type=StreamEventType.DONE, response=LLMResponse(text="first"))
            finally:
                observed.append(telemetry._run_id.get())

    a = Agent(name="ctx", prompt="p", provider=Streaming([]), loop=loop, strict_tenancy=False)
    stream = a.stream("go")
    it = stream.__aiter__()
    while (await anext(it)).type != RunEventType.TEXT_DELTA:
        pass
    assert telemetry._run_id.get() is None, "telemetry must not remain bound across a yielded delta"
    await asyncio.create_task(stream.aclose())
    await stream.aclose()
    assert observed and all(rid == stream.run_id for rid in observed)
    assert telemetry._run_id.get() is None


async def test_unawaited_run_task_exception_is_retrieved():
    manager = RunTaskManager()
    failures = []
    loop = asyncio.get_running_loop()
    old = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: failures.append(context))

    async def fail():
        raise RuntimeError("fixture failure")

    try:
        manager.spawn("failure", fail())
        for _ in range(4):
            await asyncio.sleep(0)
        gc.collect()
        assert not failures, "spawned tasks must retrieve exceptions even without an awaiter"
    finally:
        loop.set_exception_handler(old)


async def test_run_task_logging_does_not_format_vendor_cause(caplog):
    from dendrux.errors import ProviderTimeoutError

    async def fail():
        try:
            raise RuntimeError("vendor-private-request-body")
        except RuntimeError as cause:
            raise ProviderTimeoutError(provider="fixture") from cause

    manager = RunTaskManager()
    with pytest.raises(ProviderTimeoutError):
        await manager.spawn("safe-log", fail())
    assert "vendor-private-request-body" not in caplog.text, (
        "task logs must not format provider causes"
    )


def test_public_retry_telemetry_hook():
    factory = getattr(telemetry, "telemetry_event_hooks", None)
    assert callable(factory), "provide the public response-hook factory"
    first, second = factory(), factory()
    assert callable(first["response"][0])
    assert first is not second and first["response"] is not second["response"]


@pytest.mark.parametrize("loop", [None, SingleCall()])
async def test_image_block_also_has_structured_evidence(loop):
    from dendrux.loops.react import ReActLoop
    from dendrux.strategies.native import NativeToolCalling
    from tests.unit.test_guardrail_images import NsfwScan, setup
    from tests.unit.test_loop_images import ImageLLM, RecordingRecorder, user_msg

    runtime, part = await setup()
    llm = ImageLLM([LLMResponse(text="unused")])
    recorder = RecordingRecorder()
    a = Agent(prompt="p", provider=llm, guardrails=[NsfwScan("block")], strict_tenancy=False)
    result = await (loop or ReActLoop()).run(
        agent=a,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="go",
        initial_history=[user_msg("see", part)],
        recorder=recorder,
        image_runtime=runtime,
    )
    assert result.meta.get("failure_reason") == "guardrail_blocked"
    (data,) = recorder.of("guardrail.blocked")
    assert data.get("code") == "guardrail_blocked", (
        "image blocks share the structured block contract"
    )
    assert data["location"] == "incoming_image"
    assert data["counts"] == {"NSFW": 1} and data["kinds"] == ["NSFW"]


@pytest.mark.parametrize("loop", [None, SingleCall()])
async def test_text_block_structured_evidence_without_persistence(loop):
    from dendrux.guardrails import SecretDetection
    from dendrux.loops.react import ReActLoop
    from dendrux.strategies.native import NativeToolCalling
    from tests.unit.test_loop_images import RecordingRecorder

    llm = ScriptedLLM([])
    recorder = RecordingRecorder()
    a = Agent(
        prompt="p", provider=llm, guardrails=[SecretDetection(action="block")], strict_tenancy=False
    )
    result = await (loop or ReActLoop()).run(
        agent=a,
        provider=llm,
        strategy=NativeToolCalling(),
        user_input="api_key=sk_test_abcdefghijklmnopqrstuvwxyz123456",
        recorder=recorder,
    )
    assert result.status == RunStatus.ERROR
    (data,) = recorder.of("guardrail.blocked")
    assert data.get("code") == "guardrail_blocked", "block events need stable structured fields"
    assert data["location"] == "message 1 (role=user)"
    assert sum(data["counts"].values()) == 1
