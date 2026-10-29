"""Provider-form persistence must not manufacture whole-conversation rewrites."""

import asyncio
from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.guardrails import PII, Pattern
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import get_delegation_context
from dendrux.runtime.replay import rebuild_history
from dendrux.types import Limits, LLMResponse, ToolCall, ToolResult


async def test_large_parallel_history_never_rewrites(store):
    fast_finished = asyncio.Event()

    @tool(parallel=True)
    async def slow() -> str:
        """Complete after the other observation is durable."""
        await fast_finished.wait()
        run_id = get_delegation_context().run_id
        for _ in range(500):
            rows = await store.get_tool_calls(run_id)
            if sum(r.tool_name == "fast" and r.outcome == "completed" for r in rows) > sum(
                r.tool_name == "slow" and r.outcome != "started" for r in rows
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("fast observation was not committed")
        fast_finished.clear()
        return "s" * 200_000

    @tool(parallel=True)
    async def fast() -> str:
        """Return a large observation first."""
        fast_finished.set()
        return "f" * 200_000

    provider = MockLLM(
        [
            *[LLMResponse(tool_calls=[ToolCall("slow"), ToolCall("fast")]) for _ in range(4)],
            LLMResponse(text="done"),
        ]
    )
    result = await Agent(
        prompt="p",
        provider=provider,
        tools=[slow, fast],
        state_store=store,
        strict_tenancy=False,
        max_iterations=8,
    ).run("go")
    assert result.status.value == "success"
    assert not await store.get_traces_after(
        result.run_id, after_order_index=None, kinds=("rewrite",)
    )


@pytest.mark.parametrize("large", [False, True], ids=["substring", "over_one_mib"])
async def test_pii_provider_form_survives_three_resumes(store, large):
    @tool(target="client")
    async def lookup() -> str:
        """Fetch a client value."""
        return ""

    views = []

    class Provider(MockLLM):
        async def complete(self, messages, tools=None, **kwargs):
            views.append([m.content for m in messages])
            if kwargs.get("run_id"):
                rebuilt = await rebuild_history(store, kwargs["run_id"])
                assert [m.content for m in rebuilt.messages] == views[-1]
            return await super().complete(messages, tools, **kwargs)

    provider = Provider(
        [
            *[LLMResponse(tool_calls=[ToolCall("lookup")]) for _ in range(3)],
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        prompt="p",
        provider=provider,
        tools=[lookup],
        state_store=store,
        strict_tenancy=False,
        limits=Limits(max_input_bytes=2 * 1024 * 1024),
        guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bAB12\b")], restore="all")],
    )
    # Individual messages remain below the guardrail scan bound; their total exceeds 1 MiB.
    from dendrux.types import Message, Role

    history = (
        [
            Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content="x " * 100_000)
            for i in range(6)
        ]
        if large
        else None
    )
    result = await agent.run(
        "mail jane@example.com" if large else "my code is AB12 and the product AB12X",
        history=history,
    )
    initial = views[0]
    for _ in range(3):
        assert result.status.value == "waiting_client_tool"
        call = result.pending.tool_calls[0]
        result = await agent.resume(
            result.run_id,
            expected_pause_id=result.pause_id,
            tool_results=[
                ToolResult(
                    name="lookup",
                    call_id=call.id,
                    payload='"jane@example.com"' if large else '"AB12 and AB12X"',
                )
            ],
        )
    assert result.status.value == "success"
    assert all(view[: len(initial)] == initial for view in views)
    if not large:
        assert all("AB12X" in view[len(initial) - 1] for view in views)
    assert not await store.get_traces_after(
        result.run_id, after_order_index=None, kinds=("rewrite",)
    )


async def test_provider_form_seeds_and_mapping_commit_together(store, monkeypatch):
    create = store.create_run
    inspected = []

    @wraps(create)
    async def inspect_creation(*args, **kwargs):
        result = await create(*args, **kwargs)
        # This is the crash boundary before the runner's first checkpoint.
        traces = await store.get_traces_after(
            result.run_id, after_order_index=None, kinds=("seed",)
        )
        assert result.storage_format == 2
        system = next(t for t in traces if t.role == "system")
        assert system.content == "code <<CODE_1>>; product AB12X"
        mapping = await store.get_pii_mapping(result.run_id)
        assert mapping and mapping["<<CODE_1>>"] == "AB12"
        inspected.append(result.run_id)
        return result

    monkeypatch.setattr(store, "create_run", inspect_creation)
    result = await Agent(
        prompt="code AB12; product AB12X",
        provider=MockLLM([LLMResponse(text="done")]),
        guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bAB12\b")], restore="all")],
        state_store=store,
        strict_tenancy=False,
    ).run("go")
    assert inspected == [result.run_id]
    assert result.status.value == "success"


async def test_rebuild_does_not_redact_provider_form_again(store):
    from dendrux.types import generate_ulid

    run_id = generate_ulid()
    await store.create_run(run_id, "provider-form")
    content = "code <<CODE_1>>; product AB12X"
    await store.save_trace(run_id, "user", content, order_index=0)
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "loop": "react"},
        pii_mapping={"<<CODE_1>>": "AB12"},
        trace=None,
    )
    rebuilt = await rebuild_history(store, run_id)
    assert [m.content for m in rebuilt.messages] == [content]


async def test_checkpoint_view_runs_after_span_redaction(store, monkeypatch):
    from dendrux.runtime.persistence import PersistenceRecorder

    checkpoint = PersistenceRecorder.checkpoint_view
    views = []

    async def capture(recorder, messages, iteration):
        views.append([m.content for m in messages])
        await checkpoint(recorder, messages, iteration)

    monkeypatch.setattr(PersistenceRecorder, "checkpoint_view", capture)
    provider = MockLLM([LLMResponse(text="done")])
    result = await Agent(
        prompt="p",
        provider=provider,
        guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bAB12\b")], restore="all")],
        state_store=store,
        strict_tenancy=False,
    ).run("code AB12; product AB12X")
    assert result.status.value == "success"
    assert views[0] == [m.content for m in provider.wire_messages[0]]


async def test_parallel_observations_commit_their_shared_engine_mapping(store, monkeypatch):
    from dendrux.guardrails import GuardrailEngine

    constructed = []
    construct = GuardrailEngine.__init__

    def count_engine(engine, *args, **kwargs):
        constructed.append(engine)
        construct(engine, *args, **kwargs)

    monkeypatch.setattr(GuardrailEngine, "__init__", count_engine)
    complete = store.complete_tool_call
    committed = []

    @wraps(complete)
    async def inspect_completion(run_id, **kwargs):
        await complete(run_id, **kwargs)
        observation = kwargs["observation"]
        mapping = await store.get_pii_mapping(run_id)
        state = await store.get_loop_state(run_id)
        assert state["pii"]["mapping_version"] == len(mapping)
        assert "<<CODE_" in observation.content
        assert mapping and any(value.startswith("CODE") for value in mapping.values())
        committed.append(observation.content)

    monkeypatch.setattr(store, "complete_tool_call", inspect_completion)

    @tool(parallel=True)
    async def read_code(n: int) -> str:
        """Return a different code from each concurrent effect."""
        await asyncio.sleep(0)
        return f"CODE{n}"

    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read_code", {"n": i}) for i in range(20)]),
            LLMResponse(text="done"),
        ]
    )
    result = await Agent(
        prompt="p",
        provider=provider,
        tools=[read_code],
        state_store=store,
        strict_tenancy=False,
        guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bCODE\d+\b")], restore="all")],
    ).run("go")
    assert result.status.value == "success"
    assert len(constructed) == 1
    assert len(committed) == 20
    mapping = await store.get_pii_mapping(result.run_id)
    assert {f"CODE{i}" for i in range(20)} <= set(mapping.values())
    assert not await store.get_traces_after(
        result.run_id, after_order_index=None, kinds=("rewrite",)
    )


async def test_submission_commits_new_placeholder_mapping(store, monkeypatch):
    @tool(target="client")
    async def lookup_code() -> str:
        """Return a client code."""
        return ""

    agent = Agent(
        prompt="p",
        provider=MockLLM(
            [LLMResponse(tool_calls=[ToolCall("lookup_code")]), LLMResponse(text="done")]
        ),
        tools=[lookup_code],
        state_store=store,
        strict_tenancy=False,
        guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bCODE\d+\b")], restore="all")],
    )
    paused = await agent.run("CODE1")
    consume = store.consume_submission
    checked = []

    @wraps(consume)
    async def inspect_submission(run_id, **kwargs):
        consumed = await consume(run_id, **kwargs)
        mapping = await store.get_pii_mapping(run_id)
        state = await store.get_loop_state(run_id)
        assert state["pii"]["mapping_version"] == len(mapping)
        # Counters store the next unused placeholder index.
        assert state["pii"]["counters"]["CODE"] == 3
        assert "CODE2" in mapping.values()
        assert "<<CODE_2>>" in kwargs["traces"][0].content
        checked.append(consumed)
        return consumed

    monkeypatch.setattr(store, "consume_submission", inspect_submission)
    result = await agent.resume(
        paused.run_id,
        expected_pause_id=paused.pause_id,
        tool_results=[
            ToolResult(
                name="lookup_code", call_id=paused.pending.tool_calls[0].id, payload='"CODE2"'
            )
        ],
    )
    assert result.status.value == "success"
    assert checked == [True]
