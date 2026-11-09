"""`runs=` replay omits whole identified blocked runs before windowing (R1.9, R1.10)."""

import pytest
from sqlalchemy import update

from dendrux import Agent, tool
from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.runtime.replay import REPLAY_META_KEY
from dendrux.types import LLMResponse, Role, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format

SECRET = "AKIAIOSFODNN7EXAMPLE"
LEGACY = "legacy-raw-history-text"
THINK = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]


def omitted(run_id):
    return (
        f"Run {run_id} omitted because it contains blocked content; "
        "prior effects may have occurred."
    )


@tool()
async def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def _agent(store, mock, **kwargs):
    kwargs.setdefault("tools", [lookup])
    kwargs.setdefault("strict_tenancy", False)
    return Agent(provider=mock, prompt="You are helpful.", state_store=store, **kwargs)


async def _clean(store, text, answer="ok", **kwargs):
    agent = _agent(store, MockLLM([LLMResponse(text=answer)]))
    try:
        result = await agent.run(text, **kwargs)
    finally:
        await agent.close()
    assert result.status == RunStatus.SUCCESS
    return result.run_id


async def _blocked(store):
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, provider, guardrails=[SecretDetection()])
    try:
        result = await agent.run(f"secret {SECRET}")
    finally:
        await agent.close()
    assert result.status == RunStatus.ERROR and provider._call_count == 0
    assert (await store.get_run(result.run_id)).meta["dendrux.blocked"] is True
    return result.run_id


async def _errored(store, text):
    agent = _agent(store, MockLLM([]))
    try:
        with pytest.raises(Exception):  # noqa: B017 - any provider failure
            await agent.run(text)
    finally:
        await agent.close()
    records = [r for r in await store.list_runs() if r.status == "error"]
    run = max(records, key=lambda r: r.created_at)
    assert run.failure_reason != "guardrail_blocked"
    return run.id


async def _strip_mark(engine, store, run_id, **values):
    record = await store.get_run(run_id)
    meta = {k: v for k, v in (record.meta or {}).items() if k != "dendrux.blocked"}
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(meta=meta, **values)
        )


async def _wire(store, runs, replay_last, **kwargs):
    mock = MockLLM([LLMResponse(text="final")])
    agent = _agent(store, mock)
    try:
        result = await agent.run("next question", runs=runs, replay_last=replay_last, **kwargs)
    finally:
        await agent.close()
    assert result.status == RunStatus.SUCCESS
    sent = [(m.role, m.content, m.tool_calls, m.call_id) for m in mock.call_history[0]["messages"]]
    return result, sent


@pytest.mark.parametrize("fmt", [1, 2])
async def test_061_replay_omits_blocked_before_window(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    first = await _clean(store, "question A", "answer A")
    blocked = await _blocked(store)
    last = await _clean(store, "question C", "answer C")

    result, sent = await _wire(store, [first, blocked, last], 1)
    _, expected = await _wire(store, [first, last], 1)

    # The window applies to survivors: C is full, A is text-only; B is gone entirely.
    assert sent == expected
    assert result.replay.runs_full == [last] and result.replay.runs_text == [first]
    assert result.replay.warnings == [omitted(blocked)]
    assert all(SECRET not in str(item) and "blocked" not in str(item[1]) for item in sent)
    stored = (await store.get_run(result.run_id)).meta[REPLAY_META_KEY]
    assert stored["runs_full"] == [last] and stored["runs_text"] == [first]


@pytest.mark.parametrize("fmt", [1, 2])
async def test_legacy_blocked_runs_are_omitted_but_errors_are_kept(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    by_reason = await _clean(store, f"question {LEGACY} 1")
    await _strip_mark(engine, store, by_reason, status="error", failure_reason="guardrail_blocked")
    by_event = await _clean(store, f"question {LEGACY} 2")
    await store.save_run_event(by_event, event_type="guardrail.blocked", data={})
    await _strip_mark(engine, store, by_event)
    errored = await _errored(store, "question E")
    clean = await _clean(store, "question F", "answer F")

    result, sent = await _wire(store, [by_reason, by_event, errored, clean], "all")
    _, expected = await _wire(store, [errored, clean], "all")

    assert sent == expected and LEGACY not in str(sent)
    assert "question E" in str(sent)
    assert result.replay.runs_full == [errored, clean] and result.replay.runs_text == []
    assert result.replay.warnings == [omitted(by_reason), omitted(by_event)]


@pytest.mark.parametrize("fmt", [1, 2])
async def test_all_sources_omitted_runs_fresh_input(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    blocked = await _blocked(store)

    result, sent = await _wire(store, [blocked], "all")
    mock = MockLLM([LLMResponse(text="final")])
    agent = _agent(store, mock)
    try:
        await agent.run("next question")
    finally:
        await agent.close()
    fresh = [(m.role, m.content, m.tool_calls, m.call_id) for m in mock.call_history[0]["messages"]]

    assert sent == fresh
    assert result.replay.runs_full == [] and result.replay.runs_text == []
    assert result.replay.warnings == [omitted(blocked)]


@pytest.mark.parametrize("fmt", [1, 2])
async def test_downstream_reasoning_bound_to_omitted_run_is_dropped(store, engine, fmt):
    # C was created replaying B; once B is identified as blocked, C's reasoning was
    # bound to a prefix that no longer exists and must not be sent.
    await pin_storage_format(engine, fmt)

    def turn(q, answer):
        return [
            LLMResponse(
                tool_calls=[ToolCall(name="lookup", params={"q": q}, provider_tool_call_id=q)],
                reasoning_blocks=THINK,
            ),
            LLMResponse(text=answer, reasoning_blocks=THINK),
        ]

    ids = []
    for index in range(3):
        agent = _agent(store, MockLLM(turn(f"q{index}", f"a{index}")))
        try:
            result = await agent.run(f"question {index}", runs=list(ids), replay_last="all")
        finally:
            await agent.close()
        assert result.status == RunStatus.SUCCESS
        ids.append(result.run_id)
    first, middle, last = ids
    await _strip_mark(engine, store, middle, failure_reason="guardrail_blocked")

    result, sent = await _wire(store, ids, "all", replay_reasoning=True)
    assert result.replay.runs_full == [first, last]
    assert result.replay.reasoning_kept == [first]
    assert result.replay.reasoning_dropped == [last]
    assert omitted(middle) in result.replay.warnings
    assert "question 1" not in str(sent)


@pytest.mark.parametrize("fmt", [1, 2])
async def test_failed_verdict_lookup_sends_nothing(store, engine, fmt, monkeypatch):
    await pin_storage_format(engine, fmt)
    source = await _clean(store, "question A")

    async def unavailable(*args, **kwargs):
        raise ValueError(SECRET)

    monkeypatch.setattr(store, "_blocked_run_ids", unavailable)
    mock = MockLLM([LLMResponse(text="final")])
    agent = _agent(store, mock)
    runs_before = len(await store.list_runs())
    try:
        with pytest.raises(PersistenceError) as caught:
            await agent.run("next question", runs=[source])
    finally:
        await agent.close()
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert mock._call_count == 0
    assert len(await store.list_runs()) == runs_before


@pytest.mark.parametrize("failure", [None, "raise", "unfiltered"])
async def test_custom_store_uses_filtered_event_fallback(store, engine, failure):
    from tests._fixtures.stores import UndeclaredSameMethodsStore

    source = await _clean(store, f"question {LEGACY}")
    await store.save_run_event(source, event_type="guardrail.blocked", data={})
    await _strip_mark(engine, store, source)
    clean = await _clean(store, "question F", "answer F")
    reads = []

    class Custom(UndeclaredSameMethodsStore):
        async def get_run_events(self, run_id, **kwargs):
            reads.append(kwargs)
            if failure == "raise":
                raise ValueError(SECRET)
            if failure == "unfiltered":
                return await self.wrapped.get_run_events(run_id)
            return await self.wrapped.get_run_events(run_id, **kwargs)

    mock = MockLLM([LLMResponse(text="final")])
    agent = _agent(Custom(store), mock)
    try:
        if failure is None:
            result = await agent.run("next", runs=[source, clean], replay_last="all")
            assert result.replay.runs_full == [clean]
            assert result.replay.warnings == [omitted(source)]
            assert LEGACY not in str(mock.call_history[0]["messages"])
            assert {"event_types": ["guardrail.blocked"], "limit": 1} in reads
        else:
            with pytest.raises(PersistenceError) as caught:
                await agent.run("next", runs=[source, clean], replay_last="all")
            assert SECRET not in str(caught.value) and mock._call_count == 0
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_tool_and_context_of_omitted_run_never_reach_the_provider(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    provider = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="lookup", params={"q": LEGACY}, provider_tool_call_id="x")
                ]
            ),
            LLMResponse(text=f"answer {LEGACY}"),
        ]
    )
    agent = _agent(store, provider)
    try:
        source = (await agent.run(f"question {LEGACY}")).run_id
    finally:
        await agent.close()
    await _strip_mark(engine, store, source, failure_reason="guardrail_blocked")
    clean = await _clean(store, "question F", "answer F")
    result, sent = await _wire(store, [source, clean], "all")
    assert LEGACY not in str(sent)
    assert all(m[0] != Role.TOOL for m in sent)
    assert result.replay.runs_full == [clean]


@pytest.mark.parametrize("fmt", [1, 2])
async def test_stream_replay_omits_blocked_sources(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    blocked = await _blocked(store)
    clean = await _clean(store, "question F", "answer F")
    mock = MockLLM([LLMResponse(text="final")])
    agent = _agent(store, mock)
    try:
        stream = agent.stream("next question", runs=[blocked, clean], replay_last="all")
        async for _ in stream:
            pass
        result = stream.result
    finally:
        await agent.close()
    assert result.status == RunStatus.SUCCESS
    assert result.replay.runs_full == [clean]
    assert result.replay.warnings == [omitted(blocked)]
    assert "blocked" not in str([m.content for m in mock.call_history[0]["messages"]])


@pytest.mark.parametrize("fmt", [1])
async def test_retry_rebuilds_the_replay_without_a_later_identified_source(store, engine, fmt):
    # Format 1 retry reloads the retried run's replay sources: one identified as blocked
    # afterwards is omitted with the same notice. (Format 2 retry rebuilds from the
    # retried run's own stored copy of that history: an earlier copy, see R1.11.)
    await pin_storage_format(engine, fmt)
    source = await _clean(store, f"question {LEGACY}")
    clean = await _clean(store, "question F", "answer F")
    first = MockLLM([LLMResponse(text="final")])
    agent = _agent(store, first)
    try:
        replayed = await agent.run("next question", runs=[source, clean], replay_last="all")
    finally:
        await agent.close()
    assert LEGACY in str(first.call_history[0]["messages"])
    await _strip_mark(engine, store, source, failure_reason="guardrail_blocked")
    retry_provider = MockLLM([LLMResponse(text="again")])
    agent = _agent(store, retry_provider)
    try:
        retried = await agent.retry(replayed.run_id)
    finally:
        await agent.close()
    assert retried.status == RunStatus.SUCCESS
    assert LEGACY not in str(retry_provider.call_history[0]["messages"])
    assert omitted(source) in retried.replay.warnings
