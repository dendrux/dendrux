"""Retry refuses identified blocked history before any new run (R1.8, R1.10)."""

import logging

import pytest
from sqlalchemy import update

from dendrux import Agent, ReplayRunError, tool
from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, RunNotFoundError
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format

SECRET = "AKIAIOSFODNN7EXAMPLE"
LEGACY = "legacy-raw-history-text"
DETAIL = "Source history contains blocked content; start a new run with clean input."


@tool()
async def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def _agent(store, mock, **kwargs):
    kwargs.setdefault("tools", [lookup])
    kwargs.setdefault("strict_tenancy", False)
    return Agent(provider=mock, prompt="You are helpful.", state_store=store, **kwargs)


async def _run(store, text, responses=None, **kwargs):
    agent = _agent(store, MockLLM(responses or [LLMResponse(text="ok")]))
    try:
        return await agent.run(text, **kwargs)
    finally:
        await agent.close()


async def _strip_mark(engine, store, run_id, **values):
    record = await store.get_run(run_id)
    meta = {k: v for k, v in (record.meta or {}).items() if k != "dendrux.blocked"}
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(meta=meta, **values)
        )


async def _source(store, engine, kind):
    if kind == "mark":
        provider = MockLLM([LLMResponse(text="unreachable")])
        agent = _agent(store, provider, guardrails=[SecretDetection()])
        try:
            result = await agent.run(f"secret {SECRET}")
        finally:
            await agent.close()
        assert result.status == RunStatus.ERROR
        return result.run_id
    result = await _run(store, f"question {LEGACY}")
    if kind == "reason":
        await _strip_mark(
            engine, store, result.run_id, status="error", failure_reason="guardrail_blocked"
        )
    else:
        await store.save_run_event(result.run_id, event_type="guardrail.blocked", data={})
        await _strip_mark(engine, store, result.run_id)
    return result.run_id


async def _refused(store, run_id, identified, caplog):
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, provider)
    before = len(await store.list_runs())
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ReplayRunError) as caught:
            await agent.retry(run_id)
    finally:
        await agent.close()
    error = caught.value
    assert error.reason == "blocked_content" and error.run_id == identified
    assert str(error) == f"Cannot replay run '{identified}': {DETAIL}"
    assert provider._call_count == 0
    assert len(await store.list_runs()) == before
    records = [r for r in caplog.records if getattr(r, "code", None) == "blocked_content"]
    assert [r.run_id for r in records] == [identified]
    assert SECRET not in caplog.text and LEGACY not in caplog.text


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["mark", "reason", "event"])
async def test_061_retry_blocked_has_no_new_run(store, engine, fmt, kind, caplog):
    await pin_storage_format(engine, fmt)
    source = await _source(store, engine, kind)
    await _refused(store, source, source, caplog)


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("kind", ["mark", "reason", "event"])
async def test_retry_refuses_a_run_whose_replay_source_is_identified(
    store, engine, fmt, kind, caplog
):
    await pin_storage_format(engine, fmt)
    clean = (await _run(store, f"question {LEGACY} clean")).run_id
    replayed = await _run(store, "next question", runs=[clean], replay_last="all")
    assert replayed.status == RunStatus.SUCCESS
    if kind == "mark":
        async with engine.begin() as connection:
            await connection.execute(
                update(AgentRun).where(AgentRun.id == clean).values(meta={"dendrux.blocked": True})
            )
    elif kind == "reason":
        await _strip_mark(engine, store, clean, status="error", failure_reason="guardrail_blocked")
    else:
        await store.save_run_event(clean, event_type="guardrail.blocked", data={})
        await _strip_mark(engine, store, clean)
    await _refused(store, replayed.run_id, clean, caplog)


@pytest.mark.parametrize("fmt", [1, 2])
async def test_retry_validates_scope_and_status_before_refusing(store, engine, fmt, caplog):
    await pin_storage_format(engine, fmt)
    source = await _source(store, engine, "mark")
    agent = _agent(store, MockLLM([]))
    try:
        with pytest.raises(RunNotFoundError):
            await agent.retry(source, tenant_id="other")
    finally:
        await agent.close()
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun).where(AgentRun.id == source).values(status="running")
        )
    agent = _agent(store, MockLLM([]))
    try:
        with pytest.raises(ValueError) as caught:
            await agent.retry(source)
    finally:
        await agent.close()
    assert not isinstance(caught.value, ReplayRunError)
    assert not [r for r in caplog.records if getattr(r, "code", None) == "blocked_content"]


@pytest.mark.parametrize("fmt", [1, 2])
async def test_clean_retry_still_runs(store, engine, fmt):
    await pin_storage_format(engine, fmt)
    clean = (await _run(store, "question A")).run_id
    replayed = (await _run(store, "next", runs=[clean], replay_last="all")).run_id
    for run_id in (clean, replayed):
        provider = MockLLM([LLMResponse(text="again")])
        agent = _agent(store, provider)
        try:
            result = await agent.retry(run_id)
        finally:
            await agent.close()
        assert result.status == RunStatus.SUCCESS and provider._call_count == 1


@pytest.mark.parametrize("fmt", [1, 2])
async def test_failed_retry_verdict_sends_nothing(store, engine, fmt, monkeypatch):
    await pin_storage_format(engine, fmt)
    source = (await _run(store, "question A")).run_id

    async def unavailable(*args, **kwargs):
        raise ValueError(SECRET)

    monkeypatch.setattr(store, "_blocked_run_ids", unavailable)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(store, provider)
    before = len(await store.list_runs())
    try:
        with pytest.raises(PersistenceError) as caught:
            await agent.retry(source)
    finally:
        await agent.close()
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert provider._call_count == 0 and len(await store.list_runs()) == before


@pytest.mark.parametrize("failure", [None, "raise", "unfiltered"])
async def test_custom_store_retry_uses_filtered_event_fallback(store, engine, failure, caplog):
    from tests._fixtures.stores import UndeclaredSameMethodsStore

    source = await _source(store, engine, "event")

    class Custom(UndeclaredSameMethodsStore):
        async def get_run_events(self, run_id, **kwargs):
            if failure == "raise":
                raise ValueError(SECRET)
            if failure == "unfiltered":
                return await self.wrapped.get_run_events(run_id)
            return await self.wrapped.get_run_events(run_id, **kwargs)

    custom = Custom(store)
    if failure is None:
        await _refused(custom, source, source, caplog)
        return
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(custom, provider)
    try:
        with pytest.raises(PersistenceError) as caught:
            await agent.retry(source)
    finally:
        await agent.close()
    assert SECRET not in str(caught.value) and provider._call_count == 0


async def test_custom_store_that_drops_the_mark_retries_only_the_placeholder(store, engine):
    # initial input: a store that cannot keep the mark refuses the blocked write
    # safely. The row exists, sanitized and unmarked, with no stored turn, so a later retry
    # stops with the existing "no persisted traces" error and calls no provider.
    from tests._fixtures.stores import UndeclaredSameMethodsStore

    custom = UndeclaredSameMethodsStore(store)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = _agent(custom, provider, guardrails=[SecretDetection()])
    try:
        with pytest.raises(PersistenceError):
            await agent.run(f"secret {SECRET}")
    finally:
        await agent.close()
    assert provider._call_count == 0
    rows = await store.list_runs()
    assert len(rows) == 1
    row = await store.get_run(rows[0].id)
    assert "dendrux.blocked" not in (row.meta or {})
    assert row.failure_reason != "guardrail_blocked"
    assert SECRET not in str(row.input_data)
    retry_provider = MockLLM([LLMResponse(text="again")])
    agent = _agent(store, retry_provider)
    try:
        try:
            await agent.retry(row.id)
        except ValueError as exc:  # a source with no usable history is the existing error
            assert not isinstance(exc, ReplayRunError)
    finally:
        await agent.close()
    assert SECRET not in str([call["messages"] for call in retry_provider.call_history])


async def test_tool_history_of_identified_source_is_never_resent(store, engine, caplog):
    source = await _run(
        store,
        f"question {LEGACY}",
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="lookup", params={"q": LEGACY}, provider_tool_call_id="x")
                ]
            ),
            LLMResponse(text=f"answer {LEGACY}"),
        ],
    )
    await _strip_mark(engine, store, source.run_id, failure_reason="guardrail_blocked")
    await _refused(store, source.run_id, source.run_id, caplog)


@pytest.mark.parametrize("fmt", [1, 2])
async def test_retry_of_identified_run_with_sources_reads_no_source_history(
    store, engine, fmt, caplog, monkeypatch
):
    # One verdict covers the retried run and its sources; when the retried run itself is
    # identified, no source traces are read and the refusal names the retried run.
    await pin_storage_format(engine, fmt)
    clean = (await _run(store, "question A")).run_id
    replayed = (await _run(store, "next", runs=[clean], replay_last="all")).run_id
    await _strip_mark(engine, store, replayed, status="error", failure_reason="guardrail_blocked")
    reads = []
    for name in ("get_traces", "conversation_traces"):
        original = getattr(store, name, None)
        if original is None:
            continue

        async def spy(run_id, *args, _original=original, **kwargs):
            reads.append(run_id)
            return await _original(run_id, *args, **kwargs)

        monkeypatch.setattr(store, name, spy)
    await _refused(store, replayed, replayed, caplog)
    assert clean not in reads


@pytest.mark.parametrize("fmt", [1, 2])
async def test_custom_store_that_drops_the_mark_after_a_stored_turn_retries_clean_history(
    store, engine, fmt
):
    # later block: the first turn is stored, then a tool result is blocked. The
    # store cannot keep the mark, so the blocked write is refused before any evidence:
    # the run ends unmarked with its clean turn only. A retry is not refused, reaches the
    # provider and sends exactly that stored clean history (no secret, nothing blocked).
    from tests._fixtures.stores import UndeclaredSameMethodsStore

    await pin_storage_format(engine, fmt)

    @tool()
    async def read_value() -> str:
        """Return a value."""
        return SECRET

    provider = MockLLM(
        [LLMResponse(tool_calls=[ToolCall(name="read_value", params={})]), LLMResponse(text="x")]
    )
    agent = _agent(
        UndeclaredSameMethodsStore(store),
        provider,
        tools=[read_value],
        guardrails=[SecretDetection()],
    )
    try:
        with pytest.raises(PersistenceError):
            await agent.run("go")
    finally:
        await agent.close()
    (run,) = await store.list_runs()
    row = await store.get_run(run.id)
    assert row.status == "error" and row.failure_reason != "guardrail_blocked"
    assert "dendrux.blocked" not in (row.meta or {})
    stored = [(t.role, t.content) for t in await store.get_traces(row.id)]
    assert stored == [("user", "go"), ("assistant", "")]
    retry_provider = MockLLM([LLMResponse(text="again")])
    agent = _agent(store, retry_provider, tools=[read_value])
    try:
        result = await agent.retry(row.id)
    finally:
        await agent.close()
    assert result.status == RunStatus.SUCCESS and retry_provider._call_count == 1
    sent = [(m.role.value, m.content) for m in retry_provider.call_history[0]["messages"]]
    assert sent == [("system", "You are helpful."), ("user", "go")]
    assert SECRET not in str(retry_provider.call_history)
