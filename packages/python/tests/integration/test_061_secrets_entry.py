"""Entry identity and evidence use rejected content's one-way projection."""

import pytest
from sqlalchemy import select

from dendrux import Agent
from dendrux.chat import ChatMessage
from dendrux.context_blocks import ContextBlock
from dendrux.db.models import AgentRun, Base
from dendrux.guardrails import SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.types import LLMResponse, RunStatus, compute_idempotency_fingerprint
from tests._fixtures.storage import pin_storage_format

SECRET = "AKIAIOSFODNN7EXAMPLE"
PLACEHOLDER = "[blocked: 1 finding]"


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("surface", ["input", "history", "stable", "dynamic"])
async def test_061_blocked_identity_uses_projected_input(store, engine, fmt, single, surface):
    await pin_storage_format(engine, fmt)
    provider = MockLLM([LLMResponse(text="unreachable")])
    agent = Agent(
        name="entry",
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[SecretDetection()],
        loop=SingleCall() if single else None,
    )
    text = SECRET if surface == "input" else "go"
    history = (
        [ChatMessage.user(SECRET), ChatMessage.assistant("okay")] if surface == "history" else None
    )
    context = (
        [ContextBlock(SECRET, placement=surface)] if surface in ("stable", "dynamic") else None
    )
    clean_text = PLACEHOLDER if surface == "input" else text
    clean_history = (
        [ChatMessage.user(PLACEHOLDER), ChatMessage.assistant("okay")] if history else None
    )
    clean_context = [ContextBlock(PLACEHOLDER, placement=surface)] if context else None
    from dendrux.chat import normalize_chat_history

    expected = compute_idempotency_fingerprint(
        "entry",
        clean_text,
        history=normalize_chat_history(clean_history) or None,
        context=clean_context,
    )
    raw = compute_idempotency_fingerprint(
        "entry", text, history=normalize_chat_history(history) or None, context=context
    )
    try:
        result = await agent.run(
            text, history=history, context=context, idempotency_key="same", tenant_id="tenant"
        )
        assert result.status == RunStatus.ERROR
        assert provider._call_count == 0
        async with engine.connect() as conn:
            row = (
                (await conn.execute(select(AgentRun).where(AgentRun.id == result.run_id)))
                .mappings()
                .one()
            )
            assert row["idempotency_fingerprint"] == expected != raw
            assert row["input_data"]["input"] == clean_text
            assert row["meta"]["dendrux.blocked"] is True
            assert row["failure_reason"] == "guardrail_blocked"
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert SECRET not in repr(rows), table.name
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("single", [False, True])
async def test_outgoing_evidence_is_marked_before_callback(store, engine, monkeypatch, fmt, single):
    from dendrux.loops.base import BaseNotifier

    await pin_storage_format(engine, fmt)
    captured = []
    marks = []
    save = store.save_llm_interaction

    from functools import wraps

    @wraps(save)
    async def inspect_write(run_id, **kwargs):
        await save(run_id, **kwargs)
        marks.append((await store.get_run(run_id)).meta.get("dendrux.blocked"))

    monkeypatch.setattr(store, "save_llm_interaction", inspect_write)

    class Capture(BaseNotifier):
        async def on_llm_call_completed(self, run_id, response, iteration, **kwargs):
            captured.append(response)

    provider = MockLLM(
        [
            LLMResponse(
                text=SECRET,
                raw={"text": SECRET},
                provider_request={"duplicate": SECRET},
                provider_response={"text": SECRET},
                reasoning=SECRET,
                raw_stop_reason=SECRET,
            )
        ]
    )
    agent = Agent(
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[SecretDetection()],
        loop=SingleCall() if single else None,
    )
    try:
        result = await agent.run("go", tenant_id="tenant", notifier=Capture())
        assert result.status == RunStatus.ERROR and provider._call_count == 1
        assert marks == [True]
        assert captured and captured[0].text == PLACEHOLDER
        assert SECRET not in repr(captured)
        async with engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert SECRET not in repr(rows), table.name
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_initial_block_mark_is_in_insert(store, engine, fmt):
    from sqlalchemy import event

    from dendrux.runtime._blocked_content import _blocked_write
    from dendrux.types import generate_ulid

    await pin_storage_format(engine, fmt)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        run_id = generate_ulid()
        with _blocked_write(store, run_id):
            await store.create_run(run_id, "entry", input_data={"input": PLACEHOLDER})
        assert not any(sql.lstrip().upper().startswith("UPDATE") for sql in statements)
        assert (await store.get_run(run_id)).meta["dendrux.blocked"] is True
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def test_blocked_system_prefix_uses_projection(store, engine):
    from dendrux.runtime.replay import _hash

    agent = Agent(
        prompt=SECRET,
        state_store=store,
        provider=MockLLM([LLMResponse(text="never")]),
        guardrails=[SecretDetection()],
    )
    try:
        result = await agent.run("go", tenant_id="tenant")
        row = await store.get_run(result.run_id)
        assert row.meta["dendrux.prefix"]["system_hash"] == _hash(PLACEHOLDER) != _hash(SECRET)
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
async def test_initial_skill_context_is_projected_before_create(
    store, engine, monkeypatch, tmp_path, fmt
):
    from functools import wraps

    from dendrux.skills import Skill

    await pin_storage_format(engine, fmt)
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: sample\ndescription: Example\n---\n" + SECRET)
    marks = []
    create = store.create_run

    @wraps(create)
    async def inspect_create(*args, **kwargs):
        result = await create(*args, **kwargs)
        marks.append((await store.get_run(result.run_id)).meta.get("dendrux.blocked"))
        return result

    monkeypatch.setattr(store, "create_run", inspect_create)
    provider = MockLLM([LLMResponse(text="never")])
    agent = Agent(
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        skills=[Skill.from_dir(folder)],
        guardrails=[SecretDetection()],
    )
    try:
        result = await agent.run("go", activate_skills=["sample"], tenant_id="tenant")
        assert result.status == RunStatus.ERROR and provider._call_count == 0
        assert marks == [True]
        async with engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert SECRET not in repr(rows), table.name
    finally:
        await agent.close()


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("purge", [False, True])
async def test_blocked_image_identity_deduplicates_projected_text(store, engine, fmt, purge):
    import hashlib
    from pathlib import Path

    from dendrux.media import Image, InMemoryBlobStore
    from dendrux.types import content_part_from_dict

    await pin_storage_format(engine, fmt)
    png = (Path(__file__).resolve().parents[1] / "fixtures/images/zebra42.png").read_bytes()
    provider = MockLLM([LLMResponse(text="never")])
    agent = Agent(
        name="image-entry",
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        blob_store=InMemoryBlobStore(),
        guardrails=[SecretDetection()],
    )
    try:
        first = await agent.run(
            [SECRET, Image.from_bytes(png)], idempotency_key="image", tenant_id="tenant"
        )
        assert first.status == RunStatus.ERROR and provider._call_count == 0
        row = await store.get_run(first.run_id)
        parts = tuple(content_part_from_dict(part) for part in row.input_data["dendrux.parts"])
        assert parts[0].text == PLACEHOLDER
        expected = compute_idempotency_fingerprint(
            "image-entry", PLACEHOLDER, user_input_parts=parts
        )
        async with engine.connect() as conn:
            actual = (
                await conn.execute(
                    select(AgentRun.idempotency_fingerprint).where(AgentRun.id == first.run_id)
                )
            ).scalar_one()
        assert actual == expected
        if purge:
            await agent._blob_services(store).catalog.mark_purging(
                "t:tenant", hashlib.sha256(png).hexdigest(), None
            )
        second = await agent.run(
            ["different " + SECRET, Image.from_bytes(png)],
            idempotency_key="image",
            tenant_id="tenant",
        )
        assert second.run_id == first.run_id and second.status == RunStatus.ERROR
        assert provider._call_count == 0
    finally:
        await agent.close()


@pytest.mark.parametrize("honors_meta", [False, True])
@pytest.mark.parametrize("single", [False, True])
async def test_custom_entry_requires_atomic_mark_contract(store, engine, honors_meta, single):
    from contextlib import nullcontext

    from dendrux.errors import PersistenceError
    from dendrux.runtime._blocked_content import _blocked_write
    from tests._fixtures.stores import LegacyInMemoryStore

    class ContractStore(LegacyInMemoryStore):
        async def create_run(self, run_id, agent_name, **kwargs):
            marked = honors_meta and (kwargs.get("meta") or {}).get("dendrux.blocked") is True
            with _blocked_write(self.wrapped, run_id) if marked else nullcontext():
                return await super().create_run(run_id, agent_name, **kwargs)

    provider = MockLLM([LLMResponse(text="never")])
    agent = Agent(
        provider=provider,
        state_store=ContractStore(store),
        prompt="Answer briefly.",
        guardrails=[SecretDetection()],
        loop=SingleCall() if single else None,
    )
    try:
        if honors_meta:
            result = await agent.run(SECRET, tenant_id="tenant")
            assert result.status == RunStatus.ERROR
            row = await store.get_run(result.run_id)
            assert row.meta["dendrux.blocked"] is True
            assert row.failure_reason == "guardrail_blocked"
        else:
            with pytest.raises(PersistenceError):
                await agent.run(SECRET, tenant_id="tenant")
        assert provider._call_count == 0
        async with engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert SECRET not in repr(rows), table.name
    finally:
        await agent.close()


async def test_rendered_context_block_projects_its_identity_components(store, engine):
    from dendrux.guardrails._protocol import Finding

    class RenderedGuard:
        action = "block"

        async def scan(self, text):
            if "ADDITIONAL CONTEXT:" in text:
                return [Finding("RENDERED", 0, len(text), 1.0, text)]
            return []

    provider = MockLLM([LLMResponse(text="never")])
    agent = Agent(
        name="rendered",
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[RenderedGuard()],
    )
    try:
        result = await agent.run(
            "INPUT_SENTINEL",
            context=[ContextBlock("CONTEXT_SENTINEL")],
            idempotency_key="rendered",
            tenant_id="tenant",
        )
        assert result.status == RunStatus.ERROR and provider._call_count == 0
        expected = compute_idempotency_fingerprint(
            "rendered", PLACEHOLDER, context=[ContextBlock(PLACEHOLDER)]
        )
        async with engine.connect() as conn:
            actual = (
                await conn.execute(
                    select(AgentRun.idempotency_fingerprint).where(AgentRun.id == result.run_id)
                )
            ).scalar_one()
            assert actual == expected
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert "INPUT_SENTINEL" not in repr(rows) and "CONTEXT_SENTINEL" not in repr(rows)
    finally:
        await agent.close()


async def test_user_trace_callback_keeps_recorded_projection(store, engine):
    from dendrux.guardrails import GuardrailEngine
    from dendrux.loops.base import BaseNotifier
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.runner import _record_user_trace
    from dendrux.types import Message, Role, generate_ulid

    await pin_storage_format(engine, 2)
    run_id = generate_ulid()
    await store.create_run(run_id, "entry")
    from dendrux.runtime.state import ClaimResult, claimed_store

    recorder = PersistenceRecorder(
        claimed_store(store, run_id, ClaimResult(0, 2, None, None)),
        run_id,
        guardrail_engine=GuardrailEngine([SecretDetection()]),
    )
    captured = []

    class Capture(BaseNotifier):
        async def on_message_appended(self, run_id, message, iteration):
            captured.append(message)

    await _record_user_trace(recorder, Capture(), run_id, Message(Role.USER, SECRET))
    assert captured[0].content == PLACEHOLDER
    assert (await store.get_run(run_id)).meta["dendrux.blocked"] is True


@pytest.mark.parametrize("fmt", [1, 2])
@pytest.mark.parametrize("outgoing", [False, True])
async def test_blocked_evidence_survives_crash_before_terminal(
    store, engine, monkeypatch, fmt, outgoing
):
    from functools import wraps

    from dendrux.runtime import runner

    class Crash(BaseException):
        pass

    await pin_storage_format(engine, fmt)
    operation = store.save_llm_interaction if outgoing else runner._create_runtime_run

    @wraps(operation)
    async def crash(*args, **kwargs):
        await operation(*args, **kwargs)
        raise Crash()

    if outgoing:
        monkeypatch.setattr(store, "save_llm_interaction", crash)
    else:
        monkeypatch.setattr(runner, "_create_runtime_run", crash)
    provider = MockLLM([LLMResponse(text=SECRET)])
    agent = Agent(
        prompt="Answer briefly.",
        provider=provider,
        state_store=store,
        guardrails=[SecretDetection()],
    )
    try:
        with pytest.raises(Crash):
            await agent.run("go" if outgoing else SECRET, tenant_id="tenant")
        assert provider._call_count == int(outgoing)
        async with engine.connect() as conn:
            row = (await conn.execute(select(AgentRun.__table__))).mappings().one()
            assert row["status"] == "running" and row["meta"]["dendrux.blocked"] is True
            for table in Base.metadata.sorted_tables:
                rows = (await conn.execute(select(table))).mappings().all()
                assert SECRET not in repr(rows), table.name
    finally:
        await agent.close()


@pytest.mark.parametrize("failure", ["cancel", "fence"])
async def test_entry_scan_preserves_cancellation_and_fencing(store, engine, failure):
    import asyncio

    from dendrux.errors import RunFencedError

    error = asyncio.CancelledError if failure == "cancel" else RunFencedError

    class Guard:
        action = "block"

        async def scan(self, text):
            raise error() if failure == "cancel" else error("run")

    provider = MockLLM([LLMResponse(text="never")])
    agent = Agent(
        prompt="Answer briefly.", provider=provider, state_store=store, guardrails=[Guard()]
    )
    try:
        with pytest.raises(error):
            await agent.run(SECRET, tenant_id="tenant")
        assert provider._call_count == 0
        async with engine.connect() as conn:
            assert (await conn.execute(select(AgentRun.id))).all() == []
    finally:
        await agent.close()


def _entry_race_worker(url, barrier, done, text):
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.types import RunAlreadyActiveError

    async def run():
        engine = create_async_engine(url)
        provider = MockLLM([LLMResponse(text="never")])
        agent = Agent(
            name="race",
            prompt="Answer briefly.",
            provider=provider,
            state_store=store_for_engine(engine),
            guardrails=[SecretDetection()],
        )
        try:
            await asyncio.to_thread(barrier.wait, 15)
            try:
                result = await agent.run(text, idempotency_key="race", tenant_id="tenant")
                assert result.status == RunStatus.ERROR
            except RunAlreadyActiveError:
                pass
            assert provider._call_count == 0
            done.set()
        finally:
            await agent.close()
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.parametrize("fmt", [1, 2])
async def test_two_process_projected_identity_race(store, engine, fmt):
    import asyncio
    import multiprocessing

    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres process race")
    await pin_storage_format(engine, fmt)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    signals = [ctx.Event(), ctx.Event()]
    processes = [
        ctx.Process(
            target=_entry_race_worker,
            args=(
                engine.url.render_as_string(hide_password=False),
                barrier,
                signals[i],
                prefix + SECRET,
            ),
        )
        for i, prefix in enumerate(("first ", "second "))
    ]
    try:
        for process in processes:
            process.start()
        for signal in signals:
            assert await asyncio.to_thread(signal.wait, 25)
        for process in processes:
            await asyncio.to_thread(process.join, 5)
            assert process.exitcode == 0
        async with engine.connect() as conn:
            row = (await conn.execute(select(AgentRun.__table__))).mappings().one()
            assert row["status"] == "error" and row["meta"]["dendrux.blocked"] is True
            assert row["input_data"] == {"input": PLACEHOLDER}
            assert row["idempotency_fingerprint"] == compute_idempotency_fingerprint(
                "race", PLACEHOLDER
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 5)
