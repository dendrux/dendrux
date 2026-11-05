import pytest
from sqlalchemy import event, insert

from dendrux import Agent, tool
from dendrux.db.models import ReactTrace
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ToolCall, ToolResult, generate_ulid
from tests._fixtures.storage import pin_storage_format


@tool()
async def slot_a() -> str:
    return "A"


@tool()
async def slot_b() -> str:
    return "B"


@tool()
async def slot_s() -> str:
    return "S"


@tool(target="client")
async def slot_c() -> str:
    return "C"


@pytest.mark.parametrize(
    "names",
    [
        ["slot_a", "slot_s", "nope", "slot_b"],
        ["slot_a", "nope", "slot_s", "bad"],
        ["slot_c", "slot_b"],
    ],
)
async def test_batch_order_matches_legacy(store, engine, names):
    views = []
    for fmt in (1, 2):
        await pin_storage_format(engine, fmt)
        calls = [
            ToolCall("slot_a" if n == "bad" else n, {"bad": 1} if n == "bad" else {}, id=str(i))
            for i, n in enumerate(names)
        ]
        provider = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
        agent = Agent(
            prompt="p", provider=provider, tools=[slot_a, slot_b, slot_s, slot_c], state_store=store
        )
        result = await agent.run("go", tenant_id="t")
        reader = RunStore(store)
        before = [
            (r.role, r.content, r.order_index)
            for r in await reader.get_traces(result.run_id, tenant_id="t", limit=50)
        ]
        if result.pending:
            result = await agent.submit_tool_results(
                result.run_id,
                [
                    ToolResult(name=tc.name, call_id=tc.id, payload='"ok"')
                    for tc in result.pending.tool_calls
                ],
                tenant_id="t",
                expected_pause_id=result.pause_id,
            )
        after = [
            (r.role, r.content, r.order_index)
            for r in await reader.get_traces(result.run_id, tenant_id="t", limit=50)
        ]
        wire = [
            [(m.role, m.content, m.call_id) for m in request] for request in provider.wire_messages
        ]
        views.append((before, after, wire))
    assert views[0] == views[1]


async def test_late_governance_denial_keeps_executed_prefix(store, engine):
    views = []
    for fmt in (1, 2):
        await pin_storage_format(engine, fmt)
        provider = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("slot_a", id="a"), ToolCall("slot_b", id="b")]),
                LLMResponse(text="done"),
            ]
        )
        agent = Agent(
            prompt="p",
            provider=provider,
            tools=[slot_a, slot_b],
            require_approval=["slot_a", "slot_b"],
            state_store=store,
        )
        paused = await agent.run("go", tenant_id="t")
        changed = Agent(
            prompt="p",
            provider=provider,
            tools=[slot_a, slot_b],
            require_approval=["slot_a"],
            deny=["slot_b"],
            on_governance_change="warn",
            state_store=store,
        )
        result = await changed.submit_approval(
            paused.run_id, approved=True, tenant_id="t", expected_pause_id=paused.pause_id
        )
        views.append(
            (
                [(m.content, m.call_id) for m in provider.wire_messages[-1]],
                [
                    (r.content, r.order_index)
                    for r in await RunStore(store).get_traces(
                        result.run_id, tenant_id="t", limit=50
                    )
                ],
            )
        )
    assert views[0] == views[1]


async def test_small_page_does_not_materialize_skipped_payloads(store, engine):
    await pin_storage_format(engine, 2)
    run_id = generate_ulid()
    await store.create_run(run_id, "paging", tenant_id="t")
    async with engine.begin() as connection:
        await connection.execute(
            insert(ReactTrace),
            [
                dict(
                    id=generate_ulid(),
                    agent_run_id=run_id,
                    trace_kind="turn",
                    role="assistant",
                    content="x",
                    order_index=i,
                    meta={},
                )
                for i in range(1242)
            ],
        )
    materialized = []
    statements = []

    def loaded(row, context):
        materialized.append(row.order_index)

    def queried(connection, cursor, statement, parameters, context, many):
        if "react_traces" in statement:
            statements.append(statement)

    event.listen(ReactTrace, "load", loaded)
    event.listen(engine.sync_engine, "before_cursor_execute", queried)
    try:
        reader = RunStore(store)
        rows = await reader.get_traces(run_id, tenant_id="t", limit=1)
        assert len(rows) == 1 and materialized == [0]
        assert len(statements) == 1
        materialized.clear()
        statements.clear()
        rows = await reader.get_traces(run_id, tenant_id="t", limit=50, offset=900)
        assert [row.order_index for row in rows] == list(range(900, 950))
        assert materialized == list(range(900, 950))
        assert len(statements) == 1
    finally:
        event.remove(ReactTrace, "load", loaded)
        event.remove(engine.sync_engine, "before_cursor_execute", queried)


async def test_public_page_does_not_use_recovery_byte_limit(store, engine):
    await pin_storage_format(engine, 2)
    run_id = generate_ulid()
    await store.create_run(run_id, "large audit", tenant_id="t")
    async with engine.begin() as connection:
        await connection.execute(
            insert(ReactTrace),
            [
                dict(
                    id=generate_ulid(),
                    agent_run_id=run_id,
                    trace_kind="turn",
                    role="assistant",
                    content="x" * (1024 * 1024),
                    order_index=i,
                    meta={},
                )
                for i in range(18)
            ],
        )
    rows = await RunStore(store).get_traces(run_id, tenant_id="t", limit=50)
    assert len(rows) == 18
    rows = await RunStore(store).get_traces(run_id, tenant_id="t", limit=1, offset=17)
    assert len(rows) == 1 and rows[0].order_index == 17
