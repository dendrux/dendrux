"""Scoped persisted result reconstruction and pause-view parity on both databases."""

import json
import re

import pytest

from dendrux import Agent
from dendrux.errors import RunNotFoundError
from dendrux.llm.mock import MockLLM
from dendrux.store import UNSCOPED, RunStore
from dendrux.types import ErrorInfo, RunStatus
from tests.integration.test_tenant_run_control import _pause


@pytest.mark.parametrize(
    "status", [s.value for s in RunStatus if s not in (RunStatus.PENDING, RunStatus.RUNNING)]
)
async def test_result_wire_json_roundtrip_every_terminal_and_pause(store, status):
    await store.create_run("result", "agent", tenant_id="A")
    if status.startswith("waiting_") or status == "paused":
        await store.pause_run(
            "result",
            status=status,
            pause_data={
                "agent_name": "agent",
                "pause_id": "P",
                "history": [{"content": "PROMPT_SENTINEL"}],
                "pending_tool_calls": [
                    {"id": "C", "name": "read", "params": {"data": "PII_SENTINEL"}}
                ],
            },
        )
    else:
        await store.finalize_run("result", status=status, answer="safe", error="VENDOR_SENTINEL")
    agent = Agent(
        name="agent",
        prompt="PROMPT_SENTINEL",
        provider=MockLLM([]),
        state_store=store,
        strict_tenancy=True,
    )
    if status == "error":
        from dendrux.errors import ProviderAuthError

        safe = ErrorInfo.from_error(ProviderAuthError(provider="test", message="VENDOR_SENTINEL"))
        await store.update_run_meta("result", {"dendrux.error_info": safe.to_dict()})
    result = await agent.get_result("result", tenant_id="A")
    expected = await RunStore(store).get_pending("result", tenant_id="A")
    assert result.status.value == status
    assert (result.pending.to_dict() if result.pending else None) == (
        expected.to_dict() if expected else None
    )
    wire = result.to_dict()
    assert json.loads(json.dumps(wire)) == wire
    if status == "error":
        assert wire["error_info"] == safe.to_dict()
    assert "PROMPT_SENTINEL" not in json.dumps(wire) and "VENDOR_SENTINEL" not in json.dumps(wire)
    if status in ("waiting_approval", "waiting_human_input"):
        assert "PII_SENTINEL" not in json.dumps(wire), "missing model arguments must be hidden"
    for tenant in ("B", None):
        with pytest.raises(RunNotFoundError):
            await agent.get_result("result", tenant_id=tenant)
    with pytest.raises(ValueError):
        await agent.get_result("result", tenant_id=UNSCOPED)


@pytest.mark.parametrize("kind", ["client", "approval", "paused"])
async def test_live_paused_result_and_fetched_result_match(store, kind):
    fixture = await _pause(kind, store, "A")
    fetched = await fixture.agent.get_result(fixture.run_id, tenant_id="A")
    assert fetched.pending is not None
    pending = await RunStore(store).get_pending(fixture.run_id, tenant_id="A")
    assert fetched.pending.to_dict() == pending.to_dict()
    # Also exercise direct run() rather than only the reconstruction API.
    from dendrux import tool
    from tests._helpers.scripted_provider import ScriptedLLM, tool_call

    @tool(target="client")
    def read() -> str:
        return "unused"

    agent = Agent(
        name="live",
        prompt="p",
        tools=[read],
        state_store=store,
        provider=ScriptedLLM([tool_call("read", pid="live-call")]),
        strict_tenancy=False,
    )
    paused = await agent.run("go", tenant_id="A")
    assert paused.pending is not None
    assert (
        paused.pending.to_dict()
        == (await RunStore(store).get_pending(paused.run_id, tenant_id="A")).to_dict()
    )


async def test_cached_result_carries_active_pending_and_mirrors_persisted_values(store):
    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.runtime.runner import _build_cached_result

    await store.create_run("cached", "agent", tenant_id="A")
    await store.pause_run(
        "cached",
        status="waiting_client_tool",
        pause_data={
            "pause_id": "P",
            "pending_tool_calls": [
                {"id": "C", "name": "read", "params": {"email": "private@example.test"}}
            ],
        },
    )
    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == "cached")
            .values(
                pii_mapping={"<<EMAIL_1>>": "private@example.test"},
            )
        )
        await session.commit()
    cached = await _build_cached_result(store, "cached")
    assert cached.pending is not None, "cached pause must carry the active pause"
    assert cached.to_dict()["pending"] == cached.pending.to_dict()
    assert "pii_mapping" not in json.dumps(cached.to_dict())
    await store.finalize_run("cached", status="success", answer="private@example.test")
    cached = await _build_cached_result(store, "cached")
    assert cached.pending is None
    assert cached.to_dict()["answer"] == cached.answer == "private@example.test"


@pytest.mark.parametrize("persisted", [True, False])
@pytest.mark.parametrize("streaming", [True, False])
async def test_active_pending_from_live_run_and_stream(store, persisted, streaming):
    from dendrux import tool
    from tests._helpers.scripted_provider import ScriptedLLM, tool_call
    from tests.integration.test_050_t7_integrity import consume

    @tool(target="client")
    def read() -> str:
        return "unused"

    agent = Agent(
        name="active",
        prompt="PROMPT_SENTINEL",
        tools=[read],
        provider=ScriptedLLM([tool_call("read", pid="read")]),
        state_store=store if persisted else None,
        strict_tenancy=False,
    )
    try:
        if streaming:
            result, _ = await consume(agent.stream("go", tenant_id="A"))
        else:
            result = await agent.run("go", tenant_id="A")
        assert result.pending is not None and result.pending.pause_id == result.pause_id
        assert result.to_dict()["pending"]["pause_id"] == result.pause_id
        assert "PROMPT_SENTINEL" not in json.dumps(result.to_dict())
        if persisted:
            fetched = await agent.get_result(result.run_id, tenant_id="A")
            assert fetched.pending.to_dict() == result.pending.to_dict()
            other = Agent(name="other", prompt="p", provider=MockLLM([]), state_store=store)
            with pytest.raises(RunNotFoundError):
                await other.get_result(result.run_id, tenant_id="A")
    finally:
        await agent.close()


async def test_get_result_projects_only_the_browser_read_fields(store, engine, record_property):
    from sqlalchemy import event, insert, text

    from dendrux.db.models import AgentRun

    if engine.dialect.name == "postgresql":
        async with engine.begin() as conn:
            for first in range(0, 20000, 5000):
                await conn.execute(
                    insert(AgentRun),
                    [
                        dict(
                            id=f"scale-{i:06}",
                            agent_name="scale",
                            tenant_id="A" if i % 2 else "B",
                            status=("running", "success", "waiting_approval")[i % 3],
                        )
                        for i in range(first, first + 5000)
                    ],
                )
            await conn.execute(text("ANALYZE agent_runs"))

    await store.create_run(
        "project", "projected", tenant_id="A", input_data={"prompt": "PROMPT_SENTINEL"}
    )
    await store.pause_run(
        "project",
        status="waiting_client_tool",
        pause_data={
            "pause_id": "P",
            "history": [{"content": "HISTORY_SENTINEL" * 10000}],
            "pending_tool_calls": [{"id": "C", "name": "read", "params": {}}],
        },
    )
    statements = []
    parameters_by_statement = []

    def capture(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)
            parameters_by_statement.append(parameters)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    a = Agent(name="projected", prompt="p", provider=MockLLM([]), state_store=store)
    try:
        result = await a.get_result("project", tenant_id="A")
        assert result.pending is not None
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await a.close()
    assert not any("agent_runs.input_data" in s for s in statements), (
        "result read must not load prompts"
    )
    assert not any(
        re.search(r"(?:SELECT|,)\s+agent_runs\.pause_data(?:,|\s+FROM)", s) for s in statements
    ), "result read must not load pause history"
    assert len(statements) <= 3, "client result is a constant-cost read, independent of history"
    record_property("get_result_select_count", len(statements))

    if engine.dialect.name == "postgresql":
        plans = []
        async with engine.connect() as conn:
            for statement, params in zip(statements, parameters_by_statement, strict=True):
                plan = (
                    await conn.exec_driver_sql(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, params
                    )
                ).scalar_one()[0]
                rendered = json.dumps(plan)
                assert "agent_runs_pkey" in rendered, rendered
                assert plan["Plan"]["Actual Rows"] == 1
                plans.append(plan)
        record_property("get_result_20k_plans", json.dumps(plans))


async def test_live_pii_approval_wire_is_safe_in_both_store_payload_views(store):
    from dendrux import tool
    from dendrux.guardrails import PII
    from dendrux.types import LLMResponse, ToolCall

    @tool()
    def send(to: str) -> str:
        return "not called before approval"

    a = Agent(
        name="pii-wire",
        prompt="PROMPT_SENTINEL",
        tools=[send],
        require_approval=["send"],
        provider=MockLLM(
            [LLMResponse(tool_calls=[ToolCall(name="send", params={"to": "<<EMAIL_ADDRESS_1>>"})])]
        ),
        guardrails=[PII()],
        state_store=store,
        strict_tenancy=False,
    )
    try:
        paused = await a.run("Email alice@example.com", tenant_id="A")
        fetched = await a.get_result(paused.run_id, tenant_id="A")
        facade = RunStore(store)
        assert (
            fetched.pending.to_dict()
            == (await facade.get_pending(paused.run_id, tenant_id="A")).to_dict()
        )
        for payloads in (False, True):
            await facade.get_pause_page(paused.run_id, tenant_id="A", include_payloads=payloads)
            for result in (paused, fetched):
                text = json.dumps(result.to_dict())
                assert "alice@example.com" not in text
                assert "PROMPT_SENTINEL" not in text
                assert "<<EMAIL_ADDRESS_1>>" in text
    finally:
        await a.close()
