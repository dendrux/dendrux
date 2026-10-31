"""A bare ``mysql+asyncmy://`` URL works end to end on MySQL 8.4 and MariaDB 10.11.

The library creates the engine with READ COMMITTED and UTC at connect; the developer passes no
extra options. Driver spellings other than ``asyncmy`` are normalised.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from dendrux import Agent, tool
from dendrux.cli.main import app
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM

from .conftest import run_sync


@pytest.fixture
def migrated_url(mysql_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DENDRUX_DATABASE_URL", mysql_url)
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 0, result.output
    monkeypatch.delenv("DENDRUX_DATABASE_URL")
    return mysql_url


@tool(target="client")
async def read_cells() -> str:
    return ""


async def session_settings(engine: Any) -> tuple[str, str]:
    async with engine.connect() as conn:
        isolation = await conn.get_isolation_level()
        zone = await conn.scalar(text("SELECT @@session.time_zone"))
    return str(isolation), str(zone)


def with_driver(url: str, driver: str) -> str:
    return url.replace("mysql+asyncmy://", f"{driver}://", 1)


@pytest.mark.parametrize("driver", ["mysql+asyncmy", "mysql+aiomysql", "mysql", "mariadb"])
async def test_bare_database_url_pauses_submits_resumes_and_finishes(
    migrated_url: str, driver: str
) -> None:
    provider = ScriptedLLM(
        [LLMResponse(tool_calls=[ToolCall("read_cells", {})]), LLMResponse(text="done")]
    )
    async with Agent(
        prompt="p",
        tools=[read_cells],
        provider=provider,
        database_url=with_driver(migrated_url, driver),
    ) as agent:
        paused = await agent.run("go", tenant_id="t")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        call = paused.pending.tool_calls[0]
        finished = await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(call.name, call.id, '"ok"')],
            tenant_id="t",
            expected_pause_id=paused.pause_id,
        )
        assert finished.status == RunStatus.SUCCESS
        assert finished.answer == "done"
        engine = agent._private_engine
        assert engine.url.drivername == "mysql+asyncmy"
        assert await session_settings(engine) == ("READ COMMITTED", "+00:00")


async def test_run_store_from_url_reads_with_the_same_settings(migrated_url: str) -> None:
    async with RunStore.from_database_url(with_driver(migrated_url, "mysql+aiomysql")) as reader:
        assert await reader.list_runs() == []
        assert await session_settings(reader._owned_engine) == ("READ COMMITTED", "+00:00")


async def test_shared_env_engine_uses_the_same_settings(
    migrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dendrux.db.session import get_engine, reset_engine

    monkeypatch.setenv("DENDRUX_DATABASE_URL", with_driver(migrated_url, "mysql+aiomysql"))
    try:
        engine = await get_engine()
        assert await session_settings(engine) == ("READ COMMITTED", "+00:00")
    finally:
        await reset_engine()


def test_cli_accepts_the_aiomysql_spelling(
    migrated_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DENDRUX_DATABASE_URL", with_driver(migrated_url, "mysql+aiomysql"))
    for command in (["db", "check"], ["db", "status"], ["db", "migrate"]):
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 0, (command, result.output)


def test_sweep_helper_builds_a_store_on_a_bare_mysql_url(migrated_url: str) -> None:
    from dendrux.runtime.sweep import _create_temp_store

    async def go() -> tuple[str, str]:
        store, engine = await _create_temp_store(with_driver(migrated_url, "mysql+aiomysql"))
        try:
            return await session_settings(engine)
        finally:
            await engine.dispose()

    assert run_sync(go()) == ("READ COMMITTED", "+00:00")


async def test_store_from_url_shares_the_library_settings(migrated_url: str) -> None:
    from dendrux.store import MySQLStateStore

    store = MySQLStateStore.from_url(with_driver(migrated_url, "mysql+aiomysql"))
    try:
        assert await store.get_run("absent") is None
        assert await session_settings(store._engine) == ("READ COMMITTED", "+00:00")
    finally:
        await store._engine.dispose()


async def test_a_large_frontier_is_not_capped_by_the_class_default_budget(
    migrated_url: str,
) -> None:
    """A frontier larger than ``Budget.max_subtree_nodes`` (default 10,000) is chunked, not refused.

    The cascade that builds a frontier is bounded by the Agent's configured budget, so the
    store must not apply the class default a second time.
    """
    from dendrux.db.models import AgentRun
    from dendrux.store import MySQLStateStore
    from dendrux.types import Budget

    ids = [f"absent-{i}" for i in range(Budget.max_subtree_nodes + 5_000)]
    store = MySQLStateStore.from_url(migrated_url)
    try:
        clause = store.id_in(AgentRun.id, ids)
        assert len(clause.clauses) == 15  # chunks of 1,000 joined by OR
        assert await store.mark_cancel_requested(ids, tenant_id=None) == 0
        assert await store.get_children_page(ids, after=None, limit=10, tenant_id=None) == []
    finally:
        await store._engine.dispose()


async def server_packet(url: str) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            return int(await conn.scalar(text("SELECT @@max_allowed_packet")))
    finally:
        await engine.dispose()


async def test_the_store_reports_the_servers_statement_capacity(migrated_url: str) -> None:
    from dendrux.store import MySQLStateStore

    store = MySQLStateStore.from_url(migrated_url)
    try:
        assert await store.statement_capacity() == await server_packet(migrated_url)
    finally:
        await store._engine.dispose()


async def test_default_limits_pass_and_an_oversized_configured_limit_is_refused(
    migrated_url: str,
) -> None:
    from dendrux.runtime.stores._limits_fit import required_statement_bytes
    from dendrux.types import Limits

    packet = await server_packet(migrated_url)
    provider = ScriptedLLM([LLMResponse(text="done")])
    async with Agent(prompt="p", provider=provider, database_url=migrated_url) as agent:
        assert (await agent.run("go", tenant_id="t")).status == RunStatus.SUCCESS

    oversized = Limits(max_tool_result_bytes=packet)  # twelve times the packet once escaped
    async with Agent(
        prompt="p",
        provider=ScriptedLLM([LLMResponse(text="never")]),
        database_url=migrated_url,
        limits=oversized,
    ) as refused:
        with pytest.raises(ValueError) as caught:
            await refused.run("go", tenant_id="t")
        message = str(caught.value)
        assert str(required_statement_bytes(oversized)) in message
        assert str(packet) in message
        # Only the earlier successful run exists: the refused call created nothing.
        assert len(await refused._lazy_store.list_runs()) == 1
    assert provider.calls == 1


async def test_limits_sized_to_the_server_are_accepted(migrated_url: str) -> None:
    from dendrux.types import Limits

    packet = await server_packet(migrated_url)
    sized = Limits(max_input_bytes=packet // 24, max_tool_result_bytes=packet // 24)
    async with Agent(
        prompt="p",
        provider=ScriptedLLM([LLMResponse(text="done")]),
        database_url=migrated_url,
        limits=sized,
    ) as agent:
        assert (await agent.run("go", tenant_id="t")).status == RunStatus.SUCCESS


def test_the_migration_engine_gets_the_same_settings(
    mysql_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlalchemy.ext.asyncio as asyncio_ext

    real = asyncio_ext.async_engine_from_config
    seen: list[dict[str, Any]] = []

    def recording(configuration: Any, prefix: str = "sqlalchemy.", **kwargs: Any) -> Any:
        seen.append({"url": configuration[f"{prefix}url"], **kwargs})
        return real(configuration, prefix, **kwargs)

    monkeypatch.setattr(asyncio_ext, "async_engine_from_config", recording)
    monkeypatch.setenv("DENDRUX_DATABASE_URL", with_driver(mysql_url, "mysql+aiomysql"))
    result = CliRunner().invoke(app, ["db", "migrate"])
    assert result.exit_code == 0, result.output
    [engine] = seen
    assert engine["url"].startswith("mysql+asyncmy://")
    assert engine["isolation_level"] == "READ COMMITTED"
    assert engine["connect_args"]["init_command"] == "SET time_zone='+00:00'"
