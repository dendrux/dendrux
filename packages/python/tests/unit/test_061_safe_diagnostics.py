"""Closed diagnostics for best-effort failures and result transforms."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from dendrux.agent import Agent
from dendrux.errors import ProviderError, RunFencedError
from dendrux.llm import _resilience
from dendrux.loops.react import _execute_tool
from dendrux.media._ids import image_id
from dendrux.runtime._blob_catalog import CatalogBlobAccessor, InMemoryBlobCatalog, _GcStats
from dendrux.runtime.context import _touch
from dendrux.runtime.replay import CurrentPrefix, record_run_prefix
from dendrux.tool import tool
from dendrux.types import RunStream, ToolCall

SENTINEL = "PRIVATE_DIAGNOSTIC_SENTINEL"
SAFE_TRANSFORM = "Tool result transform failed."
RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
SITES = (
    "heartbeat",
    "blob_orphans",
    "blob_step",
    "mcp_optional",
    "mcp_discovery_cleanup",
    "mcp_refresh",
    "mcp_close",
    "runstream",
    "replay",
    "response_hook",
    "provider_close",
    "transform",
)


def arbitrary_error(kind):
    if kind == "value":
        return ValueError(SENTINEL)
    if kind == "custom":
        return type(f"{SENTINEL}Error", (Exception,), {})(SENTINEL)
    try:
        raise ValueError(SENTINEL)
    except ValueError as cause:
        try:
            raise ProviderError(message=SENTINEL, provider="mock") from cause
        except ProviderError as provider:
            try:
                raise RuntimeError(SENTINEL) from provider
            except RuntimeError as wrapper:
                return wrapper


def assert_safe_records(records):
    assert records
    for record in records:
        assert SENTINEL not in repr(vars(record))
        assert SENTINEL not in record.getMessage()
        assert not record.exc_info
        assert not record.exc_text
        assert not record.stack_info


@tool()
async def diagnostic_value() -> str:
    """Return a harmless tool result."""
    return "value"


async def exercise_site(site, exc, monkeypatch):
    failure = AsyncMock(side_effect=exc)
    if site == "heartbeat":
        await _touch(SimpleNamespace(touch_progress=failure), RUN_ID)
    elif site == "blob_orphans":
        catalog = InMemoryBlobCatalog()
        monkeypatch.setattr(catalog, "delete_orphan_edges", failure)
        await CatalogBlobAccessor(catalog).collect_garbage()
    elif site == "blob_step":
        info = SimpleNamespace(tenant_scope="-", sha256="a" * 64, state="live")
        await CatalogBlobAccessor(None)._guarded(failure, None, info, datetime.now(UTC), _GcStats())
    elif site.startswith("mcp_"):
        agent = Agent(prompt="diagnostics")
        source = SimpleNamespace(name="configured", close=failure)
        agent._tool_sources = [source]
        if site == "mcp_optional":
            source.failure_mode = "best_effort"
            source._discover = failure
            await agent._ensure_discovered()
        elif site == "mcp_discovery_cleanup":
            source._discover = AsyncMock(return_value=[])
            trigger = RuntimeError("discovery stopped")
            agent._tool_sources.append(SimpleNamespace(_discover=AsyncMock(side_effect=trigger)))
            with pytest.raises(RuntimeError) as caught:
                await agent._ensure_discovered()
            assert caught.value is trigger
        else:
            await getattr(agent, "refresh" if site == "mcp_refresh" else "close")()
            assert agent._discovered_tool_defs is None
    elif site == "runstream":
        generator = SimpleNamespace(aclose=AsyncMock())
        stream = RunStream(RUN_ID, generator, failure)
        try:
            await stream.aclose()
        finally:
            generator.aclose.assert_awaited_once()
        await stream.aclose()
        failure.assert_awaited_once()
    elif site == "replay":
        await record_run_prefix(
            SimpleNamespace(update_run_meta=failure),
            RUN_ID,
            current=CurrentPrefix("mock", "mock", "system", "tools"),
            seed={},
            plan=None,
        )
    elif site == "response_hook":

        def fail(*args):
            raise exc

        monkeypatch.setattr(_resilience, "parse_retry_after", fail)
        monkeypatch.setattr(_resilience, "_call_state", SimpleNamespace(get=SimpleNamespace))
        await _resilience._on_response(
            httpx.Response(200, request=httpx.Request("GET", "https://example.invalid"))
        )
    elif site == "provider_close":
        await _resilience.close_quietly(SimpleNamespace(aclose=failure))
    else:
        agent = Agent(prompt="diagnostics", tools=[diagnostic_value])
        call = ToolCall(name="diagnostic_value", params={}, provider_tool_call_id="provider-call")
        result = await _execute_tool(
            call,
            await agent.get_tool_lookups(),
            transform=failure,
            run_id=RUN_ID,
        )
        assert result.name == call.name
        assert result.call_id == call.id
        assert result.success is False
        assert result.duration_ms >= 0
        assert result.error == SAFE_TRANSFORM
        assert json.loads(result.payload) == {"error": SAFE_TRANSFORM}
    if site != "response_hook":
        failure.assert_awaited_once()


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("kind", ["value", "custom", "chain"])
async def test_061_arbitrary_exception_text_never_leaks(site, kind, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="dendrux")
    await exercise_site(site, arbitrary_error(kind), monkeypatch)
    records = [r for r in caplog.records if r.name.startswith("dendrux")]
    assert_safe_records(records)
    assert records[-1].levelno == (
        logging.DEBUG
        if site in ("response_hook", "provider_close")
        else logging.ERROR
        if site.startswith("blob_")
        else logging.WARNING
    )
    if site in ("heartbeat", "replay", "runstream", "transform"):
        assert records[-1].run_id == RUN_ID
    if site == "heartbeat":
        assert records[-1].getMessage() == "Heartbeat failed"
    if site == "replay":
        assert records[-1].getMessage() == "Failed to record prefix descriptor"
    if site.startswith("mcp_"):
        assert records[-1].args == ("configured",)
        assert "configured" in records[-1].getMessage()
    if site == "blob_step":
        assert records[-1].args == ("-", image_id("a" * 64), "live")
        assert "scope=-" in records[-1].getMessage()
        assert image_id("a" * 64) in records[-1].getMessage()
        assert "state=live" in records[-1].getMessage()
    if site == "transform":
        assert records[-1].getMessage() == SAFE_TRANSFORM


@pytest.mark.parametrize("site", SITES)
async def test_cancellation_is_not_swallowed(site, monkeypatch, caplog):
    cancellation = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as caught:
        await exercise_site(site, cancellation, monkeypatch)
    assert caught.value is cancellation
    assert not [r for r in caplog.records if r.name.startswith("dendrux")]


async def test_heartbeat_fencing_is_not_swallowed(caplog):
    failure = RunFencedError(RUN_ID)
    with pytest.raises(RunFencedError) as caught:
        await _touch(SimpleNamespace(touch_progress=AsyncMock(side_effect=failure)), RUN_ID)
    assert caught.value is failure
    assert not caplog.records


@pytest.mark.parametrize("site", ["heartbeat", "replay"])
async def test_best_effort_diagnostics_preserve_existing_run_ids(site, monkeypatch):
    monkeypatch.setenv("DENDRUX_STRICT_LOG_FIELDS", "1")
    monkeypatch.setattr(f"{__name__}.RUN_ID", "short-run-id")
    await exercise_site(site, ValueError(SENTINEL), monkeypatch)
