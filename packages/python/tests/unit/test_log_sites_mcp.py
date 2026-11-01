"""MCP diagnostics contain registered codes and counts, never transport text."""

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from dendrux.mcp import MCPSource
from dendrux.mcp._client import MCPClientAdapter


@pytest.mark.parametrize("grouped", [False, True])
async def test_two_adapters_keep_diagnostics_closed(grouped, caplog):
    sentinel = "PRIVATE_TRANSPORT_SENTINEL"
    errors = [RuntimeError(sentinel), ValueError(sentinel)]
    if grouped:
        errors = [ExceptionGroup(sentinel, [e]) for e in errors]
    adapters = [
        MCPClientAdapter(MCPSource.http(f"SOURCE_SENTINEL_{i}", "https://mcp.example.invalid"))
        for i in range(2)
    ]

    async def project(adapter, error):
        await asyncio.sleep(0)
        return adapter._safe_error("connect", error)

    with caplog.at_level(logging.DEBUG, logger="dendrux.mcp._client"):
        projected = await asyncio.gather(
            *(project(a, e) for a, e in zip(adapters, errors, strict=True))
        )
    assert all(error.transport_detail == "mcp_transport_failed count=1" for error in projected)
    assert len(caplog.records) == 2
    standard = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
    for record in caplog.records:
        assert record.code == "mcp_transport_failed" and record.count == 1
        assert record.levelno == logging.DEBUG
        assert set(record.__dict__) - standard == {"code", "kind", "count"}
    assert sentinel not in caplog.text + repr([r.__dict__ for r in caplog.records])
    assert "SOURCE_SENTINEL" not in caplog.text


def test_mcp_debug_sites_use_registry_and_cleanup_never_formats_exception_text():
    source = Path(__file__).parents[2] / "src/dendrux/mcp"
    failures = []
    for path in source.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "debug":
                failures.append((path.name, node.lineno, "direct debug"))
            if isinstance(node.func, ast.Name) and node.func.id == "safe_source_exception_detail":
                failures.append((path.name, node.lineno, "exception text"))
    assert not failures, failures
