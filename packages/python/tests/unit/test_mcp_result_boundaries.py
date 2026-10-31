"""MCP discovery is bounded: aggregate deadline, page cap, repeated-cursor refusal."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.mcp import MCPConnectionError, MCPServer, MCPSource
from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import MCPDiscoveryLimitError, error_code_for

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CURSOR_SECRET = "cursor-sentinel-1234"


def _tool(name: str) -> Any:
    return SimpleNamespace(name=name)


class _PagedClient:
    """Stands in for a connected SDK client serving scripted ``tools/list`` pages."""

    def __init__(self, cursors: Any, *, delay: float = 0.0, hang_at: int | None = None) -> None:
        self.cursors = cursors  # callable(page_index) -> next cursor or None
        self.delay = delay
        self.hang_at = hang_at  # from this page on, a request never returns
        self.pages = 0

    async def list_tools(self, cursor: Any = None, cache_mode: Any = None) -> Any:
        index = self.pages
        self.pages += 1
        if self.hang_at is not None and index >= self.hang_at:
            await asyncio.Event().wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(tools=[_tool(f"t{index}")], next_cursor=self.cursors(index))


def _adapter(source: MCPSource, client: _PagedClient) -> MCPClientAdapter:
    adapter = MCPClientAdapter(source)
    adapter._client = client  # type: ignore[assignment]
    return adapter


def _source(**kw: Any) -> MCPSource:
    return MCPSource.http("remote", "https://mcp.example.com/mcp", **kw)


class TestRepeatedCursor:
    async def test_same_cursor_twice_in_a_row_is_refused(self) -> None:
        client = _PagedClient(lambda i: CURSOR_SECRET)
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(), client).list_tools()
        assert caught.value.reason == "repeated_cursor"
        assert client.pages == 2

    async def test_cursor_that_loops_back_is_refused(self) -> None:
        client = _PagedClient(lambda i: ["a", "b", "c", "a"][i % 4])
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(), client).list_tools()
        assert caught.value.reason == "repeated_cursor"
        assert client.pages == 4

    async def test_error_is_typed_coded_and_carries_no_cursor(self) -> None:
        client = _PagedClient(lambda i: CURSOR_SECRET)
        with pytest.raises(MCPConnectionError) as caught:
            await _adapter(_source(), client).list_tools()
        assert isinstance(caught.value, MCPDiscoveryLimitError)
        assert error_code_for(caught.value) == "discovery_limit"
        assert CURSOR_SECRET not in str(caught.value)
        assert CURSOR_SECRET not in repr(caught.value.transport_detail)
        assert "remote" in str(caught.value)

    async def test_logs_a_stable_code_without_the_cursor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _PagedClient(lambda i: CURSOR_SECRET)
        with caplog.at_level(logging.WARNING), pytest.raises(MCPDiscoveryLimitError):
            await _adapter(_source(), client).list_tools()
        assert "mcp_discovery_repeated_cursor" in caplog.text
        (record,) = [
            r for r in caplog.records if getattr(r, "code", None) == "mcp_discovery_repeated_cursor"
        ]
        assert record.pages == 2
        assert "remote" not in caplog.text
        assert CURSOR_SECRET not in caplog.text

    async def test_distinct_cursors_still_page_through(self) -> None:
        client = _PagedClient(lambda i: f"c{i}" if i < 4 else None)
        tools = await _adapter(_source(), client).list_tools()
        assert [t.name for t in tools] == ["t0", "t1", "t2", "t3", "t4"]


class TestPageCap:
    async def test_endless_distinct_cursors_stop_at_the_cap(self) -> None:
        client = _PagedClient(lambda i: f"c{i}")
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(max_discovery_pages=5), client).list_tools()
        assert caught.value.reason == "page_cap"
        assert client.pages == 5

    async def test_exactly_the_cap_is_allowed(self) -> None:
        client = _PagedClient(lambda i: f"c{i}" if i < 4 else None)
        tools = await _adapter(_source(max_discovery_pages=5), client).list_tools()
        assert len(tools) == 5

    async def test_default_cap_is_bounded(self) -> None:
        assert 1 <= _source().max_discovery_pages <= 1000
        client = _PagedClient(lambda i: f"c{i}")
        with pytest.raises(MCPDiscoveryLimitError):
            await _adapter(_source(), client).list_tools()
        assert client.pages == _source().max_discovery_pages


class TestAggregateDeadline:
    # The deadline is the only thing that can end these requests, so passing means it fired.
    # No elapsed-time assertion: the hang guard is generous and does not depend on load.
    async def test_pages_that_stop_answering_end_at_the_deadline(self) -> None:
        client = _PagedClient(lambda i: f"c{i}", hang_at=3)
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await asyncio.wait_for(
                _adapter(_source(discovery_timeout=0.5), client).list_tools(), timeout=60
            )
        assert caught.value.reason == "deadline"
        assert client.pages >= 1

    async def test_one_page_that_never_answers_is_cut_off(self) -> None:
        client = _PagedClient(lambda i: None, hang_at=0)
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await asyncio.wait_for(
                _adapter(_source(discovery_timeout=0.2), client).list_tools(), timeout=60
            )
        assert caught.value.reason == "deadline"
        assert client.pages == 1

    async def test_fast_discovery_is_unaffected(self) -> None:
        client = _PagedClient(lambda i: f"c{i}" if i < 2 else None)
        assert len(await _adapter(_source(discovery_timeout=5), client).list_tools()) == 3

    async def test_an_outer_cancellation_is_not_swallowed(self) -> None:
        client = _PagedClient(lambda i: f"c{i}", delay=0.05)
        task = asyncio.create_task(_adapter(_source(), client).list_tools())
        await asyncio.sleep(0.12)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestOneOversizedPage:
    async def test_too_many_tools_in_one_page(self) -> None:
        class Huge:
            async def list_tools(self, cursor: Any = None, cache_mode: Any = None) -> Any:
                tools = [_tool(f"t{i}") for i in range(100_000)]
                return SimpleNamespace(tools=tools, next_cursor=None)

        adapter = MCPClientAdapter(_source(max_discovery_pages=1))
        adapter._client = Huge()  # type: ignore[assignment]
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await adapter.list_tools()
        assert caught.value.reason == "tool_cap"

    async def test_tool_cap_counts_across_pages(self) -> None:
        client = _PagedClient(lambda i: f"c{i}")
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(max_discovery_tools=3), client).list_tools()
        assert caught.value.reason == "tool_cap"
        assert client.pages == 4

    async def test_exactly_the_tool_cap_is_allowed(self) -> None:
        client = _PagedClient(lambda i: f"c{i}" if i < 2 else None)
        assert len(await _adapter(_source(max_discovery_tools=3), client).list_tools()) == 3

    async def test_overlong_cursor_is_refused(self) -> None:
        client = _PagedClient(lambda i: "c" * 100_000)
        with pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(), client).list_tools()
        assert caught.value.reason == "cursor_length"
        assert client.pages == 1

    async def test_error_text_and_log_name_the_reason_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _PagedClient(lambda i: "c" * 100_000)
        with caplog.at_level(logging.WARNING), pytest.raises(MCPDiscoveryLimitError) as caught:
            await _adapter(_source(), client).list_tools()
        assert "mcp_discovery_cursor_length" in caplog.text
        assert "ccccc" not in caplog.text
        assert "ccccc" not in str(caught.value)


class TestSourceConfiguration:
    def test_new_bound_defaults(self) -> None:
        source = _source()
        assert source.max_discovery_tools == 2000
        assert source.max_discovery_bytes == 16 * 1024 * 1024

    @pytest.mark.parametrize("field", ["max_discovery_tools", "max_discovery_bytes"])
    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
    def test_new_bounds_must_be_positive_ints(self, field: str, bad: Any) -> None:
        with pytest.raises(ValueError, match=field):
            _source(**{field: bad})

    def test_new_bounds_pass_through_every_constructor(self) -> None:
        stdio = MCPSource.stdio("l", ["echo"], max_discovery_tools=5)
        assert stdio.max_discovery_tools == 5
        server = MCPServer(
            "r", url="https://mcp.example.com/mcp", max_discovery_tools=7, max_discovery_bytes=8
        )
        assert (server.source.max_discovery_tools, server.source.max_discovery_bytes) == (7, 8)

    def test_defaults(self) -> None:
        source = _source()
        assert source.max_discovery_pages == 100
        assert source.discovery_timeout == 60.0

    @pytest.mark.parametrize("pages", [0, -1, True, 1.5, "10"])
    def test_page_cap_must_be_a_positive_int(self, pages: Any) -> None:
        with pytest.raises(ValueError, match="max_discovery_pages"):
            _source(max_discovery_pages=pages)

    @pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
    def test_deadline_must_be_a_positive_finite_number(self, seconds: float) -> None:
        with pytest.raises(ValueError, match="discovery_timeout"):
            _source(discovery_timeout=seconds)

    def test_stdio_source_takes_the_same_overrides(self) -> None:
        source = MCPSource.stdio("local", ["echo"], max_discovery_pages=3, discovery_timeout=2)
        assert (source.max_discovery_pages, source.discovery_timeout) == (3, 2)

    def test_server_facade_passes_the_overrides_through(self) -> None:
        server = MCPServer(
            "remote", url="https://mcp.example.com/mcp", max_discovery_pages=7, discovery_timeout=9
        )
        assert (server.source.max_discovery_pages, server.source.discovery_timeout) == (7, 9)

    def test_connection_configuration_difference_is_visible(self) -> None:
        assert _source(max_discovery_pages=5) != _source()


@asynccontextmanager
async def _serve_pages(*, delay: float = 0.0) -> AsyncIterator[tuple[str, dict[str, int]]]:
    """A local streamable-HTTP server whose ``tools/list`` repeats one cursor forever."""
    stats = {"pages": 0}

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next(
                (
                    int(line.split(":", 1)[1])
                    for line in head.split("\r\n")
                    if line.lower().startswith("content-length:")
                ),
                0,
            )
            body = await reader.readexactly(length) if length else b"{}"
            req = json.loads(body)
            method = req.get("method")
            if method == "tools/list":
                stats["pages"] += 1
                await asyncio.sleep(delay)
                result: dict[str, Any] = {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "local fake",
                            "inputSchema": {"type": "object"},
                        }
                    ],
                    "nextCursor": "repeat",
                }
            elif method == "initialize":
                result = {
                    "protocolVersion": req.get("params", {}).get("protocolVersion", "2025-03-26"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "local", "version": "1"},
                }
            else:
                result = {}
            encoded = json.dumps({"jsonrpc": "2.0", "id": req.get("id", 1), "result": result})
            payload = encoded.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/mcp", stats
    finally:
        server.close()
        await server.wait_closed()


@asynccontextmanager
async def _serve_huge_page(*, tool_count: int) -> AsyncIterator[tuple[str, dict[str, int]]]:
    """A local server whose one ``tools/list`` page is ``tool_count`` fat tools, no cursor."""
    stats = {"requests": 0, "sent": 0}

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next(
                (
                    int(line.split(":", 1)[1])
                    for line in head.split("\r\n")
                    if line.lower().startswith("content-length:")
                ),
                0,
            )
            req = json.loads(await reader.readexactly(length) if length else b"{}")
            method = req.get("method")
            if method == "tools/list":
                stats["requests"] += 1
                fat = "x" * 200
                result: dict[str, Any] = {
                    "tools": [
                        {"name": f"t{i}", "description": fat, "inputSchema": {"type": "object"}}
                        for i in range(tool_count)
                    ]
                }
            elif method == "initialize":
                result = {
                    "protocolVersion": req.get("params", {}).get("protocolVersion", "2025-03-26"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "local", "version": "1"},
                }
            else:
                result = {}
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": req.get("id", 1), "result": result}
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
            )
            await writer.drain()
            stats["sent"] += len(payload)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/mcp", stats
    finally:
        server.close()
        await server.wait_closed()


class TestResponseBytesOnTheRealTransport:
    async def test_an_oversized_page_stops_at_the_byte_cap(self) -> None:
        async with _serve_huge_page(tool_count=5000) as (url, stats):
            source = MCPSource.http(
                "fake", url, max_discovery_bytes=100_000, max_discovery_tools=1_000_000
            )
            adapter = MCPClientAdapter(source)
            await adapter.connect()
            try:
                with pytest.raises(MCPDiscoveryLimitError) as caught:
                    await asyncio.wait_for(adapter.list_tools(), timeout=10)
            finally:
                await adapter.close()
            assert caught.value.reason == "byte_cap"
            assert error_code_for(caught.value) == "discovery_limit"
            assert stats["requests"] == 1

    async def test_a_page_within_the_cap_is_read(self) -> None:
        async with _serve_huge_page(tool_count=20) as (url, _):
            adapter = MCPClientAdapter(MCPSource.http("fake", url))
            await adapter.connect()
            try:
                tools = await asyncio.wait_for(adapter.list_tools(), timeout=10)
            finally:
                await adapter.close()
            assert len(tools) == 20

    async def test_a_too_big_catalog_stops_at_the_tool_cap(self) -> None:
        async with _serve_huge_page(tool_count=50) as (url, _):
            adapter = MCPClientAdapter(MCPSource.http("fake", url, max_discovery_tools=10))
            await adapter.connect()
            try:
                with pytest.raises(MCPDiscoveryLimitError) as caught:
                    await asyncio.wait_for(adapter.list_tools(), timeout=10)
            finally:
                await adapter.close()
            assert caught.value.reason == "tool_cap"


class TestAgainstARealServer:
    async def test_repeating_cursor_stops_on_its_own(self) -> None:
        async with _serve_pages() as (url, stats):
            adapter = MCPClientAdapter(MCPSource.http("fake", url, call_timeout=0.2))
            await adapter.connect()
            try:
                with pytest.raises(MCPDiscoveryLimitError) as caught:
                    await asyncio.wait_for(adapter.list_tools(), timeout=5)
            finally:
                await adapter.close()
            assert caught.value.reason == "repeated_cursor"
            assert stats["pages"] <= 3


_STDIO_SERVER = """
from mcp.server.mcpserver import MCPServer
server = MCPServer(name="small")
@server.tool(name="one", description="a" * 2000)
def one() -> str:
    return "ok"
@server.tool(name="two", description="b")
def two() -> str:
    return "ok"
server.run(transport="stdio")
"""


class TestStdioSources:
    def _source(self, tmp_path: Any, **kw: Any) -> MCPSource:
        import sys

        script = tmp_path / "small_server.py"
        script.write_text(_STDIO_SERVER)
        return MCPSource.stdio("local", [sys.executable, str(script)], **kw)

    async def test_discovery_works_and_keeps_the_tool_cap(self, tmp_path: Any) -> None:
        adapter = MCPClientAdapter(self._source(tmp_path))
        await adapter.connect()
        try:
            tools = await asyncio.wait_for(adapter.list_tools(), timeout=30)
        finally:
            await adapter.close()
        assert sorted(t.name for t in tools) == ["one", "two"]
        capped = MCPClientAdapter(self._source(tmp_path, max_discovery_tools=1))
        await capped.connect()
        try:
            with pytest.raises(MCPDiscoveryLimitError) as caught:
                await asyncio.wait_for(capped.list_tools(), timeout=30)
        finally:
            await capped.close()
        assert caught.value.reason == "tool_cap"

    def test_a_byte_cap_on_a_stdio_source_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="only valid for HTTP sources"):
            MCPSource(name="l", command=("echo",), max_discovery_bytes=1000)
        with pytest.raises(ValueError, match="only valid for HTTP sources"):
            MCPServer("l", command=["echo"], max_discovery_bytes=1000)

    def test_the_stdio_factory_has_no_byte_cap_option(self) -> None:
        with pytest.raises(TypeError):
            MCPSource.stdio("l", ["echo"], max_discovery_bytes=1000)  # type: ignore[call-arg]

    def test_the_default_byte_cap_is_accepted_on_stdio(self) -> None:
        source = MCPSource.stdio("l", ["echo"])
        assert source.max_discovery_bytes == 16 * 1024 * 1024
        assert MCPSource(name="l", command=("echo",)).max_discovery_bytes == 16 * 1024 * 1024

    def test_the_other_caps_apply_to_stdio_sources(self) -> None:
        source = MCPSource.stdio(
            "l", ["echo"], max_discovery_pages=3, discovery_timeout=2, max_discovery_tools=5
        )
        assert (source.max_discovery_pages, source.discovery_timeout) == (3, 2)
        assert source.max_discovery_tools == 5


class TestTheSdkIsNotPatched:
    def test_importing_dendrux_leaves_the_sdk_stdio_module_untouched(self) -> None:
        import subprocess
        import sys

        code = (
            "import mcp.client.stdio as m\n"
            "before = {k: id(v) for k, v in vars(m).items()}\n"
            "import dendrux, dendrux.mcp, dendrux.mcp._client\n"
            "after = {k: id(v) for k, v in vars(m).items()}\n"
            "assert before == after, sorted(k for k in after if before.get(k) != after[k])\n"
            "from anyio.streams.text import TextReceiveStream\n"
            "assert m.TextReceiveStream is TextReceiveStream\n"
        )
        done = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
