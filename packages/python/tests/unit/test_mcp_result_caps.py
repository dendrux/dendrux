"""Transport and Agent result caps remain independent, with documented defaults."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dendrux import types
from dendrux.mcp import MCPServer, MCPSource
from dendrux.mcp._errors import MCPResultTooLargeError
from dendrux.mcp._http import wire_cap
from dendrux.mcp._result import normalize_mcp_result
from dendrux.mcp._server import _normalize_mcp_result
from dendrux.mcp._source import DEFAULT_MAX_RESULT_BYTES
from dendrux.tools._limits import cap_value_payload, result_cap
from dendrux.types import Limits


def test_result_caps_have_separate_named_defaults():
    assert DEFAULT_MAX_RESULT_BYTES == 1_000_000
    assert hasattr(types, "_DEFAULT_MAX_TOOL_RESULT_BYTES")
    assert types._DEFAULT_MAX_TOOL_RESULT_BYTES == Limits().max_tool_result_bytes == 1_048_576
    for consumer in (
        MCPServer,
        MCPSource.http,
        MCPSource.stdio,
        normalize_mcp_result,
        _normalize_mcp_result,
    ):
        assert (
            inspect.signature(consumer).parameters["max_result_bytes"].default
            == DEFAULT_MAX_RESULT_BYTES
        )
    assert MCPSource.__dataclass_fields__["max_result_bytes"].default == DEFAULT_MAX_RESULT_BYTES


@pytest.mark.parametrize("size", [1_000_000, 1_000_001, 1_048_576])
def test_payload_between_caps_needs_transport_override(size):
    value = "x" * (size - 2)
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text=value)])
    if size <= DEFAULT_MAX_RESULT_BYTES:
        assert normalize_mcp_result(result) == value
    else:
        with pytest.raises(MCPResultTooLargeError):
            normalize_mcp_result(result)
    source = MCPSource.http(
        "remote", "https://example.com/mcp", max_result_bytes=Limits().max_tool_result_bytes
    )
    assert normalize_mcp_result(result, max_result_bytes=source.max_result_bytes) == value
    payload = json.dumps(value)
    assert cap_value_payload(value, payload, Limits().max_tool_result_bytes) == (payload, None)


def test_http_read_cap_uses_smaller_knob_and_keeps_framing_allowance():
    with result_cap(Limits().max_tool_result_bytes):
        assert wire_cap(DEFAULT_MAX_RESULT_BYTES) == 4_000_000
    with result_cap(500_000):
        assert wire_cap(DEFAULT_MAX_RESULT_BYTES) == 2_000_000
    with result_cap(None, image_bytes=100):
        assert wire_cap(DEFAULT_MAX_RESULT_BYTES) == 4_000_100


def test_docs_explain_both_result_caps_together():
    page = Path(__file__).resolve().parents[4] / "docs/architecture/mcp.mdx"
    section = (
        page.read_text()
        .split("## Calls, errors, and result limits", 1)[1]
        .split("**Session loss.**", 1)[0]
    )
    for text in (
        "1,000,000",
        "1,048,576",
        "max_result_bytes",
        "Limits.max_tool_result_bytes",
        "smaller",
        "4,000,000",
    ):
        assert text in section
