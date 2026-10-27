"""Decoder syntax and resource failures produce the same safe tool rejection."""

import sys

import pytest

from dendrux.llm._helpers import build_tool_call, parse_tool_json


@pytest.mark.parametrize("kind", ["syntax", "integer_limit", "recursion_limit"])
def test_decoder_failures_are_fixed_rejections(kind):
    raw = {
        "syntax": '{"sentinel":',
        "integer_limit": '{"sentinel":' + "1" * max(5000, sys.get_int_max_str_digits() + 1) + "}",
        "recursion_limit": '{"sentinel":'
        + "[" * max(10000, sys.getrecursionlimit() + 100)
        + "0"
        + "]" * max(10000, sys.getrecursionlimit() + 100)
        + "}",
    }[kind]
    params, error = parse_tool_json(raw)
    assert params == {}
    assert error == "arguments must be valid JSON"
    call = build_tool_call("f", raw, provider_tool_call_id="c", provider="fixture", model="m")
    assert call.params == {}
    assert call.parse_error == error
    assert "sentinel" not in error
