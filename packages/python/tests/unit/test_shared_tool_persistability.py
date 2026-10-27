"""Tool preflight uses the same finite JSON and resource bounds as persistence."""

import json

import pytest

from dendrux.tools._validate import check_tool_call
from dendrux.types import ToolCall, ToolDef


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize(
    "kind", ["nul", "surrogate", "nan", "infinity", "depth", "nodes", "key", "opaque"]
)
def test_tool_preflight_rejects_every_unpersistable_shape_without_echo(known, kind):
    value = {
        "nul": "PRIVATE_PARAMS\x00",
        "surrogate": "PRIVATE_PARAMS\ud800",
        "nan": float("nan"),
        "infinity": float("inf"),
        "depth": None,
        "nodes": ["PRIVATE_PARAMS"] * (1048576 // 2 + 1),
        "key": {17: "PRIVATE_PARAMS"},
        "opaque": object(),
    }[kind]
    if kind == "depth":
        value = "PRIVATE_PARAMS"
        for _ in range(40):
            value = [value]
    call = ToolCall("lookup", {"private": value})
    definition = ToolDef("lookup", "", {"type": "object"}) if known else None
    result = check_tool_call(call, definition)
    assert result is not None, "tool preflight must reject what persistence rejects"
    assert result.success is False and result.meta["code"] == "invalid_params"
    assert json.loads(result.payload)["code"] == "invalid_params"
    assert "PRIVATE_PARAMS" not in json.dumps([result.payload, result.error, result.meta])


@pytest.mark.parametrize("known", [False, True])
def test_shared_tool_boundary_retains_ordinary_finite_unicode_json(known):
    call = ToolCall("lookup", {"text": "café \U0001f600", "nested": [None, True, 3, 1.25]})
    definition = ToolDef("lookup", "", {"type": "object"}) if known else None
    assert check_tool_call(call, definition) is None
