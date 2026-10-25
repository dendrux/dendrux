"""Wire and in-process submissions share database-safe scalar limits."""

import pytest

from dendrux.types import ToolResult


@pytest.mark.parametrize(
    "patch",
    [
        {"duration_ms": 2**40},
        {"payload": "x\x00"},
        {"payload": '"x\\u0000"'},
        {"error": "x\x00"},
        {"name": "x\x00"},
        {"call_id": "c" * 256},
        {"payload": '"x\\ud800"'},
    ],
)
def test_wire_rejects_unpersistable_result_scalars(patch):
    with pytest.raises(ValueError):
        ToolResult.from_dict({**dict(name="read", call_id="c", payload='"ok"'), **patch})
