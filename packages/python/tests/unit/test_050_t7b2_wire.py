"""Explicit browser DTO and bounded client-result parsing contracts."""

import json

import pytest

from dendrux.errors import ProviderAuthError
from dendrux.store import PendingPause, PendingToolCall
from dendrux.types import ErrorInfo, Limits, RunResult, RunStatus, ToolResult


def test_run_wire_allowlist_and_pii_redaction():
    result = RunResult(
        run_id="wire",
        status=RunStatus.ERROR,
        answer="Hello private@example.test",
        output={"payload": "PAYLOAD_SENTINEL"},
        error="VENDOR_SENTINEL",
        error_info=ErrorInfo.from_error(
            ProviderAuthError(provider="test", message="VENDOR_SENTINEL")
        ),
        meta={"prompt": "PROMPT_SENTINEL", "pii_mapping": {"<<EMAIL_1>>": "private@example.test"}},
        commentary=["TRACE_SENTINEL"],
    )
    wire = result.to_dict()
    assert set(wire) == {
        "wire_version",
        "answer_omitted",
        "run_id",
        "status",
        "pending",
        "answer",
        "output",
        "error_info",
    }
    assert type(wire["wire_version"]) is int and wire["wire_version"] == 1
    assert wire["answer"] == result.answer
    assert wire["output"] is None
    text = json.dumps(wire)
    for secret in (
        "PROMPT_SENTINEL",
        "TRACE_SENTINEL",
        "VENDOR_SENTINEL",
        "PAYLOAD_SENTINEL",
    ):
        assert secret not in text
    assert json.loads(text) == wire


def test_wire_pending_is_current_and_does_not_alias_mutable_params():
    pending = PendingPause(
        run_id="wire",
        status="waiting_approval",
        kind="approval",
        pause_id="P",
        tool_calls=[PendingToolCall("C", "refund", "server", {"email": "<<EMAIL_1>>"})],
    )
    result = RunResult("wire", RunStatus.WAITING_APPROVAL, pending=pending)
    wire = result.to_dict()
    assert wire["pending"] == pending.to_dict()
    wire["pending"]["pending_tool_calls"][0]["params"]["email"] = "changed"
    assert pending.tool_calls[0].params == {"email": "<<EMAIL_1>>"}
    result.status = RunStatus.SUCCESS
    assert result.to_dict()["pending"] is None


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"name": "t", "call_id": "c", "payload": 3},
        {"name": "t", "call_id": "c", "payload": "NaN"},
        {"name": "t", "call_id": "c", "payload": "{}", "success": 1},
        {"name": "", "call_id": "c", "payload": "{}"},
        {"name": "t", "call_id": "c", "payload": "{}", "meta": {"code": "trusted"}},
        {"name": "t", "call_id": "c", "payload": "{}", "duration_ms": -1},
    ],
)
def test_tool_result_parser_validates_untrusted_fields(data):
    with pytest.raises((ValueError, TypeError)):
        ToolResult.from_dict(data)


def test_tool_result_parser_caps_and_bounded_batch():
    data = {"name": "read", "call_id": "C", "payload": json.dumps({"large": "x" * 4000})}
    result = ToolResult.from_dict(data, limits=Limits(max_tool_result_bytes=256))
    assert not result.success and result.meta["code"] == "size_limit"
    assert "x" * 300 not in result.payload
    small = {**data, "payload": '{"ok": true}'}
    assert ToolResult.list_from_dict([small]) == [ToolResult("read", "C", '{"ok": true}')]
    with pytest.raises(ValueError):
        ToolResult.list_from_dict([small, small])
    with pytest.raises(ValueError):
        ToolResult.list_from_dict([{**small, "call_id": str(i)} for i in range(3)], max_results=2)


def test_wire_rejects_unserializable_integer_in_client_params():
    pending = PendingPause(
        run_id="wire",
        status="waiting_client_tool",
        kind="client_tool",
        pause_id="P",
        tool_calls=[PendingToolCall("C", "read", "client", {"large": 10**10000})],
    )
    result = RunResult("wire", RunStatus.WAITING_CLIENT_TOOL, pending=pending)
    # Application-created params can bypass the JSON decoder's integer bound.
    # DTO construction must fail closed rather than returning unserializable JSON.
    wire = result.to_dict()
    assert wire["pending"]["pause_id"] == "P"
    assert wire["pending"]["pending_tool_calls"][0]["params"] is None
    assert wire["pending"]["pending_tool_calls"][0]["args_hidden"] is True
    json.dumps(wire)


def test_image_bearing_and_opaque_outputs_never_run_custom_serializers():
    from dendrux.media import Image
    from dendrux.types import ImageRef

    class PrivateOutput:
        def __str__(self):
            raise AssertionError("wire must not inspect private application objects")

        def model_dump(self):
            raise AssertionError("wire must not run app serializers")

    for output in (
        PrivateOutput(),
        ToolResult(
            "read",
            "C",
            '"PAYLOAD_SENTINEL"',
            images=(Image.from_bytes(b"IMAGE_SENTINEL", media_type="image/png"),),
        ),
        ImageRef("a" * 64, "image/png", 10, 1, 1),
    ):
        wire = RunResult("R", RunStatus.SUCCESS, answer="ok", output=output).to_dict()
        assert wire["output"] is None
        assert "SENTINEL" not in json.dumps(wire)


def test_tool_result_parser_image_refs_and_aggregate_bound():
    from dendrux.types import ImageRef

    ref = ImageRef("a" * 64, "image/png", 10, 1, 1)
    data = dict(name="read", call_id="C", payload='"safe"', images=[ref.to_dict()])
    assert ToolResult.from_dict(data).images == (ref,)
    with pytest.raises(ValueError):
        ToolResult.from_dict({**data, "images": [{"bytes": "RAW_IMAGE_SENTINEL"}]})
    with pytest.raises(ValueError):
        ToolResult.list_from_dict(
            [data, {**data, "call_id": "C2"}], limits=Limits(max_input_bytes=10)
        )


@pytest.mark.parametrize("payload", ["1e999", '{"nested":[-1e999]}'])
def test_tool_result_parser_rejects_finite_syntax_that_overflows(payload):
    with pytest.raises(ValueError, match="finite"):
        ToolResult.from_dict(dict(name="read", call_id="C", payload=payload))


def test_wire_mapping_bound_is_checked_before_allocating_a_regex(monkeypatch):
    import dendrux._wire as wire

    def must_not_escape_oversize(value):
        raise AssertionError("oversized PII must be refused before building a regexp")

    result = RunResult(
        "R",
        RunStatus.WAITING_APPROVAL,
        pending=PendingPause("R", "waiting_approval", "approval", "P"),
        meta={"pii_mapping": {"<<DATA_1>>": "x" * 65537}},
    )
    monkeypatch.setattr(wire.re, "escape", must_not_escape_oversize)
    assert result.to_dict()["pending"]["pause_id"] == "P"


@pytest.mark.parametrize("bound", ["depth", "nodes", "chars", "utf8", "unicode"])
def test_wire_display_limits_fail_closed(bound):
    value = "safe"
    if bound == "depth":
        for _ in range(40):
            value = [value]
    elif bound == "nodes":
        value = [None] * 10001
    elif bound == "chars":
        value = "x" * 1048577
    elif bound == "utf8":
        value = "é" * 600000
    else:
        value = "\ud800"
    pending = PendingPause(
        "R",
        "waiting_client_tool",
        "client_tool",
        "P",
        [PendingToolCall("C", "read", "client", {"value": value})],
    )
    display = RunResult("R", RunStatus.WAITING_CLIENT_TOOL, pending=pending).to_dict()["pending"]
    assert display["pause_id"] == "P"
    assert display["pending_tool_calls"] == [
        dict(id="C", name="read", target="client", params=None, approval=None, args_hidden=True)
    ]


def test_wire_only_copies_finite_json_values():
    pending = PendingPause(
        "R",
        "waiting_client_tool",
        "client_tool",
        "P",
        [
            PendingToolCall(
                "C",
                "read",
                "client",
                {
                    "finite": 1.5,
                    "infinite": float("inf"),
                    "integer": 12,
                    "bool": True,
                    "opaque": object(),
                    "tuple": (None, "ok"),
                    3: "PRIVATE_KEY_SENTINEL",
                },
            )
        ],
    )
    wire = RunResult("R", RunStatus.WAITING_CLIENT_TOOL, pending=pending).to_dict()
    params = wire["pending"]["pending_tool_calls"][0]["params"]
    assert params == dict(
        finite=1.5, infinite=None, integer=12, bool=True, opaque=None, tuple=[None, "ok"]
    )
    json.dumps(wire, allow_nan=False)


@pytest.mark.parametrize(
    "mapping",
    [
        {str(i): str(i) for i in range(1001)},
        {"<<DATA_1>>": "?" * 32768, "<<DATA_2>>": "!" * 32768},
    ],
)
def test_wire_aggregate_mapping_budget_fails_closed(mapping):
    display = RunResult(
        "R",
        RunStatus.WAITING_APPROVAL,
        pending=PendingPause("R", "waiting_approval", "approval", "P"),
        meta={"pii_mapping": mapping},
    ).to_dict()["pending"]
    assert display["pause_id"] == "P"
    assert display["pending_tool_calls"] == []
