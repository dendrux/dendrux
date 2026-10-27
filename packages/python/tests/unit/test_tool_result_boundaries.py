"""Tool validation: value-free errors and typed rejection of malformed schemas."""

from __future__ import annotations

import enum
import json
import logging
import uuid  # noqa: TC003
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import pytest
from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, Json, field_validator

from dendrux.tool import get_tool_def, tool
from dendrux.tools._validate import check_tool_call, prepare_tool_call, validate_tool_params
from dendrux.types import ToolCall, ToolDef, ToolResult

PRIVATE = "astra.person@example.com"
SENTINEL = "zq7Sentinel"


def _prepare(fn: Any, params: dict[str, Any]) -> ToolResult:
    result = prepare_tool_call(ToolCall("t", params), get_tool_def(fn), fn)
    assert isinstance(result, ToolResult)
    assert result.success is False
    return result


def _assert_value_free(result: ToolResult, *needles: str) -> None:
    blob = json.dumps([result.payload, result.error, result.meta])
    for needle in needles:
        assert needle not in blob, f"{needle!r} escaped through the validation error"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int


class Cat(BaseModel):
    kind: Literal["cat"]


class Dog(BaseModel):
    kind: Literal["dog"]


class Pet(BaseModel):
    animal: Annotated[Cat | Dog, Field(discriminator="kind")]


class WithUuid(BaseModel):
    ident: uuid.UUID


class WithBase64(BaseModel):
    blob: Base64Bytes


class WithJson(BaseModel):
    blob: Json[int]


class IntKeys(BaseModel):
    values: dict[int, int]


class Row(TypedDict):
    n: int


class Color(enum.Enum):
    RED = "red"


@dataclass
class Item:
    qty: int


class Checked(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def check(cls, value: str) -> str:
        raise ValueError(f"bad value {value}")


class TestInputKeysNeverEchoed:
    def test_dict_key_in_the_location(self) -> None:
        @tool()
        def ingest(values: dict[str, int]) -> str:
            """Ingest numeric values."""
            return "ok"

        result = _prepare(ingest, {"values": {PRIVATE: "not-an-int"}})
        _assert_value_free(result, PRIVATE, "astra")
        assert "values.<key>" in (result.error or "")

    def test_dict_key_that_fails_key_validation(self) -> None:
        @tool()
        def ingest(model: IntKeys) -> str:
            """Ingest numeric values."""
            return "ok"

        result = _prepare(ingest, {"model": {"values": {PRIVATE: 1}}})
        _assert_value_free(result, PRIVATE)

    def test_nested_dict_keys_inside_a_list(self) -> None:
        @tool()
        def ingest(rows: list[dict[str, int]]) -> str:
            """Ingest rows."""
            return "ok"

        result = _prepare(ingest, {"rows": [{"ok": 1}, {PRIVATE: "x"}]})
        _assert_value_free(result, PRIVATE)
        assert "rows[1].<key>" in (result.error or "")

    def test_extra_key_in_a_forbidding_model(self) -> None:
        @tool()
        def make(person: Strict) -> str:
            """Make a person."""
            return "ok"

        result = _prepare(make, {"person": {"age": 3, PRIVATE: 1}})
        _assert_value_free(result, PRIVATE)

    def test_extra_key_in_a_typed_dict(self) -> None:
        @tool()
        def make(row: Row) -> str:
            """Make a row."""
            return "ok"

        result = _prepare(make, {"row": {"n": "x", PRIVATE: 1}})
        _assert_value_free(result, PRIVATE)
        assert "row.n" in (result.error or "")

    def test_declared_field_names_are_kept(self) -> None:
        @tool()
        def make(person: Strict, items: list[Item]) -> str:
            """Make."""
            return "ok"

        result = _prepare(make, {"person": {"age": "old"}, "items": [{"qty": "many"}]})
        assert "person.age" in (result.error or "")
        assert "items[0].qty" in (result.error or "")

    @pytest.mark.parametrize(
        "name", [PRIVATE, "alice_smith", "sk_live_a1b2c3", "Alice Smith", "555-123-4567", 'r4<|"|>']
    )
    def test_unknown_top_level_names_are_never_echoed(self, name: str) -> None:
        @tool()
        def look(city: str) -> str:
            """Look."""
            return "ok"

        result = _prepare(look, {"city": "Oslo", name: "x"})
        _assert_value_free(result, name)
        assert "1 unknown argument" in (result.error or "")
        assert "Valid arguments: city" in (result.error or "")

    def test_unknown_names_in_the_json_schema_check_report_a_count(self) -> None:
        td = ToolDef(
            name="t",
            description="",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        problems = validate_tool_params(td, {"Alice Smith": 1, "bogus": 2, "city": "x"})
        assert problems == ["t has 2 unknown arguments"]

    def test_many_unknown_names_are_one_short_problem(self) -> None:
        td = ToolDef(
            name="t",
            description="",
            parameters={"type": "object", "properties": {"a": {}}},
        )
        assert validate_tool_params(td, {f"k{i}": i for i in range(50)}) == [
            "t has 50 unknown arguments"
        ]

    def test_exception_text_carries_the_same_path(self) -> None:
        from dendrux.tools._coerce import ArgumentCoercionError, get_args_spec, normalize_arguments

        @tool()
        def ingest(values: dict[str, int]) -> str:
            """Ingest numeric values."""
            return "ok"

        spec = get_args_spec(ingest)
        assert spec is not None
        with pytest.raises(ArgumentCoercionError) as caught:
            normalize_arguments(spec, {"values": {PRIVATE: "x"}})
        assert PRIVATE not in str(caught.value)


class TestMessageTextNeverEchoesInput:
    """Pydantic message templates that quote the input are replaced by fixed text."""

    def test_wrong_discriminator_tag(self) -> None:
        @tool()
        def pet(model: Pet) -> str:
            """Pet."""
            return "ok"

        result = _prepare(pet, {"model": {"animal": {"kind": SENTINEL}}})
        _assert_value_free(result, SENTINEL)

    def test_bad_uuid(self) -> None:
        @tool()
        def look(model: WithUuid) -> str:
            """Look."""
            return "ok"

        result = _prepare(
            look, {"model": {"ident": f"1234567{SENTINEL}-1234-1234-1234-123456789012"}}
        )
        _assert_value_free(result, SENTINEL, "found `")

    def test_bad_base64(self) -> None:
        @tool()
        def look(model: WithBase64) -> str:
            """Look."""
            return "ok"

        result = _prepare(look, {"model": {"blob": SENTINEL + "!!"}})
        _assert_value_free(result, SENTINEL)

    def test_bad_embedded_json(self) -> None:
        @tool()
        def look(model: WithJson) -> str:
            """Look."""
            return "ok"

        result = _prepare(look, {"model": {"blob": '{"' + SENTINEL + '": x}'}})
        _assert_value_free(result, SENTINEL)

    def test_custom_validator_text_is_not_shown(self) -> None:
        @tool()
        def look(item: Checked) -> str:
            """Look."""
            return "ok"

        result = _prepare(look, {"item": {"name": SENTINEL}})
        _assert_value_free(result, SENTINEL)

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [
            (int, SENTINEL),
            (float, SENTINEL),
            (bool, SENTINEL),
            (Color, SENTINEL),
            (Literal["a", "b"], SENTINEL),
            (list[int], SENTINEL),
        ],
    )
    def test_common_types_stay_value_free(self, annotation: Any, value: Any) -> None:
        def fn(x: annotation) -> str:  # type: ignore[valid-type]
            """Fn."""
            return "ok"

        fn.__annotations__["x"] = annotation
        decorated = tool()(fn)
        result = _prepare(decorated, {"x": value})
        _assert_value_free(result, SENTINEL)


def _td(parameters: Any) -> ToolDef:
    return ToolDef(name="remote", description="", parameters=parameters)


class TestMalformedSchemaIsTypedRejection:
    def test_required_entry_that_is_not_a_string(self) -> None:
        td = _td({"type": "object", "properties": {"x": {"type": "string"}}, "required": [{}]})
        result = check_tool_call(ToolCall("remote", {"x": "ok"}), td)
        assert result is not None
        assert result.success is False
        assert result.meta == {"code": "invalid_tool_schema"}
        assert json.loads(result.payload)["code"] == "invalid_tool_schema"
        assert "remote" in (result.error or "")

    @pytest.mark.parametrize(
        "parameters",
        [
            {"type": "object", "required": [1, 2]},
            {"type": "object", "required": [["a"]]},
            {"type": "object", "required": [None]},
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": [{}, "x"]},
            {"type": "object", "properties": {"x": {"type": [{}]}}},
            {"type": "object", "properties": {"x": {"type": {"a": 1}}}},
            {"type": "object", "properties": {"x": {"enum": "abc"}}},
            {"type": "object", "properties": {"x": {"enum": [{}, []]}}},
            {"type": "object", "properties": {"x": []}},
            {"type": "object", "properties": {"x": None}},
            {"type": "object", "properties": []},
            {"type": "object", "properties": "nope"},
            {"type": "object", "required": "x"},
            {"type": "object", "required": None},
            {"type": "object", "additionalProperties": []},
            {"type": "object", "patternProperties": 3},
            {"type": []},
            {"type": 5},
            {"properties": {"x": {"type": ["string", None]}}},
            {"properties": {"x": True}},
            {"properties": {"x": {"anyOf": 3}}},
            {"properties": {"x": {"$ref": []}}},
            {},
            {"properties": {}},
        ],
    )
    def test_no_malformed_shape_crashes(self, parameters: Any) -> None:
        td = _td(parameters)
        for params in ({}, {"x": "ok"}, {"x": 1, "y": [1]}, {"x": None}, {"x": {"a": []}}):
            result = check_tool_call(ToolCall("remote", params), td)
            assert result is None or isinstance(result, ToolResult)
            problems = validate_tool_params(td, params)
            assert isinstance(problems, list)

    def test_non_dict_schema_is_ignored(self) -> None:
        td = ToolDef(name="remote", description="", parameters={})
        object.__setattr__(td, "parameters", "not a dict")
        assert check_tool_call(ToolCall("remote", {"x": 1}), td) is None

    def test_rejection_is_logged_with_a_code_and_no_schema_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        td = _td({"type": "object", "required": [{"secret": SENTINEL}]})
        with caplog.at_level(logging.WARNING):
            check_tool_call(ToolCall("remote", {}), td)
        assert "invalid_tool_schema" in caplog.text
        assert SENTINEL not in caplog.text

    def test_well_formed_schemas_are_unchanged(self) -> None:
        td = _td({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})
        assert check_tool_call(ToolCall("remote", {"x": "ok"}), td) is None
        result = check_tool_call(ToolCall("remote", {}), td)
        assert result is not None
        assert json.loads(result.payload)["code"] == "invalid_params"


class TestMalformedSchemaAtDiscovery:
    def _tools(self) -> list[Any]:
        from mcp.types import Tool

        return [
            Tool(name="good", description="", input_schema={"type": "object"}),
            Tool(
                name="bad",
                description="",
                input_schema={"type": "object", "required": [{}]},
            ),
        ]

    def _build(self, *, strict: bool = False) -> Any:
        from types import SimpleNamespace

        from dendrux.mcp._server import build_mcp_tool_defs

        source = SimpleNamespace(name="srv", transport="stdio", call_timeout=30.0)
        return build_mcp_tool_defs(
            source=source,  # type: ignore[arg-type]  # minimal stand-in
            namespace="srv",
            tools=self._tools(),
            connection_info=None,
            strict=strict,
        )

    def test_tool_with_a_malformed_schema_is_skipped_and_reported(self) -> None:
        from dendrux.mcp._server import MCPSkippedTool

        defs, skipped = self._build()
        assert [d.name for d in defs] == ["srv__good"]
        assert skipped == [
            MCPSkippedTool(source_name="srv", namespace="srv", tool="bad", reason="invalid_schema")
        ]

    def test_strict_view_raises_a_value_error_without_schema_text(self) -> None:
        with pytest.raises(ValueError, match="malformed") as caught:
            self._build(strict=True)
        assert "bad" in str(caught.value)


class TestRequiredPresenceIsTestedSeparatelyFromValue:
    @pytest.mark.parametrize("required", [None, "x", 3, {"x": 1}, [None], ["x", 1]])
    def test_a_present_but_malformed_required_is_a_typed_rejection(self, required: Any) -> None:
        td = _td({"type": "object", "properties": {"x": {"type": "string"}}, "required": required})
        result = check_tool_call(ToolCall("remote", {}), td)
        assert result is not None
        assert result.meta == {"code": "invalid_tool_schema"}
        assert validate_tool_params(td, {}) == ["remote has a malformed argument schema"]

    def test_an_absent_required_is_fine(self) -> None:
        td = _td({"type": "object", "properties": {"x": {"type": "string"}}})
        assert check_tool_call(ToolCall("remote", {}), td) is None


class TestMalformedSchemaViews:
    @pytest.mark.parametrize("bad", [None, [{}], "x"])
    def test_both_views_reject_a_malformed_required(self, bad: Any) -> None:
        from types import SimpleNamespace

        from mcp.types import Tool

        from dendrux.mcp import MCPInvalidToolSchemaError, error_code_for
        from dendrux.mcp._server import build_mcp_tool_defs

        source = SimpleNamespace(name="srv", transport="stdio", call_timeout=30.0)
        tool_obj = Tool(
            name="bad",
            description="",
            input_schema={"type": "object", "properties": {"x": {}}, "required": bad},
        )
        defs, skipped = build_mcp_tool_defs(
            source=source,  # type: ignore[arg-type]
            namespace="srv",
            tools=[tool_obj],
            connection_info=None,
        )
        assert defs == []
        assert [s.reason for s in skipped] == ["invalid_schema"]
        with pytest.raises(MCPInvalidToolSchemaError) as caught:
            build_mcp_tool_defs(
                source=source,  # type: ignore[arg-type]
                namespace="srv",
                tools=[tool_obj],
                connection_info=None,
                strict=True,
            )
        assert isinstance(caught.value, ValueError)
        assert error_code_for(caught.value) == "invalid_tool_schema"

    def test_a_bad_name_keeps_its_own_code(self) -> None:
        from dendrux.mcp import error_code_for

        assert error_code_for(ValueError("bad rename")) == "invalid_tool_names"


NUL = "\x00"
LONE_SURROGATE = "\ud800"
CALLS: list[dict[str, Any]] = []


@tool()
def record(note: str, extra: dict[str, Any] | None = None) -> str:
    """Record."""
    CALLS.append({"note": note})
    return "ok"


def _assert_no_content_echo(result: ToolResult) -> None:
    blob = json.dumps([result.payload, result.error, result.meta]).lower()
    for needle in ("\\u0000", "\\ud800", "ud800", "u0000", "surrogate:"):
        assert needle not in blob.replace("u+0000", "").replace("u+d800", "")


class TestUnpersistableCharactersAreRejected:
    @pytest.mark.parametrize("bad", [NUL, LONE_SURROGATE, f"a{NUL}b", f"x{LONE_SURROGATE}y"])
    @pytest.mark.parametrize(
        "params",
        [
            lambda bad: {"note": bad},
            lambda bad: {"note": "ok", "extra": {"deep": [{"v": bad}]}},
            lambda bad: {"note": "ok", "extra": {bad: 1}},
            lambda bad: {"note": "ok", "extra": {"a": {"b": {f"k{bad}": "v"}}}},
            lambda bad: {f"n{bad}": 1},
            lambda bad: {"note": "ok", "extra": {"a": [[["z", bad]]]}},
        ],
    )
    def test_rejected_before_anything_runs(self, bad: str, params: Any) -> None:
        call = ToolCall("record", params(bad))
        result = prepare_tool_call(call, get_tool_def(record), record)
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.meta == {"code": "invalid_params"}
        assert json.loads(result.payload)["code"] == "invalid_params"
        _assert_no_content_echo(result)
        assert bad not in (result.error or "")
        assert bad not in result.payload

    def test_rejected_for_unknown_and_client_tools_too(self) -> None:
        call = ToolCall("nope", {"a": [{"b": NUL}]})
        result = check_tool_call(call, None)
        assert result is not None
        assert result.meta == {"code": "invalid_params"}
        td = ToolDef(name="c", description="", parameters={"type": "object"})
        assert check_tool_call(ToolCall("c", {NUL: 1}), td) is not None

    def test_ordinary_unicode_is_fine(self) -> None:
        call = ToolCall(
            "record",
            {"note": "café \U0001f600 😀".encode("utf-16", "surrogatepass").decode("utf-16")},
        )
        assert not isinstance(prepare_tool_call(call, get_tool_def(record), record), ToolResult)

    async def test_the_tool_never_runs_and_the_model_gets_a_value_free_result(self) -> None:
        from dendrux.agent import Agent
        from dendrux.llm.mock import MockLLM
        from dendrux.loops.base import BaseRecorder
        from dendrux.loops.react import ReActLoop
        from dendrux.strategies.native import NativeToolCalling
        from dendrux.types import LLMResponse, Role

        CALLS.clear()
        call = ToolCall(name="record", params={"note": "ok", "extra": {"k": [NUL]}})
        llm = MockLLM([LLMResponse(tool_calls=[call]), LLMResponse(text="done")])
        agent = Agent(prompt="p", tools=[record], provider=llm)
        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=BaseRecorder(),
        )
        assert CALLS == []
        (tool_msg,) = [m for m in llm.call_history[1]["messages"] if m.role == Role.TOOL]
        assert json.loads(tool_msg.content)["code"] == "invalid_params"
        assert "\\u0000" not in tool_msg.content


class TestNestedSchemaShapes:
    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "object", "properties": {"x": {"type": "object", "required": None}}},
            {"type": "object", "properties": {"x": {"type": "array", "items": {"required": 3}}}},
            {"type": "object", "properties": {"x": {"anyOf": [{"required": [1]}]}}},
            {"type": "object", "properties": {"x": {"properties": {"y": {"required": "z"}}}}},
            {"type": "object", "additionalProperties": {"required": None}},
            {"type": "object", "$defs": {"d": {"required": [{}]}}},
            {"type": "object", "patternProperties": {"^a": {"required": None}}},
            {"type": "object", "properties": {"x": {"properties": []}}},
            {"type": "object", "properties": []},
        ],
    )
    def test_malformed_nested_keywords_are_found(self, schema: Any) -> None:
        from dendrux.tools._validate import schema_problem

        assert schema_problem(schema) is not None

    @pytest.mark.parametrize("strict", [False, True])
    def test_discovery_rejects_or_skips_a_nested_malformed_required(self, strict: bool) -> None:
        from types import SimpleNamespace

        from mcp.types import Tool

        from dendrux.mcp import MCPInvalidToolSchemaError
        from dendrux.mcp._server import build_mcp_tool_defs

        source = SimpleNamespace(name="srv", transport="stdio", call_timeout=30.0)
        bad = Tool(
            name="bad",
            description="",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "object", "required": None}},
            },
        )
        good = Tool(name="good", description="", input_schema={"type": "object"})
        if strict:
            with pytest.raises(MCPInvalidToolSchemaError):
                build_mcp_tool_defs(
                    source=source,  # type: ignore[arg-type]
                    namespace="srv",
                    tools=[good, bad],
                    connection_info=None,
                    strict=True,
                )
            return
        defs, skipped = build_mcp_tool_defs(
            source=source,  # type: ignore[arg-type]
            namespace="srv",
            tools=[good, bad],
            connection_info=None,
        )
        assert [d.name for d in defs] == ["srv__good"]
        assert [(s.tool, s.reason) for s in skipped] == [("bad", "invalid_schema")]

    def test_well_formed_deep_schemas_pass(self) -> None:
        from dendrux.tools._validate import schema_problem

        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {
                    "type": "object",
                    "required": ["b"],
                    "properties": {"b": {"type": "array", "items": {"required": ["c"]}}},
                },
                "u": {"anyOf": [{"type": "string"}, {"type": "object", "required": []}]},
            },
            "$defs": {"d": {"type": "object", "properties": {}}},
        }
        assert schema_problem(schema) is None

    def test_a_schema_past_the_depth_or_node_budget_fails_closed(self) -> None:
        from dendrux.tools._validate import schema_problem

        deep: dict[str, Any] = {"type": "object"}
        for _ in range(200):
            deep = {"type": "object", "properties": {"n": deep}}
        assert schema_problem(deep) is not None
        wide = {
            "type": "object",
            "properties": {f"p{i}": {"type": "string"} for i in range(50_000)},
        }
        assert schema_problem(wide) is not None

    def test_the_call_time_check_stays_cheap_and_root_only(self) -> None:
        # Nested keywords are not read by the validator, so a call is not refused for them.
        td = _td({"type": "object", "properties": {"x": {"type": "object", "required": None}}})
        assert check_tool_call(ToolCall("remote", {"x": {}}), td) is None


_BAD = {"required": None}
# Every JSON Schema 2020-12 keyword (plus the draft-07 spellings) whose value is a
# schema, a map of schemas or an array of schemas.
_SCHEMA_VALUED = [
    "items",
    "contains",
    "additionalProperties",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
    "additionalItems",
]
_SCHEMA_MAPS = ["properties", "patternProperties", "dependentSchemas", "$defs", "definitions"]
_SCHEMA_ARRAYS = ["allOf", "anyOf", "oneOf", "prefixItems", "items"]


class TestEverySchemaKeywordIsWalked:
    @pytest.mark.parametrize("keyword", _SCHEMA_VALUED)
    def test_schema_valued_keyword(self, keyword: str) -> None:
        from dendrux.tools._validate import schema_problem

        assert schema_problem({"type": "object", keyword: dict(_BAD)}) is not None

    @pytest.mark.parametrize("keyword", _SCHEMA_MAPS)
    def test_schema_map_keyword(self, keyword: str) -> None:
        from dendrux.tools._validate import schema_problem

        assert schema_problem({"type": "object", keyword: {"k": dict(_BAD)}}) is not None

    @pytest.mark.parametrize("keyword", _SCHEMA_ARRAYS)
    def test_schema_array_keyword(self, keyword: str) -> None:
        from dendrux.tools._validate import schema_problem

        assert (
            schema_problem({"type": "object", keyword: [{"type": "string"}, dict(_BAD)]})
            is not None
        )

    @pytest.mark.parametrize("keyword", [*_SCHEMA_VALUED, *_SCHEMA_ARRAYS, *_SCHEMA_MAPS])
    def test_nested_one_level_deeper(self, keyword: str) -> None:
        from dendrux.tools._validate import schema_problem

        inner = {
            keyword: {"k": dict(_BAD)}
            if keyword in _SCHEMA_MAPS
            else ([dict(_BAD)] if keyword in _SCHEMA_ARRAYS else dict(_BAD))
        }
        assert schema_problem({"type": "object", "properties": {"x": inner}}) is not None

    @pytest.mark.parametrize("keyword", [*_SCHEMA_VALUED, *_SCHEMA_ARRAYS, *_SCHEMA_MAPS])
    def test_well_formed_values_pass(self, keyword: str) -> None:
        from dendrux.tools._validate import schema_problem

        good = {"type": "object", "required": ["a"]}
        value = (
            {"k": good}
            if keyword in _SCHEMA_MAPS
            else ([good] if keyword in _SCHEMA_ARRAYS else good)
        )
        assert schema_problem({"type": "object", keyword: value}) is None

    def test_booleans_and_scalars_in_schema_positions_are_ignored(self) -> None:
        from dendrux.tools._validate import schema_problem

        assert (
            schema_problem({"items": True, "unevaluatedItems": False, "contentSchema": 3}) is None
        )

    def test_the_budget_still_applies_to_the_new_keywords(self) -> None:
        from dendrux.tools._validate import schema_problem

        deep: dict[str, Any] = {"type": "string"}
        for _ in range(100):
            deep = {"unevaluatedItems": {"contentSchema": deep}}
        assert schema_problem(deep) is not None
