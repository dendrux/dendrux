"""Tests for pre-execution tool parameter validation (dendrux.tools._validate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseRecorder
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import get_tool_def, tool
from dendrux.tools._validate import (
    INVALID_PARAMS_CODE,
    check_tool_call,
    format_invalid_params,
    validate_tool_params,
)
from dendrux.types import (
    LLMResponse,
    Role,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolDef,
    ToolTarget,
)


def _td(parameters: Any, name: str = "show") -> ToolDef:
    return ToolDef(name=name, description="", parameters=parameters)


SHOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["table", "chart"]},
        "title": {"type": "string"},
        "result": {"type": "object"},
        "limit": {"type": "integer"},
    },
    "required": ["kind"],
}


# ------------------------------------------------------------------
# validate_tool_params — pure rules
# ------------------------------------------------------------------


class TestUnknownNames:
    def test_unknown_name_rejected_when_additional_properties_absent(self) -> None:
        problems = validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "r4<|": 1})
        assert problems == ["show has 1 unknown argument"]

    def test_unknown_name_rejected_when_additional_properties_false(self) -> None:
        schema = {**SHOW_SCHEMA, "additionalProperties": False}
        assert validate_tool_params(_td(schema), {"kind": "table", "x": 1}) == [
            "show has 1 unknown argument"
        ]

    def test_multiple_unknown_names_grouped(self) -> None:
        problems = validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "x": 1, "y": 2})
        assert problems == ["show has 2 unknown arguments"]

    @pytest.mark.parametrize("additional", [True, {"type": "string"}, {}])
    def test_permissive_additional_properties_allows_unknown(self, additional: Any) -> None:
        schema = {**SHOW_SCHEMA, "additionalProperties": additional}
        assert validate_tool_params(_td(schema), {"kind": "table", "x": 1}) == []

    def test_pattern_properties_allows_unknown(self) -> None:
        schema = {**SHOW_SCHEMA, "patternProperties": {"^x": {}}}
        assert validate_tool_params(_td(schema), {"kind": "table", "x1": 1}) == []

    @pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf", "$ref"])
    def test_top_level_composition_allows_unknown(self, keyword: str) -> None:
        schema = {**SHOW_SCHEMA, keyword: [{"properties": {"extra": {}}}]}
        assert validate_tool_params(_td(schema), {"kind": "table", "extra": 1}) == []

    def test_no_properties_skips_name_check(self) -> None:
        assert validate_tool_params(_td({"type": "object"}), {"anything": 1}) == []

    def test_empty_properties_rejects_every_name(self) -> None:
        schema = {"type": "object", "properties": {}}
        assert validate_tool_params(_td(schema), {"a": 1}) == ["show has 1 unknown argument"]

    def test_non_dict_schema_is_ignored(self) -> None:
        assert validate_tool_params(_td(None), {"a": 1}) == []


class TestRequired:
    def test_missing_required(self) -> None:
        assert validate_tool_params(_td(SHOW_SCHEMA), {"title": "t"}) == [
            "show is missing required argument(s): kind"
        ]

    def test_multiple_missing_required(self) -> None:
        schema = {**SHOW_SCHEMA, "required": ["kind", "title"]}
        assert validate_tool_params(_td(schema), {}) == [
            "show is missing required argument(s): kind, title"
        ]

    def test_required_checked_without_properties(self) -> None:
        schema = {"type": "object", "required": ["q"]}
        assert validate_tool_params(_td(schema), {}) == ["show is missing required argument(s): q"]


class TestTypes:
    @pytest.mark.parametrize(
        ("json_type", "good", "bad"),
        [
            ("string", "x", 1),
            ("integer", 3, "3"),
            ("number", 1.5, "1.5"),
            ("boolean", True, "true"),
            ("array", [1], "[1]"),
            ("object", {"a": 1}, [1]),
            ("null", None, 0),
        ],
    )
    def test_basic_types(self, json_type: str, good: Any, bad: Any) -> None:
        schema = {"type": "object", "properties": {"v": {"type": json_type}}, "required": ["v"]}
        assert validate_tool_params(_td(schema), {"v": good}) == []
        assert len(validate_tool_params(_td(schema), {"v": bad})) == 1

    def test_type_error_message(self) -> None:
        problems = validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "limit": "5"})
        assert problems == ["show argument 'limit' must be integer, got string"]

    def test_integer_rejects_bool(self) -> None:
        problems = validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "limit": True})
        assert problems == ["show argument 'limit' must be integer, got boolean"]

    def test_integer_accepts_integral_float(self) -> None:
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "limit": 2.0}) == []
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "limit": 2.5}) == [
            "show argument 'limit' must be integer, got number"
        ]

    def test_number_accepts_int_rejects_bool(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "number"}}}
        assert validate_tool_params(_td(schema), {"n": 3}) == []
        assert validate_tool_params(_td(schema), {"n": False}) == [
            "show argument 'n' must be number, got boolean"
        ]

    def test_type_array_nullable(self) -> None:
        schema = {
            "type": "object",
            "properties": {"t": {"type": ["string", "null"]}},
            "required": ["t"],
        }
        assert validate_tool_params(_td(schema), {"t": None}) == []
        assert validate_tool_params(_td(schema), {"t": "x"}) == []
        assert validate_tool_params(_td(schema), {"t": 1}) == [
            "show argument 't' must be string or null, got integer"
        ]

    def test_null_allowed_for_optional_argument(self) -> None:
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": "table", "title": None}) == []

    def test_null_rejected_for_required_non_nullable(self) -> None:
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": None}) == [
            "show argument 'kind' must be string, got null"
        ]

    def test_enum_membership(self) -> None:
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": "chart"}) == []
        # The constraint is named, the value never echoed.
        assert validate_tool_params(_td(SHOW_SCHEMA), {"kind": "pie"}) == [
            'show argument \'kind\' must be one of "table", "chart"'
        ]

    def test_enum_does_not_confuse_bool_and_int(self) -> None:
        schema = {"type": "object", "properties": {"v": {"enum": [1, 2]}}}
        assert validate_tool_params(_td(schema), {"v": 1}) == []
        assert len(validate_tool_params(_td(schema), {"v": True})) == 1

    def test_enum_with_nullable_type(self) -> None:
        schema = {
            "type": "object",
            "properties": {"v": {"type": ["string", "null"], "enum": ["a", None]}},
            "required": ["v"],
        }
        assert validate_tool_params(_td(schema), {"v": None}) == []
        assert len(validate_tool_params(_td(schema), {"v": "b"})) == 1

    @pytest.mark.parametrize(
        "prop",
        [
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            {"oneOf": [{"type": "string"}]},
            {"allOf": [{"type": "string"}]},
            {"$ref": "#/$defs/Thing"},
            {"type": "string", "not": {"const": "x"}},
            {"type": "custom-type"},
            {"type": ["string", {"weird": True}]},
        ],
    )
    def test_unknown_constructs_are_tolerated(self, prop: dict[str, Any]) -> None:
        schema = {"type": "object", "properties": {"v": prop}}
        assert validate_tool_params(_td(schema), {"v": 12345}) == []

    def test_format_keyword_ignored(self) -> None:
        schema = {"type": "object", "properties": {"d": {"type": "string", "format": "date"}}}
        assert validate_tool_params(_td(schema), {"d": "not a date"}) == []

    def test_nested_objects_not_recursed(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "o": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                }
            },
        }
        assert validate_tool_params(_td(schema), {"o": {"y": "z"}}) == []

    def test_non_dict_property_schema_skipped(self) -> None:
        schema = {"type": "object", "properties": {"v": True}}
        assert validate_tool_params(_td(schema), {"v": 1}) == []

    def test_all_problems_collected_in_order(self) -> None:
        problems = validate_tool_params(_td(SHOW_SCHEMA), {"bogus": 1, "limit": "x"})
        assert problems == [
            "show has 1 unknown argument",
            "show is missing required argument(s): kind",
            "show argument 'limit' must be integer, got string",
        ]


class TestMessageAndResult:
    def test_oreo_message_shape(self) -> None:
        schema = {
            "type": "object",
            "properties": {"kind": {}, "title": {}, "result": {}},
        }
        tc = ToolCall(name="show", params={'r4<|"|>,split': "x"})
        result = check_tool_call(tc, _td(schema))
        assert result is not None
        assert result.error == (
            "show has 1 unknown argument. Valid arguments: kind, title, result."
        )

    def test_type_only_message_omits_valid_names(self) -> None:
        tc = ToolCall(name="show", params={"kind": "table", "limit": "5"})
        result = check_tool_call(tc, _td(SHOW_SCHEMA))
        assert result is not None
        assert result.error == "show argument 'limit' must be integer, got string."

    def test_multiple_problems_joined(self) -> None:
        tc = ToolCall(name="show", params={"limit": "5"})
        result = check_tool_call(tc, _td(SHOW_SCHEMA))
        assert result is not None
        assert result.error == (
            "show is missing required argument(s): kind; "
            "show argument 'limit' must be integer, got string. "
            "Valid arguments: kind, title, result, limit."
        )

    def test_result_payload_shape(self) -> None:
        tc = ToolCall(name="show", params={})
        result = check_tool_call(tc, _td(SHOW_SCHEMA))
        assert result is not None
        assert result.success is False
        assert result.call_id == tc.id
        assert result.name == "show"
        payload = json.loads(result.payload)
        assert payload == {"error": result.error, "code": INVALID_PARAMS_CODE}

    def test_valid_call_returns_none(self) -> None:
        assert (
            check_tool_call(ToolCall(name="show", params={"kind": "table"}), _td(SHOW_SCHEMA))
            is None
        )

    def test_unknown_tool_passes_through(self) -> None:
        assert check_tool_call(ToolCall(name="nope", params={"x": 1}), None) is None

    def test_no_valid_names(self) -> None:
        msg = format_invalid_params(_td({"type": "object", "properties": {}}), ["p"])
        assert msg == "p. Valid arguments: (none)."

    def test_unserializable_enum_value_reported_without_echo(self) -> None:
        schema = {"type": "object", "properties": {"v": {"enum": ["a"]}}}
        problems = validate_tool_params(_td(schema), {"v": object()})
        assert problems == ["show argument 'v' must be one of \"a\""]

    def test_decorated_tool_schema(self) -> None:
        @tool()
        def search(query: str, limit: int = 10, tag: str | None = None) -> str:
            """Search."""
            return query

        td = get_tool_def(search)
        assert validate_tool_params(td, {"query": "q", "tag": None}) == []
        assert validate_tool_params(td, {"query": "q", "limit": "ten", "qury": 1}) == [
            "search has 1 unknown argument",
            "search argument 'limit' must be integer, got string",
        ]


# ------------------------------------------------------------------
# Loop integration
# ------------------------------------------------------------------

CALLS: dict[str, int] = {}


@tool()
async def show(kind: str, title: str = "", limit: int = 10) -> str:
    """Show a result."""
    CALLS["show"] = CALLS.get("show", 0) + 1
    return f"shown {kind}"


@tool()
async def lookup(query: str) -> str:
    """Look something up."""
    CALLS["lookup"] = CALLS.get("lookup", 0) + 1
    return f"found {query}"


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Read a sheet range on the client."""
    return ""


class _Capture(BaseRecorder):
    def __init__(self) -> None:
        self.tool_results: list[Any] = []
        self.started: list[str] = []

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        self.started.append(tool_call.name)

    async def on_tool_completed(
        self, run_id: str, tool_call: ToolCall, tool_result: Any, iteration: int
    ) -> None:
        self.tool_results.append(tool_result)


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def _tool_messages(llm: MockLLM, call_index: int) -> list[Any]:
    return [m for m in llm.call_history[call_index]["messages"] if m.role == Role.TOOL]


class TestLoopValidation:
    async def test_invalid_call_skips_tool_and_reports_error(self) -> None:
        tc = ToolCall(name="show", params={'r4<|"|>,split': "x"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = Agent(prompt="p", tools=[show], provider=llm)
        recorder = _Capture()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="go",
            recorder=recorder,
        )

        assert result.status == RunStatus.SUCCESS
        assert CALLS.get("show", 0) == 0
        tool_msgs = _tool_messages(llm, 1)
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0].content)
        assert payload["code"] == "invalid_params"
        assert payload["error"] == (
            "show has 1 unknown argument; "
            "show is missing required argument(s): kind. "
            "Valid arguments: kind, title, limit."
        )
        assert tool_msgs[0].meta == {"is_error": True}
        assert recorder.started == []
        assert len(recorder.tool_results) == 1
        assert recorder.tool_results[0].success is False
        assert result.steps[0].meta["all_tool_calls"][0].name == "show"

    async def test_valid_call_unaffected(self) -> None:
        tc = ToolCall(name="show", params={"kind": "table"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = Agent(prompt="p", tools=[show], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert CALLS["show"] == 1
        assert json.loads(_tool_messages(llm, 1)[0].content) == "shown table"

    async def test_parallel_batch_runs_valid_calls(self) -> None:
        bad = ToolCall(
            name="show", params={"kind": "table", "limit": "x"}, provider_tool_call_id="a"
        )
        good1 = ToolCall(name="lookup", params={"query": "q1"}, provider_tool_call_id="b")
        good2 = ToolCall(name="show", params={"kind": "chart"}, provider_tool_call_id="c")
        llm = MockLLM([LLMResponse(tool_calls=[bad, good1, good2]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[show, lookup], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert CALLS == {"lookup": 1, "show": 1}
        msgs = {m.call_id: m for m in _tool_messages(llm, 1)}
        assert len(msgs) == 3
        assert "must be integer, got string" in msgs[bad.id].content
        assert msgs[good1.id].meta == {}
        assert msgs[good2.id].meta == {}

    async def test_invalid_client_tool_does_not_pause(self) -> None:
        tc = ToolCall(name="read_range", params={"sheeet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[read_range], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert "read_range has 1 unknown argument" in _tool_messages(llm, 1)[0].content

    async def test_valid_client_tool_still_pauses(self) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = Agent(prompt="p", tools=[read_range], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.WAITING_CLIENT_TOOL

    async def test_invalid_call_does_not_count_toward_max_calls(self) -> None:
        @tool(max_calls_per_run=1)
        async def once(x: int) -> int:
            """Only once."""
            CALLS["once"] = CALLS.get("once", 0) + 1
            return x

        bad = ToolCall(name="once", params={"y": 1}, provider_tool_call_id="a")
        good = ToolCall(name="once", params={"x": 1}, provider_tool_call_id="b")
        llm = MockLLM([LLMResponse(tool_calls=[bad, good]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[once], provider=llm)

        await agent.run("go")

        assert CALLS == {"once": 1}

    async def test_denied_tool_not_validated(self) -> None:
        tc = ToolCall(name="show", params={"bogus": 1}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[show, lookup], deny=["show"], provider=llm)

        await agent.run("go")

        content = _tool_messages(llm, 1)[0].content
        assert "denied" in content and "invalid_params" not in content

    async def test_stream_emits_failed_tool_result(self) -> None:
        tc = ToolCall(name="show", params={"kind": 5}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tools=[show], provider=llm)

        events = [e async for e in agent.stream("go")]

        results = [e for e in events if e.type == RunEventType.TOOL_RESULT]
        assert len(results) == 1
        assert results[0].tool_result is not None
        assert results[0].tool_result.success is False
        assert results[0].tool_result.error == "show argument 'kind' must be string, got integer."
        assert events[-1].type == RunEventType.RUN_COMPLETED
        assert CALLS.get("show", 0) == 0


class TestUseSkillValidation:
    async def test_use_skill_without_name_is_rejected(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "greet"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: greet\ndescription: Greeting skill.\n---\nSay hello warmly.\n"
        )
        bad = ToolCall(name="use_skill", params={"skill": "greet"}, provider_tool_call_id="a")
        good = ToolCall(name="use_skill", params={"name": "greet"}, provider_tool_call_id="b")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[bad]),
                LLMResponse(tool_calls=[good]),
                LLMResponse(text="ok"),
            ]
        )
        agent = Agent(prompt="p", tools=[lookup], skills_dir=str(tmp_path), provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        first = _tool_messages(llm, 1)[0].content
        assert "use_skill has 1 unknown argument" in first
        second = _tool_messages(llm, 2)[-1].content
        assert "Say hello warmly." in second


# ------------------------------------------------------------------
# MCP tools
# ------------------------------------------------------------------


def _mock_mcp_source(parameters: dict[str, Any], counter: dict[str, int]) -> Any:
    from unittest.mock import AsyncMock, MagicMock

    from dendrux.mcp._server import MCPServer

    source = MagicMock(spec=MCPServer)
    source.name = "fs"
    td = ToolDef(
        name="fs__read_file",
        description="Read a file.",
        parameters=parameters,
        target=ToolTarget.SERVER,
        meta={"source_name": "fs", "mcp_tool_name": "read_file"},
    )

    async def executor(**kwargs: Any) -> Any:
        counter["n"] = counter.get("n", 0) + 1
        return {"content": "data"}

    async def discover() -> list[ToolDef]:
        return [td]

    source._discover = discover
    source._create_executor = lambda name, exposed_name=None: executor
    source.close = AsyncMock()
    return source


class TestMCPValidation:
    async def test_mcp_schema_rejects_unknown_argument_without_calling_server(self) -> None:
        counter: dict[str, int] = {}
        source = _mock_mcp_source(
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            counter,
        )
        bad = ToolCall(name="fs__read_file", params={"pth": "/a"}, provider_tool_call_id="a")
        good = ToolCall(name="fs__read_file", params={"path": "/a"}, provider_tool_call_id="b")
        llm = MockLLM(
            [LLMResponse(tool_calls=[bad]), LLMResponse(tool_calls=[good]), LLMResponse(text="ok")]
        )
        agent = Agent(prompt="p", tool_sources=[source], provider=llm)

        result = await agent.run("go")

        assert result.status == RunStatus.SUCCESS
        assert counter == {"n": 1}
        assert _tool_messages(llm, 1)[0].content == json.dumps(
            {
                "error": "fs__read_file has 1 unknown argument; fs__read_file is missing "
                "required argument(s): path. Valid arguments: path.",
                "code": "invalid_params",
            }
        )

    async def test_mcp_permissive_schema_passes_through(self) -> None:
        counter: dict[str, int] = {}
        source = _mock_mcp_source(
            {
                "type": "object",
                "properties": {"path": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                "additionalProperties": True,
            },
            counter,
        )
        tc = ToolCall(name="fs__read_file", params={"path": 3, "x": 1}, provider_tool_call_id="a")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="ok")])
        agent = Agent(prompt="p", tool_sources=[source], provider=llm)

        await agent.run("go")

        assert counter == {"n": 1}

    async def test_real_sdk_stdio_server_never_sees_invalid_call(self) -> None:
        from dendrux.mcp._server import MCPServer

        fixture = Path(__file__).parents[1] / "fixtures" / "mcp_echo_server.py"
        server = MCPServer(
            "echo",
            command=[sys.executable, str(fixture)],
            connect_timeout=10.0,
            call_timeout=10.0,
        )
        bad = ToolCall(name="echo__echo", params={"msg": "hi"}, provider_tool_call_id="a")
        good = ToolCall(name="echo__echo", params={"message": "hi"}, provider_tool_call_id="b")
        llm = MockLLM(
            [LLMResponse(tool_calls=[bad]), LLMResponse(tool_calls=[good]), LLMResponse(text="ok")]
        )
        agent = Agent(prompt="p", tool_sources=[server], provider=llm)
        try:
            result = await agent.run("go")
        finally:
            await agent.close()

        assert result.status == RunStatus.SUCCESS
        first = json.loads(_tool_messages(llm, 1)[0].content)
        assert first == {
            "error": "echo__echo has 1 unknown argument; echo__echo is missing required "
            "argument(s): message. Valid arguments: message.",
            "code": "invalid_params",
        }
        second = _tool_messages(llm, 2)[-1]
        assert second.meta == {}
        assert "hi" in second.content
