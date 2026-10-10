"""Tests for @tool decorator and schema generation."""

import dataclasses
import enum
import json
from typing import Annotated, Any, Literal, NotRequired, Optional, TypedDict

import pytest
from pydantic import BaseModel, Field

from dendrux.tool import get_tool_def, is_tool, tool
from dendrux.types import ToolTarget


class TestToolDecorator:
    def test_basic_server_tool(self) -> None:
        @tool(target="server")
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        td = get_tool_def(add)
        assert td.name == "add"
        assert td.description == "Add two numbers."
        assert td.target == ToolTarget.SERVER

    def test_client_tool(self) -> None:
        @tool(target="client")
        async def read_range(sheet: str, range: str) -> list:
            """Read cells from Excel."""
            pass

        td = get_tool_def(read_range)
        assert td.target == ToolTarget.CLIENT

    def test_human_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):

            @tool(target="human")
            async def ask_user(question: str) -> str:
                """Ask user a question."""
                pass

    def test_agent_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):

            @tool(target="agent")
            async def delegate(task: str) -> str:
                """Delegate to sub-agent."""
                pass

    def test_default_target_is_server(self) -> None:
        @tool()
        async def ping() -> str:
            """Ping."""
            return "pong"

        td = get_tool_def(ping)
        assert td.target == ToolTarget.SERVER

    def test_decorator_options(self) -> None:
        @tool(target="server", parallel=False, max_calls_per_run=3, timeout_seconds=60.0)
        async def slow_search(query: str) -> str:
            """Search slowly."""
            return query

        td = get_tool_def(slow_search)
        assert td.parallel is False
        assert td.max_calls_per_run == 3
        assert td.timeout_seconds == 60.0

    def test_function_stays_callable(self) -> None:
        @tool(target="server")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(1, 2) == 3

    async def test_async_function_stays_callable(self) -> None:
        @tool(target="server")
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add(1, 2)
        assert result == 3

    def test_no_docstring_uses_empty_description(self) -> None:
        @tool(target="server")
        async def mystery(x: int) -> int:
            return x

        td = get_tool_def(mystery)
        assert td.description == ""

    def test_is_tool_returns_true_for_decorated(self) -> None:
        @tool(target="server")
        async def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert is_tool(add) is True

    def test_is_tool_returns_false_for_plain_function(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b

        assert is_tool(add) is False

    def test_get_tool_def_raises_for_plain_function(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ValueError, match="not a Dendrux tool"):
            get_tool_def(add)


class TestSchemaGeneration:
    def test_int_param(self) -> None:
        @tool()
        async def fn(x: int) -> int:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "integer"
        assert "x" in schema["required"]

    def test_str_param(self) -> None:
        @tool()
        async def fn(x: str) -> str:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "string"

    def test_float_param(self) -> None:
        @tool()
        async def fn(x: float) -> float:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "number"

    def test_bool_param(self) -> None:
        @tool()
        async def fn(x: bool) -> bool:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "boolean"

    def test_list_param(self) -> None:
        @tool()
        async def fn(x: list[str]) -> list:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "array"
        assert schema["properties"]["x"]["items"]["type"] == "string"

    def test_dict_param(self) -> None:
        @tool()
        async def fn(x: dict) -> dict:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == "object"

    def test_optional_param_not_required(self) -> None:
        @tool()
        async def fn(x: str, y: str | None = None) -> str:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_default_value_in_schema(self) -> None:
        @tool()
        async def fn(x: str, limit: int = 10) -> str:
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["limit"]["default"] == 10
        assert "limit" not in schema["required"]

    def test_multiple_params(self) -> None:
        @tool()
        async def read_range(sheet: str, range: str, include_headers: bool = True) -> list:
            """Read cells from a range."""
            return []

        schema = get_tool_def(read_range).parameters
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"sheet", "range"}
        assert schema["properties"]["include_headers"]["type"] == "boolean"
        assert schema["properties"]["include_headers"]["default"] is True

    def test_no_params(self) -> None:
        @tool()
        async def get_time() -> str:
            """Get current time."""
            return "now"

        schema = get_tool_def(get_time).parameters
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_typing_optional_not_required(self) -> None:
        """M1: typing.Optional[X] must also be detected as optional."""

        @tool()
        async def fn(x: str, y: Optional[str] = None) -> str:  # noqa: UP045
            """Test."""
            return x

        schema = get_tool_def(fn).parameters
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_invalid_target_raises(self) -> None:
        """M8: Invalid target string raises ValueError."""
        with pytest.raises(ValueError, match="invalid"):

            @tool(target="invalid")
            async def fn() -> str:
                """Test."""
                return ""

    def test_untyped_param_raises_type_error(self) -> None:
        """F-06: Parameters without type hints must fail fast at decoration time."""
        with pytest.raises(TypeError, match="has no type hint"):

            @tool()
            async def fn(x: int, y) -> int:  # y has no type hint
                """Test."""
                return x

    def test_all_untyped_params_raise_type_error(self) -> None:
        """F-06: Even a single untyped param (the first one) raises."""
        with pytest.raises(TypeError, match="has no type hint"):

            @tool()
            async def fn(a) -> str:  # noqa: ANN001
                """Test."""
                return str(a)

    def test_union_type_raises_type_error(self) -> None:
        """C-05: Multi-type unions like str | int are rejected."""
        with pytest.raises(TypeError, match="Union type"):

            @tool()
            async def fn(x: str | int) -> str:
                """Test."""
                return str(x)

    def test_optional_union_with_none_is_valid(self) -> None:
        """Optional[str] (str | None) should still work."""

        @tool()
        async def fn(x: str | None = None) -> str:
            """Test."""
            return str(x)

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"]["type"] == ["string", "null"]

    def test_non_json_serializable_default_raises(self) -> None:
        """A-02: Defaults that can't be JSON-serialized raise at decoration time."""
        from datetime import datetime

        with pytest.raises(TypeError, match="not JSON-serializable"):

            @tool()
            async def fn(x: str = datetime.now()) -> str:  # noqa: B008
                """Test."""
                return x

    def test_invalid_tool_name_raises(self) -> None:
        """A-01: Non-identifier names are rejected."""
        import types

        fn = types.FunctionType((lambda: None).__code__, {}, "<lambda>")
        fn.__doc__ = "Test."
        fn.__annotations__ = {"return": str}
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            tool()(fn)


# ---------------------------------------------------------------------------
# Rich schema types
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class Priority(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Node(BaseModel):
    id: str = Field(description="Node identifier")
    label: str | None = None


class Graph(BaseModel):
    nodes: list[Node]
    root: Node


class TreeNode(BaseModel):
    name: str
    children: list["TreeNode"] = []  # noqa: UP037


@dataclasses.dataclass
class Point:
    x: float
    y: float = 0.0
    tag: str | None = None


@dataclasses.dataclass
class Segment:
    start: Point
    end: Point


class Series(TypedDict):
    name: str
    values: list[float]
    color: NotRequired[str]


@dataclasses.dataclass
class Loop:
    next: "Loop | None" = None  # noqa: UP037


class TestLiteralAndEnum:
    def test_literal_strings(self) -> None:
        @tool()
        async def fn(kind: Literal["chart", "table"]) -> str:
            """Test."""
            return kind

        prop = get_tool_def(fn).parameters["properties"]["kind"]
        assert prop == {"type": "string", "enum": ["chart", "table"]}

    def test_literal_ints(self) -> None:
        @tool()
        async def fn(n: Literal[1, 2, 3]) -> int:
            """Test."""
            return n

        assert get_tool_def(fn).parameters["properties"]["n"] == {
            "type": "integer",
            "enum": [1, 2, 3],
        }

    def test_literal_bools(self) -> None:
        @tool()
        async def fn(flag: Literal[True]) -> bool:
            """Test."""
            return flag

        assert get_tool_def(fn).parameters["properties"]["flag"] == {
            "type": "boolean",
            "enum": [True],
        }

    def test_literal_mixed_types_raises(self) -> None:
        with pytest.raises(TypeError, match="mixed value types"):

            @tool()
            async def fn(x: Literal["a", 1]) -> str:
                """Test."""
                return str(x)

    def test_literal_with_default(self) -> None:
        @tool()
        async def fn(kind: Literal["a", "b"] = "a") -> str:
            """Test."""
            return kind

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["kind"] == {
            "type": "string",
            "enum": ["a", "b"],
            "default": "a",
        }
        assert schema["required"] == []

    def test_str_enum(self) -> None:
        @tool()
        async def fn(color: Color) -> str:
            """Test."""
            return str(color)

        assert get_tool_def(fn).parameters["properties"]["color"] == {
            "type": "string",
            "enum": ["red", "green"],
        }

    def test_int_enum(self) -> None:
        @tool()
        async def fn(p: Priority) -> str:
            """Test."""
            return str(p)

        assert get_tool_def(fn).parameters["properties"]["p"] == {
            "type": "integer",
            "enum": [1, 2],
        }

    def test_mixed_enum_raises(self) -> None:
        class Mixed(enum.Enum):
            A = "a"
            B = 2

        with pytest.raises(TypeError, match="mixed value types"):

            @tool()
            async def fn(m: Mixed) -> str:
                """Test."""
                return str(m)

    def test_optional_literal_includes_null_in_enum(self) -> None:
        @tool()
        async def fn(kind: Literal["a", "b"] | None = None) -> str:
            """Test."""
            return str(kind)

        assert get_tool_def(fn).parameters["properties"]["kind"] == {
            "type": ["string", "null"],
            "enum": ["a", "b", None],
            "default": None,
        }


class TestDescriptions:
    def test_annotated_pydantic_field(self) -> None:
        @tool()
        async def fn(q: Annotated[str, Field(description="Search query")]) -> str:
            """Test."""
            return q

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["q"] == {"type": "string", "description": "Search query"}
        assert schema["required"] == ["q"]

    def test_annotated_plain_string(self) -> None:
        @tool()
        async def fn(n: Annotated[int, "How many"] = 3) -> int:
            """Test."""
            return n

        assert get_tool_def(fn).parameters["properties"]["n"] == {
            "type": "integer",
            "description": "How many",
            "default": 3,
        }

    def test_annotated_without_description_is_plain(self) -> None:
        @tool()
        async def fn(n: Annotated[int, 42]) -> int:
            """Test."""
            return n

        assert get_tool_def(fn).parameters["properties"]["n"] == {"type": "integer"}

    def test_annotated_optional_stays_optional(self) -> None:
        @tool()
        async def fn(x: Annotated[str | None, "Maybe"]) -> str:
            """Test."""
            return str(x)

        schema = get_tool_def(fn).parameters
        assert schema["required"] == []
        assert schema["properties"]["x"]["description"] == "Maybe"

    def test_annotated_inside_list(self) -> None:
        @tool()
        async def fn(ids: list[Annotated[int, "An id"]]) -> int:
            """Test."""
            return 0

        assert get_tool_def(fn).parameters["properties"]["ids"]["items"] == {
            "type": "integer",
            "description": "An id",
        }

    def test_docstring_args_section(self) -> None:
        @tool()
        async def fn(query: str, limit: int = 5) -> str:
            """Search things.

            Args:
                query: What to search for.
                limit (int): Max results
                    to return.

            Returns:
                Results.
            """
            return query

        td = get_tool_def(fn)
        props = td.parameters["properties"]
        assert props["query"]["description"] == "What to search for."
        assert props["limit"]["description"] == "Max results to return."
        assert td.description.startswith("Search things.")
        assert "Args:" in td.description

    def test_annotated_wins_over_docstring(self) -> None:
        @tool()
        async def fn(q: Annotated[str, "From annotation"]) -> str:
            """Test.

            Args:
                q: From docstring.
            """
            return q

        assert get_tool_def(fn).parameters["properties"]["q"]["description"] == "From annotation"

    def test_docstring_unknown_args_ignored(self) -> None:
        @tool()
        async def fn(q: str) -> str:
            """Test.

            Args:
                other: Not a parameter.
                garbage line without colon
            """
            return q

        assert get_tool_def(fn).parameters["properties"]["q"] == {"type": "string"}

    def test_docstring_without_args_section(self) -> None:
        @tool()
        async def fn(q: str) -> str:
            """Just a description: with a colon."""
            return q

        assert get_tool_def(fn).parameters["properties"]["q"] == {"type": "string"}


class TestNestedObjects:
    def test_pydantic_model_inlined(self) -> None:
        @tool()
        async def fn(node: Node) -> str:
            """Test."""
            return node.id

        prop = get_tool_def(fn).parameters["properties"]["node"]
        assert prop["type"] == "object"
        assert prop["properties"]["id"]["type"] == "string"
        assert prop["properties"]["id"]["description"] == "Node identifier"
        assert prop["required"] == ["id"]

    def test_pydantic_nested_refs_resolved(self) -> None:
        @tool()
        async def fn(graph: Graph) -> str:
            """Test."""
            return "ok"

        prop = get_tool_def(fn).parameters["properties"]["graph"]
        assert "$ref" not in json.dumps(prop)
        assert "$defs" not in prop
        assert prop["properties"]["nodes"]["items"]["properties"]["id"]["type"] == "string"
        assert prop["properties"]["root"]["type"] == "object"

    def test_list_of_models(self) -> None:
        @tool()
        async def fn(nodes: list[Node]) -> str:
            """Test."""
            return "ok"

        prop = get_tool_def(fn).parameters["properties"]["nodes"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "object"
        assert "id" in prop["items"]["properties"]

    def test_recursive_model_raises(self) -> None:
        with pytest.raises(TypeError, match="[Rr]ecursive"):

            @tool()
            async def fn(tree: TreeNode) -> str:
                """Test."""
                return "ok"

    def test_dataclass(self) -> None:
        @tool()
        async def fn(seg: Segment) -> str:
            """Test."""
            return "ok"

        prop = get_tool_def(fn).parameters["properties"]["seg"]
        assert prop["type"] == "object"
        assert prop["required"] == ["start", "end"]
        assert prop["properties"]["start"] == {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number", "default": 0.0},
                "tag": {"type": ["string", "null"], "default": None},
            },
            "required": ["x"],
        }

    def test_recursive_dataclass_raises(self) -> None:
        with pytest.raises(TypeError, match="[Rr]ecursive"):

            @tool()
            async def fn(loop: Loop) -> str:
                """Test."""
                return "ok"

    def test_typeddict(self) -> None:
        @tool()
        async def fn(series: list[Series]) -> str:
            """Test."""
            return "ok"

        assert get_tool_def(fn).parameters["properties"]["series"]["items"] == {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "array", "items": {"type": "number"}},
                "color": {"type": "string"},
            },
            "required": ["name", "values"],
        }

    def test_dict_with_value_type(self) -> None:
        @tool()
        async def fn(scores: dict[str, int]) -> str:
            """Test."""
            return "ok"

        assert get_tool_def(fn).parameters["properties"]["scores"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_dict_str_any_is_plain_object(self) -> None:
        @tool()
        async def fn(data: dict[str, Any]) -> str:
            """Test."""
            return "ok"

        assert get_tool_def(fn).parameters["properties"]["data"] == {"type": "object"}

    def test_dict_non_str_key_raises(self) -> None:
        with pytest.raises(TypeError, match="keys must be str"):

            @tool()
            async def fn(data: dict[int, str]) -> str:
                """Test."""
                return "ok"

    def test_unsupported_type_raises(self) -> None:
        class Opaque:
            pass

        with pytest.raises(TypeError, match="not supported"):

            @tool()
            async def fn(x: Opaque) -> str:
                """Test."""
                return "ok"


class TestNullable:
    def test_optional_str_is_nullable(self) -> None:
        @tool()
        async def fn(title: str | None = None) -> str:
            """Test."""
            return str(title)

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["title"] == {"type": ["string", "null"], "default": None}
        assert schema["required"] == []

    def test_optional_without_default_not_required(self) -> None:
        @tool()
        async def fn(x: Optional[int]) -> str:  # noqa: UP045
            """Test."""
            return str(x)

        schema = get_tool_def(fn).parameters
        assert schema["properties"]["x"] == {"type": ["integer", "null"]}
        assert schema["required"] == []

    def test_optional_model_is_nullable_object(self) -> None:
        @tool()
        async def fn(node: Node | None = None) -> str:
            """Test."""
            return "ok"

        prop = get_tool_def(fn).parameters["properties"]["node"]
        assert prop["type"] == ["object", "null"]
        assert "id" in prop["properties"]

    def test_optional_list(self) -> None:
        @tool()
        async def fn(tags: list[str] | None = None) -> str:
            """Test."""
            return "ok"

        assert get_tool_def(fn).parameters["properties"]["tags"] == {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "default": None,
        }

    def test_simple_tool_schema_unchanged(self) -> None:
        @tool()
        async def read_range(sheet: str, rows: list[int], include_headers: bool = True) -> list:
            """Read cells."""
            return []

        assert get_tool_def(read_range).parameters == {
            "type": "object",
            "properties": {
                "sheet": {"type": "string"},
                "rows": {"type": "array", "items": {"type": "integer"}},
                "include_headers": {"type": "boolean", "default": True},
            },
            "required": ["sheet", "rows"],
        }


class TestExplicitParameters:
    def test_parameters_override_bypasses_generation(self) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["x"],
        }

        @tool(parameters=params)
        async def fn(x) -> str:  # noqa: ANN001
            """Test."""
            return str(x)

        td = get_tool_def(fn)
        assert td.parameters == params
        assert td.parameters is not params

    def test_parameters_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="must be a dict"):

            @tool(parameters=["nope"])  # type: ignore[arg-type]
            async def fn() -> str:
                """Test."""
                return ""

    def test_parameters_must_be_object_type(self) -> None:
        with pytest.raises(ValueError, match='"type": "object"'):

            @tool(parameters={"type": "string"})
            async def fn() -> str:
                """Test."""
                return ""


class TestStrict:
    def test_strict_default_false(self) -> None:
        @tool()
        async def fn(x: str) -> str:
            """Test."""
            return x

        assert get_tool_def(fn).strict is False

    def test_strict_normalizes_schema(self) -> None:
        @tool(strict=True)
        async def fn(
            kind: Literal["chart", "table"],
            nodes: list[Node],
            title: str | None = None,
        ) -> str:
            """Test."""
            return kind

        td = get_tool_def(fn)
        assert td.strict is True
        schema = td.parameters
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["kind", "nodes", "title"]
        assert schema["properties"]["title"]["type"] == ["string", "null"]
        node = schema["properties"]["nodes"]["items"]
        assert node["additionalProperties"] is False
        assert node["required"] == ["id", "label"]

    def test_strict_nullable_object_gets_additional_properties(self) -> None:
        @tool(strict=True)
        async def fn(p: Point | None = None) -> str:
            """Test."""
            return "ok"

        prop = get_tool_def(fn).parameters["properties"]["p"]
        assert prop["additionalProperties"] is False
        assert prop["required"] == ["x", "y", "tag"]

    def test_strict_rejects_open_dict(self) -> None:
        with pytest.raises(TypeError, match="strict"):

            @tool(strict=True)
            async def fn(data: dict[str, int]) -> str:
                """Test."""
                return "ok"

    def test_strict_rejects_bare_dict(self) -> None:
        with pytest.raises(TypeError, match="strict"):

            @tool(strict=True)
            async def fn(data: dict) -> str:
                """Test."""
                return "ok"

    def test_strict_with_explicit_parameters(self) -> None:
        @tool(
            strict=True,
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        async def fn(q: str) -> str:
            """Test."""
            return q

        assert get_tool_def(fn).parameters == {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        }
