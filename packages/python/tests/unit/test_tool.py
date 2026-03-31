"""Tests for @tool decorator and schema generation."""

from typing import Optional

import pytest

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
        @tool(
            target="server", parallel=False, priority=5, max_calls_per_run=3, timeout_seconds=60.0
        )
        async def slow_search(query: str) -> str:
            """Search slowly."""
            return query

        td = get_tool_def(slow_search)
        assert td.parallel is False
        assert td.priority == 5
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
        assert schema["properties"]["x"]["type"] == "string"

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
