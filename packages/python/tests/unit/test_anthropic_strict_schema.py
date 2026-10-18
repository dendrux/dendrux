"""Tests for Anthropic strict-mode schema helpers in dendrux.llm._schema."""

from __future__ import annotations

import copy
import json
from decimal import Decimal  # noqa: TC003 (pydantic resolves it at runtime)
from enum import StrEnum
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field, conlist

from dendrux.llm._schema import (
    ANTHROPIC_STRICT_MAX_OPTIONAL,
    ANTHROPIC_STRICT_MAX_UNIONS,
    anthropic_strict_budget,
    anthropic_strict_problem,
    find_open_object,
    is_anthropic_strict_compatible,
    normalize_for_anthropic_strict,
)


def _obj(props: dict[str, Any], required: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": list(props) if required is None else required,
    }
    schema.update(extra)
    return schema


def _x(result: dict[str, Any]) -> dict[str, Any]:
    return result["properties"]["x"]


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class Address(BaseModel):
    city: str
    zip_code: str = Field(pattern=r"^\d{5}$")


class Invoice(BaseModel):
    number: int = Field(ge=1, description="Invoice number.")
    amount: float = Field(gt=0)
    color: Color
    note: Optional[str] = None  # noqa: UP045
    address: Address
    tags: list[str] = Field(default_factory=list, max_length=3)


# ------------------------------------------------------------------
# normalize_for_anthropic_strict
# ------------------------------------------------------------------


class TestNormalizeClosesObjects:
    def test_additional_properties_false_everywhere(self) -> None:
        inner = _obj({"a": {"type": "integer"}})
        schema = _obj(
            {
                "nested": inner,
                "list": {"type": "array", "items": copy.deepcopy(inner)},
                "maybe": {"anyOf": [copy.deepcopy(inner), {"type": "null"}]},
                "ref": {"$ref": "#/$defs/D"},
            },
            **{"$defs": {"D": copy.deepcopy(inner)}},
        )
        result = normalize_for_anthropic_strict(schema)

        assert result["additionalProperties"] is False
        assert result["properties"]["nested"]["additionalProperties"] is False
        assert result["properties"]["list"]["items"]["additionalProperties"] is False
        assert result["properties"]["maybe"]["anyOf"][0]["additionalProperties"] is False
        assert result["$defs"]["D"]["additionalProperties"] is False

    def test_empty_properties_object_is_closed(self) -> None:
        result = normalize_for_anthropic_strict(_obj({"x": _obj({})}))
        assert _x(result)["additionalProperties"] is False

    def test_required_unchanged(self) -> None:
        schema = _obj({"a": {"type": "integer"}, "b": {"type": "string"}}, required=["a"])
        assert normalize_for_anthropic_strict(schema)["required"] == ["a"]

    def test_input_not_mutated(self) -> None:
        schema = _obj({"x": {"type": "integer", "minimum": 1}})
        before = copy.deepcopy(schema)
        normalize_for_anthropic_strict(schema)
        assert schema == before

    def test_deterministic_bytes(self) -> None:
        schema = Invoice.model_json_schema()
        first = json.dumps(normalize_for_anthropic_strict(schema))
        second = json.dumps(normalize_for_anthropic_strict(schema))
        assert first == second

    def test_idempotent(self) -> None:
        once = normalize_for_anthropic_strict(Invoice.model_json_schema())
        assert normalize_for_anthropic_strict(once) == once


class TestNormalizeMovesConstraintsToDescription:
    @pytest.mark.parametrize(
        ("keyword", "value", "note"),
        [
            ("minimum", 1, "Must be >= 1."),
            ("maximum", 10, "Must be <= 10."),
            ("exclusiveMinimum", 0, "Must be > 0."),
            ("exclusiveMaximum", 100, "Must be < 100."),
            ("multipleOf", 5, "Must be a multiple of 5."),
        ],
    )
    def test_numeric_bounds(self, keyword: str, value: int, note: str) -> None:
        for type_ in ("integer", "number"):
            result = normalize_for_anthropic_strict(_obj({"x": {"type": type_, keyword: value}}))
            assert keyword not in _x(result)
            assert _x(result)["description"] == note

    def test_appends_to_existing_description(self) -> None:
        schema = _obj({"x": {"type": "integer", "description": "Count.", "minimum": 1}})
        assert (
            _x(normalize_for_anthropic_strict(schema))["description"] == "Count.\n\nMust be >= 1."
        )

    def test_multiple_notes_in_fixed_order(self) -> None:
        schema = _obj({"x": {"type": "integer", "maximum": 9, "minimum": 1}})
        assert _x(normalize_for_anthropic_strict(schema))["description"] == (
            "Must be >= 1. Must be <= 9."
        )

    @pytest.mark.parametrize("value", [0, 1])
    def test_min_items_zero_or_one_kept(self, value: int) -> None:
        schema = _obj({"x": {"type": "array", "items": {"type": "integer"}, "minItems": value}})
        assert _x(normalize_for_anthropic_strict(schema))["minItems"] == value

    def test_array_constraints_stripped(self) -> None:
        schema = _obj(
            {
                "x": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 5,
                    "uniqueItems": True,
                }
            }
        )
        x = _x(normalize_for_anthropic_strict(schema))
        assert not {"minItems", "maxItems", "uniqueItems"} & set(x)
        assert x["description"] == (
            "Must have at least 2 items. Must have at most 5 items. Items must be unique."
        )

    def test_unique_items_false_dropped_silently(self) -> None:
        schema = _obj({"x": {"type": "array", "items": {"type": "integer"}, "uniqueItems": False}})
        x = _x(normalize_for_anthropic_strict(schema))
        assert "uniqueItems" not in x
        assert "description" not in x

    def test_contains_family_stripped(self) -> None:
        schema = _obj(
            {
                "x": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "contains": {"const": 1},
                    "minContains": 1,
                    "maxContains": 2,
                }
            }
        )
        x = _x(normalize_for_anthropic_strict(schema))
        assert not {"contains", "minContains", "maxContains"} & set(x)
        assert "contains" in x["description"]

    def test_object_constraints_stripped(self) -> None:
        inner = _obj(
            {"a": {"type": "string"}},
            minProperties=1,
            maxProperties=2,
            propertyNames={"pattern": "^a$"},
            patternProperties={"^a": {"type": "string"}},
            dependentRequired={"a": []},
            dependentSchemas={"a": {"required": ["a"]}},
            unevaluatedProperties=False,
        )
        x = _x(normalize_for_anthropic_strict(_obj({"x": inner})))
        stripped = {
            "minProperties",
            "maxProperties",
            "propertyNames",
            "patternProperties",
            "dependentRequired",
            "dependentSchemas",
            "unevaluatedProperties",
        }
        assert not stripped & set(x)
        assert x["description"].startswith(
            "Must have at least 1 properties. Must have at most 2 properties."
        )

    def test_draft7_dependencies_stripped(self) -> None:
        inner = _obj({"a": {"type": "string"}}, dependencies={"a": ["a"]})
        x = _x(normalize_for_anthropic_strict(_obj({"x": inner})))
        assert "dependencies" not in x
        assert x["description"] == 'Constraint dependencies: {"a": ["a"]}.'

    def test_posix_class_pattern_does_not_warn(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalize_for_anthropic_strict(
                _obj({"x": {"type": "string", "pattern": "^[[:alpha:]]+$"}})
            )

    def test_not_stripped(self) -> None:
        schema = _obj({"x": {"type": "integer", "not": {"const": 3}}})
        x = _x(normalize_for_anthropic_strict(schema))
        assert "not" not in x
        assert x["description"] == 'Constraint not: {"const": 3}.'

    def test_kept_keywords(self) -> None:
        kept = {
            "type": "string",
            "minLength": 1,
            "maxLength": 5,
            "pattern": "^[a-z]+$",
            "format": "date",
            "default": "a",
            "title": "X",
            "examples": ["a"],
        }
        assert _x(normalize_for_anthropic_strict(_obj({"x": dict(kept)}))) == kept

    @pytest.mark.parametrize("fmt", ["binary", "path", "password", "ipvanyaddress", "json-string"])
    def test_unsupported_format_stripped(self, fmt: str) -> None:
        x = _x(normalize_for_anthropic_strict(_obj({"x": {"type": "string", "format": fmt}})))
        assert "format" not in x
        assert x["description"] == f"Format: {fmt}."

    @pytest.mark.parametrize(
        "fmt",
        [
            "date-time",
            "time",
            "date",
            "duration",
            "email",
            "hostname",
            "uri",
            "ipv4",
            "ipv6",
            "uuid",
        ],
    )
    def test_supported_format_kept(self, fmt: str) -> None:
        x = _x(normalize_for_anthropic_strict(_obj({"x": {"type": "string", "format": fmt}})))
        assert x["format"] == fmt

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$",  # pydantic Decimal: lookahead
            r"(?<=a)b",
            r"^(?P<n>ab)$",
            r"(?i)^ab$",
            r"^(a)\1$",
            r"\bfoo\b",
            r"^\p{L}+$",
            r"^[^c]{3}$",
            r"^\D{3}$",
            r"^.{3}$",
            r"^\W{2}$",
            r"^(unclosed",
        ],
    )
    def test_unsupported_pattern_stripped(self, pattern: str) -> None:
        x = _x(normalize_for_anthropic_strict(_obj({"x": {"type": "string", "pattern": pattern}})))
        assert "pattern" not in x
        assert x["description"] == f"Must match the regex pattern {json.dumps(pattern)}."

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^\d{3}-\d{4}$",
            r"^[A-Z]{2}\d{2}$",
            r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$",
            r"^(a|b){2,5}$",
            r"^\w{1,10}$",
            r"^\D+$",
            r"^[^c]+$",
            r"^(?:\d{3})$",
            r"^a\?$",
            r"^[0-9]+(\.[0-9]{1,2})?$",
            r"^[\d]{2}$",
        ],
    )
    def test_simple_pattern_kept(self, pattern: str) -> None:
        x = _x(normalize_for_anthropic_strict(_obj({"x": {"type": "string", "pattern": pattern}})))
        assert x["pattern"] == pattern

    def test_one_of_becomes_any_of_and_discriminator_dropped(self) -> None:
        schema = _obj(
            {
                "x": {
                    "oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}],
                    "discriminator": {"propertyName": "kind"},
                }
            },
            **{
                "$defs": {
                    "A": _obj({"kind": {"type": "string", "const": "a"}}),
                    "B": _obj({"kind": {"type": "string", "const": "b"}}),
                }
            },
        )
        x = _x(normalize_for_anthropic_strict(schema))
        assert "oneOf" not in x
        assert "discriminator" not in x
        assert x["anyOf"] == [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]

    def test_empty_items_dropped(self) -> None:
        schema = _obj(
            {"x": {"type": "array", "items": {}}, "y": {"type": "array", "items": {"title": "I"}}}
        )
        result = normalize_for_anthropic_strict(schema)
        assert result["properties"]["x"] == {"type": "array"}
        assert result["properties"]["y"] == {"type": "array"}

    def test_constraints_in_defs_and_any_of_variants(self) -> None:
        schema = _obj(
            {
                "x": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
                "y": {"$ref": "#/$defs/D"},
            },
            **{"$defs": {"D": _obj({"a": {"type": "integer", "maximum": 3}})}},
        )
        result = normalize_for_anthropic_strict(schema)
        assert _x(result)["anyOf"][0] == {"type": "integer", "description": "Must be >= 1."}
        assert result["$defs"]["D"]["properties"]["a"] == {
            "type": "integer",
            "description": "Must be <= 3.",
        }

    def test_pydantic_model(self) -> None:
        result = normalize_for_anthropic_strict(Invoice.model_json_schema())
        props = result["properties"]
        assert props["number"]["description"] == "Invoice number.\n\nMust be >= 1."
        assert "minimum" not in props["number"]
        assert props["amount"]["description"] == "Must be > 0."
        assert props["tags"]["description"] == "Must have at most 3 items."
        assert result["$defs"]["Address"]["properties"]["zip_code"]["pattern"] == r"^\d{5}$"
        assert result["required"] == Invoice.model_json_schema()["required"]

    def test_pydantic_decimal(self) -> None:
        class Price(BaseModel):
            value: Decimal

        result = normalize_for_anthropic_strict(Price.model_json_schema())
        variants = result["properties"]["value"]["anyOf"]
        assert all("pattern" not in v for v in variants)


# ------------------------------------------------------------------
# Compatibility
# ------------------------------------------------------------------


class TestCompatibility:
    def test_pydantic_model_compatible(self) -> None:
        assert is_anthropic_strict_compatible(Invoice.model_json_schema())
        assert anthropic_strict_problem(Invoice.model_json_schema()) is None

    def test_conlist_and_set_compatible(self) -> None:
        class M(BaseModel):
            a: conlist(int, min_length=2)  # type: ignore[valid-type]
            b: set[int]

        assert is_anthropic_strict_compatible(M.model_json_schema())

    def test_dict_field_incompatible(self) -> None:
        class M(BaseModel):
            labels: dict[str, str]

        problem = anthropic_strict_problem(M.model_json_schema())
        assert problem is not None and "labels" in problem
        assert not is_anthropic_strict_compatible(M.model_json_schema())

    def test_object_without_properties_incompatible(self) -> None:
        assert not is_anthropic_strict_compatible(_obj({"x": {"type": "object"}}))

    @pytest.mark.parametrize("extra", [True, {"type": "string"}])
    def test_additional_properties_not_false_incompatible(self, extra: Any) -> None:
        schema = _obj({"x": _obj({"a": {"type": "string"}}, additionalProperties=extra)})
        assert not is_anthropic_strict_compatible(schema)

    def test_additional_properties_false_compatible(self) -> None:
        schema = _obj({"x": _obj({"a": {"type": "string"}}, additionalProperties=False)})
        assert is_anthropic_strict_compatible(schema)

    def test_self_recursion_incompatible(self) -> None:
        class Node(BaseModel):
            value: int
            children: list[Node] = []

        problem = anthropic_strict_problem(Node.model_json_schema())
        assert problem is not None and "recursive" in problem

    def test_mutual_recursion_incompatible(self) -> None:
        schema = _obj(
            {"a": {"$ref": "#/$defs/A"}},
            **{
                "$defs": {
                    "A": _obj({"b": {"anyOf": [{"$ref": "#/$defs/B"}, {"type": "null"}]}}),
                    "B": _obj({"a": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}}),
                }
            },
        )
        assert not is_anthropic_strict_compatible(schema)

    def test_root_ref_incompatible(self) -> None:
        schema = _obj({"children": {"type": "array", "items": {"$ref": "#"}}})
        assert not is_anthropic_strict_compatible(schema)

    @pytest.mark.parametrize(
        "ref", ["https://example.com/s.json", "#/$defs/Missing", "#/properties/a"]
    )
    def test_unresolvable_ref_incompatible(self, ref: str) -> None:
        assert not is_anthropic_strict_compatible(_obj({"x": {"$ref": ref}}))

    def test_definitions_ref_compatible(self) -> None:
        schema = _obj(
            {"x": {"$ref": "#/definitions/P"}}, definitions={"P": _obj({"a": {"type": "string"}})}
        )
        assert is_anthropic_strict_compatible(schema)

    def test_untyped_property_incompatible(self) -> None:
        class M(BaseModel):
            anything: Any

        problem = anthropic_strict_problem(M.model_json_schema())
        assert problem is not None and "anything" in problem

    def test_list_of_any_compatible(self) -> None:
        class M(BaseModel):
            items: list[Any]

        assert is_anthropic_strict_compatible(M.model_json_schema())

    def test_const_and_enum_without_type_compatible(self) -> None:
        schema = _obj({"x": {"const": "a"}, "y": {"enum": ["a", 1, None]}})
        assert is_anthropic_strict_compatible(schema)

    def test_complex_enum_incompatible(self) -> None:
        assert not is_anthropic_strict_compatible(_obj({"x": {"enum": [{"a": 1}]}}))

    def test_complex_const_incompatible(self) -> None:
        problem = anthropic_strict_problem(_obj({"x": {"const": {"a": 1}}}))
        assert problem is not None and "const" in problem

    def test_tuple_incompatible(self) -> None:
        class M(BaseModel):
            pair: tuple[int, str]

        assert not is_anthropic_strict_compatible(M.model_json_schema())

    def test_one_of_with_any_of_incompatible(self) -> None:
        schema = _obj(
            {
                "x": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "oneOf": [{"type": "integer"}],
                }
            }
        )
        assert not is_anthropic_strict_compatible(schema)

    def test_union_limit(self) -> None:
        def unions(n: int) -> dict[str, Any]:
            return _obj(
                {f"p{i}": {"anyOf": [{"type": "integer"}, {"type": "null"}]} for i in range(n)}
            )

        assert is_anthropic_strict_compatible(unions(ANTHROPIC_STRICT_MAX_UNIONS))
        problem = anthropic_strict_problem(unions(ANTHROPIC_STRICT_MAX_UNIONS + 1))
        assert problem is not None and "union" in problem

    def test_type_array_counts_as_union(self) -> None:
        schema = _obj({f"p{i}": {"type": ["integer", "null"]} for i in range(17)})
        assert not is_anthropic_strict_compatible(schema)

    def test_optional_limit(self) -> None:
        def optional(n: int) -> dict[str, Any]:
            return _obj({f"p{i}": {"type": "integer"} for i in range(n)}, required=[])

        assert is_anthropic_strict_compatible(optional(ANTHROPIC_STRICT_MAX_OPTIONAL))
        problem = anthropic_strict_problem(optional(ANTHROPIC_STRICT_MAX_OPTIONAL + 1))
        assert problem is not None and "optional" in problem

    def test_pydantic_many_optional_fields_incompatible(self) -> None:
        fields = {f"f{i}": (Optional[int], None) for i in range(17)}  # noqa: UP045
        from pydantic import create_model

        model = create_model("Wide", **fields)  # type: ignore[call-overload]
        assert not is_anthropic_strict_compatible(model.model_json_schema())


class TestBudget:
    def test_counts(self) -> None:
        schema = _obj(
            {
                "a": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "b": {"type": "string"},
                "c": _obj(
                    {"d": {"type": ["string", "null"]}, "e": {"type": "integer"}}, required=[]
                ),
            },
            required=["b", "c"],
        )
        assert anthropic_strict_budget(schema) == (3, 2)

    def test_defs_counted_once(self) -> None:
        d = _obj({f"p{i}": {"anyOf": [{"type": "integer"}, {"type": "null"}]} for i in range(9)})
        schema = _obj(
            {"a": {"$ref": "#/$defs/D"}, "b": {"$ref": "#/$defs/D"}}, **{"$defs": {"D": d}}
        )
        assert anthropic_strict_budget(schema) == (0, 9)

    def test_unused_defs_not_counted(self) -> None:
        unused = _obj({f"p{i}": {"type": "integer"} for i in range(30)}, required=[])
        schema = _obj({"a": {"type": "integer"}}, **{"$defs": {"U": unused}})
        assert anthropic_strict_budget(schema) == (0, 0)


class TestFindOpenObject:
    def test_reports_path(self) -> None:
        schema = _obj({"outer": _obj({"inner": {"type": "object"}})})
        assert find_open_object(schema, "parameters") == "parameters.outer.inner"

    def test_none_when_closed(self) -> None:
        assert find_open_object(_obj({"a": {"type": "string"}}), "p") is None


class TestDraft7Definitions:
    SCHEMA: dict[str, Any] = _obj(  # noqa: RUF012
        {"x": {"$ref": "#/definitions/D"}}, definitions={"D": {"type": "object"}}
    )

    def test_find_open_object_walks_definitions(self) -> None:
        assert find_open_object(self.SCHEMA, "parameters") == "parameters.definitions.D"

    def test_open_object_in_definitions_incompatible(self) -> None:
        problem = anthropic_strict_problem(self.SCHEMA)
        assert problem is not None and "definitions.D" in problem

    def test_strict_tool_decoration_rejects_it(self) -> None:
        from dendrux.tool import tool

        with pytest.raises(TypeError, match="definitions.D"):

            @tool(strict=True, parameters=copy.deepcopy(self.SCHEMA))
            async def f(x: Any) -> None: ...
