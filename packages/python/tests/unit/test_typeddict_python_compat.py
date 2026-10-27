"""Standard TypedDict tool arguments behave alike on supported Python versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, ClassVar, Generic, NotRequired, Required, TypedDict, TypeVar

import pytest
from pydantic import Field

from dendrux.tool import get_tool_def, tool
from dendrux.tools._coerce import (
    ArgumentCoercionError,
    build_args_spec,
    build_call,
    get_args_spec,
    normalize_arguments,
)


@pytest.fixture(autouse=True, params=["native", "compatibility"])
def annotation_mode(request, monkeypatch):
    if request.param == "compatibility":
        from dendrux.tools import _typing

        # Exercise the compatibility implementation unchanged on all Pythons,
        # as well as each interpreter's native version-selection path.
        monkeypatch.setattr(_typing, "sys", SimpleNamespace(version_info=(3, 11)))


class Row(TypedDict):
    count: int


class Options(TypedDict, total=False):
    label: str
    mandatory: Required[int]
    row: NotRequired[Row]


class Inherited(Options):
    name: str


class Batch(TypedDict):
    rows: list[Row]
    options: Options


@dataclass
class Envelope:
    row: Row
    label: str = "default"


T = TypeVar("T")


class GenericRow(TypedDict, Generic[T]):
    value: T


def function_for(annotation):
    def apply(value):
        return value

    apply.__annotations__ = {"value": annotation}
    return apply


@pytest.mark.parametrize(
    "annotation,payload,expected",
    [
        (Row, {"count": "3"}, {"count": 3}),
        (list[Row], [{"count": "3"}], [{"count": 3}]),
        (dict[str, Row], {"one": {"count": "3"}}, {"one": {"count": 3}}),
        (Annotated[Row, "row details"], {"count": "3"}, {"count": 3}),
        (Row | None, None, None),
        (Row | None, {"count": "3"}, {"count": 3}),
        (
            Batch,
            {"rows": [{"count": "3"}], "options": {"mandatory": "4"}},
            {"rows": [{"count": 3}], "options": {"mandatory": 4}},
        ),
        (Inherited, {"mandatory": "4", "name": "batch"}, {"mandatory": 4, "name": "batch"}),
    ],
)
def test_stdlib_typeddict_decorates_and_coerces(annotation, payload, expected):
    fn = tool()(function_for(annotation))
    spec = get_args_spec(fn)
    normalized = normalize_arguments(spec, {"value": payload})
    assert normalized == {"value": expected}
    args, kwargs = build_call(spec, normalized)
    assert fn(*args, **kwargs) == expected
    assert get_tool_def(fn).parameters["type"] == "object"
    assert fn.__annotations__["value"] is annotation, "never mutate developer annotations"


def test_required_notrequired_and_constraints_are_retained():
    fn = tool()(function_for(Options))
    spec = get_args_spec(fn)
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {}})
    assert normalize_arguments(spec, {"value": {"mandatory": 2}}) == {"value": {"mandatory": 2}}
    constrained = build_args_spec(function_for(Annotated[list[Row], Field(min_length=1)]))
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(constrained, {"value": []})


def test_typeddict_inside_dataclass_keeps_the_declared_runtime_type():
    fn = tool()(function_for(Envelope))
    spec = get_args_spec(fn)
    normalized = normalize_arguments(spec, {"value": {"row": {"count": "3"}}})
    args, kwargs = build_call(spec, normalized)
    value = fn(*args, **kwargs)
    assert type(value) is Envelope
    assert value.row == {"count": 3}
    assert value.label == "default"


@pytest.mark.parametrize("argument,raw,expected", [(int, "3", 3), (str, "three", "three")])
def test_generic_typeddict_specialization_at_argument_model_boundary(argument, raw, expected):
    # Generic aliases already work at Pydantic's arguments boundary; public
    # tool schema generation does not advertise generic TypedDict aliases.
    spec = build_args_spec(function_for(GenericRow[argument]))
    normalized = normalize_arguments(spec, {"value": {"value": raw}})
    assert normalized == {"value": {"value": expected}}
    assert build_call(spec, normalized)[1] == normalized


class AnnotatedOptions(TypedDict, total=False):
    count: Annotated[Required[int], Field(gt=0)]
    note: Annotated[NotRequired[str], Field(min_length=2)]


def test_annotated_presence_qualifiers_are_kept_at_the_pydantic_boundary():
    spec = build_args_spec(function_for(AnnotatedOptions))
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {}})
    assert normalize_arguments(spec, {"value": {"count": "2"}}) == {"value": {"count": 2}}
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {"count": 0}})
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {"count": 1, "note": "x"}})


class RecursiveReply(TypedDict):
    value: int
    children: list[RecursiveReply]


def test_return_and_ignored_variadic_annotations_are_not_argument_models():
    def apply(value: int, *args: RecursiveReply, **kwargs: RecursiveReply) -> RecursiveReply:
        return {"value": value, "children": []}

    decorated = tool()(apply)
    spec = get_args_spec(decorated)
    assert normalize_arguments(spec, {"value": "3"}) == {"value": 3}
    assert tuple(get_tool_def(decorated).parameters["properties"]) == ("value",)


@dataclass(slots=True, frozen=True)
class SlottedEnvelope:
    row: Row
    tags: list[str] = field(default_factory=list)
    cache: ClassVar[RecursiveReply | None] = None


def test_dataclass_slots_defaults_and_ignored_classvars_are_preserved():
    fn = tool()(function_for(SlottedEnvelope))
    spec = get_args_spec(fn)
    normalized = normalize_arguments(spec, {"value": {"row": {"count": "3"}}})
    args, kwargs = build_call(spec, normalized)
    value = fn(*args, **kwargs)
    assert type(value) is SlottedEnvelope
    assert value.row == {"count": 3}
    assert value.tags == []
    assert "cache" not in normalized["value"]


def test_existing_extension_typeddict_validators_are_not_replaced():
    from pydantic import field_validator
    from typing_extensions import TypedDict as ExtensionTypedDict

    class Checked(ExtensionTypedDict):
        count: int

        @field_validator("count")
        @classmethod
        def positive(cls, value):
            if value <= 0:
                raise ValueError("count must be positive")
            return value

    spec = build_args_spec(function_for(Checked))
    assert spec.model.model_fields["a0"].annotation is Checked
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {"count": 0}})
    assert normalize_arguments(spec, {"value": {"count": "2"}}) == {"value": {"count": 2}}


def test_stdlib_typeddict_validator_is_preserved_when_conversion_is_needed():
    from pydantic import field_validator

    class Checked(TypedDict):
        count: int

        @field_validator("count")
        @classmethod
        def positive(cls, value):
            if value <= 0:
                raise ValueError("count must be positive")
            return value

    spec = build_args_spec(function_for(Checked))
    with pytest.raises(ArgumentCoercionError):
        normalize_arguments(spec, {"value": {"count": 0}})
    assert normalize_arguments(spec, {"value": {"count": "2"}}) == {"value": {"count": 2}}
