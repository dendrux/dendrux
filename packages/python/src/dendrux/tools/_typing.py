"""Compatibility annotations for Pydantic's tool argument models."""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import sys
import types
import typing
from typing import Annotated, Any, NotRequired, Required, get_args, get_origin, get_type_hints

from pydantic import AfterValidator
from typing_extensions import TypedDict, is_typeddict


def normalize_tool_hints(hints: dict[str, Any], signature: inspect.Signature) -> dict[str, Any]:
    if sys.version_info >= (3, 12):
        return hints
    normalizer = _Annotations()
    return {
        name: normalizer.annotation(hint, {})
        for name, hint in hints.items()
        if name in signature.parameters
        and name != "self"
        and signature.parameters[name].kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


_PYDANTIC_CLASS_HOOKS = frozenset(
    {"__pydantic_config__", "__get_pydantic_core_schema__", "__get_pydantic_json_schema__"}
)


def _typed_dict_class(cls: Any, fields: dict[str, Any], options: dict[str, Any]) -> Any:
    # Class creation preserves validator descriptors and their __set_name__
    # protocol; the functional TypedDict factory would discard those methods.
    namespace = {
        name: member
        for name, member in vars(cls).items()
        if not name.startswith("__") or name in _PYDANTIC_CLASS_HOOKS
    }
    namespace.update(
        __annotations__=fields,
        __module__=cls.__module__,
        __qualname__=cls.__qualname__,
        __doc__=cls.__doc__,
    )
    base: Any = TypedDict
    return types.new_class(cls.__name__, (base,), options, lambda ns: ns.update(namespace))


class _Annotations:
    def __init__(self) -> None:
        self.memo: dict[Any, Any] = {}
        self.active: set[Any] = set()

    def annotation(self, hint: Any, bindings: dict[Any, Any]) -> Any:
        if isinstance(hint, typing.TypeVar):
            return bindings.get(hint, hint)
        origin = get_origin(hint)
        if is_typeddict(origin):
            arguments = tuple(self.annotation(arg, bindings) for arg in get_args(hint))
            parameters = getattr(origin, "__parameters__", ())
            return self.typed_dict(
                origin, {**bindings, **dict(zip(parameters, arguments, strict=True))}
            )
        if is_typeddict(hint):
            return self.typed_dict(hint, bindings)
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            return self.dataclass(hint, bindings)
        if origin is None:
            return hint
        return self.compound(hint, origin, bindings)

    def compound(self, hint: Any, origin: Any, bindings: dict[Any, Any]) -> Any:
        arguments = get_args(hint)
        if origin is Annotated:
            return Annotated[self.annotation(arguments[0], bindings), *arguments[1:]]
        normalized = tuple(self.annotation(arg, bindings) for arg in arguments)
        if normalized == arguments:
            return hint
        if origin is types.UnionType:
            return typing.Union[normalized]  # noqa: UP007
        copier = getattr(hint, "copy_with", None)
        if copier is not None:
            return copier(normalized)
        return origin[normalized[0] if len(normalized) == 1 else normalized]

    def typed_dict(self, cls: Any, bindings: dict[Any, Any]) -> Any:
        key = (cls, tuple(bindings.items()))
        if key in self.memo:
            return self.memo[key]
        self.enter(key)
        hints = get_type_hints(cls, include_extras=True)
        if type(cls).__module__ != "typing" and not bindings:
            annotations = {name: self.annotation(hint, bindings) for name, hint in hints.items()}
            raw_extra = getattr(cls, "__extra_items__", None)
            if annotations == hints and self.annotation(raw_extra, bindings) == raw_extra:
                self.memo[key] = cls
                self.active.remove(key)
                return cls
        fields = {
            name: self.typed_field(hint, name in cls.__required_keys__, bindings)
            for name, hint in hints.items()
        }
        options = {"total": cls.__total__}
        for keyword, attribute in (("closed", "__closed__"), ("extra_items", "__extra_items__")):
            if hasattr(cls, attribute):
                options[keyword] = self.annotation(getattr(cls, attribute), bindings)
        compatible = _typed_dict_class(cls, fields, options)
        self.memo[key] = compatible
        self.active.remove(key)
        return compatible

    def typed_field(self, hint: Any, required: bool, bindings: dict[Any, Any]) -> Any:
        origin = get_origin(hint)
        if origin is Annotated:
            arguments = get_args(hint)
            qualified = self.typed_field(arguments[0], required, bindings)
            wrapped: Any = Annotated[get_args(qualified)[0], *arguments[1:]]
            return get_origin(qualified)[wrapped]
        if origin in (typing.Required, typing.NotRequired):
            required = origin is typing.Required
            hint = get_args(hint)[0]
        normalized = self.annotation(hint, bindings)
        return Required[normalized] if required else NotRequired[normalized]

    def enter(self, key: Any) -> None:
        if key in self.active:
            raise TypeError("Recursive tool argument types require an explicit schema")
        self.active.add(key)

    def dataclass(self, cls: Any, bindings: dict[Any, Any]) -> Any:
        key = (cls, tuple(bindings.items()))
        if key in self.memo:
            return self.memo[key]
        self.enter(key)
        all_hints = get_type_hints(cls, include_extras=True)
        dataclass_fields = dataclasses.fields(cls)
        hints = {field.name: all_hints[field.name] for field in dataclass_fields}
        annotations = {name: self.annotation(hint, bindings) for name, hint in hints.items()}
        if annotations == hints:
            compatible = cls
        else:
            params = cls.__dataclass_params__
            fields = [
                (field.name, annotations[field.name], copy.copy(field))
                for field in dataclass_fields
            ]
            clone = dataclasses.make_dataclass(
                cls.__name__,
                fields,
                bases=(cls,),
                namespace={"__module__": cls.__module__},
                frozen=params.frozen,
                init=params.init,
                repr=params.repr,
                eq=params.eq,
                order=params.order,
                unsafe_hash=params.unsafe_hash,
            )
            compatible = Annotated[
                clone, AfterValidator(functools.partial(_restore_dataclass, cls))
            ]
        self.memo[key] = compatible
        self.active.remove(key)
        return compatible


def _restore_dataclass(cls: type, value: Any) -> Any:
    result: Any = object.__new__(cls)
    if hasattr(result, "__dict__"):
        result.__dict__.update(value.__dict__)
    for field in dataclasses.fields(cls):
        object.__setattr__(result, field.name, getattr(value, field.name))
    return result
