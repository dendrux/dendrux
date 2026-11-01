"""Everything a third-party store needs to write the dialect hooks comes from ``dendrux.store``."""

from __future__ import annotations

import pytest

import dendrux.store as public

HOOK_TYPES = (
    "ConflictSpec",
    "InsertPlan",
    "SelectSource",
    "ValuesSource",
    "SetFields",
    "AppendArray",
    "PatchLoopState",
    "JsonPatch",
    "JsonPath",
    "JsonReadType",
    "JsonValue",
    "ScalarKey",
    "OrderedAssignments",
    "RunLockScope",
    "RunParentCursor",
    "DialectFacts",
    "DbError",
    "DbErrorKind",
)


@pytest.mark.parametrize("name", HOOK_TYPES)
def test_each_hook_type_is_exported_from_dendrux_store(name: str) -> None:
    assert name in public.__all__
    assert getattr(public, name) is not None


def test_exported_hook_types_are_the_ones_the_base_class_uses() -> None:
    from dendrux.runtime.stores._errors import DbError
    from dendrux.runtime.stores.sql import _types

    assert public.DbError is DbError
    for name in ("ConflictSpec", "InsertPlan", "SelectSource", "SetFields", "AppendArray"):
        assert getattr(public, name) is getattr(_types, name)
