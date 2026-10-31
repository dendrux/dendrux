"""Frozen relational hook inputs shared by SQL store adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.sql import Executable
    from sqlalchemy.sql.elements import ColumnElement

    from dendrux.runtime.state import LoopStatePatch, TenantScope

ScalarKey: TypeAlias = str | int | bytes | datetime
JsonPath: TypeAlias = tuple[str | int, ...]
JsonReadType: TypeAlias = Literal["json", "str", "int", "bool"]
JsonValue: TypeAlias = Any
ValuesSource: TypeAlias = Mapping[str, Any]
OrderedAssignments: TypeAlias = Sequence[tuple["ColumnElement[Any]", Any]]


@dataclass(frozen=True)
class SelectSource:
    """An INSERT SELECT source with an explicit target mapping."""

    columns: Sequence[str]
    select: Select[Any]


@dataclass(frozen=True)
class ConflictSpec:
    """The only conflict an idempotent insert is permitted to absorb."""

    target: Sequence[ColumnElement[Any]]
    predicate: ColumnElement[bool] | None = None
    constraints: frozenset[str] = frozenset()


@dataclass(frozen=True)
class InsertPlan:
    """A dialect statement and its targeted conflict handling strategy."""

    statement: Executable
    conflict_mode: Literal["targeted_ignore", "raise_unique"]


@dataclass(frozen=True)
class SetFields:
    """Set literal object members, including members whose names contain dots."""

    fields: Mapping[str, JsonValue]


@dataclass(frozen=True)
class AppendArray:
    """Append one JSON value to the array at a literal path."""

    path: JsonPath
    value: JsonValue


@dataclass(frozen=True)
class PatchLoopState:
    """Apply a bounded loop patch and optionally increment a tool count."""

    patch: LoopStatePatch
    count_tool: str | None = None


JsonPatch: TypeAlias = SetFields | AppendArray | PatchLoopState


@dataclass(frozen=True)
class RunLockScope:
    """Owning run identity and an explicit tenant scope for serialization."""

    run_id: str
    tenant_scope: TenantScope


@dataclass(frozen=True)
class RunParentCursor:
    """One parent and its immutable child after-ID."""

    run_id: str
    after: str | None = None


@dataclass(frozen=True)
class DialectFacts:
    """First-connection facts resolving independent feature defaults."""

    server_family: str
    server_version: tuple[int, ...] = ()
    features: Mapping[str, bool] = field(default_factory=dict)
    max_allowed_packet: int | None = None
