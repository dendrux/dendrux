"""SQLite statement and expression quirks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from dendrux.runtime.stores.sql.base import SQLStateStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class SQLiteStateStore(SQLStateStore):
    """SQLite state store, scoped to one process or one shared filesystem host."""

    dialect_names = ("sqlite",)
    supports_update_returning = True
    supports_insert_returning = True
    supports_row_locks = False
    ddl_transactional = True

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
        memory = (
            engine.url.database in (None, "", ":memory:", "file::memory:")
            or engine.url.query.get("mode") == "memory"
        )
        self.capabilities = replace(
            self.capabilities,
            deployment="single_process" if memory else "single_host",
            notify=False,
        )

    def _insert(self, table: Any) -> Any:
        from sqlalchemy.dialects.sqlite import insert

        return insert(table)

    def insert_if_absent_stmt(
        self, table: Any, source: Any, *, conflict: Any, returning: Any = ()
    ) -> Any:
        from dendrux.runtime.stores.sql._types import InsertPlan

        plan = super().insert_if_absent_stmt(table, source, conflict=conflict)
        stmt = cast("Any", plan.statement).on_conflict_do_nothing(
            index_elements=conflict.target, index_where=conflict.predicate
        )
        if returning and self.supports_insert_returning:
            stmt = stmt.returning(*returning)
        return InsertPlan(stmt, "targeted_ignore")

    def upsert_setting(self, table: Any, *, key: str, value: str, updated_at: Any) -> Any:
        return (
            self._insert(table)
            .values(key=key, value=value, updated_at=updated_at)
            .on_conflict_do_update(
                index_elements=[table.c.key], set_={"value": value, "updated_at": updated_at}
            )
        )

    def _json_path(self, path: Any) -> str:
        import json

        result = "$"
        for part in path:
            if isinstance(part, int):
                result += f"[{part}]" if part >= 0 else f"[#{part}]"
            else:
                result += "." + json.dumps(part, ensure_ascii=False)
        return result

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        from sqlalchemy import JSON, Boolean, case, func, type_coerce

        path = self._json_path(path)
        kind = func.json_type(col, path)
        value = func.json_extract(col, path)
        if as_type == "json":
            encoded = case(
                (kind == "true", "true"), (kind == "false", "false"), else_=func.json_quote(value)
            )
            return type_coerce(encoded, JSON)
        if as_type == "str":
            return case((kind == "text", value)).collate("BINARY")
        if as_type == "int":
            return case((kind == "integer", value))
        if as_type == "bool":
            return type_coerce(case((kind == "true", True), (kind == "false", False)), Boolean)
        raise ValueError("unsupported JSON read type")

    def metadata_filter_predicate(self, col: Any, metadata_filter: dict[str, Any]) -> Any:
        from sqlalchemy import and_, func, true

        return and_(
            true(), *(func.json_extract(col, f"$.{k}") == v for k, v in metadata_filter.items())
        )

    def json_contains(self, col: Any, path: Any, value: Any) -> Any:
        import json

        from sqlalchemy import and_, false, func, literal, select

        target = self._json_path(path)
        if isinstance(value, (dict, list)):
            # Structural containment for object members/array values is handled
            # by a correlated json_tree comparison, preserving JSON types.
            wanted = func.json_tree(json.dumps(value)).table_valued("fullkey", "value", "type")
            actual = func.json_tree(func.json_extract(col, target)).table_valued(
                "fullkey", "value", "type"
            )
            missing = (
                select(literal(1))
                .select_from(wanted)
                .where(
                    ~select(literal(1))
                    .select_from(actual)
                    .where(
                        actual.c.fullkey == wanted.c.fullkey,
                        actual.c.type == wanted.c.type,
                        actual.c.value.is_not_distinct_from(wanted.c.value),
                    )
                    .exists()
                )
                .exists()
            )
            return func.coalesce(and_(func.json_type(col, target).is_not(None), ~missing), false())
        kind = (
            "null"
            if value is None
            else "true"
            if value is True
            else "false"
            if value is False
            else "text"
            if isinstance(value, str)
            else "integer"
            if isinstance(value, int)
            else "real"
        )
        return func.coalesce(
            and_(
                func.json_type(col, target) == kind,
                func.json_extract(col, target).is_not_distinct_from(value),
            ),
            false(),
        )

    def json_is_null(self, col: Any, path: Any) -> Any:
        from sqlalchemy import false, func

        return func.coalesce(func.json_type(col, self._json_path(path)) == "null", false())

    def json_patch_expr(self, col: Any, patch: Any) -> Any:
        import json

        from sqlalchemy import func

        from dendrux.runtime.stores.sql._types import AppendArray, SetFields

        base = func.coalesce(func.nullif(col, "null"), "{}")
        if isinstance(patch, SetFields):
            args = [
                item
                for key, value in patch.fields.items()
                for item in (self._json_path((key,)), func.json(json.dumps(value)))
            ]
            return func.json_set(base, *args) if args else base
        if isinstance(patch, AppendArray):
            return func.json_insert(
                base, self._json_path(patch.path) + "[#]", func.json(json.dumps(patch.value))
            )
        return None

    async def serialize_in_tx(self, session: Any, *, owner: Any) -> None:
        # The conditional note INSERT reserves the writer before its count subquery.
        return None

    async def reserve_parent_writer(self, session: Any, run_id: str) -> None:
        from sqlalchemy import text

        await session.execute(
            text("UPDATE agent_runs SET cancel_requested = cancel_requested WHERE id = :parent"),
            {"parent": run_id},
        )

    def id_in(self, col: Any, ids: Any) -> Any:
        import json

        from sqlalchemy import func, select

        values = func.json_each(json.dumps(sorted(set(ids)))).table_valued("value")
        return col.in_(select(values.c.value))

    def _frontier_scope(self, statement: Any, tenant_id: Any) -> Any:
        from sqlalchemy import String, bindparam, literal_column

        from dendrux.runtime.state import _check_scope, _Unscoped

        if isinstance(tenant_id, _Unscoped):
            return statement
        _check_scope(tenant_id)
        return statement.where(
            literal_column("+agent_runs.tenant_id").is_not_distinct_from(
                bindparam("frontier_tenant", tenant_id, type_=String())
            )
        )
