"""PostgreSQL statement and expression quirks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from dendrux.runtime.stores.sql.base import SQLStateStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresStateStore(SQLStateStore):
    """Durable, multi-server state store for PostgreSQL."""

    dialect_names = ("postgresql",)
    supports_update_returning = True
    supports_insert_returning = True
    supports_row_locks = True
    ddl_transactional = True

    def __init__(self, engine: AsyncEngine, *, notify_channel: str | None = None) -> None:
        from dendrux.db._notify import validate_channel

        if notify_channel is not None:
            validate_channel(notify_channel)
        super().__init__(engine)
        self.notify_channel = notify_channel
        self.capabilities = replace(self.capabilities, deployment="multi_server")

    def _insert(self, table: Any) -> Any:
        from sqlalchemy.dialects.postgresql import insert

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

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        from sqlalchemy import Boolean, Integer, case, cast, func
        from sqlalchemy.dialects.postgresql import JSONB

        node: Any = cast(col, JSONB)
        for part in path:
            node = node[part]
        if as_type == "json":
            return node
        kind = func.jsonb_typeof(node)
        if as_type == "str":
            return case((kind == "string", node.astext)).collate("C")
        if as_type == "int":
            return case(
                (
                    (kind == "number") & node.astext.op("~")(r"^-?[0-9]+$"),
                    cast(node.astext, Integer),
                )
            )
        if as_type == "bool":
            return case((kind == "boolean", cast(node.astext, Boolean)))
        raise ValueError("unsupported JSON read type")

    def metadata_filter_predicate(self, col: Any, metadata_filter: dict[str, Any]) -> Any:
        from sqlalchemy import literal
        from sqlalchemy.dialects.postgresql import JSONB

        return _jsonb_or_null(col).contains(literal(metadata_filter, JSONB))

    def json_contains(self, col: Any, path: Any, value: Any) -> Any:
        from sqlalchemy import false, func, literal
        from sqlalchemy.dialects.postgresql import JSONB

        return func.coalesce(
            self.json_get(_jsonb_or_null(col), path, "json").contains(literal(value, JSONB)),
            false(),
        )

    def pause_projection(self, col: Any) -> list[Any]:
        from dendrux.runtime.state import PAUSE_SUMMARY_KEYS

        doc = _readable_json_document(col)
        return [doc[key] for key in PAUSE_SUMMARY_KEYS] + [doc["steps"][-1]]

    def pause_summary_query(self, run_id: str, tenant_scope: Any, *, pending: bool) -> Any:
        if not pending:
            return super().pause_summary_query(run_id, tenant_scope, pending=False)
        from sqlalchemy import cast, func, select
        from sqlalchemy.dialects.postgresql import JSONB

        from dendrux.db.models import AgentRun, RunEvent
        from dendrux.runtime.state import PAUSE_SUMMARY_KEYS, _apply_tenant_scope

        sequence = (
            select(func.max(RunEvent.sequence_index))
            .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
            .scalar_subquery()
        )
        inner = (
            _apply_tenant_scope(
                select(
                    AgentRun.status.label("status"),
                    sequence.label("seq"),
                    AgentRun.pii_mapping.label("pii"),
                    cast(AgentRun.pause_data, JSONB).label("doc"),
                ).where(AgentRun.id == run_id),
                tenant_scope,
            )
            .offset(0)
            .subquery()
        )
        doc = inner.c.doc
        return select(
            inner.c.status,
            inner.c.seq,
            inner.c.pii,
            *[doc[key] for key in PAUSE_SUMMARY_KEYS],
            doc["steps"][-1],
        )

    def json_conversion_error(self, exc: BaseException) -> bool:
        return self.classify_db_error(exc).code in ("22P02", "22P05")

    def json_is_null(self, col: Any, path: Any) -> Any:
        from sqlalchemy import false, func, literal
        from sqlalchemy.dialects.postgresql import JSONB

        return func.coalesce(self.json_get(col, path, "json") == literal(None, JSONB), false())

    def lock_clause(self, stmt: Any, intent: Any) -> Any:
        return stmt.with_for_update(read=intent == "share", key_share=intent == "update_no_key")

    async def serialize_in_tx(self, session: Any, *, owner: Any) -> None:
        from sqlalchemy import Integer, func, literal, select

        from dendrux.runtime.state import _NOTE_LOCK_NAMESPACE

        await session.execute(
            select(
                func.pg_advisory_xact_lock(
                    literal(_NOTE_LOCK_NAMESPACE, Integer()), func.hashtext(owner.run_id)
                )
            )
        )

    def id_in(self, col: Any, ids: Any) -> Any:
        from sqlalchemy import String, any_, bindparam
        from sqlalchemy.dialects.postgresql import ARRAY

        return col == any_(
            bindparam("frontier", sorted(set(ids)), type_=ARRAY(String()), unique=True)
        )

    def blob_now(self) -> Any:
        from sqlalchemy import func

        return func.now()

    async def _after_event_insert(self, session: Any, run_id: str) -> None:
        if self.notify_channel is not None:
            from sqlalchemy import text

            await session.execute(
                text("SELECT pg_notify(:channel, :run_id)"),
                {"channel": self.notify_channel, "run_id": run_id},
            )

    def json_patch_expr(self, col: Any, patch: Any) -> Any:
        from sqlalchemy import Text, case, func, literal, select
        from sqlalchemy import cast as sql_cast
        from sqlalchemy.dialects.postgresql import JSONB, array

        from dendrux.runtime.stores.sql._types import AppendArray, PatchLoopState, SetFields

        base = func.coalesce(
            func.nullif(sql_cast(col, JSONB), literal(None, JSONB)), literal({}, JSONB)
        )
        if isinstance(patch, SetFields):
            return base.op("||")(literal(dict(patch.fields), JSONB))
        if isinstance(patch, AppendArray):
            path = array([str(part) for part in patch.path], type_=Text)
            current = self.json_get(base, patch.path, "json")
            return func.jsonb_set(
                base,
                path,
                func.coalesce(current, literal([], JSONB)).op("||")(literal([patch.value], JSONB)),
                True,
            )
        if not isinstance(patch, PatchLoopState):
            return None
        count_tool = patch.count_tool
        patch = patch.patch
        from sqlalchemy import Text
        from sqlalchemy.dialects.postgresql import JSONB, aggregate_order_by, array

        base = func.coalesce(col, literal({}, JSONB))
        changed: Any = base
        if patch.remove_pending or patch.set_call_phase:
            pending = (
                func.jsonb_array_elements(func.coalesce(base["pending"], literal([], JSONB)))
                .table_valued("value", with_ordinality="ordinality")
                .alias("pending_call")
            )
            value = sql_cast(pending.c.value, JSONB)
            phase_value: Any = value
            if patch.set_call_phase:
                phase_value = case(
                    *[
                        (
                            value["call_id"].astext == call_id,
                            value.op("||")(literal({"phase": phase}, JSONB)),
                        )
                        for call_id, phase in patch.set_call_phase.items()
                    ],
                    else_=value,
                )
            kept = (
                select(
                    func.coalesce(
                        func.jsonb_agg(aggregate_order_by(phase_value, pending.c.ordinality)),
                        literal([], JSONB),
                    )
                )
                .where(value["call_id"].astext.not_in(patch.remove_pending))
                .select_from(pending)
                .scalar_subquery()
            )
            changed = func.jsonb_set(changed, array(["pending"], type_=Text), kept, True)
        fields = dict(patch.set_fields)
        if patch.set_phase is not None:
            if patch.set_phase == "model_next" and patch.remove_pending:
                phase = case(
                    (func.jsonb_array_length(kept) == 0, literal("model_next")),
                    else_=base["phase"].as_string(),
                )
                changed = func.jsonb_set(
                    changed,
                    array(["phase"], type_=Text),
                    func.to_jsonb(sql_cast(phase, Text)),
                    True,
                )
            else:
                fields["phase"] = patch.set_phase
        if fields:
            changed = changed.op("||")(literal(fields, JSONB))
        if count_tool is not None:
            counts: Any = func.coalesce(base["tool_counts"], literal({}, JSONB))
            count = func.coalesce(base["tool_counts"][count_tool].as_integer(), 0) + 1
            counts = counts.op("||")(func.jsonb_build_object(literal(count_tool, Text), count))
            changed = func.jsonb_set(changed, array(["tool_counts"], type_=Text), counts, True)
        return changed

    def child_page_query(
        self, *, runs: Any, parents: Any, tenant_scope: Any, page_size: int
    ) -> Any:
        from sqlalchemy import String, bindparam, func, select, true

        from dendrux.db.models import AgentRun

        parent_ids = [parent.run_id for parent in parents]
        cursors = {parent.after for parent in parents}
        if len(cursors) != 1:
            return super().child_page_query(
                runs=runs, parents=parents, tenant_scope=tenant_scope, page_size=page_size
            )
        after = next(iter(cursors))
        tenant_id = tenant_scope
        size = page_size
        from sqlalchemy.dialects.postgresql import ARRAY

        parents = (
            func.unnest(bindparam("frontier", sorted(set(parent_ids)), type_=ARRAY(String())))
            .table_valued("id")
            .render_derived(name="parents")
        )
        # Fence parent selection before the cursor/sort: otherwise even LATERAL
        # can choose a history-wide primary-key range scan for every parent.
        scoped = (
            self._frontier_scope(
                select(AgentRun.id).where(AgentRun.parent_run_id == parents.c.id), tenant_id
            )
            .correlate(parents)
            .offset(0)
            .subquery("scoped_children")
        )
        per_parent = select(scoped.c.id)
        if after is not None:
            per_parent = per_parent.where(scoped.c.id > after)
        children = per_parent.order_by(scoped.c.id).limit(size).lateral("children")
        statement = (
            select(children.c.id)
            .select_from(parents.join(children, true()))
            .order_by(children.c.id)
            .limit(size)
        )
        return statement


def _jsonb_or_null(column: Any) -> Any:
    """``column::jsonb``, or NULL for a stored document ``jsonb`` cannot hold.

    The ``json`` column accepts ``\\u0000`` and unpaired surrogate escapes;
    converting such a row raises and fails the whole statement, so one row
    would break a scan for every tenant. The text is checked first: escaped
    backslashes and valid surrogate pairs are removed, and any ``\\u0000`` or
    surrogate escape left means the row is not convertible (it matches no
    filter). Rows without any ``\\u`` escape skip the check.
    """
    from sqlalchemy import Text, case, cast, func
    from sqlalchemy.dialects.postgresql import JSONB

    raw = cast(column, Text)
    no_backslash_pairs = func.regexp_replace(raw, r"\\\\", "", "g")
    no_pairs = func.regexp_replace(
        no_backslash_pairs,
        r"\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}",
        "",
        "g",
    )
    unconvertible = no_pairs.op("~")(r"\\u(0000|[dD][89a-fA-F])")
    return case(
        (func.strpos(raw, "\\u") == 0, cast(column, JSONB)),
        (~unconvertible, cast(column, JSONB)),
        else_=None,
    )


def _readable_json_document(column: Any) -> Any:
    from sqlalchemy import JSON, Text, cast, func

    # Older plain-JSON columns can contain escaped NUL. Preserve escaped
    # backslashes while replacing only an active NUL escape before extraction.
    cleaned = func.regexp_replace(
        cast(column, Text), r"((?:^|[^\\])(?:\\\\)*)\\u0000", r"\1\\ufffd", "g"
    )
    return cast(cleaned, JSON)
