"""MySQL 8.4 and MariaDB 10.11 relational store hooks."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import String
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement

from dendrux.runtime.state import _TERMINAL_STATUS_VALUES
from dendrux.runtime.stores.sql.base import SQLStateStore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.runtime.state import ClaimResult, TenantScope

_increment_site: ContextVar[tuple[Any, str] | None] = ContextVar(
    "mysql_increment_site", default=None
)


class _NoPadJSONText(FunctionElement[str]):
    type = String()
    inherit_cache = True


@compiles(_NoPadJSONText)
def _compile_json_text(element: Any, compiler: Any, **kwargs: Any) -> str:
    from dendrux.db._dialects._mysql_types import nopad_collation

    return (
        f"({compiler.process(element.clauses, **kwargs)}) "
        f"COLLATE {nopad_collation(compiler.dialect)}"
    )


class MySQLStateStore(SQLStateStore):
    """InnoDB store requiring READ COMMITTED and NO PAD identifier collations."""

    dialect_names = ("mysql",)
    supports_update_returning = False
    supports_insert_returning = False
    supports_row_locks = True
    ddl_transactional = False

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
        self.capabilities = replace(self.capabilities, deployment="multi_server", notify=False)

    @classmethod
    def from_url(cls, url: str, **engine_options: Any) -> MySQLStateStore:
        """Create a shared engine with connection-time UTC and READ COMMITTED."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.db._engine_options import mysql_engine_options, normalized_url

        options = mysql_engine_options(engine_options)
        return cls(create_async_engine(normalized_url(url), **options))

    async def connection_checks(self, conn: Any) -> Any:
        from sqlalchemy import bindparam, text

        from dendrux.db._dialects._mysql_types import is_mariadb
        from dendrux.db.models import Base
        from dendrux.runtime.stores.sql._types import DialectFacts
        from dendrux.types import Limits

        maria = is_mariadb(conn.dialect)
        version = tuple(conn.dialect.server_version_info or ())
        if version < ((10, 11) if maria else (8, 4)):
            raise ValueError("MySQLStateStore requires MySQL 8.4 or MariaDB 10.11 or newer")
        isolation_variable = "tx_isolation" if maria else "transaction_isolation"
        facts = (
            await conn.execute(
                text(
                    f"SELECT @@max_allowed_packet, @@session.{isolation_variable}, "
                    "(SELECT count(*) FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :tables "
                    "AND (ENGINE IS NULL OR ENGINE <> 'InnoDB'))"
                ).bindparams(bindparam("tables", expanding=True)),
                {"tables": sorted(Base.metadata.tables)},
            )
        ).one()
        if str(facts[1]).replace("-", " ").upper() != "READ COMMITTED":
            raise ValueError(
                'MySQLStateStore requires an engine with isolation_level="READ COMMITTED"'
            )
        if facts[2]:
            raise ValueError("MySQLStateStore requires every Dendrux table to use InnoDB")
        # TODO: pass Agent's configured Limits through the store construction seam.
        limits = Limits()
        maximum = max(
            limits.max_input_bytes, limits.max_tool_result_bytes, limits.max_metadata_bytes
        )
        packet = int(facts[0])
        if maximum > packet:
            raise ValueError(
                f"Dendrux maximum payload {maximum} exceeds max_allowed_packet {packet}"
            )
        return DialectFacts(
            "mariadb" if maria else "mysql", version, {"supports_insert_returning": maria}, packet
        )

    async def claim_paused_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        expected_pause_id: str | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> ClaimResult | Literal[False]:
        """Claim with a locked integer readback saving scoped to this operation."""
        token = _increment_site.set((self, "claim_paused_run"))
        try:
            return await super().claim_paused_run(
                run_id,
                expected_status=expected_status,
                expected_pause_id=expected_pause_id,
                validate=validate,
            )
        finally:
            _increment_site.reset(token)

    async def reclaim_run(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        agent_name: str,
        expected_claim_epoch: int | None,
        governance: dict[str, Any],
        governance_policy: Literal["warn", "raise"],
        skill_pins: dict[str, str],
        loop_type: str,
    ) -> ClaimResult | Literal[False]:
        """Reuse the validated reclaim lock for the two plain integer increments."""
        token = _increment_site.set((self, "reclaim_run"))
        try:
            return await super().reclaim_run(
                run_id,
                tenant_id=tenant_id,
                agent_name=agent_name,
                expected_claim_epoch=expected_claim_epoch,
                governance=governance,
                governance_policy=governance_policy,
                skill_pins=skill_pins,
                loop_type=loop_type,
            )
        finally:
            _increment_site.reset(token)

    async def update_one(
        self,
        session: Any,
        *,
        table: Any,
        pk: Any,
        predicate: Any,
        values: Any,
        returning: Any,
        lock_projection: Any = (),
        validate_locked: Any = None,
    ) -> Any:
        """Save readback only at the two explicitly named locked increment sites."""
        from sqlalchemy import and_, select, update

        site = _increment_site.get()
        if site is None or site[0] is not self or table.name != "agent_runs":
            return await super().update_one(
                session,
                table=table,
                pk=pk,
                predicate=predicate,
                values=values,
                returning=returning,
                lock_projection=lock_projection,
                validate_locked=validate_locked,
            )
        columns = tuple(table.primary_key.columns)
        if len(pk) != len(columns) or not columns or not returning:
            raise ValueError("a complete primary key and returning projection are required")
        session.info["dendrux.store"] = self
        key = and_(*(col == value for col, value in zip(columns, pk, strict=True)))
        projection = list(dict.fromkeys((*columns, *lock_projection, *returning)))
        locked = (
            (
                await session.execute(
                    self.lock_clause(select(*projection).where(key, predicate), "update")
                )
            )
            .mappings()
            .first()
        )
        if locked is None:
            return None
        if validate_locked is not None:
            validate_locked(locked)
        primary_names = {col.name for col in columns}
        if any(col.name in primary_names for col, _ in values):
            raise ValueError("primary keys are immutable")
        increments = (
            {"claim_epoch"}
            if site[1] == "claim_paused_run"
            else {"claim_epoch", "reclaim_attempts"}
        )
        predicted: dict[str, Any] = {}
        can_predict = True
        assignments = {col.name: value for col, value in values}
        for col in returning:
            name = col.name
            column = table.c[name]
            if any(
                getattr(column, attr) is not None
                for attr in (
                    "onupdate",
                    "server_onupdate",
                    "computed",
                )
            ):
                can_predict = False
                break
            if name not in assignments:
                predicted[name] = locked[name]
            elif (
                name in increments
                and hasattr(assignments[name], "compare")
                and assignments[name].compare(table.c[name] + 1)
            ):
                predicted[name] = locked[name] + 1
            else:
                # Any future expression change retains the protected DB readback.
                can_predict = False
                break
        await session.execute(update(table).where(key).ordered_values(*values))
        if can_predict:
            return predicted
        return (await session.execute(select(*returning).where(key))).mappings().one()

    def id_in(self, col: Any, ids: Sequence[Any]) -> Any:
        """Bind complete frontiers in chunks within the subtree accounting bound."""
        from sqlalchemy import false, or_

        from dendrux.types import Budget

        if len(ids) > Budget.max_subtree_nodes:
            raise ValueError(
                f"MySQL frontier membership must not exceed {Budget.max_subtree_nodes} identifiers"
            )
        if not ids:
            return false()
        return or_(*(col.in_(ids[start : start + 1000]) for start in range(0, len(ids), 1000)))

    def idempotency_scope_predicate(self, tenant_id: str | None) -> Any:
        """Use the generated NO PAD scope for lookups and lost-ack proofs."""
        from dendrux.db._dialects.mysql import tenant_scope_column
        from dendrux.db.models import AgentRun

        return tenant_scope_column(AgentRun.__table__) == (tenant_id or "")

    def event_sequence_expr(self, next_index: Any) -> Any:
        """Expose MySQL's allocated sequence through its connection-local result ID."""
        from sqlalchemy import func

        return next_index if self.supports_insert_returning else func.last_insert_id(next_index)

    def inserted_event_sequence(self, result: Any) -> int | None:
        """Use the successful INSERT result, including sequence zero."""
        return int(result.lastrowid) if result.lastrowid is not None else None

    def _insert(self, table: Any) -> Any:
        from sqlalchemy.dialects.mysql import insert

        return insert(table)

    def _conflict(self, table: Any, names: Any, *constraints: str) -> Any:
        conflict = super()._conflict(table, names, *constraints)
        if list(names) == [column.name for column in table.primary_key.columns]:
            return replace(conflict, constraints=conflict.constraints | {"PRIMARY"})
        return conflict

    async def _execute_insert_plan(self, session: Any, plan: Any, conflict: Any) -> Any:
        from sqlalchemy.exc import IntegrityError

        try:
            return await session.execute(plan.statement)
        except IntegrityError as exc:
            error = self.classify_db_error(exc)
            if error.kind == "unique_violation" and error.constraint in conflict.constraints:
                return None
            raise

    def upsert_setting(self, table: Any, *, key: str, value: str, updated_at: Any) -> Any:
        return (
            self._insert(table)
            .values(key=key, value=value, updated_at=updated_at)
            .on_duplicate_key_update(value=value, updated_at=updated_at)
        )

    def _json_path(self, col: Any, path: Any) -> Any:
        from sqlalchemy import case, func, literal

        result: Any = literal("$")
        for part in path:
            if isinstance(part, int) and part < 0:
                result = case(
                    (
                        func.json_length(col, result) >= -part,
                        func.concat(result, "[", func.json_length(col, result) + part, "]"),
                    ),
                    else_=None,
                )
            else:
                result = func.concat(
                    result,
                    f"[{part}]"
                    if isinstance(part, int)
                    else "." + json.dumps(part, ensure_ascii=False),
                )
        return result

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        from sqlalchemy import JSON, Boolean, Integer, case, cast, func, type_coerce

        value = func.json_extract(col, self._json_path(col, path))
        kind = func.json_type(value)
        raw = func.json_unquote(value)
        if as_type == "json":
            return type_coerce(value, JSON)
        if as_type == "str":
            return _NoPadJSONText(case((kind == "STRING", raw)))
        if as_type == "int":
            return case((kind == "INTEGER", cast(raw, Integer)))
        if as_type == "bool":
            return type_coerce(case((kind == "BOOLEAN", raw == "true")), Boolean)
        raise ValueError("unsupported JSON read type")

    def json_contains(self, col: Any, path: Any, value: Any) -> Any:
        from sqlalchemy import Boolean, false, func, type_coerce

        return type_coerce(
            func.coalesce(
                func.json_contains(
                    col,
                    json.dumps(value, ensure_ascii=False, allow_nan=False),
                    self._json_path(col, path),
                ),
                false(),
            ),
            Boolean,
        )

    def json_is_null(self, col: Any, path: Any) -> Any:
        from sqlalchemy import Boolean, false, func, type_coerce

        return type_coerce(
            func.coalesce(
                func.json_type(func.json_extract(col, self._json_path(col, path))) == "NULL",
                false(),
            ),
            Boolean,
        )

    def json_patch_expr(self, col: Any, patch: Any) -> Any:
        from sqlalchemy import case, func

        from dendrux.runtime.stores.sql._types import AppendArray, SetFields

        base = case((func.json_type(col) == "NULL", "{}"), else_=func.coalesce(col, "{}"))
        if isinstance(patch, SetFields):
            args = [
                item
                for key, value in patch.fields.items()
                for item in (
                    self._json_path(col, (key,)),
                    func.json_extract(json.dumps(value, ensure_ascii=False, allow_nan=False), "$"),
                )
            ]
            return func.json_set(base, *args) if args else base
        if isinstance(patch, AppendArray):
            return func.json_array_append(
                base,
                self._json_path(col, patch.path),
                func.json_extract(
                    json.dumps(patch.value, ensure_ascii=False, allow_nan=False), "$"
                ),
            )
        return None

    async def _check_idempotency(
        self, idempotency_key: str, fingerprint: str, *, tenant_id: str | None = None
    ) -> Any:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun
        from dendrux.types import CreateRunResult, IdempotencyConflictError, RunStatus

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AgentRun).where(
                        self.idempotency_scope_predicate(tenant_id),
                        AgentRun.idempotency_key == idempotency_key,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            if row.idempotency_fingerprint != fingerprint:
                raise IdempotencyConflictError(run_id=row.id, idempotency_key=idempotency_key)
            return CreateRunResult(
                run_id=row.id,
                outcome="existing_terminal"
                if row.status in _TERMINAL_STATUS_VALUES
                else "existing_active",
                status=RunStatus(row.status),
                storage_format=row.storage_format,
            )
