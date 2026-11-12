"""MySQL 8.4 and MariaDB 10.11 relational store hooks."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import String
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement

from dendrux._meta_keys import (
    STORE_SESSION_KEY,
)
from dendrux.runtime.durability import _safe_database_call
from dendrux.runtime.state import _TERMINAL_STATUS_VALUES
from dendrux.runtime.stores.sql.base import SQLStateStore

_ERROR_WALK_LIMIT = 64
_JSON_WALK_LIMIT = 524288
_IN_BATCH = 1000

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.db._live_updates import LiveUpdates
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


def _packet_limit(facts: Any) -> int | None:
    """The server's ``max_allowed_packet`` when the capacity probe has run."""
    return None if facts is None else facts.max_allowed_packet


def _nested_errors(error: BaseException) -> Iterator[BaseException]:
    """``orig``, cause and context, in the order they are pushed on the walk stack.

    The getters are read eagerly, outside the generator, so an exception they
    raise propagates unchanged; only the ``isinstance`` test runs inside it.
    """
    nested = (getattr(error, "orig", None), error.__cause__, error.__context__)
    return (item for item in nested if isinstance(item, BaseException))


def _value_over_limit(value: str | bytes | bytearray | memoryview, limit: int) -> bool:
    if isinstance(value, str):
        return len(value) > limit or len(value.encode("utf-8", "surrogatepass")) > limit
    return (value.nbytes if isinstance(value, memoryview) else len(value)) > limit


def _nested_values(value: Any) -> Iterator[Any] | None:
    """Dict values and list/tuple items; keys and other containers are not inspected."""
    if isinstance(value, dict):
        return iter(value.values())
    if isinstance(value, (list, tuple)):
        return iter(value)
    return None


def _params_over_limit(params: Any, limit: int) -> bool:
    """True when a bound value is larger than ``limit``; inspects at most 524,288 values."""
    pending = [iter((params,))]
    inspected = 0
    while pending and inspected < _JSON_WALK_LIMIT:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        inspected += 1
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            if _value_over_limit(value, limit):
                return True
            continue
        nested = _nested_values(value)
        if nested is not None:
            pending.append(nested)
    return False


def _binds_over_limit(exc: BaseException, limit: int) -> bool:
    """Walk at most 64 distinct errors (context first) looking for an oversize bound value.

    A plain loop, not a generator: an exception raised by a metadata getter
    must propagate unchanged.
    """
    errors = [exc]
    seen: set[int] = set()
    while errors and len(seen) < _ERROR_WALK_LIMIT:
        error = errors.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        errors.extend(_nested_errors(error))
        if _params_over_limit(getattr(error, "params", None), limit):
            return True
    return False


class MySQLStateStore(SQLStateStore):
    """InnoDB store requiring READ COMMITTED and NO PAD identifier collations."""

    dialect_names = ("mysql", "mariadb")
    supports_update_returning = False
    supports_insert_returning = False
    supports_row_locks = True
    ddl_transactional = False

    def __init__(
        self, engine: AsyncEngine, *, live_updates: LiveUpdates | bool | None = None
    ) -> None:
        super().__init__(engine, live_updates=live_updates)
        self.capabilities = replace(self.capabilities, deployment="multi_server")

    def _waiting_sweep_page(
        self, statement: Any, after: tuple[datetime, str] | None, limit: int
    ) -> Any:
        from sqlalchemy import literal, select, union_all

        from dendrux.db.models import AgentRun

        # Waiting classification never uses the started-event lookup. Avoid
        # its correlated probe (and MariaDB expression cache) for each row.
        statement = statement.with_only_columns(
            *[
                literal(None).label("started") if c.key == "started" else c
                for c in statement.selected_columns
            ]
        )
        if after is None:
            return statement
        age, run_id = after
        same_time = (
            statement.where(AgentRun.status_changed_at == age, AgentRun.id > run_id)
            .order_by(None)
            .order_by(AgentRun.id)
            .subquery("same_time")
        )
        later_time = statement.where(AgentRun.status_changed_at > age).subquery("later_time")
        # LIMIT inside each derived table prevents merging it into an unbounded
        # status-prefix scan. The outer sort sees at most two bounded pages.
        window = union_all(select(same_time), select(later_time)).subquery("waiting_window")
        return select(window).order_by(window.c.sweep_age, window.c.id).limit(limit)

    def classify_db_error(self, exc: BaseException) -> Any:
        from dendrux.runtime.stores._errors import DbError

        classified = super().classify_db_error(exc)
        limit = _packet_limit(self._dialect_facts)
        if (
            classified.kind == "connection_lost"
            and limit is not None
            and _binds_over_limit(exc, limit)
        ):
            return DbError("statement_too_large", code=classified.code)
        return classified

    @classmethod
    def from_url(
        cls, url: str, *, live_updates: LiveUpdates | bool | None = None, **engine_options: Any
    ) -> MySQLStateStore:
        """Create a shared engine with connection-time UTC and READ COMMITTED."""
        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.db._engine_options import (
            create_engine_with_hint,
            mysql_engine_options,
            normalized_url,
        )

        options = mysql_engine_options(engine_options)
        return cls(
            create_engine_with_hint(create_async_engine, normalized_url(url), **options),
            live_updates=live_updates,
        )

    @_safe_database_call
    async def connection_checks(self, conn: Any) -> Any:
        from sqlalchemy import bindparam, text

        from dendrux.db._dialects._mysql_types import is_mariadb
        from dendrux.db.models import Base
        from dendrux.runtime.stores.sql._types import DialectFacts

        maria = is_mariadb(conn.dialect)
        version = tuple(conn.dialect.server_version_info or ())
        if version < ((10, 11) if maria else (8, 4)):
            raise ValueError("MySQLStateStore requires MySQL 8.4 or MariaDB 10.11 or newer")
        isolation_variable = "tx_isolation" if maria else "transaction_isolation"
        facts = (
            await conn.execute(
                text(
                    f"SELECT @@max_allowed_packet, @@session.{isolation_variable}, "  # noqa: S608 - Allow-list: isolation_variable is tx_isolation or transaction_isolation; tables are bound.
                    "(SELECT count(*) FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :tables "
                    "AND (ENGINE IS NULL OR ENGINE <> 'InnoDB')), "
                    "@@session.autocommit, @@session.sql_mode, "
                    "@@session.character_set_client, @@session.character_set_connection, "
                    "@@session.character_set_results"
                ).bindparams(bindparam("tables", expanding=True)),
                {"tables": sorted(Base.metadata.tables)},
            )
        ).one()
        if facts[3]:
            raise ValueError("MySQLStateStore requires transactions; AUTOCOMMIT is not supported")
        if str(facts[1]).replace("-", " ").upper() != "READ COMMITTED":
            raise ValueError(
                'MySQLStateStore requires an engine with isolation_level="READ COMMITTED"'
            )
        if facts[2]:
            raise ValueError("MySQLStateStore requires every Dendrux table to use InnoDB")
        if not {"STRICT_TRANS_TABLES", "STRICT_ALL_TABLES"}.intersection(
            str(facts[4]).upper().split(",")
        ):
            raise ValueError(
                "MySQLStateStore requires strict sql_mode "
                "(STRICT_TRANS_TABLES or STRICT_ALL_TABLES)"
            )
        if any(str(charset).lower() != "utf8mb4" for charset in facts[5:8]):
            raise ValueError(
                "MySQLStateStore requires utf8mb4 character_set_client/connection/results"
            )
        packet = int(facts[0])
        return DialectFacts(
            "mariadb" if maria else "mysql", version, {"supports_insert_returning": maria}, packet
        )

    @_safe_database_call
    async def statement_capacity(self) -> int | None:
        """Most bytes one SQL statement may carry (the server's ``max_allowed_packet``).

        The Agent compares its configured ``Limits`` with this before its first use of the
        store; the value is read once, with the first-use connection checks.
        """
        if self._dialect_facts is None:
            async with self._session_factory():
                pass
        facts = self._dialect_facts
        return None if facts is None else facts.max_allowed_packet

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

    @_safe_database_call
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
        session.info[STORE_SESSION_KEY] = self
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
        """Bind a frontier in chunks of 1,000; the caller's traversal bounds its size."""
        from sqlalchemy import false, or_

        if not ids:
            return false()
        return or_(
            *(col.in_(ids[start : start + _IN_BATCH]) for start in range(0, len(ids), _IN_BATCH))
        )

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
        from sqlalchemy import Boolean, and_, false, func, type_coerce

        node = func.json_extract(col, self._json_path(col, path))
        kind = func.json_type(node)
        matches_type = (
            kind == "NULL"
            if value is None
            else kind == "BOOLEAN"
            if isinstance(value, bool)
            else kind == "STRING"
            if isinstance(value, str)
            else kind.in_(("INTEGER", "UNSIGNED INTEGER", "DOUBLE", "DECIMAL"))
            if isinstance(value, (int, float))
            else kind.in_(("OBJECT", "ARRAY"))
        )

        return type_coerce(
            func.coalesce(
                and_(
                    matches_type,
                    _NoPadJSONText(func.json_unquote(node)) == value
                    if isinstance(value, str)
                    else func.json_contains(
                        col,
                        json.dumps(value, ensure_ascii=False, allow_nan=False),
                        self._json_path(col, path),
                    ),
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
        from sqlalchemy import case, func, literal

        from dendrux.db._dialects._mysql_types import json_document
        from dendrux.runtime.stores.sql._types import AppendArray, SetFields

        base = case((func.json_type(col) == "NULL", "{}"), else_=func.coalesce(col, "{}"))
        if isinstance(patch, SetFields):
            args = [
                item
                for key, value in patch.fields.items()
                for item in (
                    self._json_path(col, (key,)),
                    func.json_extract(literal(value, json_document(col.key)), "$"),
                )
            ]
            return func.json_set(base, *args) if args else base
        if isinstance(patch, AppendArray):
            return func.json_array_append(
                base,
                self._json_path(col, patch.path),
                func.json_extract(literal(patch.value, json_document(col.key)), "$"),
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
            if (row.idempotency_fingerprint or "") != fingerprint:
                raise IdempotencyConflictError(run_id=row.id, idempotency_key=idempotency_key)
            return CreateRunResult(
                run_id=row.id,
                outcome="existing_terminal"
                if row.status in _TERMINAL_STATUS_VALUES
                else "existing_active",
                status=RunStatus(row.status),
                storage_format=row.storage_format,
            )
