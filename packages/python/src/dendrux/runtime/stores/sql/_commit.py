"""Read back ambiguous commits without changing the transaction's retry identity."""

from __future__ import annotations

import datetime as dt
import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from dendrux import _codes
from dendrux._meta_keys import (
    BLOCKED_META_KEY,
    BLOCKED_RUNS_SESSION_KEY,
    COMMIT_PROOFS_SESSION_KEY,
    FRESH_EVIDENCE_SESSION_KEY,
    TOUCHED_RUNS_SESSION_KEY,
)
from dendrux.db.models import AgentRun
from dendrux.errors import PersistenceError, StoreOutcomeUnknown
from dendrux.runtime.durability import (
    _CommitNotAppliedError,
    _is_driver_failure,
    is_transient_db_error,
)
from dendrux.runtime.state import _to_aware_utc
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.stores.sql.base import SQLStateStore

logger = logging.getLogger(__name__)
_TRANSITIONS = frozenset(
    {
        "finalize_run",
        "finalize_run_if_status_in",
        "finalize_with_event",
        "try_pause_run",
        "pause_with_event",
        "pause_run",
        "consume_submission",
    }
)
_EVIDENCE = frozenset(
    {
        "save_trace",
        "save_tool_call",
        "save_usage",
        "save_llm_interaction",
        "save_run_event",
        "start_tool_call",
        "complete_tool_call",
        "consume_submission",
    }
)
_RETRY_IDS = {
    "save_trace": "trace_id",
    "save_tool_call": "row_id",
    "save_usage": "usage_id",
    "save_llm_interaction": "interaction_id",
    "save_run_event": "event_id",
    "start_tool_call": "row_id",
}


def _comparable(actual: Any, value: Any) -> tuple[Any, Any]:
    if isinstance(actual, Decimal) and value is not None:
        value = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    if isinstance(actual, dt.datetime) and isinstance(value, dt.datetime):
        actual, value = _to_aware_utc(actual), _to_aware_utc(value)
    if isinstance(actual, (dict, list)) or isinstance(value, (dict, list)):
        from dendrux.runtime._digest import content_digest

        actual, value = content_digest(actual), content_digest(value)
    return actual, value


class _CommitReadback:
    def __init__(self, store: SQLStateStore, session: AsyncSession, operation: Any) -> None:
        self.store, self.session = store, session
        self.name: str
        self.arguments: dict[str, Any]
        context = operation if operation is not None else (None, "", {})
        self.name = context[1]
        self.arguments = dict(context[2])
        self.run_id = self.arguments.get("run_id")
        self.fresh = session.info.pop(FRESH_EVIDENCE_SESSION_KEY, False)
        self.epoch = self.arguments.get("claim_epoch")
        self.prove_fence = self.name in _TRANSITIONS and self.epoch is not None and not self.fresh
        self.proofs = self._proofs(session.info.pop(COMMIT_PROOFS_SESSION_KEY, []))
        self.blocked = session.info.pop(BLOCKED_RUNS_SESSION_KEY, ())

    def unknown(self) -> NoReturn:
        _codes.log_code(logger, _codes.STORE_OUTCOME_UNKNOWN, run_id=self.run_id)
        raise StoreOutcomeUnknown() from None

    def _proofs(self, proofs: list[Any]) -> list[Any]:
        if self.name in _EVIDENCE:
            evidence = [proof for proof in proofs if proof[0].name != "agent_runs"]
            if evidence and not self.prove_fence:
                return evidence
        return proofs

    @staticmethod
    def _database_error(error: BaseException) -> bool:
        return isinstance(error, Exception) and _CommitReadback._database_exception(error)

    @staticmethod
    def _database_exception(error: Exception) -> bool:
        return isinstance(
            error, (SQLAlchemyError, OSError, PersistenceError)
        ) or _is_driver_failure(error)

    def _needs_readback(self, error: Exception, classified: Any) -> bool:
        # A raw OS failure (including asyncio's bare timeout) during COMMIT
        # carries no proof that the transaction was rejected.
        if isinstance(error, OSError):
            return True
        return classified.kind not in (
            "deadlock_or_serialization",
            "lock_timeout",
        ) and is_transient_db_error(error, classify=self.store.classify_db_error)

    async def _rollback(self, *, ambiguous: bool) -> None:
        try:
            await self.session.rollback()
        except BaseException as error:
            if not self._database_error(error):
                self._reraise_clean(error)
            _codes.log_code(logger, _codes.PERSISTENCE_ERROR, run_id=self.run_id)
            if ambiguous:
                self.unknown()
            raise PersistenceError() from None

    async def resolve(self, error: Exception) -> None:
        if isinstance(error, PersistenceError) or not self._database_error(error):
            raise error
        classified = self.store.classify_db_error(error)
        await self._rollback(
            ambiguous=isinstance(error, OSError) or classified.kind == "connection_lost"
        )
        if not self._needs_readback(error, classified):
            raise error
        if self.name in (
            "claim_paused_run",
            "submit_and_claim",
            "reclaim_run",
            "release_run_claim",
        ):
            self.unknown()
        if not self.proofs:
            self.unknown()
        await self._prove()

    async def _prove(self) -> None:
        try:
            await self.store._prepare_commit_readback(self.session)
            await self._lock_touched()
            for run_id in self.blocked:
                await self._check_proof(
                    AgentRun.__table__,
                    (AgentRun.id == run_id)
                    & self.store.json_get(AgentRun.meta, (BLOCKED_META_KEY,), "bool").is_(True),
                    {"id": run_id},
                )
            fenced = False
            for table, predicate, expected in self.proofs:
                check_epoch = self.prove_fence and table.name == "agent_runs"
                if check_epoch:
                    expected = {**expected, "claim_epoch": self.epoch}
                await self._check_proof(table, predicate, expected)
                fenced |= check_epoch
            if self.name in _TRANSITIONS and not self.fresh and not fenced:
                self.unknown()
        except _CommitNotAppliedError:
            raise
        except BaseException as error:
            self._readback_failed(error)

    def _readback_failed(self, error: BaseException) -> NoReturn:
        if self._database_error(error):
            self.unknown()
        self._reraise_clean(error)

    @staticmethod
    def _reraise_clean(error: BaseException) -> NoReturn:
        # A logic failure or cancellation during readback must not retain the
        # COMMIT driver's credentials through Python's implicit exception chain.
        error.__context__ = None
        error.__cause__ = None
        raise error from None

    async def _lock_touched(self) -> None:
        for run_id in sorted(self.session.info[TOUCHED_RUNS_SESSION_KEY]):
            if not self.store.supports_row_locks:
                await self.store.reserve_parent_writer(self.session, run_id)
            await self.session.execute(
                self.store.lock_clause(select(AgentRun.id).where(AgentRun.id == run_id), "share")
            )

    async def _check_proof(self, table: Any, predicate: Any, expected: dict[str, Any]) -> None:
        stored = (
            (
                await self.session.execute(
                    self.store.lock_clause(
                        select(*[table.c[key] for key in expected]).where(predicate), "share"
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if stored is None:
            await self._lost(absent=True)
        for key, value in expected.items():
            actual, value = _comparable(stored[key], value)
            if actual != value:
                await self._lost()

    async def _lost(self, *, absent: bool = False) -> NoReturn:
        retry_id = _RETRY_IDS.get(self.name)
        if absent and (
            self.name == "create_run"
            or (retry_id is not None and self.arguments.get(retry_id) is not None)
        ):
            raise _CommitNotAppliedError() from None
        if self.name not in _TRANSITIONS or self.run_id is None:
            self.unknown()
        if (
            await self.session.scalar(select(AgentRun.id).where(self._retry_predicate()))
            is not None
        ):
            raise _CommitNotAppliedError() from None
        self.unknown()

    def _retry_predicate(self) -> Any:
        predicate = AgentRun.id == self.run_id
        if self.epoch is not None:
            predicate &= AgentRun.claim_epoch == self.epoch
        predicate = self._status_predicate(predicate)
        if self.name == "consume_submission":
            predicate &= (
                self.store.json_get(AgentRun.loop_state, ("submission", "submission_id"), "str")
                == self.arguments["submission_id"]
            )
        return predicate

    def _status_predicate(self, predicate: Any) -> Any:
        statuses = self.arguments.get("allowed_current_statuses")
        if statuses is not None:
            return predicate & AgentRun.status.in_(statuses)
        expected = self.arguments.get("expected_current_status")
        if self.name == "finalize_run" and expected is not None:
            return predicate & (AgentRun.status == expected)
        if self.name in ("try_pause_run", "pause_with_event"):
            return (
                predicate
                & (AgentRun.status == RunStatus.RUNNING.value)
                & AgentRun.cancel_requested.is_(False)
            )
        if self.name in ("pause_run", "consume_submission") and self.epoch is not None:
            return predicate & (AgentRun.status == RunStatus.RUNNING.value)
        return predicate
