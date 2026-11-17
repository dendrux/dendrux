"""Accept submissions and acquire or release pause claim epochs."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import select

import dendrux.runtime._blob_catalog as _blobs
from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, CancelReason, StoredEventType
from dendrux.runtime import state as _state
from dendrux.runtime.durability import _rollback_unless_cancelled, retry_transient_db
from dendrux.runtime.state import (
    ClaimResult,
    EventWrite,
    LoopStatePatch,
    TraceWrite,
    _apply_loop_patch,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_lifecycle import _SqlLifecycle
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.media import BlobEdge


async def _write_release_event(
    session: AsyncSession, run_id: str, won: Any, event: EventWrite | None
) -> None:
    if won and event is not None:
        await _state._write_event(session, run_id, event)


class _SqlClaims(_SqlLifecycle):
    async def consume_submission(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        submission_id: str,
        traces: Sequence[TraceWrite],
        patch: LoopStatePatch,
        pii_mapping: dict[str, str] | None = None,
    ) -> bool:
        """Consume a durable submission once, even after an acknowledgement is lost."""
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            await self._fence(
                session,
                run_id,
                claim_epoch,
                **({"pii_mapping": pii_mapping} if pii_mapping is not None else {}),
            )
            row = (
                await session.execute(
                    select(AgentRun.loop_state, AgentRun.pause_data).where(AgentRun.id == run_id)
                )
            ).one()
            current = row.loop_state or {}
            if (current.get("submission") or {}).get("submission_id") != submission_id:
                await session.rollback()
                return False
            changed = _apply_loop_patch(current, patch)
            changed["submission"] = None
            values = {"loop_state": changed, **_sql_support._submission_progress(current, traces)}
            self._remember_run_values(session, run_id, values)
            _sql_support._validate_checkpoint(run_id, changed)
            pause = dict(row.pause_data or {})
            pause.pop("submission", None)
            for trace in traces:
                await self._insert_trace(session, run_id, trace)
            await session.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(**values, pause_data=pause)
            )
            self._mark_announce(session, (run_id,))
            await self._write_tx(session)
            return True

    async def claim_paused_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        expected_pause_id: str | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> ClaimResult | Literal[False]:
        """Atomically transition a paused run to RUNNING.

        Returns the new claim epoch if the claim succeeded.
        Returns False if someone else already claimed it or status didn't match.
        Uses UPDATE ... WHERE status=? for atomicity — no race window.
        The claim clears ``pause_requested`` in the same statement, so a
        resumed PAUSED run does not immediately re-pause on a stale flag.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> ClaimResult | Literal[False]:
            async with self._session_factory() as session:
                if not self.supports_update_returning:

                    def validate_locked(row: Any) -> None:
                        pause = row.get("pause_data") or {}
                        if (
                            expected_pause_id is not None
                            and pause.get("pause_id") != expected_pause_id
                        ):
                            from dendrux.errors import PauseStatusMismatchError

                            _codes.log_code(logger, _codes.PAUSE_ID_MISMATCH, run_id=run_id)
                            raise PauseStatusMismatchError(
                                run_id,
                                RunStatus(expected_status),
                                RunStatus(expected_status),
                                expected_pause_id=expected_pause_id,
                                current_pause_id=pause.get("pause_id"),
                                reason="pause_id",
                            )
                        if validate is not None:
                            validate(pause)

                    row = await self.update_one(
                        session,
                        table=AgentRun.__table__,
                        pk=(run_id,),
                        predicate=AgentRun.status == expected_status,
                        values=self._ordered(
                            AgentRun.__table__,
                            {
                                **_sql_support._status_values(RunStatus.RUNNING.value),
                                "last_progress_at": _dt.datetime.now(_dt.UTC),
                                "pause_requested": False,
                                "claim_epoch": AgentRun.claim_epoch + 1,
                            },
                        ),
                        returning=[
                            AgentRun.claim_epoch,
                            AgentRun.storage_format,
                            AgentRun.loop_state,
                        ],
                        lock_projection=[AgentRun.__table__.c.pause_data]
                        if validate is not None or expected_pause_id is not None
                        else (),
                        validate_locked=validate_locked
                        if validate is not None or expected_pause_id is not None
                        else None,
                    )
                    self._mark_announce(session, (run_id,))
                    await self._write_tx(session)
                    return (
                        False
                        if row is None
                        else ClaimResult(
                            row["claim_epoch"],
                            row["storage_format"],
                            row["loop_state"],
                            (row["loop_state"] or {}).get("submission"),
                        )
                    )
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__,
                            dict(
                                **_sql_support._status_values(RunStatus.RUNNING.value),
                                last_progress_at=_dt.datetime.now(_dt.UTC),
                                pause_requested=False,
                                claim_epoch=AgentRun.claim_epoch + 1,
                            ),
                        )
                    )
                )
                result = await session.execute(stmt)
                if result.rowcount:
                    row = (
                        await session.execute(
                            select(
                                AgentRun.claim_epoch, AgentRun.storage_format, AgentRun.loop_state
                            ).where(AgentRun.id == run_id)
                        )
                    ).one()
                    token = ClaimResult(
                        row.claim_epoch,
                        row.storage_format,
                        row.loop_state,
                        (row.loop_state or {}).get("submission"),
                    )
                if result.rowcount and (validate is not None or expected_pause_id is not None):
                    pause = (
                        await session.execute(
                            select(AgentRun.pause_data).where(AgentRun.id == run_id)
                        )
                    ).scalar_one_or_none()
                    try:
                        if expected_pause_id is not None:
                            from dendrux.errors import PauseStatusMismatchError

                            current_id = (pause or {}).get("pause_id")
                            if current_id != expected_pause_id:
                                _codes.log_code(logger, _codes.PAUSE_ID_MISMATCH, run_id=run_id)
                                raise PauseStatusMismatchError(
                                    run_id,
                                    RunStatus(expected_status),
                                    RunStatus(expected_status),
                                    expected_pause_id=expected_pause_id,
                                    current_pause_id=current_id,
                                    reason="pause_id",
                                )
                        if validate is not None:
                            validate(pause or {})
                    except BaseException as error:
                        await _rollback_unless_cancelled(session, error)
                        raise
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token if result.rowcount else False

        return await retry_transient_db(
            _attempt,
            label="claim_paused_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
        submission: dict[str, Any] | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> ClaimResult | Literal[False]:
        """Atomically save submitted data and claim the run.

        Implementation (one transaction):
          1. Conditional UPDATE status -> 'running' WHERE status still equals
             *expected_status*. Losing this CAS returns False. Winning it
             takes the row's write lock (Postgres row lock / SQLite write
             lock), so the pause_data read next cannot change underneath.
          2. Read pause_data and run *validate* on it. A failure rolls the
             whole transaction back (status stays *expected_status*) and
             re-raises.
          3. Merge submitted_data into pause_data.
          4. Write ``blob_refs`` (tenant asserted, blobs claimed) and commit.
             A failure rolls the claim back too.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> ClaimResult | Literal[False]:
            async with self._session_factory() as session:
                # 1. Claim first — the CAS on status is the concurrency
                # boundary. updated_at is filled by the model's onupdate hook.
                claim = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__,
                            dict(
                                **_sql_support._status_values(RunStatus.RUNNING.value),
                                last_progress_at=_dt.datetime.now(_dt.UTC),
                                pause_requested=False,
                                claim_epoch=AgentRun.claim_epoch + 1,
                            ),
                        )
                    )
                )
                if not claim.rowcount:
                    await session.rollback()
                    return False

                # 2. Read the pause being claimed, under the claim's lock.
                run = (
                    await session.execute(
                        select(
                            AgentRun.pause_data, AgentRun.storage_format, AgentRun.loop_state
                        ).where(AgentRun.id == run_id)
                    )
                ).one()
                current_pause_data = run.pause_data
                if current_pause_data is None:
                    await session.rollback()
                    return False
                if validate is not None:
                    try:
                        validate(current_pause_data)
                    except BaseException as error:
                        await _rollback_unless_cancelled(session, error)
                        raise

                # 3. Merge submitted_data into pause_data.
                merged = dict(current_pause_data)
                values: dict[str, Any] = {}
                if run.storage_format == STORAGE_FORMAT_CHECKPOINTED:
                    if submission is None:
                        from dendrux.errors import PersistenceError

                        _codes.log_code(logger, _codes.SUBMISSION_MISSING, run_id=run_id)
                        raise PersistenceError(code="submission_missing")
                    merged["submission"] = {
                        **{
                            k: submission[k]
                            for k in ("submission_id", "kind", "pause_id", "call_ids")
                        },
                        "payload": dict(submitted_data),
                    }
                    loop_state = dict(run.loop_state or {})
                    loop_state["submission"] = {
                        k: v for k, v in submission.items() if k != "payload"
                    }
                    _sql_support._validate_checkpoint(run_id, loop_state)
                    values["loop_state"] = loop_state
                else:
                    merged.update(submitted_data)
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .ordered_values(
                        *self._ordered(AgentRun.__table__, dict(pause_data=merged, **values))
                    )
                )
                if blob_refs:
                    await _blobs.write_blob_refs(
                        session,
                        run_id=run_id,
                        blob_refs=blob_refs,
                        blob_refs_scope=blob_refs_scope,
                    )
                row = (
                    await session.execute(
                        select(
                            AgentRun.claim_epoch, AgentRun.storage_format, AgentRun.loop_state
                        ).where(AgentRun.id == run_id)
                    )
                ).one()
                token = ClaimResult(
                    row.claim_epoch,
                    row.storage_format,
                    row.loop_state,
                    (row.loop_state or {}).get("submission"),
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            _attempt,
            label="submit_and_claim",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _finish_claim_release(
        self,
        session: AsyncSession,
        run_id: str,
        claim_epoch: int,
        won: bool,
        event: EventWrite | None,
    ) -> bool:
        if won:
            await _write_release_event(session, run_id, won, event)
            return True
        return await self._cancel_claim_release(session, run_id, claim_epoch)

    async def _cancel_claim_release(
        self, session: AsyncSession, run_id: str, claim_epoch: int
    ) -> bool:
        from dendrux.db.models import AgentRun

        requested = await session.scalar(
            self.lock_clause(
                select(AgentRun.cancel_requested).where(
                    AgentRun.id == run_id,
                    AgentRun.status == RunStatus.RUNNING.value,
                    AgentRun.claim_epoch == claim_epoch,
                    AgentRun.cancel_requested.is_(True),
                ),
                "update",
            )
        )
        if not requested:
            return False
        won = await self._cas_finalize(
            session,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[RunStatus.RUNNING.value],
            claim_epoch=claim_epoch,
            answer=None,
            error=None,
            iteration_count=None,
            total_usage=None,
            pii_mapping=None,
        )
        if won:
            await _state._insert_event_locked(
                session,
                run_id,
                StoredEventType.RUN_CANCELLED.value,
                {"reason": CancelReason.CANCEL_REQUESTED.value},
            )
        return won

    async def release_run_claim(
        self, run_id: str, *, status: str, claim_epoch: int, event: EventWrite | None = None
    ) -> bool:
        """Return an owned claim to its pause after post-claim validation fails."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.claim_epoch == claim_epoch,
                        AgentRun.cancel_requested.is_(False),
                    )
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__, dict(**_sql_support._status_values(status))
                        )
                    )
                )
                won = await self._finish_claim_release(
                    session, run_id, claim_epoch, bool(result.rowcount), event
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return won

        return await retry_transient_db(
            attempt,
            label="release_run_claim",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )
