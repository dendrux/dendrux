"""A stand-in for track A2's announcing store, to test the kit's live-update contract.

``AnnouncingSqliteStore`` does what the spec asks of a store (section 4.1): note the runs a write
touched, enqueue them once COMMIT was issued (also when the call fails afterwards, never when the
commit is proved not to have applied), and flush on the way out of a call that succeeded. ``mode``
breaks exactly one of those promises, so the kit's contract can be shown to catch each.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import event

from dendrux._meta_keys import TOUCHED_RUNS_SESSION_KEY
from dendrux.runtime.durability import _CommitNotAppliedError, _database_operation
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.db._live_updates import LiveUpdates

MODES = (
    "correct",
    "never",  # never enqueues
    "no_flush",  # enqueues, never flushes at the end of a call
    "early",  # announces before COMMIT
    "on_not_applied",  # announces a write that was proved not to have applied
    "only_on_success",  # forgets writes whose call failed after COMMIT
    "flush_on_failure",  # flushes on the way out of a call that raised
)


class AnnouncingSqliteStore(SQLiteStateStore):
    def __init__(
        self, engine: AsyncEngine, *, live_updates: LiveUpdates | None = None, mode: str = "correct"
    ) -> None:
        super().__init__(engine)
        assert mode in MODES
        self._mode = mode
        # Keep inherited production announcements disabled so they cannot repair broken modes.
        self._test_updates = live_updates
        if live_updates is not None:
            live_updates.bind(self)

    async def close(self) -> None:
        if self._test_updates is not None:
            await self._test_updates.close()

    # -- send side: what track A2 adds to _write_tx and to the flush boundary ---------------

    def _announce(self, session: AsyncSession, run_ids: Sequence[str] = ()) -> None:
        if self._test_updates is None or self._mode == "never":
            return
        operation = _database_operation.get()
        named = operation[2].get("run_id") if operation is not None else None
        touched = {*session.info.get(TOUCHED_RUNS_SESSION_KEY, ()), *run_ids}
        self._test_updates.enqueue(sorted({*touched, *([named] if named else [])}))

    async def _write_tx(self, session: AsyncSession, *, run_ids: Sequence[str] = ()) -> None:
        issued = {"commit": False}
        event.listen(
            session.sync_session,
            "before_commit",
            lambda _session: issued.__setitem__("commit", True),
            once=True,
        )
        completed = not_applied = False
        try:
            if self._mode == "early":
                self._announce(session, run_ids)
                assert self._test_updates is not None
                await self._test_updates.flush()
            await super()._write_tx(session, run_ids=run_ids)
            completed = True
        except _CommitNotAppliedError:
            not_applied = True
            raise
        finally:
            if self._should_announce(issued["commit"], completed, not_applied):
                self._announce(session, run_ids)

    def _should_announce(self, issued: bool, completed: bool, not_applied: bool) -> bool:
        if self._mode == "early":
            return False  # it announced before the commit
        if not_applied:
            return self._mode == "on_not_applied"
        # the spec: announce whenever COMMIT was issued, whether or not the call then succeeded
        return issued and (completed or self._mode != "only_on_success")

    async def _flushed(self, call: Any) -> Any:
        if self._test_updates is None or self._mode == "no_flush":
            return await call
        if self._mode == "flush_on_failure":
            try:
                return await call
            finally:
                await self._test_updates.flush()
        result = await call
        await self._test_updates.flush()
        return result

    async def create_run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._flushed(super().create_run(*args, **kwargs))

    async def save_run_event(self, *args: Any, **kwargs: Any) -> Any:
        return await self._flushed(super().save_run_event(*args, **kwargs))

    async def finalize_run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._flushed(super().finalize_run(*args, **kwargs))
