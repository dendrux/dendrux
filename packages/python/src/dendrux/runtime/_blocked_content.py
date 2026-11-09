"""Private provenance for projected blocked writes and scoped source reads."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from dendrux.errors import PersistenceError, RunFencedError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from dendrux.runtime._continuation_guardrails import _SubmissionBlock
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.runtime.stores.sql.base import SQLStateStore


class _DispatchUndecided(Exception):  # noqa: N818 - private signal, never surfaced
    """A dispatch write was refused and the read that would explain why failed."""


_blocked_target: ContextVar[tuple[object, str] | None] = ContextVar(
    "dendrux_blocked_target", default=None
)


def _native_store(store: StateStore) -> SQLStateStore:
    from dendrux.runtime.state import unclaimed_store
    from dendrux.runtime.stores.sql.base import SQLStateStore

    native = unclaimed_store(store)
    if not isinstance(native, SQLStateStore):
        raise PersistenceError()
    return native


@contextmanager
def _blocked_write(store: StateStore, run_id: str) -> Iterator[None]:
    # Call only around persistence of an already-projected blocked value.
    token = _blocked_target.set((_native_store(store), run_id))
    try:
        yield
    finally:
        _blocked_target.reset(token)


async def _blocked_run_ids(
    store: StateStore, run_ids: Sequence[str], *, tenant_id: str | None
) -> set[str]:
    return cast(
        "set[str]", await _native_store(store)._blocked_run_ids(run_ids, tenant_id=tenant_id)
    )


@asynccontextmanager
async def _blocked_evidence(store: StateStore, run_id: str) -> AsyncIterator[None]:
    from dendrux._meta_keys import BLOCKED_META_KEY
    from dendrux.runtime.state import unclaimed_store
    from dendrux.runtime.stores.sql.base import SQLStateStore

    if isinstance(unclaimed_store(store), SQLStateStore):
        with _blocked_write(store, run_id):
            yield
        return
    failed = False
    try:
        await store.update_run_meta(run_id, {BLOCKED_META_KEY: True})
    except RunFencedError:
        raise
    except Exception:
        failed = True
    if failed:
        raise PersistenceError()
    await _require_block_mark(store, run_id)
    yield


async def _require_block_mark(store: StateStore, run_id: str) -> None:
    from dendrux._meta_keys import BLOCKED_META_KEY

    valid = False
    try:
        record = await store.get_run(run_id)
        valid = (
            record is not None
            and record.id == run_id
            and (record.meta or {}).get(BLOCKED_META_KEY) is True
        )
    except RunFencedError:
        raise
    except Exception:
        pass
    if not valid:
        raise PersistenceError()


@dataclass
class _ClaimCall:
    """One public continuation call: whether it won a claim, and its safe block."""

    run_id: str
    store: object | None = None
    block: _SubmissionBlock | None = None


_claim_call: ContextVar[_ClaimCall | None] = ContextVar("dendrux_claim_call", default=None)


@contextmanager
def _claim_call_scope(run_id: str) -> Iterator[None]:
    """Scope a continuation call; nested scopes for the same run share it."""
    current = _claim_call.get()
    if current is not None and current.run_id == run_id:
        yield
        return
    token = _claim_call.set(_ClaimCall(run_id))
    try:
        yield
    finally:
        _claim_call.reset(token)


def _remember_claim(store: StateStore, run_id: str, block: _SubmissionBlock | None) -> None:
    # Called only after a won claim; a blocked submission committed its mark with it.
    from dendrux.runtime.state import unclaimed_store

    current = _claim_call.get()
    if current is not None and current.run_id == run_id:
        current.store, current.block = unclaimed_store(store), block


def _claimed_here(store: StateStore, run_id: str) -> bool:
    """Whether this call won the claim it is continuing (as opposed to adopting one)."""
    from dendrux.runtime.state import unclaimed_store

    current = _claim_call.get()
    return (
        current is not None
        and current.run_id == run_id
        and current.store is not None
        and current.store is unclaimed_store(store)
    )


def _claimed_block_detail(store: StateStore, run_id: str) -> _SubmissionBlock | None:
    """This call's own submission block (safe error text and kind counts), if any."""
    current = _claim_call.get()
    return current.block if current is not None and _claimed_here(store, run_id) else None


async def _continuation_blocked(
    store: StateStore, run_id: str, record: RunRecord | None, *, claimed_here: bool
) -> bool:
    """Whether a validated continuation must stop for durable blocked provenance.

    A continuation whose claim this call won reuses the run row it already loaded,
    so a clean continuation adds no statements: a blocked submission marks the run
    in its claim transaction (a competing continuation then loses its claim) and
    this call's own blocked claim is remembered. A note marked after the row was
    read is handled as during a running turn: guarded runs check the mark when
    notes are pending, and native stores refuse tool dispatch on a marked run.
    Adopting a claim made elsewhere (``resume_claimed`` after a crash, raw-store
    callers) also runs the full scoped lookup, which sees a legacy blocked event.
    """
    if _claimed_block_detail(store, run_id) is not None:
        return True
    if _row_blocked(store, run_id, record):
        return True
    if claimed_here:
        return False
    return await _adopted_run_blocked(store, run_id, record)


async def _adopted_run_blocked(store: StateStore, run_id: str, record: RunRecord | None) -> bool:
    from dendrux.runtime.state import unclaimed_store
    from dendrux.runtime.stores.sql.base import SQLStateStore
    from dendrux.types import GovernanceEventType

    if isinstance(unclaimed_store(store), SQLStateStore):
        tenant = record.tenant_id if record is not None else None
        return bool(await _blocked_run_ids(store, [run_id], tenant_id=tenant))
    blocked = GovernanceEventType.GUARDRAIL_BLOCKED.value
    events = None
    try:
        events = await store.get_run_events(run_id, event_types=[blocked], limit=1)
    except RunFencedError:
        raise
    except Exception:
        events = None
    if not isinstance(events, (list, tuple)) or any(
        getattr(event, "event_type", None) != blocked for event in events
    ):
        raise PersistenceError()
    return bool(events)


def _row_blocked(store: StateStore, run_id: str, record: RunRecord | None) -> bool:
    """Read the mark from an already loaded run row; malformed rows fail closed."""
    from dendrux._meta_keys import BLOCKED_META_KEY
    from dendrux.runtime.state import unclaimed_store
    from dendrux.runtime.stores.sql.base import SQLStateStore

    if record is None:
        if isinstance(unclaimed_store(store), SQLStateStore):
            raise PersistenceError()
        # Custom blocked writes verify the mark through get_run before any related
        # evidence, so a store without a readable run row never holds a mark.
        return False
    meta = getattr(record, "meta", None)
    if getattr(record, "id", None) != run_id or (meta is not None and not isinstance(meta, dict)):
        raise PersistenceError()
    return (meta or {}).get(BLOCKED_META_KEY) is True or getattr(
        record, "failure_reason", None
    ) == "guardrail_blocked"


_dispatch_target: ContextVar[tuple[object, str] | None] = ContextVar(
    "dendrux_blocked_dispatch", default=None
)


@contextmanager
def _dispatch_write(store: StateStore, run_id: str) -> Iterator[None]:
    from dendrux.runtime.state import unclaimed_store

    token = _dispatch_target.set((unclaimed_store(store), run_id))
    try:
        yield
    finally:
        _dispatch_target.reset(token)


def _dispatch_guarded(store: object, run_id: str) -> bool:
    return _dispatch_target.get() == (store, run_id)
