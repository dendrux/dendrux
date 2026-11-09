"""Private provenance for projected blocked writes and scoped source reads."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

from dendrux.errors import PersistenceError, RunFencedError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from dendrux.runtime._continuation_guardrails import _SubmissionBlock
    from dendrux.runtime.state import RunRecord, StateStore
    from dendrux.runtime.stores.sql.base import SQLStateStore

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


_claimed_block: ContextVar[tuple[object, str, _SubmissionBlock] | None] = ContextVar(
    "dendrux_claimed_block", default=None
)


def _remember_claimed_block(store: StateStore, run_id: str, block: _SubmissionBlock) -> None:
    # Set only after a won claim that committed this call's blocked submission and
    # its monotone mark, so the flag can never disagree with durable provenance.
    from dendrux.runtime.state import unclaimed_store

    _claimed_block.set((unclaimed_store(store), run_id, block))


def _claimed_block_detail(store: StateStore, run_id: str) -> _SubmissionBlock | None:
    """This call's own submission block (safe error and findings), if any."""
    from dendrux.runtime.state import unclaimed_store

    claimed = _claimed_block.get()
    if claimed is None or claimed[:2] != (unclaimed_store(store), run_id):
        return None
    return claimed[2]


def _continuation_blocked(store: StateStore, run_id: str, record: RunRecord | None) -> bool:
    """Whether a validated continuation must stop for durable blocked provenance.

    Reuses the run row the continuation already loaded, so a clean continuation
    adds no statements. A blocked submission marks the run in its claim
    transaction (a competing continuation then loses its claim) and this call's
    own blocked claim is remembered locally. A note marked after the row was read
    is handled exactly as during a running turn: guarded runs check the mark when
    notes are pending, and native stores refuse tool dispatch on a marked run. Only the mark
    and the terminal reason can describe a continuable run: every earlier blocked
    event finalized its run as an error.
    """
    if _claimed_block_detail(store, run_id) is not None:
        return True
    return _row_blocked(store, run_id, record)


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
