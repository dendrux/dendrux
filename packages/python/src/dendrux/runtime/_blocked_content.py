"""Private provenance for projected blocked writes and scoped source reads."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

from dendrux.errors import PersistenceError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dendrux.runtime.state import StateStore
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
