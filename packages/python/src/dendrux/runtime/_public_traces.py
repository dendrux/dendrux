"""Public conversation rows, independent of internal recovery positions and metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from dendrux._lifecycle import TERMINAL_STATUS_VALUES
from dendrux._meta_keys import MESSAGE_META_KEY, SEED_PREFIX_META_KEY, STEP_META_KEY
from dendrux.runtime._bounds import MAX_SMALL_PAGE
from dendrux.runtime._pending import _PendingReader

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore, TenantScope, TraceRecord


async def public_traces(
    store: StateStore,
    run_id: str,
    *,
    tenant_id: TenantScope,
    limit: int | None,
    offset: int,
    retry: bool = False,
) -> list[TraceRecord]:
    """Page the run's own user, assistant and tool rows in their original order."""
    sql_page = getattr(store, "get_public_traces", None)
    if sql_page is not None:
        rows = await sql_page(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            retry=retry,
        )
        return [_public_record(row, offset + i) for i, row in enumerate(rows)]
    boundary = await _pending_boundary(store, run_id, tenant_id)
    return await _keyset_public_traces(store, run_id, tenant_id, limit, offset, retry, boundary)


async def _keyset_public_traces(
    store: StateStore,
    run_id: str,
    tenant_id: TenantScope,
    limit: int | None,
    offset: int,
    retry: bool,
    before_order_index: int | None,
) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    after = None
    ordinal = 0
    while limit is None or len(records) < limit:
        page = await store.get_traces_after(
            run_id,
            after_order_index=after,
            limit=_keyset_limit(limit, ordinal < offset, len(records)),
            kinds=("seed", "turn"),
            tenant_id=tenant_id,
        )
        if not page:
            break
        last = page[-1].order_index
        if before_order_index is not None:
            page = [row for row in page if row.order_index < before_order_index]
        if not page:
            break
        remaining = None if limit is None else limit - len(records)
        shown, count = _visible_page(page, retry, offset, ordinal, remaining)
        records.extend(shown)
        ordinal += count
        after = last
    return records


def _keyset_limit(limit: int | None, skipping: bool, collected: int) -> int:
    if limit is None or skipping:
        return MAX_SMALL_PAGE
    return min(MAX_SMALL_PAGE, limit - collected)


def _visible_page(
    page: list[TraceRecord], retry: bool, offset: int, ordinal: int, limit: int | None
) -> tuple[list[TraceRecord], int]:
    visible = [row for row in page if _visible(row, retry)]
    skip = max(0, offset - ordinal)
    shown = visible[skip:] if limit is None else visible[skip : skip + limit]
    return [_public_record(row, ordinal + skip + i) for i, row in enumerate(shown)], len(visible)


def _visible(row: TraceRecord, retry: bool) -> bool:
    return (
        row.role in ("user", "assistant", "tool")
        and not (row.meta or {}).get(SEED_PREFIX_META_KEY)
        and not (retry and row.trace_kind == "seed")
    )


def _public_record(row: TraceRecord, ordinal: int) -> TraceRecord:
    shown = {
        k: v
        for k, v in (row.meta or {}).items()
        if k not in (MESSAGE_META_KEY, STEP_META_KEY, SEED_PREFIX_META_KEY)
    }
    return replace(row, order_index=ordinal, meta=shown)


async def _pending_boundary(store: StateStore, run_id: str, tenant_id: TenantScope) -> int | None:
    record = await _PendingReader(store)._scoped_run(run_id, tenant_id)
    if record is None:
        return 0
    if record.status in TERMINAL_STATUS_VALUES:
        return None
    state = await store.get_loop_state(run_id) or {}
    # The allocation frontier also hides rows from a batch committed after this read.
    return int(
        min(
            (p["order_index"] for p in state.get("pending", [])),
            default=state.get("trace_order_offset", 0),
        )
    )
