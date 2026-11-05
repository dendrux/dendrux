"""Public conversation rows, independent of internal recovery positions and metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from dendrux._meta_keys import MESSAGE_META_KEY, SEED_PREFIX_META_KEY, STEP_META_KEY
from dendrux.runtime._bounds import MAX_SMALL_PAGE

if TYPE_CHECKING:
    from dendrux.runtime.state import RunRecord, StateStore, TenantScope, TraceRecord


async def public_traces(
    store: StateStore,
    run_id: str,
    *,
    tenant_id: TenantScope,
    limit: int | None,
    offset: int,
    retry: bool = False,
    before_order_index: int | None = None,
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
            before_order_index=before_order_index,
        )
        return [_public_record(row, offset + i) for i, row in enumerate(rows)]
    records: list[TraceRecord] = []
    after = None
    ordinal = 0
    while limit is None or len(records) < limit:
        page = await store.get_traces_after(
            run_id,
            after_order_index=after,
            limit=min(MAX_SMALL_PAGE, limit - len(records))
            if limit is not None
            else MAX_SMALL_PAGE,
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
        shown, count = _visible_page(page, retry, offset, ordinal)
        records.extend(shown)
        ordinal += count
        after = last
    return records


def _visible_page(
    page: list[TraceRecord], retry: bool, offset: int, ordinal: int
) -> tuple[list[TraceRecord], int]:
    visible = [row for row in page if _visible(row, retry)]
    skip = max(0, offset - ordinal)
    return [_public_record(row, ordinal + i) for i, row in enumerate(visible) if i >= skip], len(
        visible
    )


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


def pending_boundary(record: RunRecord | None) -> int | None:
    pending = (getattr(record, "loop_state", None) or {}).get("pending", [])
    return min((p["order_index"] for p in pending), default=None)
