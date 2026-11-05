"""Public conversation rows, independent of internal recovery positions and metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from dendrux import _codes
from dendrux._meta_keys import MESSAGE_META_KEY, SEED_PREFIX_META_KEY, STEP_META_KEY
from dendrux.errors import RunNotReclaimableError
from dendrux.runtime._bounds import MAX_SMALL_PAGE
from dendrux.types import Limits

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
    records: list[TraceRecord] = []
    budget = _ScanBudget(run_id)
    after = None
    ordinal = 0
    while limit is None or len(records) < limit:
        page = await store.get_traces_after(
            run_id,
            after_order_index=after,
            limit=MAX_SMALL_PAGE,
            kinds=("seed", "turn"),
            tenant_id=tenant_id,
        )
        if not page:
            break
        for row in page:
            budget.add(row)
            if not _visible(row, retry):
                continue
            if ordinal >= offset:
                records.append(_public_record(row, ordinal))
            ordinal += 1
            if len(records) == limit:
                break
        after = page[-1].order_index
    return records


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


@dataclass
class _ScanBudget:
    run_id: str
    limits: Limits = field(default_factory=Limits)
    rows: int = 0
    size: int = 0

    def add(self, row: TraceRecord) -> None:
        self.rows += 1
        self.size += len(
            json.dumps(
                {"role": row.role, "content": row.content, "meta": row.meta},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if self.rows > self.limits.max_rebuild_rows or self.size > self.limits.max_rebuild_bytes:
            _codes.log_code(
                logging.getLogger(__name__), _codes.HISTORY_TOO_LARGE, run_id=self.run_id
            )
            raise RunNotReclaimableError(self.run_id, "history_too_large")
