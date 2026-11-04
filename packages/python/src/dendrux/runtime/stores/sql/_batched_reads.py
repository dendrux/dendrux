"""Bounded inputs and deterministic grouping for authorized batched reads."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dendrux.runtime.state import RunEventRecord, _to_aware_utc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAX_BATCH_RUNS = 200
MAX_EVENTS_PER_RUN = 100


def _validate_run_count(run_ids: Any) -> None:
    if isinstance(run_ids, (str, bytes)) or len(run_ids) > MAX_BATCH_RUNS:
        raise ValueError(f"batched_read run count must not exceed {MAX_BATCH_RUNS}")


def _validate_event_batch(heads: Mapping[str, int], limit: int) -> None:
    _validate_run_count(heads)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_EVENTS_PER_RUN
    ):
        raise ValueError(f"batched_read limit must be between 1 and {MAX_EVENTS_PER_RUN}")
    if any(
        isinstance(head, bool) or not isinstance(head, int) or head < -1 for head in heads.values()
    ):
        raise ValueError("batched_read head must be an integer at least -1")


def _group_events(rows: Sequence[Any]) -> dict[str, list[RunEventRecord]]:
    grouped: dict[str, list[RunEventRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.agent_run_id].append(
            RunEventRecord(
                id=row.id,
                event_type=row.event_type,
                sequence_index=row.sequence_index,
                iteration_index=row.iteration_index,
                correlation_id=row.correlation_id,
                data=row.data,
                created_at=_to_aware_utc(row.created_at),
            )
        )
    for events in grouped.values():
        events.sort(key=lambda event: event.sequence_index)
    return dict(grouped)
