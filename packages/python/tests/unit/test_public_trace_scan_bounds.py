"""Public trace filtering cannot turn a small requested page into an unlimited scan."""

import pytest

from dendrux.errors import RunNotReclaimableError
from dendrux.runtime._public_traces import public_traces
from dendrux.runtime.state import TraceRecord


class Rows:
    def __init__(self, count, *, hidden=False, content="message"):
        self.count = count
        self.hidden = hidden
        self.content = content
        self.calls = 0

    async def get_traces_after(self, run_id, *, after_order_index, limit, kinds, tenant_id):
        self.calls += 1
        start = 0 if after_order_index is None else after_order_index + 1
        return [
            TraceRecord(
                id=str(i),
                role="user",
                content=self.content,
                order_index=i,
                meta={"dendrux.seed_prefix": self.hidden},
                trace_kind="seed",
            )
            for i in range(start, min(start + limit, self.count))
        ]


async def test_hidden_prefix_scan_is_bounded():
    store = Rows(20001, hidden=True)
    with pytest.raises(RunNotReclaimableError, match="history_too_large"):
        await public_traces(store, "0" * 26, tenant_id="t", limit=1, offset=0)
    assert store.calls <= 81


async def test_ten_times_realistic_trace_history_can_be_paged():
    store = Rows(10000)
    rows = await public_traces(store, "0" * 26, tenant_id="t", limit=10, offset=9990)
    assert [r.order_index for r in rows] == list(range(9990, 10000))


async def test_trace_byte_scan_is_bounded():
    store = Rows(1, content="x" * (17 * 1024 * 1024))
    with pytest.raises(RunNotReclaimableError, match="history_too_large"):
        await public_traces(store, "0" * 26, tenant_id="t", limit=1, offset=0)
