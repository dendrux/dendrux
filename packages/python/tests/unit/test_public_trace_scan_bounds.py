"""Custom public trace pages do not inherit recovery history limits."""

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


async def test_hidden_prefix_is_not_returned():
    store = Rows(21, hidden=True)
    assert await public_traces(store, "0" * 26, tenant_id="t", limit=1, offset=0) == []


async def test_ten_times_realistic_trace_history_can_be_paged():
    store = Rows(10000)
    rows = await public_traces(store, "0" * 26, tenant_id="t", limit=10, offset=9990)
    assert [r.order_index for r in rows] == list(range(9990, 10000))


async def test_public_page_is_independent_of_recovery_bytes():
    store = Rows(1, content="x" * (17 * 1024 * 1024))
    rows = await public_traces(store, "0" * 26, tenant_id="t", limit=1, offset=0)
    assert len(rows) == 1 and len(rows[0].content) == 17 * 1024 * 1024
