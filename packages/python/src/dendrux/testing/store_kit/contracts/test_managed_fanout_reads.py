"""``managed_fanout`` (0.8): the intent-row evidence a v4 adoption reads back.

Adoption skips a call only when its intent row proves a committed observation: the row's
``content_digest`` (written with the outcome by ``complete_tool_call``) and the binding the bind
persisted (``managed_child_run_id``, from the row's managed metadata). A store that omits either
from ``get_tool_calls`` makes adoption refuse; this contract certifies both through the protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dendrux.testing.store_kit.contracts.test_managed_fanout import (
    _claim_adoption,
    _complete,
    _parent,
    _request,
    fanout,
)

if TYPE_CHECKING:
    from dendrux.testing.store_kit._kit import Kit


@fanout
async def test_fanout_intent_reads_carry_the_binding_and_the_digest(dendrux_kit: Kit) -> None:
    store = dendrux_kit.store
    parent, calls = await _parent(dendrux_kit, 2)
    request = _request(parent, calls)
    await store.bind_managed_children(request)
    bound = {c.tool_call_id: c.child_run_id for c in request.children}

    async def rows() -> dict[str, tuple[str, str | None, str | None]]:
        read = await store.get_tool_calls(parent, iteration_index=1)
        return {r.tool_call_id: (r.outcome, r.content_digest, r.managed_child_run_id) for r in read}

    started = await rows()
    assert {c: (o, child) for c, (o, _, child) in started.items()} == {
        c: ("started", child) for c, child in bound.items()
    }
    epoch = await _claim_adoption(store, request)
    await _complete(store, parent, epoch, calls[0], 10)
    after = await rows()
    assert after[calls[1]] == started[calls[1]]
    outcome, digest, child = after[calls[0]]
    assert (outcome, child) == ("completed", bound[calls[0]])
    assert digest and digest != started[calls[0]][1], "the completion writes its own digest"
