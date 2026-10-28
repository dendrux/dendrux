"""Transactional checkpoints and bounded, rewrite-aware conversation reads."""

import json

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime import state
from dendrux.types import generate_ulid


def trace(role, content, order, kind="turn", meta=None):
    cls = getattr(state, "TraceWrite", None)
    assert cls is not None, "checkpoint traces need a caller-owned write descriptor"
    return cls(
        trace_id=generate_ulid(),
        role=role,
        content=content,
        order_index=order,
        trace_kind=kind,
        meta=meta,
    )


async def test_a7_seeds_rewrite_prefix_and_keyset_pages(store):
    run_id = generate_ulid()
    seeds = [trace("system", "system", 0, "seed"), trace("user", "seed", 1, "seed")]
    await store.create_run(run_id, "history", seed_traces=seeds)
    await store.save_trace(run_id, "assistant", "old", order_index=2)
    await store.save_trace(
        run_id, "state", '{"kind":"image_view","v":1}', order_index=3, trace_kind="state"
    )
    replacement = [
        {"role": "system", "content": "system", "meta": {}},
        {"role": "user", "content": "summary", "meta": {}},
    ]
    rewrite = trace("rewrite", json.dumps(replacement), 4, "rewrite")
    await store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "phase": "model_next"},
        pii_mapping={"x": "y"},
        trace=rewrite,
    )
    await store.save_trace(run_id, "assistant", "new", order_index=5)
    first = await store.conversation_traces(run_id, after_position=None, limit=1)
    assert [(t.position, t.content) for t in first.entries] == [(4001, "system")]
    second = await store.conversation_traces(run_id, after_position=first.next_position, limit=2)
    assert [(t.position, t.content) for t in second.entries] == [(4002, "summary"), (5000, "new")]
    assert second.next_position is None
    raw = await store.get_traces_after(run_id, after_order_index=2, limit=2)
    assert [t.trace_kind for t in raw] == ["state", "rewrite"]
    saved = await store.get_trace(run_id, seeds[0].trace_id)
    assert saved.content == "system"
    assert await store.get_trace(run_id, seeds[0].trace_id, tenant_id="foreign") is None
    assert await store.get_loop_state(run_id) == {"v": 1, "phase": "model_next"}
    assert await store.get_pii_mapping(run_id) == {"x": "y"}


@pytest.mark.parametrize("payload", [{"provider_kwargs": {"x": "a" * 2048}}, {"x": "a" * 8192}])
async def test_a8_checkpoint_bounds_are_atomic(store, engine, payload):
    run_id = generate_ulid()
    await store.create_run(run_id, "bound")
    assert hasattr(store, "checkpoint_step"), "checkpoint must enforce its byte bounds"
    with pytest.raises(PersistenceError) as caught:
        await store.checkpoint_step(
            run_id, claim_epoch=0, loop_state=payload, pii_mapping={"secret": "value"}, trace=None
        )
    assert caught.value.code == "checkpoint_too_large"
    assert await store.get_loop_state(run_id) is None
    assert await store.get_pii_mapping(run_id) is None


async def test_a8_realistic_checkpoint_and_large_state(store):
    run_id = generate_ulid()
    await store.create_run(run_id, "bound")
    assert hasattr(store, "checkpoint_step"), "checkpoint must accept realistic state"
    payload = {
        "v": 1,
        "provider_kwargs": {"model": "model", "max_tokens": 4096},
        "pending": [
            {
                "call_id": str(i),
                "tool": "search",
                "phase": "dispatchable",
                "order_index": i + 10,
                "target": "server",
            }
            for i in range(32)
        ],
    }
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state=payload, pii_mapping=None, trace=None
    )
    assert await store.get_loop_state(run_id) == payload
    await store.save_trace(
        run_id,
        "state",
        json.dumps({"kind": "clarification", "v": 1, "question": "x" * 700_000}),
        order_index=0,
        trace_kind="state",
    )
    assert not (await store.conversation_traces(run_id, after_position=None)).entries


@pytest.mark.parametrize(
    "content,kind",
    [
        (json.dumps([{"role": "user", "content": "x", "meta": {}}] * 1000), "rewrite"),
        ("x" * (1024 * 1024 + 1), "state"),
    ],
)
async def test_a8_state_and_rewrite_write_bounds(store, content, kind):
    run_id = generate_ulid()
    await store.create_run(run_id, "bound")
    assert "trace_kind" in __import__("inspect").signature(store.save_trace).parameters
    with pytest.raises(PersistenceError) as caught:
        await store.save_trace(run_id, kind, content, order_index=0, trace_kind=kind)
    assert caught.value.code == "checkpoint_too_large"


async def test_checkpoint_rejects_unpersistable_pii_before_any_write(store):
    from dendrux.types import InputValidationError

    run_id = generate_ulid()
    await store.create_run(run_id, "pii")
    with pytest.raises(InputValidationError):
        await store.checkpoint_step(
            run_id, claim_epoch=0, loop_state={"v": 1}, pii_mapping={"token": "\x00"}, trace=None
        )
    assert await store.get_loop_state(run_id) is None
    assert await store.get_pii_mapping(run_id) is None
