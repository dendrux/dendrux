"""P21: a store without the fan-out capability keeps 0.7 behaviour and refuses v4 data.

The old store is built by composition (as a third-party store would be): it declares the
0.7 managed capabilities only and forwards the 0.7 protocol methods, so the fan-out methods
are genuinely absent. A one-child workflow runs through run, stream, ``get_pendings``,
submits, resume and reclaim; a two-call batch is refused with limit 1; v4 data written by a
fan-out-capable store and then read through the old one is refused (``fanout_capability``)
before any claim, with nothing changed.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

import pytest

from dendrux import Agent
from dendrux.errors import ManagedStateError
from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime.state import ManagedStateStore, StateStore, StoreCapabilities
from dendrux.store import RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_routing import (
    TENANT,
    approver,
    assert_coherent,
    call,
    fan_out,
    pause_of,
    published,
    root_agent,
    snapshot,
    start_root,
)

_FORWARDED = sorted(
    {
        name
        for protocol in (StateStore, ManagedStateStore)
        for name, _ in inspect.getmembers(protocol, inspect.isfunction)
        if not name.startswith("_")
    }
)


def _forward(name: str) -> Any:
    async def method(self: OldStore, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    method.__signature__ = inspect.signature(  # type: ignore[attr-defined]
        getattr(StateStore, name, None) or getattr(ManagedStateStore, name)
    )
    return method


class OldStore:
    """A 0.7 managed store: no ``managed_fanout`` declaration and no fan-out methods."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.capabilities = StoreCapabilities(
            fenced_writes=True,
            retry_ids=True,
            checkpointing=True,
            atomic_completion=True,
            recovery=True,
            managed_subagents=True,
            deployment=inner.capabilities.deployment,
        )


for _name in _FORWARDED:
    setattr(OldStore, _name, _forward(_name))


@pytest.fixture(autouse=True)
def _reset_published() -> None:
    published.clear()


@pytest.fixture
def old(store: Any) -> OldStore:
    wrapped = OldStore(store)
    assert fanout_support(wrapped).capable is False
    assert not hasattr(wrapped, "get_managed_children")
    assert not hasattr(wrapped, "bind_managed_children")
    return wrapped


def lead(store: Any, child: Agent, *responses: LLMResponse) -> Agent:
    return Agent(
        name="lead",
        prompt="You lead.",
        provider=ScriptedLLM(list(responses)),
        state_store=store,
        subagents=[child],
    )


def delegate(task: str = "review") -> LLMResponse:
    return call("reviewer", {"task": task}, f"r-{task}")


async def test_a_one_child_workflow_runs_on_the_old_store(old: OldStore) -> None:
    """run, get_pendings, reclaim, submit_approval, then resume, on the 0.7 paths."""
    child = approver("reviewer", call("publish", {"text": "again"}, "p2"), LLMResponse(text="ok"))
    root = lead(old, child, delegate(), LLMResponse(text="root done"))

    paused = await root.run("go", tenant_id=TENANT)

    assert paused.status == RunStatus.WAITING_APPROVAL
    assert len(paused.pendings) == 1 and paused.pendings[0].agent_path == ["lead", "reviewer"]
    assert_coherent(paused)
    assert await RunStore(old).get_pendings(paused.run_id, tenant_id=TENANT) == list(
        paused.pendings
    )
    viewed = await root.reclaim(paused.run_id, tenant_id=TENANT)
    assert viewed.pendings == paused.pendings
    assert_coherent(viewed)

    again = await root.submit_approval(
        paused.run_id, approved=True, tenant_id=TENANT, expected_pause_id=paused.pause_id
    )
    assert again.status == RunStatus.WAITING_APPROVAL and again.pause_id != paused.pause_id
    assert_coherent(again)
    done = await root.resume(paused.run_id, tenant_id=TENANT, expected_pause_id=again.pause_id)

    assert (done.status, done.answer, done.pendings) == (RunStatus.SUCCESS, "root done", ())
    assert published == ["reviewer", "again"]


async def test_a_one_child_workflow_streams_on_the_old_store(old: OldStore) -> None:
    child = approver("reviewer")
    root = lead(old, child, delegate(), LLMResponse(text="root done"))

    paused = None
    async for item in root.stream("go", tenant_id=TENANT):
        if item.run_result is not None:
            paused = item.run_result
    assert paused is not None and paused.status == RunStatus.WAITING_APPROVAL
    assert_coherent(paused)
    final = None
    async for item in root.resume_stream(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    ):
        if item.run_result is not None:
            final = item.run_result
    assert final is not None and final.status == RunStatus.SUCCESS
    assert_coherent(final)


async def test_a_two_call_batch_is_refused_with_limit_one(
    old: OldStore, caplog: pytest.LogCaptureFixture
) -> None:
    child = approver("reviewer")
    two = LLMResponse(
        tool_calls=[
            ToolCall(name="reviewer", params={"task": "a"}, provider_tool_call_id="a"),
            ToolCall(name="reviewer", params={"task": "b"}, provider_tool_call_id="b"),
        ]
    )
    root = lead(old, child, two, LLMResponse(text="alone"))

    with caplog.at_level(logging.INFO):
        result = await root.run("go", tenant_id=TENANT)

    assert (result.status, result.answer) == (RunStatus.SUCCESS, "alone")
    limit_logs = [r for r in caplog.records if getattr(r, "code", None) == "managed_fanout_limit"]
    assert [(r.limit, r.requested, r.store_capability) for r in limit_logs] == [(1, 2, False)]
    rows = await old.get_tool_calls(result.run_id, limit=10)
    bodies = [r.result if isinstance(r.result, dict) else json.loads(r.result) for r in rows]
    assert [(b["reason"], b["limit"], b["requested"]) for b in bodies] == [
        ("fanout_limit", 1, 2)
    ] * 2
    assert await old.list_runs(parent_run_id=result.run_id, limit=5) == []
    assert published == []


async def test_v4_data_on_the_old_store_is_refused_before_any_claim(
    store: Any, old: OldStore
) -> None:
    a, b = approver("a"), approver("b")
    root_id = await start_root(store, root_agent(store, a, b))
    fan = await fan_out(store, root_agent(store, a, b), root_id, [a, b])
    before = await snapshot(store, root_id, *fan.children)
    root = root_agent(old, a, b)

    attempts = {
        "get_pendings": lambda: RunStore(old).get_pendings(root_id, tenant_id=TENANT),
        "get_result": lambda: root.get_result(root_id, tenant_id=TENANT),
        "submit_approval": lambda: root.submit_approval(
            root_id, approved=True, tenant_id=TENANT, expected_pause_id=pause_of(fan, 0)
        ),
        "reclaim": lambda: root.reclaim(root_id, tenant_id=TENANT),
    }
    for name, attempt in attempts.items():
        with pytest.raises(ManagedStateError) as caught:
            await attempt()
        assert caught.value.reason == "fanout_capability", name
        assert await snapshot(store, root_id, *fan.children) == before, name
    assert published == []
