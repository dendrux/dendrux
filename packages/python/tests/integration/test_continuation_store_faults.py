"""Fail each public post-claim store boundary and verify retry/terminal outcomes."""

import inspect
from functools import wraps

import pytest

from dendrux.errors import PersistenceError
from dendrux.types import RunEventType, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.runtime_paths import ProbeProvider, path_case
from tests.integration.test_resume_release_outcomes import ENTRIES


class StoreFaults:
    def __init__(self, store, monkeypatch, target):
        self.calls = []
        self.target = target
        self.claimed = False
        self.depth = 0
        self.fired = None
        for name in dir(type(store)):
            if name.startswith("_") or name in {"close", "release_run_claim"}:
                continue
            method = getattr(store, name)
            if inspect.iscoroutinefunction(method):
                monkeypatch.setattr(store, name, self.wrap(name, method))

    def wrap(self, name, method):
        @wraps(method)
        async def call(*args, **kwargs):
            top = self.depth == 0
            if top and self.claimed:
                self.calls.append(name)
                if len(self.calls) - 1 == self.target:
                    self.fired = name
                    raise PersistenceError()
            self.depth += 1
            try:
                result = await method(*args, **kwargs)
            finally:
                self.depth -= 1
            if top and name in {"claim_paused_run", "submit_and_claim", "reclaim_run"} and result:
                self.claimed = True
            return result

        return call


async def exercise(store, engine, monkeypatch, entry, target):
    provider = ProbeProvider()
    case = await path_case(store, engine, entry, provider)
    before = await store.get_run(case.run_id)
    traces = len(await store.get_traces(case.run_id))
    result = None
    with monkeypatch.context() as patcher:
        faults = StoreFaults(store, patcher, target)
        try:
            result = await case.invoke()
        except PersistenceError:
            assert faults.fired is not None
    row = await store.get_run(case.run_id)
    if row.status == before.status:
        assert faults.fired is not None and provider.calls == 0
        assert len(await store.get_traces(case.run_id)) == traces
        if entry == "recovery":
            item = await expire(store, engine, case.run_id)
            retry = await case.agent.reclaim(case.run_id, tenant_id="tenant", item=item)
        else:
            assert await store.get_pause_state(case.run_id) is not None
            retry = await case.invoke()
        assert retry.status == RunStatus.SUCCESS and provider.calls == 1
    else:
        assert row.status in {"success", "error"}
        if row.status == "success":
            assert result is not None and result.status == RunStatus.SUCCESS
    if entry == "resume_stream" and faults.fired and result.status != RunStatus.SUCCESS:
        assert any(e.type == RunEventType.RUN_ERROR for e in case.events)
    return faults


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_each_post_claim_store_failure(store, engine, monkeypatch, entry, storage_format):
    if entry == "recovery" and storage_format == 1:
        pytest.skip("reclaim requires checkpoints")
    await pin_storage_format(engine, storage_format)
    baseline = await exercise(store, engine, monkeypatch, entry, None)
    for target in range(len(baseline.calls)):
        faults = await exercise(store, engine, monkeypatch, entry, target)
        assert faults.fired == baseline.calls[target]
