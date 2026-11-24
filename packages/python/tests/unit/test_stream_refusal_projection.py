from types import SimpleNamespace

import pytest

from dendrux.runtime import _managed_route, _refusal_view
from dendrux.runtime.state import RunRecord


@pytest.mark.parametrize("partial", [False, True])
async def test_legacy_refusal_uses_only_the_admission_snapshot(partial):
    record = RunRecord(
        "00000000000000000000000001", "root", "paused", answer="admitted", tenant_id="original"
    )
    store = SimpleNamespace()
    if partial:

        async def unexpected(*args, **kwargs):
            pytest.fail("a partial projection cannot authorize related reads")

        store._session_factory = unexpected
        store.pause_projection = unexpected
        store.get_run = unexpected
    admitted = _refusal_view.AdmittedRoot(store, record)
    record.tenant_id = "replacement"
    record.answer = "private-replacement"
    result = await admitted.refusal()
    assert result.answer == "admitted"


async def test_admitted_reader_is_limited_to_its_root_and_restored_on_exit(monkeypatch):
    store = object()
    record = RunRecord("00000000000000000000000001", "root", "paused", tenant_id=None)
    ordinary = object()
    requests = []

    async def original(request_store, run_id):
        requests.append((request_store, run_id))
        return ordinary

    monkeypatch.setattr(_managed_route, "_build_cached_result", original)
    admitted = _refusal_view.AdmittedRoot(store, record)

    with pytest.raises(ValueError, match="opening failed"):
        assert await _managed_route.root_view(store, "another-root") is ordinary
        assert (await admitted.refusal()).run_id == record.id
        raise ValueError("opening failed")
    assert await _managed_route.root_view(store, record.id) is ordinary
    assert requests == [(store, "another-root"), (store, record.id)]


async def test_legacy_fanout_pause_read_outside_refusal():
    from dendrux.runtime._managed_pending import ManagedHop, _fanout_pause

    original = {"pause_id": "generation"}

    async def get_pause_state(run_id):
        assert run_id == "root"
        return original

    store = SimpleNamespace(get_pause_state=get_pause_state)
    hop = ManagedHop("root", "paused", "tenant", None, "agent", {}, {})
    assert await _fanout_pause(store, hop) is original


@pytest.mark.parametrize("initial", ["running", "waiting_client_tool"])
async def test_latest_refreshes_when_the_pause_changes_during_projection(initial):
    from datetime import UTC, datetime

    from dendrux.runtime.state import PauseSummary

    record = RunRecord(
        "00000000000000000000000001", "root", initial, tenant_id="t", updated_at=datetime.now(UTC)
    )
    generation = 1

    async def get_run(run_id, *, tenant_id):
        assert tenant_id == "t"
        return record

    async def summary(run_id, *, tenant_id):
        return PauseSummary(
            record.status,
            {
                "pause_id": str(generation),
                "pending_tool_calls": [{"id": "call", "name": "edit"}],
                "pending_targets": {"call": "client"},
            },
        )

    async def mapping(run_id, *, tenant_id):
        nonlocal generation
        if generation == 1:
            generation = 2
            record.claim_epoch += 1
            record.status = "waiting_client_tool"
        return {}

    store = SimpleNamespace(
        get_run=get_run, get_pending_pause_summary=summary, get_pii_mapping=mapping
    )
    result = await _refusal_view.AdmittedRoot(store, record).latest()
    assert result.status.value == "waiting_client_tool"
    assert result.pause_id == "2"


@pytest.mark.parametrize(
    "pause",
    [
        {},
        {"pending_tool_calls": [{"id": "call", "name": "edit"}]},
        {"pause_id": "token"},
        {"pause_id": "token", "pending_tool_calls": [{"id": "call"}]},
    ],
)
async def test_incomplete_client_pause_in_public_summary_is_retryable(pause):
    from dendrux.errors import PersistenceError
    from dendrux.runtime.state import PauseSummary

    record = RunRecord("00000000000000000000000001", "root", "waiting_client_tool", tenant_id="t")

    async def get_run(run_id, *, tenant_id):
        return record

    async def summary(run_id, *, tenant_id):
        return PauseSummary(record.status, pause)

    async def mapping(run_id, *, tenant_id):
        return {}

    adapter = SimpleNamespace(
        get_run=get_run, get_pending_pause_summary=summary, get_pii_mapping=mapping
    )
    with pytest.raises(PersistenceError) as failure:
        await _refusal_view.AdmittedRoot(adapter, record).latest()
    assert failure.value.retryable


async def test_terminal_public_record_does_not_need_a_pause_summary():
    async def get_run(run_id, *, tenant_id):
        assert tenant_id == "t"
        return RunRecord(run_id, "root", "success", answer="complete", tenant_id=tenant_id)

    async def mapping(run_id, *, tenant_id):
        return {}

    store = SimpleNamespace(get_run=get_run, get_pii_mapping=mapping)
    record = RunRecord("00000000000000000000000001", "root", "paused", tenant_id="t")
    result = await _refusal_view.AdmittedRoot(store, record).latest()
    assert result.answer == "complete" and not result.pendings


async def test_inconsistent_public_summary_is_retryable():
    from dendrux.errors import PersistenceError
    from dendrux.runtime.state import PauseSummary

    record = RunRecord("00000000000000000000000001", "root", "waiting_client_tool", tenant_id="t")

    async def get_run(run_id, *, tenant_id):
        return record

    async def summary(run_id, *, tenant_id):
        return PauseSummary("running", {})

    async def mapping(run_id, *, tenant_id):
        return {}

    store = SimpleNamespace(
        get_run=get_run, get_pending_pause_summary=summary, get_pii_mapping=mapping
    )
    with pytest.raises(PersistenceError) as failure:
        await _refusal_view.AdmittedRoot(store, record).latest()
    assert failure.value.retryable


@pytest.mark.parametrize("moved", ["another-tenant", "completed"])
async def test_unavailable_public_summary_rechecks_the_root(moved):
    from dendrux.errors import RunNotFoundError
    from dendrux.runtime.state import PauseSummary

    record = RunRecord("00000000000000000000000001", "root", "paused", tenant_id="t")

    async def get_run(run_id, *, tenant_id):
        return record if record.tenant_id == tenant_id else None

    async def summary(run_id, *, tenant_id):
        if moved == "another-tenant":
            record.tenant_id = "new"
        else:
            record.status, record.answer = "success", "complete"
        return PauseSummary("paused", {"pause_reason": "managed_wait"})

    async def mapping(run_id, *, tenant_id):
        return {}

    store = SimpleNamespace(
        get_run=get_run, get_pending_pause_summary=summary, get_pii_mapping=mapping
    )
    admitted = _refusal_view.AdmittedRoot(store, record)
    if moved == "another-tenant":
        with pytest.raises(RunNotFoundError):
            await admitted.latest()
    else:
        assert (await admitted.latest()).answer == "complete"


@pytest.mark.parametrize("capability", ["no-summary", "missing-run", "no-approval-display"])
async def test_public_pause_requires_scoped_evidence(capability):
    from dendrux.errors import PersistenceError, RunNotFoundError
    from dendrux.runtime.state import PauseSummary

    record = RunRecord("00000000000000000000000001", "root", "waiting_approval", tenant_id="t")

    async def get_run(run_id, *, tenant_id):
        return record

    async def mapping(run_id, *, tenant_id):
        return {}

    async def summary(run_id, *, tenant_id):
        if capability == "missing-run":
            return None
        return PauseSummary(record.status, {"pause_id": "token"})

    store = SimpleNamespace(get_run=get_run, get_pii_mapping=mapping)
    if capability != "no-summary":
        store.get_pending_pause_summary = summary
    view = _refusal_view.AdmittedRoot(store, record)
    if capability == "missing-run":
        with pytest.raises(RunNotFoundError):
            await view.latest()
    else:
        with pytest.raises(PersistenceError) as failure:
            await view.latest()
        assert failure.value.retryable


async def test_route_recovery_refusal_is_not_a_definition_event():
    from unittest.mock import AsyncMock

    from dendrux.errors import RunNotReclaimableError
    from dendrux.runtime._continuation_refusals import routed_open

    record = RunRecord("00000000000000000000000001", "root", "paused", tenant_id="t")

    async def get_run(run_id, *, tenant_id):
        return record

    agent = SimpleNamespace(
        _resolve_state_store=AsyncMock(return_value=SimpleNamespace(get_run=get_run))
    )
    failed = RunNotReclaimableError(record.id, "loop_unsupported")
    first = AsyncMock()
    checked = routed_open(first, AsyncMock(side_effect=failed), record.id, "t", None)
    with pytest.raises(RunNotReclaimableError) as error:
        await checked(SimpleNamespace(root_agent=agent))
    assert error.value is failed
    first.assert_not_awaited()
