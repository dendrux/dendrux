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
    with _refusal_view.admitted_root(store, record):
        record.tenant_id = "replacement"
        record.answer = "private-replacement"
        result = await _managed_route.root_view(store, record.id)
    assert result.answer == "admitted"


async def test_admitted_reader_is_limited_to_its_root_and_restored_on_exit():
    store = object()
    record = RunRecord("00000000000000000000000001", "root", "paused", tenant_id=None)
    ordinary = object()
    requests = []

    async def original(request_store, run_id):
        requests.append((request_store, run_id))
        return ordinary

    token = _refusal_view.root_reader.set(original)
    try:
        with (
            pytest.raises(ValueError, match="opening failed"),
            _refusal_view.admitted_root(store, record),
        ):
            assert await _managed_route.root_view(store, "another-root") is ordinary
            raise ValueError("opening failed")
        assert await _managed_route.root_view(store, record.id) is ordinary
        assert requests == [(store, "another-root"), (store, record.id)]
    finally:
        _refusal_view.root_reader.reset(token)


async def test_legacy_fanout_pause_read_outside_refusal():
    from dendrux.runtime._managed_pending import ManagedHop, _fanout_pause

    original = {"pause_id": "generation"}

    async def get_pause_state(run_id):
        assert run_id == "root"
        return original

    store = SimpleNamespace(get_pause_state=get_pause_state)
    hop = ManagedHop("root", "paused", "tenant", None, "agent", {}, {})
    assert await _fanout_pause(store, hop) is original
