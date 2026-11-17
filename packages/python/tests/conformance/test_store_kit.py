"""The published store test kit, run against the repo's own stores.

The contracts live in ``dendrux.testing.store_kit.contracts``; this module pulls every contract
module in so they run on each backend ``tests/conformance/conftest.py`` provides (in-memory and
file SQLite, SQLite pinned to format 1, Postgres, MySQL and MariaDB when configured).
``usefixtures`` puts the backend parameter into each test's static fixture closure: the kit
requests the author's fixtures dynamically, and pytest only parametrizes static ones.
"""

import importlib
import pkgutil

import pytest

from dendrux.testing.store_kit import contracts

pytestmark = pytest.mark.usefixtures("kit_backend_kind")

for _module in pkgutil.iter_modules(contracts.__path__):
    if _module.name.startswith("test_"):
        _loaded = importlib.import_module(f"{contracts.__name__}.{_module.name}")
        globals().update({k: v for k, v in vars(_loaded).items() if k.startswith("test_")})


# ---------------------------------------------------------------------------
# Managed subagents (0.7, T2a): every backend declaring ``managed_subagents`` binds a
# managed invocation exactly once; a format-1 run refuses before any write.
# ---------------------------------------------------------------------------


async def test_managed_capability_is_valid_on_every_backend(dendrux_store_factory):
    from dendrux.runtime._capabilities import validate_capabilities

    store = await dendrux_store_factory()
    caps = validate_capabilities(store)
    if caps.managed_subagents:
        assert callable(getattr(store, "bind_managed_child", None))


async def test_managed_bind_once_then_read_back(dendrux_store_factory):
    from dendrux.errors import ManagedStateError
    from dendrux.runtime._capabilities import capabilities
    from tests.integration.test_managed_bind import bind_request, managed_parent

    store = await dendrux_store_factory()
    if not capabilities(store).managed_subagents:
        pytest.skip("store does not declare managed_subagents")
    parent, call_id = await managed_parent(store)
    record = await store.get_run(parent)
    request = bind_request(parent, call_id)
    if record.storage_format != 2:
        with pytest.raises(ManagedStateError) as refused:
            await store.bind_managed_child(request)
        assert refused.value.reason == "storage_format"
        assert (await store.get_run(parent)).status == "running"
        assert await store.get_run(request.child_run_id) is None
        return
    created = await store.bind_managed_child(request)
    assert created.outcome == "created" and created.child_claim is not None
    again = await store.bind_managed_child(bind_request(parent, call_id))
    assert again.outcome == "existing" and again.child_claim is None
    assert again.child_run_id == created.child_run_id
    child = await store.get_run(created.child_run_id)
    assert child.status == "running" and child.parent_run_id == parent
    assert (await store.get_run(parent)).status == "paused"
