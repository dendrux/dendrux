"""The public live-update surface imports without the optional Redis dependency."""

import subprocess
import sys

import pytest

import dendrux.store as store
from dendrux.db._live_updates import LiveUpdates, LiveUpdatesStats
from dendrux.db._postgres_updates import PostgresUpdates
from dendrux.db._redis_updates import RedisUpdates


@pytest.mark.parametrize("kind", [LiveUpdates, LiveUpdatesStats, PostgresUpdates, RedisUpdates])
def test_live_update_types_are_public(kind):
    assert getattr(store, kind.__name__) is kind
    assert kind.__name__ in store.__all__


@pytest.mark.parametrize("kind", [LiveUpdates, PostgresUpdates, RedisUpdates])
def test_listener_coordination_is_internal(kind):
    for name in ("register", "unregister", "acquire_listener", "release_listener", "generation"):
        # ABCMeta.register is Python's virtual-subclass API, not the hint hub hook.
        assert all(name not in vars(base) for base in kind.__mro__), name
        assert hasattr(kind, "_" + name), name
        assert name not in store.__all__
    assert kind.__abstractmethods__ <= {"send", "receive"}


def test_public_redis_import_does_not_load_optional_dependency():
    code = """
import sys
from importlib.abc import MetaPathFinder

class NoRedis(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "redis" or fullname.startswith("redis."):
            raise ModuleNotFoundError("redis intentionally unavailable", name=fullname)

sys.meta_path.insert(0, NoRedis())
from dendrux.store import RedisUpdates
assert "redis" not in sys.modules
try:
    RedisUpdates("redis://localhost/0")
except ImportError as error:
    assert "pip install dendrux[redis]" in str(error)
else:
    raise AssertionError("construction must require the optional dependency")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
