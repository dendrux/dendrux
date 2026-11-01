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
