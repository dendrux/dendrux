"""Manual load repro for the file-SQLite cancelled-write lock; not part of any suite.

The file name does not match ``test_*.py``, so ``make test``, ``make ci`` and CI never
collect it. Run it by path, with CPU load, for example:
``pytest tests/integration/_sqlite_zombie_load.py -n 2 --no-cov --reruns 0``.
"""

import pytest

from tests.integration.test_fanout_driver import live_adoption  # noqa: F401, F811
from tests.integration.test_fanout_stream import (
    test_closing_a_full_queue_leaves_the_queue_alone as original,
)


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
@pytest.mark.parametrize("repetition", range(100))
async def test_loaded_full_queue(store, live_adoption, monkeypatch, repetition):  # noqa: F811
    await original(store, live_adoption, monkeypatch)
