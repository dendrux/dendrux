"""Diagnostic counts of open live-update stores; no engine or store ownership."""

from __future__ import annotations

import hashlib
import logging
import warnings
import weakref
from typing import Any

from dendrux import _codes

#: Warn above this many open live-update stores against one database per process.
MANY_STORES_THRESHOLD = 16
_open_stores: dict[str, set[weakref.ReferenceType[Any]]] = {}
_warned: set[str] = set()
logger = logging.getLogger(__name__)


def _database_key(store: Any) -> str:
    # Query options can contain credentials too; database identity excludes them.
    url = store._engine.url._replace(username=None, password=None, query={})
    return hashlib.sha256(str(url).encode()).hexdigest()


def _remove(key: str, reference: weakref.ReferenceType[Any]) -> None:
    entries = _open_stores.get(key)
    if entries is None:
        return
    entries.discard(reference)
    if not entries:
        _open_stores.pop(key, None)


def register(store: Any) -> None:
    key = _database_key(store)
    entries = _open_stores.setdefault(key, set())
    entries.add(weakref.ref(store, lambda reference: _remove(key, reference)))
    count = len(entries)
    if count <= MANY_STORES_THRESHOLD or key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"Open live-update stores count={count}; share a store and close it at shutdown",
        RuntimeWarning,
        stacklevel=4,
    )
    _codes.log_code(logger, _codes.LIVE_UPDATES_MANY_STORES, count=count, database_key=key)


def deregister(store: Any) -> None:
    _remove(_database_key(store), weakref.ref(store))
