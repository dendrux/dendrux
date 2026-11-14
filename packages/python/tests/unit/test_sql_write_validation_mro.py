"""SQL write validation follows inheritance without broadening existing providers."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError

from dendrux import types
from dendrux.errors import PersistenceError
from dendrux.runtime._blob_catalog import SQLBlobCatalog
from dendrux.runtime.durability import safe_database_operations
from dendrux.runtime.stores.sql._managed_reads import ManagedReads
from dendrux.runtime.stores.sql.base import SQLStateStore, _validated_store_writes
from dendrux.runtime.stores.sql.mysql import MySQLStateStore
from dendrux.runtime.stores.sql.postgres import PostgresStateStore
from dendrux.runtime.stores.sql.sqlite import SQLiteStateStore

STORES = (SQLStateStore, SQLiteStateStore, PostgresStateStore, MySQLStateStore)
PREFIXES = (
    "save_",
    "create_",
    "finalize_",
    "pause_",
    "try_pause_",
    "submit_",
    "add_note",
    "update_run_meta",
    "request_pause",
    "checkpoint_step",
    "start_tool_call",
    "complete_tool_call",
    "consume_submission",
    "set_trace_link",
)


def _layers(method):
    layers = []
    while method is not None:
        layers.append((Path(method.__code__.co_filename).name, method.__code__.co_name))
        method = getattr(method, "__wrapped__", None)
    return layers


def _snapshot(cls):
    members = {}
    for name in sorted(dir(cls)):
        if name.startswith("__"):
            continue
        member = inspect.getattr_static(cls, name)
        kind = type(member).__name__
        method = member.fget if isinstance(member, property) else member
        if isinstance(member, (classmethod, staticmethod)):
            method = member.__func__
        if not inspect.isfunction(method):
            continue
        members[name] = {
            "kind": kind,
            "signature": str(inspect.signature(method)),
            "layers": _layers(method),
            "owner": next(base.__name__ for base in cls.__mro__ if name in vars(base)),
        }
    return {"mro": [base.__name__ for base in cls.__mro__], "members": members}


# Captured from c2e31697 before changing the decorator. Readable snapshots are
# retained in the task evidence; the test has no dependency on private files.
BASELINE = {
    "SQLStateStore": (104, "3911714a5552c503ca99a9052a2197efd166fec0389b7192348f8a1c5378ea49"),
    "SQLiteStateStore": (103, "6fd4323ce3ff6eb9b17f527628519968ed609dd9858adc2cb7a71040cfd6ba9b"),
    "PostgresStateStore": (104, "1e7cbd76b9f31a63490dac7f41a231b9bd6ea2b3764de98bfd401540d172dbab"),
    "MySQLStateStore": (103, "eb0c1fd4ea37eea09b42875cfe8de3045871708af4316fd1b128f568daea3be4"),
}


@pytest.mark.parametrize("cls", STORES)
def test_exact_wrapper_signature_kind_and_mro_snapshot(cls):
    snapshot = _snapshot(cls)
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    wrapped = sum(len(member["layers"]) > 1 for member in snapshot["members"].values())
    assert (wrapped, digest) == BASELINE[cls.__name__]


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("value", ["valid", "invalid\x00", {"nested": "invalid\x00"}])
async def test_inherited_write_validates_once_before_effects(monkeypatch, prefix, value):
    effects = []

    class Notes:
        async def add_note(self, run_id, *, text):
            effects.append((run_id, text))
            return True

    name = prefix + "probe"
    mixin = type("WriteMixin", (), {name: Notes.add_note})
    host = safe_database_operations(_validated_store_writes(type("Host", (mixin,), {})))
    validator = Mock(wraps=types._validate_persistable)
    monkeypatch.setattr(types, "_validate_persistable", validator)
    method = getattr(host(), name)
    if value == "valid":
        assert await method("run", text=value) is True
        assert effects == [("run", value)]
    else:
        with pytest.raises(types.InputValidationError):
            await method("run", text=value)
        assert effects == []
    validator.assert_called_once_with(value, "text")
    assert [layer[1] for layer in _layers(method)] == ["call", "call", "write", "add_note"]
    assert name in vars(host)
    assert getattr(mixin, name) is Notes.add_note


@pytest.mark.parametrize("cls", STORES)
def test_every_effective_write_has_exactly_one_validation_layer(cls):
    for name, method in inspect.getmembers(cls, inspect.iscoroutinefunction):
        if name.startswith(PREFIXES):
            assert [layer[1] for layer in _layers(method)].count("write") == 1, name


def _implementation(method):
    unwrapped = inspect.unwrap(method)
    return getattr(unwrapped, "__func__", unwrapped)


def _assert_no_unreviewed_provider_writes(cls):
    newly_wrapped = set()
    for provider in (ManagedReads, SQLBlobCatalog):
        for name, method in inspect.getmembers(provider, inspect.iscoroutinefunction):
            if (
                name.startswith(PREFIXES)
                and _implementation(getattr(cls, name)) is _implementation(method)
                and "write" not in [layer[1] for layer in _layers(method)]
            ):
                newly_wrapped.add((provider.__name__, name))
    assert newly_wrapped == set(), newly_wrapped


def test_existing_provider_writes_do_not_gain_validation():
    _assert_no_unreviewed_provider_writes(SQLStateStore)


@pytest.mark.parametrize("provider", [ManagedReads, SQLBlobCatalog])
@pytest.mark.parametrize("shadowed", [False, True])
@pytest.mark.parametrize("descriptor", [lambda method: method, classmethod, staticmethod])
def test_provider_guard_checks_provenance_and_refuses_future_write(
    monkeypatch, provider, shadowed, descriptor
):
    async def save_future(self, *, text):
        return text

    async def reviewed_override(self, *, text):
        return text

    monkeypatch.setattr(provider, "save_future", descriptor(save_future), raising=False)
    members = {"save_future": reviewed_override} if shadowed else {}
    host = _validated_store_writes(type("Host", (provider,), members))
    if shadowed:
        _assert_no_unreviewed_provider_writes(host)
    else:
        with pytest.raises(AssertionError, match="save_future"):
            _assert_no_unreviewed_provider_writes(host)


@pytest.mark.parametrize("cls", STORES)
def test_properties_and_class_identity_hooks(cls):
    store = object.__new__(cls)
    assert store._uses_waiting_sweep_cursor is True
    assert store._uses_combined_sweep is True
    assert store._refuses_managed_child_reclaim is (cls is not MySQLStateStore)
    assert store._note_lock_requires_scope_precheck is (cls is PostgresStateStore)
    for name in (
        "_uses_waiting_sweep_cursor",
        "_uses_combined_sweep",
        "_refuses_managed_child_reclaim",
    ):
        assert isinstance(inspect.getattr_static(cls, name), property)


def test_dialect_overrides_stay_unwrapped():
    for name in ("claim_paused_run", "reclaim_run"):
        assert [layer[1] for layer in _layers(getattr(MySQLStateStore, name))] == [name]
        assert getattr(MySQLStateStore, name) is not getattr(SQLStateStore, name)
        assert getattr(
            super(MySQLStateStore, object.__new__(MySQLStateStore)), name
        ).__func__ is getattr(SQLStateStore, name)


CALLS = {
    "add_note": lambda s: s.add_note("r", note_id="n", text="t"),
    "get_notes": lambda s: s.get_notes("r"),
    "mark_notes_delivered": lambda s: s.mark_notes_delivered("r", ["n"], iteration=1),
    "get_lease_expired": lambda s: s.get_lease_expired(limit=5, cursor=None),
    "sweep_stale_runs": lambda s: s.sweep_stale_runs(dt.timedelta(hours=1)),
    "sweep_abandoned_runs": lambda s: s.sweep_abandoned_runs(dt.timedelta(hours=1)),
    "_sweep_runs": lambda s: s._sweep_runs(
        dt.timedelta(hours=1), waiting=False, batch_size=5, max_runs=5
    ),
}


@pytest.mark.parametrize("cls", STORES)
@pytest.mark.parametrize("name", CALLS)
async def test_explicit_private_wrapper_and_sanitized_database_failures(cls, name):
    def fail_session():
        raise OperationalError("SELECT private_sentinel", {}, Exception("private_sentinel"))

    store = object.__new__(cls)
    store._dialect_facts = None
    store._session_factory = fail_session
    if cls is SQLStateStore:
        store.json_get = lambda col, path, as_type: col
    with pytest.raises(PersistenceError) as caught:
        await CALLS[name](store)
    assert "private_sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
