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


EXPECTED_MRO = {
    "SQLStateStore": [
        "SQLStateStore",
        "_SqlRuns",
        "_SqlFanout",
        "_SqlManaged",
        "_SqlTree",
        "_SqlPause",
        "_SqlReads",
        "_SqlEvents",
        "_SqlClaims",
        "_SqlLifecycle",
        "_SqlEvidence",
        "_SqlLeases",
        "_SqlSweep",
        "_SqlWrites",
        "_StoreHost",
        "ManagedReads",
        "SQLBlobCatalog",
        "object",
    ],
    "SQLiteStateStore": [
        "SQLiteStateStore",
        "SQLStateStore",
        "_SqlRuns",
        "_SqlFanout",
        "_SqlManaged",
        "_SqlTree",
        "_SqlPause",
        "_SqlReads",
        "_SqlEvents",
        "_SqlClaims",
        "_SqlLifecycle",
        "_SqlEvidence",
        "_SqlLeases",
        "_SqlSweep",
        "_SqlWrites",
        "_StoreHost",
        "ManagedReads",
        "SQLBlobCatalog",
        "object",
    ],
    "PostgresStateStore": [
        "PostgresStateStore",
        "SQLStateStore",
        "_SqlRuns",
        "_SqlFanout",
        "_SqlManaged",
        "_SqlTree",
        "_SqlPause",
        "_SqlReads",
        "_SqlEvents",
        "_SqlClaims",
        "_SqlLifecycle",
        "_SqlEvidence",
        "_SqlLeases",
        "_SqlSweep",
        "_SqlWrites",
        "_StoreHost",
        "ManagedReads",
        "SQLBlobCatalog",
        "object",
    ],
    "MySQLStateStore": [
        "MySQLStateStore",
        "SQLStateStore",
        "_SqlRuns",
        "_SqlFanout",
        "_SqlManaged",
        "_SqlTree",
        "_SqlPause",
        "_SqlReads",
        "_SqlEvents",
        "_SqlClaims",
        "_SqlLifecycle",
        "_SqlEvidence",
        "_SqlLeases",
        "_SqlSweep",
        "_SqlWrites",
        "_StoreHost",
        "ManagedReads",
        "SQLBlobCatalog",
        "object",
    ],
}

# Original member snapshots plus the reviewed _parent_run_query hook: inherited
# from SQLStateStore on SQLite/Postgres, overridden on MySQL. No other member changed.
# 0.8 (T3): the _SqlFanout mixin adds bind_managed_children and get_managed_children (two
# more wrapped public members) and its private batch-bind helpers; _bind_child takes the
# keyword-only trusted ``blocked`` evidence of a batch child.
# Readable snapshots are retained in the task evidence, not loaded by this test.
BASELINE = {
    "SQLStateStore": (106, "3cce8c685f7013262e17f0e68c579ca2487c7b6cf60a179f869b450f57468821"),
    "SQLiteStateStore": (105, "781aa7f7fc25badf805b95c7bbce74692037ec45cf892404e8f2763940646662"),
    "PostgresStateStore": (106, "b8b243254ef4fe7cd36ace4455c0287257d752850ba6c1a43aa6f81906e9c39c"),
    "MySQLStateStore": (105, "f67c47e67b1ed2379447bda73e86ecd2142976c5e2d9dc09d0c54889dc11c9a2"),
}


WRITES_RELOCATIONS = {
    "MySQLStateStore": {
        "_blocked_run_ids": ("SQLStateStore", "_SqlWrites"),
        "_blocked_runs_query": ("SQLStateStore", "_SqlWrites"),
        "_check_dispatch_block": ("SQLStateStore", "_SqlWrites"),
        "_commit_operation": ("SQLStateStore", "_SqlWrites"),
        "_dispatch_filter": ("SQLStateStore", "_SqlWrites"),
        "_enqueue_committed": ("SQLStateStore", "_SqlWrites"),
        "_fence": ("SQLStateStore", "_SqlWrites"),
        "_insert_retry": ("SQLStateStore", "_SqlWrites"),
        "_mark_announce": ("SQLStateStore", "_SqlWrites"),
        "_mark_blocked_writes": ("SQLStateStore", "_SqlWrites"),
        "_native_blocked_mark": ("SQLStateStore", "_SqlWrites"),
        "_ordered": ("SQLStateStore", "_SqlWrites"),
        "_remember_run_values": ("SQLStateStore", "_SqlWrites"),
        "_remember_write": ("SQLStateStore", "_SqlWrites"),
        "_reservation_values": ("SQLStateStore", "_SqlWrites"),
        "_update_many_rows": ("SQLStateStore", "_SqlWrites"),
        "_update_one_rows": ("SQLStateStore", "_SqlWrites"),
        "_write_conflict": ("SQLStateStore", "_SqlWrites"),
        "_write_tx": ("SQLStateStore", "_SqlWrites"),
        "update_many": ("SQLStateStore", "SQLStateStore"),
    },
    "PostgresStateStore": {
        "_blocked_run_ids": ("SQLStateStore", "_SqlWrites"),
        "_blocked_runs_query": ("SQLStateStore", "_SqlWrites"),
        "_check_dispatch_block": ("SQLStateStore", "_SqlWrites"),
        "_commit_operation": ("SQLStateStore", "_SqlWrites"),
        "_conflict": ("SQLStateStore", "_SqlWrites"),
        "_dispatch_filter": ("SQLStateStore", "_SqlWrites"),
        "_enqueue_committed": ("SQLStateStore", "_SqlWrites"),
        "_execute_insert_plan": ("SQLStateStore", "_SqlWrites"),
        "_fence": ("SQLStateStore", "_SqlWrites"),
        "_insert_retry": ("SQLStateStore", "_SqlWrites"),
        "_mark_announce": ("SQLStateStore", "_SqlWrites"),
        "_mark_blocked_writes": ("SQLStateStore", "_SqlWrites"),
        "_native_blocked_mark": ("SQLStateStore", "_SqlWrites"),
        "_ordered": ("SQLStateStore", "_SqlWrites"),
        "_remember_run_values": ("SQLStateStore", "_SqlWrites"),
        "_remember_write": ("SQLStateStore", "_SqlWrites"),
        "_reservation_values": ("SQLStateStore", "_SqlWrites"),
        "_update_many_rows": ("SQLStateStore", "_SqlWrites"),
        "_update_one_rows": ("SQLStateStore", "_SqlWrites"),
        "_write_conflict": ("SQLStateStore", "_SqlWrites"),
        "_write_tx": ("SQLStateStore", "_SqlWrites"),
        "update_many": ("SQLStateStore", "SQLStateStore"),
        "update_one": ("SQLStateStore", "SQLStateStore"),
    },
    "SQLStateStore": {
        "_blocked_run_ids": ("SQLStateStore", "_SqlWrites"),
        "_blocked_runs_query": ("SQLStateStore", "_SqlWrites"),
        "_check_dispatch_block": ("SQLStateStore", "_SqlWrites"),
        "_commit_operation": ("SQLStateStore", "_SqlWrites"),
        "_conflict": ("SQLStateStore", "_SqlWrites"),
        "_dispatch_filter": ("SQLStateStore", "_SqlWrites"),
        "_enqueue_committed": ("SQLStateStore", "_SqlWrites"),
        "_execute_insert_plan": ("SQLStateStore", "_SqlWrites"),
        "_fence": ("SQLStateStore", "_SqlWrites"),
        "_insert_retry": ("SQLStateStore", "_SqlWrites"),
        "_mark_announce": ("SQLStateStore", "_SqlWrites"),
        "_mark_blocked_writes": ("SQLStateStore", "_SqlWrites"),
        "_native_blocked_mark": ("SQLStateStore", "_SqlWrites"),
        "_ordered": ("SQLStateStore", "_SqlWrites"),
        "_remember_run_values": ("SQLStateStore", "_SqlWrites"),
        "_remember_write": ("SQLStateStore", "_SqlWrites"),
        "_reservation_values": ("SQLStateStore", "_SqlWrites"),
        "_update_many_rows": ("SQLStateStore", "_SqlWrites"),
        "_update_one_rows": ("SQLStateStore", "_SqlWrites"),
        "_write_conflict": ("SQLStateStore", "_SqlWrites"),
        "_write_tx": ("SQLStateStore", "_SqlWrites"),
        "update_many": ("SQLStateStore", "SQLStateStore"),
        "update_one": ("SQLStateStore", "SQLStateStore"),
    },
    "SQLiteStateStore": {
        "_blocked_run_ids": ("SQLStateStore", "_SqlWrites"),
        "_blocked_runs_query": ("SQLStateStore", "_SqlWrites"),
        "_check_dispatch_block": ("SQLStateStore", "_SqlWrites"),
        "_commit_operation": ("SQLStateStore", "_SqlWrites"),
        "_conflict": ("SQLStateStore", "_SqlWrites"),
        "_dispatch_filter": ("SQLStateStore", "_SqlWrites"),
        "_enqueue_committed": ("SQLStateStore", "_SqlWrites"),
        "_execute_insert_plan": ("SQLStateStore", "_SqlWrites"),
        "_fence": ("SQLStateStore", "_SqlWrites"),
        "_insert_retry": ("SQLStateStore", "_SqlWrites"),
        "_mark_announce": ("SQLStateStore", "_SqlWrites"),
        "_mark_blocked_writes": ("SQLStateStore", "_SqlWrites"),
        "_native_blocked_mark": ("SQLStateStore", "_SqlWrites"),
        "_ordered": ("SQLStateStore", "_SqlWrites"),
        "_remember_run_values": ("SQLStateStore", "_SqlWrites"),
        "_remember_write": ("SQLStateStore", "_SqlWrites"),
        "_reservation_values": ("SQLStateStore", "_SqlWrites"),
        "_update_many_rows": ("SQLStateStore", "_SqlWrites"),
        "_update_one_rows": ("SQLStateStore", "_SqlWrites"),
        "_write_conflict": ("SQLStateStore", "_SqlWrites"),
        "_write_tx": ("SQLStateStore", "_SqlWrites"),
        "update_many": ("SQLStateStore", "SQLStateStore"),
        "update_one": ("SQLStateStore", "SQLStateStore"),
    },
}


EVIDENCE_RELOCATIONS = {
    "MySQLStateStore": {
        "_insert_trace": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "_patch_fence": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "checkpoint_step": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "complete_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "get_replayed_tool_calls": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_evidence.py",
        ),
        "save_llm_interaction": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "set_trace_link": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "start_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
    },
    "PostgresStateStore": {
        "_insert_trace": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "_patch_fence": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "checkpoint_step": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "complete_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "get_replayed_tool_calls": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_evidence.py",
        ),
        "save_llm_interaction": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "set_trace_link": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "start_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
    },
    "SQLStateStore": {
        "_insert_trace": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "_patch_fence": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "checkpoint_step": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "complete_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "get_replayed_tool_calls": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_evidence.py",
        ),
        "save_llm_interaction": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "set_trace_link": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "start_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
    },
    "SQLiteStateStore": {
        "_insert_trace": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "_patch_fence": ("SQLStateStore", "_SqlEvidence", "base.py", "_sql_evidence.py"),
        "checkpoint_step": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "complete_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "get_replayed_tool_calls": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_evidence.py",
        ),
        "save_llm_interaction": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "save_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "set_trace_link": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
        "start_tool_call": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_evidence.py"),
    },
}


RUNS_RELOCATIONS = {
    "MySQLStateStore": {
        "_creation_storage_format": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "create_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_runs.py"),
    },
    "PostgresStateStore": {
        "_check_idempotency": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "_creation_storage_format": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "create_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_runs.py"),
        "idempotency_scope_predicate": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
    },
    "SQLStateStore": {
        "_check_idempotency": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "_creation_storage_format": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "create_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_runs.py"),
        "idempotency_scope_predicate": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
    },
    "SQLiteStateStore": {
        "_check_idempotency": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "_creation_storage_format": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
        "create_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_runs.py"),
        "idempotency_scope_predicate": ("SQLStateStore", "_SqlRuns", "base.py", "_sql_runs.py"),
    },
}


MANAGED_RELOCATIONS = {
    "MySQLStateStore": {
        "_bind_admit": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_child": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_event": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_existing": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_intent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_park_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_seeds": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "bind_managed_child": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_managed.py"),
    },
    "PostgresStateStore": {
        "_bind_admit": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_child": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_event": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_existing": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_intent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_park_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_seeds": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "bind_managed_child": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_managed.py"),
    },
    "SQLStateStore": {
        "_bind_admit": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_child": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_event": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_existing": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_intent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_park_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_seeds": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "bind_managed_child": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_managed.py"),
    },
    "SQLiteStateStore": {
        "_bind_admit": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_child": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_event": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_existing": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_intent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_park_parent": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "_bind_seeds": ("SQLStateStore", "_SqlManaged", "base.py", "_sql_managed.py"),
        "bind_managed_child": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_managed.py"),
    },
}


TREE_RELOCATIONS = {
    "MySQLStateStore": {
        "_bfs_descendants": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_descendant_batch": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_traverse_subtree_bfs": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_usage_by_run": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_walk_ancestry": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "get_children_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_delegation_info": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_descendant_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_subtree": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "run_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "subtree_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
    },
    "PostgresStateStore": {
        "_bfs_descendants": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_descendant_batch": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_traverse_subtree_bfs": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_usage_by_run": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_walk_ancestry": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "get_children_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_delegation_info": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_descendant_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_subtree": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "run_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "subtree_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
    },
    "SQLStateStore": {
        "_bfs_descendants": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_descendant_batch": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_traverse_subtree_bfs": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_usage_by_run": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_walk_ancestry": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "get_children_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_delegation_info": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_descendant_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_subtree": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "run_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "subtree_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
    },
    "SQLiteStateStore": {
        "_bfs_descendants": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_descendant_batch": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_traverse_subtree_bfs": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_usage_by_run": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "_walk_ancestry": ("SQLStateStore", "_SqlTree", "base.py", "_sql_tree.py"),
        "get_children_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_delegation_info": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_descendant_page": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "get_subtree": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "run_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
        "subtree_usage": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_tree.py"),
    },
}


LIFECYCLE_RELOCATIONS = {
    "MySQLStateStore": {
        "_cas_finalize": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_finalize_error_info": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_predicate": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_scope": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "finalize_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "finalize_run_if_status_in": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_lifecycle.py",
        ),
        "finalize_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "get_run_control": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "is_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "mark_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "request_cancel": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "touch_progress": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "update_run_meta": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
    },
    "PostgresStateStore": {
        "_cas_finalize": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_finalize_error_info": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_predicate": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_scope": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "finalize_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "finalize_run_if_status_in": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_lifecycle.py",
        ),
        "finalize_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "get_run_control": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "is_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "mark_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "request_cancel": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "touch_progress": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "update_run_meta": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
    },
    "SQLStateStore": {
        "_cas_finalize": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_finalize_error_info": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_predicate": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_scope": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "finalize_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "finalize_run_if_status_in": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_lifecycle.py",
        ),
        "finalize_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "get_run_control": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "is_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "mark_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "request_cancel": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "touch_progress": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "update_run_meta": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
    },
    "SQLiteStateStore": {
        "_cas_finalize": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_finalize_error_info": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "_frontier_predicate": ("SQLStateStore", "_SqlLifecycle", "base.py", "_sql_lifecycle.py"),
        "finalize_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "finalize_run_if_status_in": (
            "SQLStateStore",
            "SQLStateStore",
            "base.py",
            "_sql_lifecycle.py",
        ),
        "finalize_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "get_run_control": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "is_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "mark_cancel_requested": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "request_cancel": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "touch_progress": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
        "update_run_meta": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_lifecycle.py"),
    },
}


PAUSE_RELOCATIONS = {
    "MySQLStateStore": {
        "_read_pause_summary": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "get_active_pause_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_approval_params": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pending_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_summary_query": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "pause_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "request_pause": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "try_pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
    },
    "PostgresStateStore": {
        "_read_pause_summary": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "get_active_pause_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_approval_params": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pending_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "request_pause": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "try_pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
    },
    "SQLStateStore": {
        "_read_pause_summary": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "get_active_pause_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_approval_params": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pending_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_summary_query": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "pause_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "request_pause": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "try_pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
    },
    "SQLiteStateStore": {
        "_read_pause_summary": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "get_active_pause_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_approval_params": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "get_pending_pause_summary": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "pause_summary_query": ("SQLStateStore", "_SqlPause", "base.py", "_sql_pause.py"),
        "pause_with_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "request_pause": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
        "try_pause_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_pause.py"),
    },
}


READS_RELOCATIONS = {
    "MySQLStateStore": {
        "_apply_metadata_filter": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_apply_run_filters": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_get_result_record": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "conversation_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_pauses_per_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_llm_interactions": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_loop_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_next_trace_index": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_pii_mapping": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_public_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_status": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_statuses": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_tool_calls": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces_after": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "list_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
    },
    "PostgresStateStore": {
        "_apply_metadata_filter": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_apply_run_filters": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_get_result_record": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "conversation_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_pauses_per_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_llm_interactions": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_loop_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_next_trace_index": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_pii_mapping": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_public_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_status": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_statuses": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_tool_calls": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces_after": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "list_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
    },
    "SQLStateStore": {
        "_apply_metadata_filter": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_apply_run_filters": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_get_result_record": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "conversation_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_pauses_per_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_llm_interactions": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_loop_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_next_trace_index": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_pii_mapping": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_public_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_status": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_statuses": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_tool_calls": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces_after": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "list_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
    },
    "SQLiteStateStore": {
        "_apply_metadata_filter": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_apply_run_filters": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "_get_result_record": ("SQLStateStore", "_SqlReads", "base.py", "_sql_reads.py"),
        "conversation_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_pauses_per_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "count_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_llm_interactions": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_loop_state": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_next_trace_index": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_pii_mapping": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_public_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_status": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_run_statuses": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_tool_calls": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_trace": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "get_traces_after": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
        "list_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_reads.py"),
    },
}


EVENTS_RELOCATIONS = {
    "MySQLStateStore": {
        "_insert_event": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "_refuse_note_insert": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "add_note": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "events_for_runs_query": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "get_events_for_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_next_event_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_notes": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_run_events": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "mark_notes_delivered": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "save_run_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
    },
    "PostgresStateStore": {
        "_insert_event": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "_refuse_note_insert": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "add_note": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_events_for_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_next_event_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_notes": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_run_events": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "mark_notes_delivered": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "save_run_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
    },
    "SQLStateStore": {
        "_insert_event": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "_refuse_note_insert": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "add_note": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "events_for_runs_query": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "get_events_for_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_next_event_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_notes": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_run_events": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "mark_notes_delivered": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "save_run_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
    },
    "SQLiteStateStore": {
        "_insert_event": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "_refuse_note_insert": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "add_note": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "events_for_runs_query": ("SQLStateStore", "_SqlEvents", "base.py", "_sql_events.py"),
        "get_events_for_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_next_event_sequence": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_notes": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "get_run_events": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "mark_notes_delivered": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
        "save_run_event": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_events.py"),
    },
}


# MySQL keeps its own claim_paused_run override (its super() target relocates).
CLAIMS_RELOCATIONS = {
    "MySQLStateStore": {
        "_cancel_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "_finish_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "consume_submission": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "release_run_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "submit_and_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
    },
    "PostgresStateStore": {
        "_cancel_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "_finish_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "claim_paused_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "consume_submission": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "release_run_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "submit_and_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
    },
    "SQLStateStore": {
        "_cancel_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "_finish_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "claim_paused_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "consume_submission": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "release_run_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "submit_and_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
    },
    "SQLiteStateStore": {
        "_cancel_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "_finish_claim_release": ("SQLStateStore", "_SqlClaims", "base.py", "_sql_claims.py"),
        "claim_paused_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "consume_submission": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "release_run_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
        "submit_and_claim": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_claims.py"),
    },
}


# MySQL keeps its own reclaim_run override (its super() target relocates).
LEASES_RELOCATIONS = {
    "MySQLStateStore": {
        "_scan_expired_runs": ("SQLStateStore", "_SqlLeases", "base.py", "_sql_leases.py"),
        "get_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "mark_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
    },
    "PostgresStateStore": {
        "_scan_expired_runs": ("SQLStateStore", "_SqlLeases", "base.py", "_sql_leases.py"),
        "get_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "mark_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "reclaim_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
    },
    "SQLStateStore": {
        "_scan_expired_runs": ("SQLStateStore", "_SqlLeases", "base.py", "_sql_leases.py"),
        "get_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "mark_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "reclaim_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
    },
    "SQLiteStateStore": {
        "_scan_expired_runs": ("SQLStateStore", "_SqlLeases", "base.py", "_sql_leases.py"),
        "get_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "mark_lease_expired": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
        "reclaim_run": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_leases.py"),
    },
}


# MySQL keeps its own _waiting_sweep_page override (its super() target relocates).
SWEEP_RELOCATIONS = {
    "MySQLStateStore": {
        "_scan_abandoned_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_sweep_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "sweep_abandoned_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
        "sweep_stale_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
    },
    "PostgresStateStore": {
        "_scan_abandoned_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_sweep_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_waiting_sweep_page": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "sweep_abandoned_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
        "sweep_stale_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
    },
    "SQLStateStore": {
        "_scan_abandoned_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_sweep_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_waiting_sweep_page": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "sweep_abandoned_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
        "sweep_stale_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
    },
    "SQLiteStateStore": {
        "_scan_abandoned_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_sweep_runs": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "_waiting_sweep_page": ("SQLStateStore", "_SqlSweep", "base.py", "_sql_sweep.py"),
        "sweep_abandoned_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
        "sweep_stale_runs": ("SQLStateStore", "SQLStateStore", "base.py", "_sql_sweep.py"),
    },
}


def _canonical_waiting_reads(members):
    # F6 changes observation APIs only. Keep the original hash for every other member.
    optional = {
        "list_runs": ", include_waiting: 'bool' = True",
        "count_runs": ", include_waiting: 'bool' = True",
        "_bfs_descendants": ", waiting_labels: 'dict[str, str | None] | None' = None",
        "_descendant_batch": ", waiting_labels: 'dict[str, str | None] | None' = None",
    }
    for name, addition in optional.items():
        assert members[name]["signature"].count(addition) == 1, name
        members[name]["signature"] = members[name]["signature"].replace(addition, "")
    assert members.get("get_waiting_for") == {
        "kind": "function",
        "signature": (
            "(self, run_id: 'str', *, tenant_id: 'TenantScope' = UNSCOPED) -> 'str | None'"
        ),
        "layers": [
            ("durability.py", "call"),
            ("durability.py", "call"),
            ("_sql_reads.py", "get_waiting_for"),
        ],
        "owner": "SQLStateStore",
    }
    del members["get_waiting_for"]


def _canonical_writes_snapshot(snapshot, cls):
    import copy

    assert snapshot["mro"] == EXPECTED_MRO[cls.__name__]
    result = copy.deepcopy(snapshot)
    _canonical_waiting_reads(result["members"])
    for name, (old_owner, new_owner) in WRITES_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == ("_sql_writes.py", name), name
        member["owner"] = old_owner
        member["layers"][-1] = ("base.py", name)
    for name, (old_owner, new_owner, old_file, new_file) in EVIDENCE_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in RUNS_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in MANAGED_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in TREE_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in LIFECYCLE_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in PAUSE_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in READS_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in EVENTS_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in CLAIMS_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in LEASES_RELOCATIONS[
        cls.__name__
    ].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    for name, (old_owner, new_owner, old_file, new_file) in SWEEP_RELOCATIONS[cls.__name__].items():
        assert name in result["members"]
        member = result["members"][name]
        assert member["owner"] == new_owner, name
        assert member["layers"][-1] == (new_file, name), name
        member["owner"] = old_owner
        member["layers"][-1] = (old_file, name)
    result["mro"].remove("_SqlSweep")
    result["mro"].remove("_SqlLeases")
    result["mro"].remove("_SqlClaims")
    result["mro"].remove("_SqlEvents")
    result["mro"].remove("_SqlReads")
    result["mro"].remove("_SqlPause")
    result["mro"].remove("_SqlLifecycle")
    result["mro"].remove("_SqlTree")
    result["mro"].remove("_SqlManaged")
    result["mro"].remove("_SqlRuns")
    result["mro"].remove("_SqlEvidence")
    result["mro"].remove("_SqlWrites")
    return result


@pytest.mark.parametrize("cls", STORES)
def test_exact_wrapper_signature_kind_and_mro_snapshot(cls):
    snapshot = _snapshot(cls)
    assert snapshot["mro"] == EXPECTED_MRO[cls.__name__]
    snapshot = _canonical_writes_snapshot(snapshot, cls)
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


@pytest.mark.parametrize("cls", STORES)
@pytest.mark.parametrize(
    "change",
    [
        "unexpected-base",
        "duplicate-base",
        "base-order",
        "moved-owner",
        "unlisted-owner",
        "moved-file",
        "evidence-owner",
        "evidence-file",
        "runs-owner",
        "runs-file",
        "managed-owner",
        "managed-file",
        "reads-owner",
        "reads-file",
        "events-owner",
        "events-file",
        "claims-owner",
        "claims-file",
        "leases-owner",
        "leases-file",
        "sweep-owner",
        "sweep-file",
        "unlisted-file",
        "wrapper-count",
        "wrapper-order",
        "signature",
        "kind",
        "missing-member",
        "extra-member",
        "waiting-reader-signature",
        "waiting-reader-wrappers",
        "waiting-list-default",
        "waiting-subtree-default",
    ],
)
def test_proposed_exact_write_relocation_still_refuses_unplanned_snapshot_changes(cls, change):
    import copy

    raw = _snapshot(cls)
    if change == "unexpected-base":
        raw["mro"].insert(1, "UnexpectedAnnotationBase")
    elif change == "duplicate-base":
        raw["mro"].insert(1, "_SqlWrites")
    elif change == "base-order":
        pos = raw["mro"].index("_SqlWrites")
        raw["mro"][pos : pos + 2] = reversed(raw["mro"][pos : pos + 2])
    elif change == "moved-owner":
        raw["members"]["_write_tx"]["owner"] = "SQLStateStore"
    elif change == "unlisted-owner":
        raw["members"]["_insert"]["owner"] = "_SqlWrites"
    elif change == "moved-file":
        raw["members"]["_write_tx"]["layers"][-1] = ("_other.py", "_write_tx")
    elif change == "evidence-owner":
        raw["members"]["_patch_fence"]["owner"] = "SQLStateStore"
    elif change == "evidence-file":
        raw["members"]["_insert_trace"]["layers"][-1] = ("_other.py", "_insert_trace")
    elif change == "runs-owner":
        raw["members"]["_check_idempotency"]["owner"] = "SQLStateStore"
    elif change == "runs-file":
        raw["members"]["_creation_storage_format"]["layers"][-1] = (
            "_other.py",
            "_creation_storage_format",
        )
    elif change == "managed-owner":
        raw["members"]["_bind_parent"]["owner"] = "SQLStateStore"
    elif change == "managed-file":
        raw["members"]["bind_managed_child"]["layers"][-1] = ("_other.py", "bind_managed_child")
    elif change == "reads-owner":
        raw["members"]["_apply_run_filters"]["owner"] = "UnplannedOwner"
    elif change == "reads-file":
        raw["members"]["get_trace"]["layers"][-1] = ("_other.py", "get_trace")
    elif change == "events-owner":
        raw["members"]["_insert_event"]["owner"] = "SQLStateStore"
    elif change == "events-file":
        raw["members"]["save_run_event"]["layers"][-1] = ("_other.py", "save_run_event")
    elif change == "claims-owner":
        raw["members"]["_cancel_claim_release"]["owner"] = "SQLStateStore"
    elif change == "claims-file":
        raw["members"]["release_run_claim"]["layers"][-1] = ("_other.py", "release_run_claim")
    elif change == "leases-owner":
        raw["members"]["_scan_expired_runs"]["owner"] = "SQLStateStore"
    elif change == "leases-file":
        raw["members"]["get_lease_expired"]["layers"][-1] = ("_other.py", "get_lease_expired")
    elif change == "sweep-owner":
        raw["members"]["_sweep_runs"]["owner"] = "SQLStateStore"
    elif change == "sweep-file":
        raw["members"]["sweep_stale_runs"]["layers"][-1] = ("_other.py", "sweep_stale_runs")
    elif change == "unlisted-file":
        raw["members"]["_insert"]["layers"][-1] = ("_sql_writes.py", "_insert")
    elif change == "wrapper-count":
        raw["members"]["update_many"]["layers"].insert(0, ("durability.py", "call"))
    elif change == "wrapper-order":
        layers = raw["members"]["save_trace"]["layers"]
        layers[0], layers[-2] = layers[-2], layers[0]
    elif change == "signature":
        raw["members"]["update_many"]["signature"] += " changed"
    elif change == "waiting-reader-signature":
        raw["members"]["get_waiting_for"]["signature"] += " changed"
    elif change == "waiting-reader-wrappers":
        raw["members"]["get_waiting_for"]["layers"].pop(0)
    elif change == "waiting-list-default":
        raw["members"]["list_runs"]["signature"] = raw["members"]["list_runs"]["signature"].replace(
            "include_waiting: 'bool' = True", "include_waiting: 'bool' = False"
        )
    elif change == "waiting-subtree-default":
        raw["members"]["_bfs_descendants"]["signature"] = raw["members"]["_bfs_descendants"][
            "signature"
        ].replace("waiting_labels: 'dict[str, str | None] | None' = None", "waiting_labels=None")
    elif change == "kind":
        raw["members"]["update_many"]["kind"] = "staticmethod"
    elif change == "missing-member":
        del raw["members"]["_write_tx"]
    else:
        raw["members"]["extra_method"] = copy.deepcopy(raw["members"]["_write_tx"])
    with pytest.raises(AssertionError):
        canonical = _canonical_writes_snapshot(raw, cls)
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        wrapped = sum(len(member["layers"]) > 1 for member in canonical["members"].values())
        assert (wrapped, digest) == BASELINE[cls.__name__]
