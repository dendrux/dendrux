"""Validate explicit StateStore promises without inferring legacy capabilities."""

from __future__ import annotations

import inspect
import logging
import weakref
from typing import Any, NoReturn

from dendrux import _codes
from dendrux.errors import StoreCapabilityError
from dendrux.runtime.state import StoreCapabilities

_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "fenced_writes": {
        name: ("claim_epoch",)
        for name in (
            "save_trace",
            "save_tool_call",
            "save_usage",
            "save_llm_interaction",
            "save_run_event",
            "add_note",
            "mark_notes_delivered",
            "update_run_meta",
            "touch_progress",
        )
    },
    "retry_ids": {
        "save_trace": ("trace_id",),
        "save_tool_call": ("row_id",),
        "save_run_event": ("event_id",),
        "save_usage": ("usage_id",),
        "save_llm_interaction": ("interaction_id",),
    },
    "checkpointing": {
        "create_run": ("seed_traces", "pii_mapping"),
        "save_tool_call": ("outcome",),
        "complete_tool_call": ("pii_mapping",),
        "consume_submission": ("pii_mapping",),
        "checkpoint_step": (),
        "get_traces_after": (),
        "conversation_traces": (),
        "save_trace": ("trace_kind",),
        "claim_paused_run": (),
        "submit_and_claim": (),
    },
    "atomic_completion": {
        "start_tool_call": (),
        "complete_tool_call": (),
        "finalize_with_event": ("trace",),
    },
    "recovery": {
        "mark_lease_expired": ("finalise_unrecoverable",),
        "get_lease_expired": (),
        "reclaim_run": (),
    },
    "notify": {"_after_event_insert": ()},
}
_CACHE: dict[int, tuple[weakref.ReferenceType[Any], StoreCapabilities]] = {}
logger = logging.getLogger(__name__)


def validate_capabilities(store: Any) -> StoreCapabilities:
    """Validate each declared capability once per live store object."""
    key = id(store)
    cached = _CACHE.get(key)
    if cached is not None and cached[0]() is store:
        return cached[1]
    caps = inspect.getattr_static(store, "capabilities", None)
    if caps is None:
        return StoreCapabilities()
    if not isinstance(caps, StoreCapabilities) or caps.version != 1:
        _refuse(store, "version", "capabilities", ("version=1",))
    for capability, methods in _REQUIREMENTS.items():
        if not getattr(caps, capability):
            continue
        for method, keywords in methods.items():
            # Signature inspection must not invoke application descriptors or spies.
            target = inspect.getattr_static(store, method, None)
            if not callable(target):
                _refuse(store, capability, method, keywords or (method,))
            try:
                params = inspect.signature(target).parameters
            except (TypeError, ValueError):
                _refuse(store, capability, method, keywords or (method,))
            missing = tuple(
                k
                for k in keywords
                if k not in params or params[k].kind is not inspect.Parameter.KEYWORD_ONLY
            )
            if missing:
                _refuse(store, capability, method, missing)
    try:
        ref = weakref.ref(store, lambda _ref: _CACHE.pop(key, None))
    except TypeError:
        return caps
    _CACHE[key] = (ref, caps)
    return caps


def _refuse(store: Any, capability: str, method: str, missing: tuple[str, ...]) -> NoReturn:
    _codes.log_code(logger, _codes.STORE_CAPABILITY, run_id=None)
    raise StoreCapabilityError(type(store).__name__, capability, method, missing)


def capabilities(store: Any) -> StoreCapabilities:
    """Read a declaration; absent means legacy regardless of method signatures."""
    declared = inspect.getattr_static(store, "capabilities", None)
    if isinstance(declared, property):
        declared = declared.__get__(store, type(store))
    return declared if isinstance(declared, StoreCapabilities) else StoreCapabilities()
