"""Traceparent validation and engine/store option separation."""

import importlib.util

import pytest

from dendrux.db import _engine_options
from dendrux.runtime import context


@pytest.mark.parametrize(
    "link",
    [
        "bad",
        "00-" + "0" * 32 + "-" + "1" * 16 + "-01",
        "00-" + "1" * 32 + "-" + "0" * 16 + "-01",
        "00-" + "A" * 32 + "-" + "1" * 16 + "-01",
        "00-" + "1" * 32 + "-" + "1" * 16 + "-01\n",
    ],
)
def test_traceparent_hostile_values_refused(link):
    assert importlib.util.find_spec("dendrux.runtime._tracelink") is not None, (
        "trace links need validation"
    )
    from dendrux.runtime import _tracelink

    assert _tracelink.validate_trace_link(link) is None


def test_traceparent_valid_and_delegation_fields():
    link = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    assert importlib.util.find_spec("dendrux.runtime._tracelink") is not None, (
        "trace links need validation"
    )
    from dendrux.runtime import _tracelink

    assert _tracelink.validate_trace_link(link) == link
    ctx = context.DelegationContext("run", trace_link=link, tool_span_link=link)
    assert ctx.trace_link == ctx.tool_span_link == link


def test_store_options_never_reach_create_async_engine():
    options = _engine_options.engine_options(
        "sqlite+aiosqlite://", {"notify_channel": "app_events", "echo": True}
    )
    assert "notify_channel" not in options, "notify_channel is a store option"
    assert options["echo"] is True


def test_delegation_context_preserves_legacy_positional_slots():
    store, loop, clock = object(), object(), [0.0]
    ctx = context.DelegationContext("run", 0, True, "db", None, set(), None, store, loop, clock)
    assert ctx.state_store is store and ctx.loop is loop and ctx._heartbeat_at is clock
