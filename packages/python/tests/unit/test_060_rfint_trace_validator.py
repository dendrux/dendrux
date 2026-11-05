"""Direct tests for the trace write validator."""

import json
import logging

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime.state import TraceWrite
from dendrux.runtime.stores.sql.base import _validate_trace_write
from dendrux.types import generate_ulid

RUN_ID = generate_ulid()
GOOD_MESSAGE = {"role": "user", "content": "hi", "meta": {}}


def _trace(kind: str, content: str, meta=None) -> TraceWrite:
    return TraceWrite("t1", "user", content, 0, trace_kind=kind, meta=meta)


@pytest.mark.parametrize(
    "kind,content",
    [
        ("turn", "plain text"),
        ("seed", "plain text"),
        ("rewrite", json.dumps([GOOD_MESSAGE, {"role": "user", "content": "no meta"}])),
        ("state", json.dumps({"kind": "image_view"})),
    ],
)
def test_valid_trace_writes_pass(kind, content):
    _validate_trace_write(RUN_ID, _trace(kind, content))


def test_unsupported_trace_kind_rejected():
    with pytest.raises(ValueError, match="Unsupported trace kind"):
        _validate_trace_write(RUN_ID, _trace("other", "x"))


@pytest.mark.parametrize(
    "content",
    ['{"not": "a list"}', json.dumps([GOOD_MESSAGE] * 100_000)],
    ids=["not-list", "count"],
)
def test_rewrite_shape_and_count_limits(content, caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(PersistenceError) as caught:
        _validate_trace_write(RUN_ID, _trace("rewrite", content))
    assert caught.value.code == "checkpoint_too_large"


@pytest.mark.parametrize(
    "entries",
    [
        ["not a dict"],
        [{"role": 1, "content": "x"}],
        [{"role": "user", "content": 1}],
        [GOOD_MESSAGE, {"role": "user"}],
    ],
)
def test_rewrite_entries_require_role_and_content(entries):
    with pytest.raises(ValueError, match="Rewrite entries require role and content"):
        _validate_trace_write(RUN_ID, _trace("rewrite", json.dumps(entries)))


def test_rewrite_entry_meta_must_be_object(caplog):
    entries = [{"role": "user", "content": "x", "meta": ["bad"]}]
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError, match="must be an object"):
        _validate_trace_write(RUN_ID, _trace("rewrite", json.dumps(entries)))
    assert "rewrite_invalid" in caplog.text


@pytest.mark.parametrize("content", ["[]", '{"kind": "unknown"}', '{"no": "kind"}'])
def test_state_trace_requires_versioned_payload(content):
    with pytest.raises(ValueError, match="versioned state payload"):
        _validate_trace_write(RUN_ID, _trace("state", content))
