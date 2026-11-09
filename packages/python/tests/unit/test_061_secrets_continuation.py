"""Continuation lookup failures preserve safe errors and cancellation."""

from types import SimpleNamespace

import pytest

from dendrux.errors import PersistenceError
from dendrux.runtime.runner import _stop_marked_continuation

SECRET = "AKIAIOSFODNN7EXAMPLE"


@pytest.mark.parametrize(
    "record",
    [
        SimpleNamespace(id="other", tenant_id="t", meta={}, failure_reason=None),
        SimpleNamespace(id="run", tenant_id="t", meta=SECRET, failure_reason=None),
        SimpleNamespace(tenant_id="t", meta={}, failure_reason=None),
    ],
)
async def test_malformed_continuation_row_is_safe(record):
    class Store:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected store call: {name}")

    with pytest.raises(PersistenceError) as caught:
        await _stop_marked_continuation(Store(), "run", record)
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize(
    "row",
    [
        None,
        SimpleNamespace(tenant_id=None),
        SimpleNamespace(id="run", meta=SECRET),
        SimpleNamespace(id="other", meta={}),
    ],
)
async def test_pending_note_malformed_lookup_is_safe(row):
    from dendrux.loops._helpers import _check_blocked_run

    class Store:
        async def get_run(self, run_id):
            return row

    with pytest.raises(PersistenceError) as caught:
        await _check_blocked_run(Store(), "run")
    assert SECRET not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize("meta", [None, {}])
async def test_pending_note_clean_lookup(meta):
    from dendrux.loops._helpers import _check_blocked_run

    class Store:
        async def get_run(self, run_id):
            return SimpleNamespace(id=run_id, meta=meta)

    await _check_blocked_run(Store(), "run")


@pytest.mark.parametrize("meta", [None, {}, {"app": "value"}])
async def test_unmarked_continuation_reads_nothing_more(meta):
    # The continuation verdict reuses the run row already loaded for validation;
    # a clean continuation adds no store reads on any store implementation.
    record = SimpleNamespace(id="run", tenant_id=None, meta=meta, failure_reason=None)

    class Store:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected store call: {name}")

    assert await _stop_marked_continuation(Store(), "run", record) is None


async def test_custom_store_without_run_row_cannot_hold_a_mark():
    # Custom blocked writes verify the mark through get_run before any evidence,
    # so a store that cannot return the run row never holds a marked run.
    class Store:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected store call: {name}")

    assert await _stop_marked_continuation(Store(), "run", None) is None


async def test_guardrail_free_custom_dispatch_adds_no_reads():
    # Custom inbox stores check pending-note marks before dispatch only when the run
    # is guarded; a guardrail-free dispatch keeps its previous reads.
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.types import ToolCall

    calls = []

    class Store:
        storage_format = 1

        async def add_note(self, *args, **kwargs):
            raise AssertionError("unexpected add_note")

        async def get_run_control(self, run_id):
            calls.append("get_run_control")
            return SimpleNamespace(
                cancel_requested=False, pause_requested=False, has_pending_notes=True
            )

        async def get_run(self, run_id):
            calls.append("get_run")
            return SimpleNamespace(id=run_id, meta={})

        async def touch_progress(self, run_id):
            calls.append("touch_progress")
            return True

    recorder = PersistenceRecorder(Store(), "run")
    await recorder.on_tool_started("run", ToolCall(name="effect", params={}), 1)
    assert calls == ["touch_progress"]  # exactly the calls 91024ac3 made
