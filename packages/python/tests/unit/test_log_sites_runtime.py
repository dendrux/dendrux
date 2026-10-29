"""Runtime logging preserves failures without projecting arbitrary exception text."""

import pytest

from dendrux.runtime.tasks import RunTaskManager


async def test_task_diagnostic_is_closed_and_original_failure_is_preserved(caplog):
    failure = ValueError("PRIVATE_EXCEPTION_SENTINEL")

    async def fail():
        raise failure

    with pytest.raises(ValueError) as caught:
        await RunTaskManager().spawn("01ARZ3NDEKTSV4RRFFQ69G5FAV", fail())
    assert caught.value is failure
    (record,) = caplog.records
    assert record.code == "run_failed"
    assert record.error_type == "ValueError"
    assert record.run_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert "PRIVATE_EXCEPTION_SENTINEL" not in caplog.text + repr(record.__dict__)
