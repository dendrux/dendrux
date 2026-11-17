"""The statuses a direct resume may ingest images for are pinned as shared constants."""

from types import SimpleNamespace

import pytest

from dendrux.loops import ReActLoop
from dendrux.runtime import _run_images, _run_pending, runner
from dendrux.types import RunStatus

RESULTS_STATUSES = {RunStatus.WAITING_CLIENT_TOOL.value, RunStatus.WAITING_APPROVAL.value}
INPUT_STATUSES = {RunStatus.WAITING_HUMAN_INPUT.value}


class _AllowedError(Exception):
    pass


class _RefusedError(Exception):
    def __init__(self, expected: str) -> None:
        super().__init__(expected)
        self.expected = expected


class _Store:
    def __init__(self, status: str) -> None:
        self._status = status

    async def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(status=self._status, tenant_id=None)

    async def get_pause_state(self, run_id: str) -> dict:
        return {"pending_tool_calls": []}


@pytest.fixture
def patched(monkeypatch):
    async def refuse(state_store, run_id, expected_status):
        raise _RefusedError(expected_status)

    def allow(*args, **kwargs):
        raise _AllowedError

    monkeypatch.setattr(_run_pending, "_raise_resume_claim_failure", refuse)
    monkeypatch.setattr(_run_images, "_image_runtime", allow)


async def _ingest(status: str, *, results: bool) -> None:
    await runner._ingest_resume_images(
        None,
        None,
        _Store(status),
        "run",
        loop=ReActLoop(),
        tool_results=[] if results else None,
        input_items=() if results else ("hello",),
    )


@pytest.mark.parametrize("status", [s.value for s in RunStatus])
@pytest.mark.parametrize("results", [True, False])
async def test_status_gate_refuses_the_same_statuses(patched, status, results):
    allowed = RESULTS_STATUSES if results else INPUT_STATUSES
    if status in allowed:
        with pytest.raises(_AllowedError):
            await _ingest(status, results=results)
        return
    with pytest.raises(_RefusedError) as refused:
        await _ingest(status, results=results)
    assert refused.value.expected == (
        RunStatus.WAITING_CLIENT_TOOL.value if results else RunStatus.WAITING_HUMAN_INPUT.value
    )


def test_resume_status_constants_match_previous_tuples():
    from dendrux._lifecycle import RESUME_INPUT_STATUS_VALUES, RESUME_RESULTS_STATUS_VALUES

    assert RESUME_RESULTS_STATUS_VALUES == ("waiting_client_tool", "waiting_approval")
    assert RESUME_INPUT_STATUS_VALUES == ("waiting_human_input",)
