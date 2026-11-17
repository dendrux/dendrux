"""T0 pure moves: managed claim and recovery code lives in its own modules.

The original modules keep every name, so callers and patch seams are unchanged.
"""

from __future__ import annotations

import pytest

from dendrux.runtime import _managed_claim, _reclaim, _reclaim_managed, submit

CLAIM_NAMES = [
    "managed_submission_descriptor",
    "refuse_managed_results",
    "_managed_adoption_validator",
    "claim_managed_adoption",
    "_admit_managed_parent",
    "_refuse_tokenless_managed",
]
RECLAIM_NAMES = [
    "_pending_adoption",
    "_finish_saved_adoption",
    "_ManagedPending",
    "_refuse_bound_pending",
    "_managed_handoff",
    "_refuse_recovered_managed_results",
    "_handed_off",
]


@pytest.mark.parametrize("name", CLAIM_NAMES)
def test_submit_reexports_moved_claim_name(name: str) -> None:
    assert getattr(submit, name) is getattr(_managed_claim, name)


@pytest.mark.parametrize("name", RECLAIM_NAMES)
def test_reclaim_imports_moved_name(name: str) -> None:
    assert getattr(_reclaim, name) is getattr(_reclaim_managed, name)


def test_core_flow_stays_in_reclaim() -> None:
    assert _reclaim._check_before_claim.__module__ == _reclaim.__name__
    assert _reclaim.reclaim.__module__ == _reclaim.__name__


async def test_moved_claim_flow_honours_submit_patch_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching ``submit.<helper>`` still takes effect on the moved adoption claim."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from typing import Any

    from dendrux.runtime import _continuation_guardrails, durability, state
    from dendrux.skills import _session
    from dendrux.types import PauseState, ToolResult

    calls: list[str] = []
    won = SimpleNamespace(claim_epoch=1)

    class Store:
        async def get_pause_state(self, run_id: str) -> dict[str, Any]:
            return {"schema_version": 3}

        async def get_run(self, run_id: str) -> Any:
            return SimpleNamespace(agent_name="a", meta={})

        async def get_loop_state(self, run_id: str) -> dict[str, Any]:
            return {}

        async def submit_and_claim(self, run_id: str, **kwargs: Any) -> Any:
            return won

    async def operation_runner(store: Any, operation: Any, cancel: Any, *, run_id: str) -> Any:
        return await operation()

    @asynccontextmanager
    async def no_evidence(*args: Any):
        yield

    async def report(agent: Any, store: Any, run_id: str, governance: Any) -> None:
        calls.append("report_governance_change")

    wait = SimpleNamespace(pause_id="p", trace_order_offset=0)
    monkeypatch.setattr(PauseState, "from_managed_dict", lambda *a, **k: wait)
    monkeypatch.setattr(
        _session.SkillSession, "prepare", lambda agent: SimpleNamespace(available={})
    )
    monkeypatch.setattr(durability, "_ownership_handoff", operation_runner)
    monkeypatch.setattr(_continuation_guardrails, "_submission_evidence", no_evidence)
    monkeypatch.setattr(state, "claimed_store", lambda *a: object())
    monkeypatch.setattr(submit, "_governance_change", lambda agent, saved: calls.append("change"))
    monkeypatch.setattr(submit, "serialize_tool_results", lambda r: calls.append("serialize") or [])
    monkeypatch.setattr(submit, "report_governance_change", report)

    agent = SimpleNamespace(name="a", _on_governance_change="warn")
    result = await _managed_claim.claim_managed_adoption(
        Store(),  # type: ignore[arg-type]
        "run",
        agent=agent,  # type: ignore[arg-type]
        call_id="c",
        child_run_id="child",
        result=ToolResult(call_id="c", name="n", payload="p"),
    )
    assert result is won
    assert calls == ["change", "serialize", "report_governance_change"]


async def test_moved_warnings_keep_the_submit_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A handler on ``dendrux.runtime.submit`` still receives both moved warnings."""
    import logging
    from types import SimpleNamespace
    from typing import Any

    from dendrux import _codes
    from dendrux.errors import GovernanceMismatchError
    from dendrux.runtime.state import MANAGED_HEADER_KEY
    from dendrux.skills import _session
    from dendrux.types import generate_ulid

    class Store:
        async def get_run(self, run_id: str) -> Any:
            return SimpleNamespace(agent_name="a", meta={})

        async def get_loop_state(self, run_id: str) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(submit, "_governance_change", lambda agent, saved: {"changed": ["x"]})
    monkeypatch.setattr(
        _session.SkillSession, "prepare", lambda agent: SimpleNamespace(available={})
    )
    agent = SimpleNamespace(name="a", _on_governance_change="raise")
    run_id = generate_ulid()
    with caplog.at_level(logging.WARNING, logger="dendrux.runtime.submit"):
        with pytest.raises(GovernanceMismatchError):
            await _managed_claim._admit_managed_parent(agent, Store(), run_id)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            submit._refuse_tokenless_managed(run_id, {MANAGED_HEADER_KEY: {"x": 1}})
    names = {r.name for r in caplog.records}
    codes = " ".join(r.getMessage() for r in caplog.records)
    assert names == {"dendrux.runtime.submit"}
    assert _codes.GOVERNANCE_MISMATCH.name in codes
    assert _codes.MANAGED_PAUSE_ID_REQUIRED.name in codes
