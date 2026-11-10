"""The docs state the blocked-content contracts that exist in code.

Expected strings come from the code (constants, messages, enum values), never a second copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dendrux import _codes
from dendrux._lifecycle import FailureReason
from dendrux.errors import ReplayRunError
from dendrux.runtime import _blocked_content
from dendrux.runtime import replay as replay_module
from dendrux.runtime import runner as runner_module
from dendrux.types import GovernanceEventType, RunStatus

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs"


def _doc(relative: str) -> str:
    return " ".join((DOCS / relative).read_text().split())


def _retry_refusal() -> ReplayRunError:
    with pytest.raises(ReplayRunError) as caught:
        runner_module._refuse_blocked_retry("01HZZZZZZZZZZZZZZZZZZZZZZZ")
    return caught.value


def _placeholder(count: int) -> str:
    from dendrux.guardrails._engine import _blocked_message
    from dendrux.types import Message, Role

    findings = [object()] * count
    return _blocked_message(Message(role=Role.USER, content="x"), findings).content  # type: ignore[arg-type]


def test_placeholder_form_is_documented():
    guardrails = _doc("architecture/guardrails.mdx")
    upgrade = _doc("upgrading-to-0.6.1.mdx")
    for count in (1, 2):
        assert f"`{_placeholder(count)}`" in guardrails
    assert f"`{_placeholder(1)}`" in upgrade


def test_blocked_content_log_code_is_documented():
    code = _codes.BLOCKED_CONTENT.name
    assert f"`{code}`" in _doc("reference/log-codes.mdx")
    for page in (
        "architecture/guardrails.mdx",
        "upgrading-to-0.6.1.mdx",
        "recipes/run-replay.mdx",
    ):
        assert f"`{code}`" in _doc(page), page
    assert f"`{code}`" in (REPO / "packages/python/CHANGELOG.md").read_text()


def test_retry_refusal_reason_and_message_are_documented():
    error = _retry_refusal()
    detail = str(error).split(": ", 1)[1]
    assert error.reason == "blocked_content"
    reason = f'reason="{error.reason}"'
    for page in (
        "reference/errors.mdx",
        "reference/agent.mdx",
        "recipes/run-replay.mdx",
        "upgrading-to-0.6.1.mdx",
    ):
        text = _doc(page)
        assert reason in text, page
        assert detail in text, page


def test_replay_omission_notice_is_documented():
    notice = replay_module._omitted_warning("<run id>")
    for page in ("recipes/run-replay.mdx", "upgrading-to-0.6.1.mdx"):
        assert notice in _doc(page), page


def test_marked_run_continuation_result_is_documented():
    reason = FailureReason.GUARDRAIL_BLOCKED.value
    message = runner_module._CONTINUATION_BLOCKED
    status = RunStatus.ERROR.value
    for page in ("architecture/guardrails.mdx", "upgrading-to-0.6.1.mdx"):
        text = _doc(page)
        assert f'`failure_reason="{reason}"`' in text, page
        assert f"`{status}`" in text, page
        assert message in text, page
    assert f"`{GovernanceEventType.GUARDRAIL_BLOCKED.value}`" in _doc("architecture/guardrails.mdx")


def test_recovery_holds_unexecuted_siblings_behind_the_approval_gate():
    waiting = RunStatus.WAITING_APPROVAL.value
    for page in ("guides/recovery.mdx", "upgrading-to-0.6.1.mdx"):
        text = _doc(page)
        assert "unexecuted" in text, page
        assert "behind the approval gate" in text, page
        assert f"`{waiting}`" in text, page


def test_limits_are_stated():
    for page in ("architecture/guardrails.mdx", "upgrading-to-0.6.1.mdx"):
        text = _doc(page)
        assert "Model reasoning text is not scanned" in text, page
        assert "0.7" in text, page
        assert "retry copies" in text, page
        assert "skill body digests" in text.lower(), page
        assert "cannot keep the blocked mark" in text, page
        assert "`PersistenceError`" in text, page


def test_changelog_lists_every_held_behaviour_change():
    text = " ".join((REPO / "packages/python/CHANGELOG.md").read_text().split())
    section = text.split("## 0.6.1", 1)[1].split("## 0.6.0", 1)[0]
    for phrase in (
        "empty `pii_mapping`",
        "outgoing tool-call parameters",
        "raw response, provider request and reasoning",
        "kind, source and metadata",
        "`PersistenceError`",
        "one extra indexed lookup",
        "`blocked_content`",
    ):
        assert phrase in section, phrase


_BLOCKED_PAGES = (
    "architecture/guardrails.mdx",
    "upgrading-to-0.6.1.mdx",
    "../packages/python/CHANGELOG.md",
)


@pytest.mark.parametrize("page", _BLOCKED_PAGES)
def test_source_lookup_costs_distinguish_native_and_custom_stores(page):
    text = _doc(page)
    assert f"chunks of {_blocked_content._SOURCE_CHUNK} source IDs" in text
    assert "per source" in text
    assert "`guardrail.blocked`" in text and "`limit=1`" in text


@pytest.mark.parametrize("page", _BLOCKED_PAGES)
def test_adoption_and_entry_mapping_contracts_are_documented(page):
    text = _doc(page)
    assert "first verdict" in text
    assert "first new event or trace" in text
    assert "verify the mark before" in text
    assert "Mappings from earlier allowed messages" in text


@pytest.mark.parametrize("page", (*_BLOCKED_PAGES, "guides/recovery.mdx"))
def test_blocked_reclaim_admission_precedes_history_limits(page):
    text = _doc(page)
    assert "before history rebuild" in text
    assert "`max_rebuild_rows`" in text
    assert "already-loaded run row" in text


@pytest.mark.parametrize("page", (*_BLOCKED_PAGES, "guides/recovery.mdx"))
def test_recovering_approval_adoption_keeps_the_blocked_stop(page):
    text = _doc(page)
    assert "recovering approval pauses" in text
    assert "before recovery resolution or approval work" in text
    assert "Clean adopted approval submissions keep the existing recovery history" in text


@pytest.mark.parametrize("page", (*_BLOCKED_PAGES, "guides/recovery.mdx"))
def test_reclaim_scans_an_unconsumed_submission_once(page):
    text = _doc(page)
    assert "submission it never consumed" in text
    assert "scans that stored submission once" in text
    assert "approved or denied" in text
    assert "no `approval.decided` event" in text
    assert "A clean stored submission is recovered as before, with no added statements" in text


@pytest.mark.parametrize("page", (*_BLOCKED_PAGES, "guides/recovery.mdx"))
def test_blocked_reclaim_keeps_checkpoint_accounting(page):
    assert "committed iteration and usage" in _doc(page)


def _sql_blocks(page: str) -> list[str]:
    text = (DOCS / page).read_text()
    return [b for b in re.findall(r"```sql\n(.*?)```", text, re.S) if "FROM agent_runs" in b]


def test_known_affected_runs_queries_run_on_sqlite():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from dendrux.db.models import AgentRun, Base, RunEvent

    blocks = _sql_blocks("upgrading-to-0.6.1.mdx")
    assert len(blocks) == 2
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                AgentRun(
                    id="a1", agent_name="a", tenant_id="t", failure_reason="guardrail_blocked"
                ),
                AgentRun(id="a2", agent_name="a", tenant_id="t"),
                AgentRun(id="a3", agent_name="a", tenant_id="t"),
                AgentRun(
                    id="a4", agent_name="a", tenant_id=None, failure_reason="guardrail_blocked"
                ),
            ]
        )
        session.flush()
        session.add(
            RunEvent(id="e1", agent_run_id="a3", event_type="guardrail.blocked", sequence_index=1)
        )
        session.commit()
        tenanted, untenanted = blocks
        first = session.execute(text(tenanted), {"tenant_id": "t", "after_id": ""}).scalars().all()
        assert first == ["a1", "a3"]
        paged = (
            session.execute(text(tenanted), {"tenant_id": "t", "after_id": "a1"}).scalars().all()
        )
        assert paged == ["a3"]
        assert session.execute(text(untenanted), {"after_id": ""}).scalars().all() == ["a4"]
