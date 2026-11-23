"""Claims made by the 0.8 docs stay true: stale wording is gone and named names exist."""

from __future__ import annotations

import inspect
from pathlib import Path

import dendrux
from dendrux import errors
from dendrux.store import RunDetail, RunStore, RunSummary
from dendrux.testing import FakeModel
from dendrux.types import Limits, RunResult

DOCS = Path(__file__).resolve().parents[4] / "docs"


def _page(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_subagents_guide_no_longer_denies_parallel_branches() -> None:
    page = _page("guides/subagents.mdx")
    assert "There are no parallel or background subagent branches" not in page
    assert "Two subagent calls in one response are refused" not in page
    assert "planned for 0.7.1" not in page
    assert "max_parallel_subagents" in page
    assert "exclusive_subagents" in page
    assert "managed_fanout_limit" in page


def test_documented_public_names_exist() -> None:
    assert callable(dendrux.cancel_run)
    assert "on_started" in inspect.signature(dendrux.Agent.run).parameters
    assert "on_started" in inspect.signature(dendrux.Agent.retry).parameters
    assert errors.OnStartedError.__mro__[1] is RuntimeError
    assert not issubclass(errors.OnStartedError, errors.RunControlError)
    assert "waiting_for" in RunSummary.__dataclass_fields__
    assert "waiting_for" in RunDetail.__dataclass_fields__
    assert "include_waiting" in inspect.signature(RunStore.list_runs).parameters
    assert "include_waiting" in inspect.signature(RunStore.count_runs).parameters
    assert hasattr(RunStore, "get_pendings")
    assert "pendings" in RunResult.__dataclass_fields__
    assert Limits().max_parallel_subagents == 4
    assert FakeModel is not None


def test_error_mapping_docstring_keeps_503_and_lists_on_started() -> None:
    doc = errors.__doc__ or ""
    line503 = next(line for line in doc.splitlines() if line.strip().startswith("503:"))
    assert "ProviderUnavailableError" in doc.split("503:")[1].split("504:")[0]
    assert "ProviderRateLimitError" in line503
    assert "OnStartedError" in doc.split("500:")[1].split("502:")[0]
    assert "OnStartedError" in _page("reference/errors.mdx")


def test_cancel_and_on_started_are_documented() -> None:
    assert "## Cancel a run without an Agent" in _page("recipes/cancel-run.mdx")
    assert "Custom recorders and a cancelled pause" in _page("architecture/cancellation.mdx")
    assert "not approval-gatable in 0.8" in _page("architecture/governance.mdx")
    assert "### `on_started`" in _page("reference/agent.mdx")
    assert "Save the id and return; keep it short" in _page("reference/agent.mdx")


def test_upgrade_page_covers_every_migration_item() -> None:
    page = _page("upgrading-to-0.8.0.mdx")
    for needle in (
        "managed_fanout_limit",
        "aiosqlite>=0.22.1",
        "waiting_for",
        "dendrux.cancel_run",
        "OnStartedError",
        "RunControlError",
        "StructuredOutputValidationError",
        "TenantScopeRequiredError",
        "binding_conflict",
        '{"task"',
        "lease_scope",
        "--exclude-waiting",
    ):
        assert needle in page, needle
