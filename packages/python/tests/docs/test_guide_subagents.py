"""Run the subagents guide's main example on SQLite with mock providers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.store import PendingPause, RunStore
from dendrux.types import LLMResponse, RunStatus, ToolCall, UsageStats

GUIDE = Path(__file__).resolve().parents[4] / "docs" / "guides" / "subagents.mdx"


def _example() -> Any:
    assert GUIDE.is_file(), "the subagents guide must exist"
    match = re.search(r"^```python\n(.*?)^```", GUIDE.read_text(), re.S | re.M)
    assert match, "the subagents guide must include its main Python example"
    assert "async def review_report(" in match[1], "the first example must define review_report"
    namespace: dict[str, Any] = {}
    exec(compile(match[1], str(GUIDE), "exec"), namespace)
    return namespace["review_report"]


def _fragment_code(marker: str) -> Any:
    blocks = re.findall(r"^```python\n(.*?)^```", GUIDE.read_text(), re.S | re.M)
    matches = [block for block in blocks if marker in block]
    assert len(matches) == 1, f"the guide must have one fragment containing {marker}"
    # Keep top-level await in the caller's task, as required by workflow streaming.
    return compile(matches[0], str(GUIDE), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def _response(*, text: str | None = None, calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=calls or [],
        usage=UsageStats(input_tokens=3, output_tokens=2, total_tokens=5),
    )


@pytest.mark.parametrize("approved", [True, False])
async def test_main_example_routes_approval_and_counts_subtree_usage(
    tmp_path: Path, approved: bool
) -> None:
    example = _example()
    url = f"sqlite+aiosqlite:///{tmp_path / 'subagents.sqlite'}"
    lead_model = MockLLM(
        [
            _response(calls=[ToolCall("reviewer", {"task": "Review: Sales rose."})]),
            _response(text="Report reviewed."),
        ]
    )
    reviewer_model = MockLLM(
        [
            _response(calls=[ToolCall("count_words", {"text": "Sales rose."})]),
            _response(text="Two words. Clear draft." if approved else "Review declined."),
        ]
    )
    pauses: list[PendingPause] = []

    async def decide(pending: PendingPause) -> bool:
        pauses.append(pending)
        assert pending.kind == "approval"
        assert pending.status == "waiting_approval"
        assert pending.pause_id
        assert pending.agent_name == "reviewer"
        assert pending.agent_path == ["lead", "reviewer"]
        assert pending.origin_run_id and pending.origin_run_id != pending.run_id
        assert [call.name for call in pending.tool_calls] == ["count_words"]
        # A restart/read route can recover the same root-facing pause without an Agent.
        async with RunStore.from_database_url(url) as store:
            namespace: dict[str, Any] = {
                "store": store,
                "root_run_id": pending.run_id,
                "tenant_id": "acme",
            }
            await eval(_fragment_code("pending = await store.get_pending("), namespace)
            reloaded = namespace["pending"]
            assert reloaded is not None
            assert reloaded.to_dict() == pending.to_dict()
        return approved

    result, tree = await example(
        url, "acme", decide, lead_provider=lead_model, reviewer_provider=reviewer_model
    )

    assert len(pauses) == 1
    assert result.run_id == pauses[0].run_id == tree.root_run_id
    assert result.status is RunStatus.SUCCESS
    assert result.answer == "Report reviewed."
    assert len(tree.nodes) == 2
    assert [node.run.agent_name for node in tree.nodes] == ["lead", "reviewer"]
    assert all(node.run.tenant_id == "acme" for node in tree.nodes)
    assert tree.nodes[1].run.run_id == pauses[0].origin_run_id
    assert result.usage.total_tokens == 10
    assert [(node.usage.input_tokens, node.usage.output_tokens) for node in tree.nodes] == [
        (6, 4),
        (6, 4),
    ]
    assert (tree.usage.input_tokens, tree.usage.output_tokens) == (12, 8)
    # The child gets a fresh brief. Its answer reaches the parent as tool data.
    assert reviewer_model.call_history[0]["messages"][-1].content == "Review: Sales rose."
    observed = [m for m in lead_model.call_history[1]["messages"] if m.role.value == "tool"]
    assert len(observed) == 1
    assert json.loads(observed[0].content) == (
        "Two words. Clear draft." if approved else "Review declined."
    )
    child_observations = [
        m for m in reviewer_model.call_history[1]["messages"] if m.role.value == "tool"
    ]
    assert len(child_observations) == 1
    if approved:
        assert json.loads(child_observations[0].content) == 2
    else:
        assert "declined" in child_observations[0].content.lower()


async def test_stream_example_yields_root_text_and_root_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _fragment_code("async with lead.stream(")
    lead_model = MockLLM(
        [
            _response(calls=[ToolCall("reviewer", {"task": "Review: Sales rose."})]),
            _response(text="Report reviewed."),
        ]
    )
    child_model = MockLLM([_response(text="Child assessment stays out of root text.")])
    url = f"sqlite+aiosqlite:///{tmp_path / 'subagents-stream.sqlite'}"
    async with (
        Agent(
            name="reviewer", description="Review one draft.", provider=child_model, prompt="Review."
        ) as reviewer,
        Agent(
            name="lead",
            provider=lead_model,
            prompt="Delegate the review, then summarize it.",
            subagents=[reviewer],
            database_url=url,
        ) as lead,
    ):
        namespace: dict[str, Any] = {"lead": lead, "tenant_id": "acme"}
        await eval(code, namespace)
        result = namespace["result"]
        assert result is namespace["stream"].result
        assert result.status is RunStatus.SUCCESS
        assert result.run_id == namespace["stream"].run_id
        assert result.answer == "Report reviewed."
        assert capsys.readouterr().out == "Report reviewed."
        async with RunStore.from_database_url(url) as store:
            tree = await store.get_subtree(result.run_id, tenant_id="acme", max_nodes=10)
            assert len(tree.nodes) == 2
            assert all(node.run.status == "success" for node in tree.nodes)
    assert child_model.calls_made == 1
    assert lead_model.calls_made == 2
