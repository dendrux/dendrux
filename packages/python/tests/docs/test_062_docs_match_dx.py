"""Wording fixed in the 0.6.2 docs pack stays fixed.

The runtime behind these pages did not change, so behaviour tests alone cannot
catch the old wording coming back. These checks pin the corrected claims.
"""

from __future__ import annotations

from pathlib import Path

from dendrux import Agent

DOCS = Path(__file__).resolve().parents[4] / "docs"


def _page(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_replay_window_is_not_described_as_a_prompt_bound() -> None:
    page = _page("recipes/run-replay.mdx")
    assert "keeps the prompt bounded" not in page
    assert "not a limit on total prompt size" in page


def test_transform_docs_cover_client_results() -> None:
    doc = Agent.tool_result_transform.__doc__ or ""
    assert "submit_tool_results" in doc
    assert "client-submitted results" not in doc
    assert "does not run on client results" not in _page("recipes/images.mdx")


def test_provider_string_is_in_the_reference() -> None:
    page = _page("reference/agent.mdx")
    assert "provider: LLMProvider | str," in page
    assert '"anthropic:claude-haiku-4-5"' in page
    assert "`openrouter`" in _page("recipes/web-endpoint.mdx")
