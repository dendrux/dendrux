"""Every Python block in docs/ and the READMEs runs against a mock model, or says why not.

Blocks on a page run in order and share one namespace, as a reader following the page
would. A block that cannot run on its own carries a reason (see ``_snippets.py``):

    [//]: # "docs-test: skip - needs a live MCP server"

Skipped blocks are still checked: if they parse, their ``dendrux`` imports must resolve.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.docs._runner import DocsProvider, run_page
from tests.docs._snippets import discover_files, extract_snippets

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[4]
PAGES = {
    path.relative_to(REPO).as_posix(): extract_snippets(path, REPO) for path in discover_files(REPO)
}
PAGES = {name: snippets for name, snippets in PAGES.items() if snippets}

pytestmark = pytest.mark.skipif(not (REPO / "docs").is_dir(), reason="repository checkout only")


@pytest.fixture
def docs_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A scratch cwd, dummy keys, and a mock model in place of every real provider."""
    import dendrux.llm.anthropic as anthropic
    import dendrux.llm.openai as openai
    import dendrux.llm.openai_responses as openai_responses
    import dendrux.llm.openrouter as openrouter

    monkeypatch.chdir(tmp_path)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "docs-test-key")
    monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
    for module, name in (
        (anthropic, "AnthropicProvider"),
        (openai, "OpenAIProvider"),
        (openai_responses, "OpenAIResponsesProvider"),
        (openrouter, "OpenRouterProvider"),
    ):
        monkeypatch.setattr(module, name, DocsProvider)
    yield
    from dendrux.db.session import reset_engine

    asyncio.run(reset_engine())


def test_the_repository_has_python_snippets() -> None:
    assert PAGES, "no Python blocks found: the extractor or the docs path is broken"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_snippets(page: str, docs_sandbox: None) -> None:
    failures = run_page(PAGES[page])
    assert not failures, "\n".join(failures)
