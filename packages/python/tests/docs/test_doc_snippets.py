"""Every Python block in docs/ and the READMEs runs against a mock model, or says why not.

Blocks on a page run in order and share one namespace, as a reader following the page
would. A block that cannot run on its own carries a reason (see ``_snippets.py``):

    [//]: # "docs-test: skip - needs a live MCP server"

Skipped blocks are still checked: if they parse, their ``dendrux`` imports must resolve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs._runner import run_page
from tests.docs._snippets import discover_files, extract_snippets

REPO = Path(__file__).resolve().parents[4]
PAGES = {
    path.relative_to(REPO).as_posix(): extract_snippets(path, REPO) for path in discover_files(REPO)
}
PAGES = {name: snippets for name, snippets in PAGES.items() if snippets}

pytestmark = pytest.mark.skipif(not (REPO / "docs").is_dir(), reason="repository checkout only")


def test_the_repository_has_python_snippets() -> None:
    assert PAGES, "no Python blocks found: the extractor or the docs path is broken"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_snippets(page: str, docs_sandbox: None) -> None:
    failures = run_page(PAGES[page])
    assert not failures, "\n".join(failures)
