"""Guard plain-Markdown docs against MDX expressions and dead links."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs"


def _without_fences(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            lines.append("\n")
        elif marker:
            fence = marker[1]
            lines.append("\n")
        else:
            lines.append(line)
    return "".join(lines)


def _without_code(text: str) -> str:
    return re.sub(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", "", _without_fences(text), flags=re.S)


def _anchors(text: str) -> set[str]:
    seen: Counter[str] = Counter()
    result = set()
    for title in re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", _without_fences(text), re.M):
        title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        suffix = f"-{seen[slug]}" if seen[slug] else ""
        seen[slug] += 1
        result.add(slug + suffix)
    return result


@pytest.mark.parametrize(
    "text, bare",
    [
        ("GET /runs/{id}", True),
        ("GET /runs/\\{id\\}", False),
        ("`GET /runs/{id}`", False),
        ("``literal ` {id}``", False),
        ("```python\nvalue = {}\n```\n", False),
        ("~~~text\n{}\n~~~\n{id}", True),
    ],
)
def test_mdx_expression_scanner(text: str, bare: bool) -> None:
    assert bool(re.search(r"(?<!\\)[{}]", _without_code(text))) is bare


@pytest.mark.skipif(not DOCS.is_dir(), reason="repository docs only")
def test_mdx_pages_have_no_bare_braces() -> None:
    issues = []
    for path in sorted(DOCS.rglob("*.mdx")):
        for number, line in enumerate(_without_code(path.read_text()).splitlines(), 1):
            if re.search(r"(?<!\\)[{}]", line):
                issues.append(f"{path.relative_to(REPO)}:{number}: {line}")
    assert not issues, "Bare MDX braces outside code:\n" + "\n".join(issues)


@pytest.mark.skipif(not DOCS.is_dir(), reason="repository docs only")
def test_repository_docs_links_resolve() -> None:
    paths = subprocess.check_output(
        ["git", "ls-files", "*.md", "*.mdx"], cwd=REPO, text=True
    ).splitlines()
    issues = []
    for name in paths:
        source = REPO / name
        if not source.is_file():
            continue
        text = _without_code(source.read_text())
        for target in re.findall(r"\[[^]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            if path.startswith("/docs/"):
                destination = DOCS / (path.removeprefix("/docs/").rstrip("/") + ".mdx")
            elif not path:
                destination = source
            elif path.endswith((".md", ".mdx")):
                destination = source.parent / path
            else:
                continue
            if not destination.is_file():
                issues.append(f"{name}: missing page {target}")
            elif url.fragment and unquote(url.fragment) not in _anchors(destination.read_text()):
                issues.append(f"{name}: missing anchor {target}")
    assert not issues, "Broken repository docs links:\n" + "\n".join(issues)
