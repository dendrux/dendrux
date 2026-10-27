"""The docs snippet extractor: fences, skip markers and file discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.docs._snippets import discover_files, extract_snippets

if TYPE_CHECKING:
    from pathlib import Path


def _extract(tmp_path: Path, text: str, name: str = "page.mdx"):
    path = tmp_path / name
    path.write_text(text)
    return extract_snippets(path, tmp_path)


def test_python_fences_are_extracted_with_file_and_line(tmp_path: Path) -> None:
    snippets = _extract(tmp_path, "# T\n\ntext\n\n```python\nx = 1\nprint(x)\n```\n")
    assert len(snippets) == 1
    assert snippets[0].file == "page.mdx"
    assert snippets[0].line == 6
    assert snippets[0].code == "x = 1\nprint(x)\n"
    assert snippets[0].skip_reason is None


def test_other_languages_are_ignored(tmp_path: Path) -> None:
    text = "```bash\npip install x\n```\n\n```json\n{}\n```\n\n```\nplain\n```\n"
    assert _extract(tmp_path, text) == []


def test_py_alias_and_fence_meta_are_python(tmp_path: Path) -> None:
    text = '```py\na = 1\n```\n\n```python title="x.py"\nb = 2\n```\n'
    assert [s.code for s in _extract(tmp_path, text)] == ["a = 1\n", "b = 2\n"]


def test_skip_marker_above_a_fence_gives_a_reason(tmp_path: Path) -> None:
    text = '[//]: # "docs-test: skip - needs a live provider"\n\n```python\nx = 1\n```\n'
    assert _extract(tmp_path, text)[0].skip_reason == "needs a live provider"


def test_a_marker_not_directly_above_a_fence_is_an_error(tmp_path: Path) -> None:
    text = '[//]: # "docs-test: skip - r"\n\nSome prose in between.\n\n```python\nx = 1\n```\n'
    with pytest.raises(ValueError, match=r"page\.mdx:1.*fence"):
        _extract(tmp_path, text)


def test_marker_covers_only_the_next_fence(tmp_path: Path) -> None:
    text = '[//]: # "docs-test: skip - r"\n```python\na = 1\n```\n\n```python\nb = 2\n```\n'
    assert [s.skip_reason for s in _extract(tmp_path, text)] == ["r", None]


@pytest.mark.parametrize(
    "marker",
    ['[//]: # "docs-test: skip"', '[//]: # "docs-test: skip - "', '[//]: # "docs-test: nope - r"'],
)
def test_malformed_markers_are_errors_naming_the_place(tmp_path: Path, marker: str) -> None:
    with pytest.raises(ValueError, match=r"page\.mdx:1"):
        _extract(tmp_path, f"{marker}\n\n```python\nx = 1\n```\n")


def test_other_reference_notes_are_not_markers(tmp_path: Path) -> None:
    text = '[//]: # "Implementation note"\n\n```python\nx = 1\n```\n'
    assert _extract(tmp_path, text)[0].skip_reason is None


def test_indented_fences_in_lists_are_dedented(tmp_path: Path) -> None:
    text = "1. step\n\n   ```python\n   if True:\n       y = 1\n   ```\n"
    assert _extract(tmp_path, text)[0].code == "if True:\n    y = 1\n"


def test_unclosed_fence_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"page\.mdx:1"):
        _extract(tmp_path, "```python\nx = 1\n")


def test_discover_finds_docs_and_readmes_but_not_vendored_trees(tmp_path: Path) -> None:
    (tmp_path / "docs" / "recipes").mkdir(parents=True)
    (tmp_path / "docs" / "a.mdx").write_text("")
    (tmp_path / "docs" / "recipes" / "b.mdx").write_text("")
    (tmp_path / "docs" / "notes.md").write_text("")
    (tmp_path / "README.md").write_text("")
    (tmp_path / "packages" / "python").mkdir(parents=True)
    (tmp_path / "packages" / "python" / "README.md").write_text("")
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "README.md").write_text("")
    found = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert found == [
        "README.md",
        "docs/a.mdx",
        "docs/recipes/b.mdx",
        "packages/python/README.md",
    ]
