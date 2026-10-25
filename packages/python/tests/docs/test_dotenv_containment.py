"""The docs sandbox blocks python-dotenv at its file boundary, not only at its public names.

References taken before the sandbox patches (early-bound aliases, a helper module's own
import) and the ``DotEnv`` class go through the library's discovery and file-open code, so
that code is cut off, and the whole environment is restored after every snippet. All data
here is synthetic: in-memory streams and files this test writes under ``tmp_path``.
"""

from __future__ import annotations

import builtins
import io
import os
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("dotenv")

from dotenv import dotenv_values as EARLY_DOTENV_VALUES  # noqa: N812, E402
from dotenv import find_dotenv as EARLY_FIND_DOTENV  # noqa: N812, E402
from dotenv import load_dotenv as EARLY_LOAD_DOTENV  # noqa: N812, E402

from tests.docs._runner import run_page  # noqa: E402
from tests.docs._snippets import Snippet  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _page(*blocks: str) -> list[str]:
    return run_page([Snippet("page.mdx", 10 * i + 1, code, None) for i, code in enumerate(blocks)])


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every open() and isfile() on a path that looks like a dotenv file."""
    touched: list[str] = []
    real_open, real_isfile = builtins.open, os.path.isfile

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if "env" in os.path.basename(str(file)):
            touched.append(f"open {file}")
        return real_open(file, *args, **kwargs)

    def spy_isfile(path: Any) -> bool:
        if "env" in os.path.basename(str(path)):
            touched.append(f"isfile {path}")
        return real_isfile(path)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os.path, "isfile", spy_isfile)
    return touched


def test_an_early_bound_find_dotenv_searches_nothing(docs_sandbox: None, spy: list[str]) -> None:
    assert EARLY_FIND_DOTENV() == ""
    assert spy == []


def test_an_early_bound_dotenv_values_with_a_path_reads_nothing(
    docs_sandbox: None, spy: list[str], tmp_path: Path
) -> None:
    target = tmp_path / "synthetic.env"
    target.write_text("SYNTHETIC_ONLY=1\n")
    spy.clear()
    assert EARLY_DOTENV_VALUES(str(target)) == {}
    assert not any(str(target) in entry for entry in spy)


def test_an_early_bound_load_dotenv_with_a_path_loads_nothing(
    docs_sandbox: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "synthetic.env"
    target.write_text("SYNTHETIC_LOAD_ONLY=1\n")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")  # prove the boundary, not the flag
    assert EARLY_LOAD_DOTENV(str(target)) is False
    assert "SYNTHETIC_LOAD_ONLY" not in os.environ


def test_the_dotenv_class_cannot_open_a_file(docs_sandbox: None, tmp_path: Path) -> None:
    from dotenv.main import DotEnv

    target = tmp_path / "synthetic.env"
    target.write_text("SYNTHETIC_CLASS_ONLY=1\n")
    assert DotEnv(str(target)).dict() == {}
    assert DotEnv(None, stream=io.StringIO("IN_MEMORY=1\n")).dict() == {"IN_MEMORY": "1"}


def test_set_key_cannot_write_a_file(docs_sandbox: None, tmp_path: Path) -> None:
    from dotenv import set_key

    target = tmp_path / "synthetic.env"
    target.write_text("A=1\n")
    with pytest.raises(RuntimeError, match="sandbox"):
        set_key(str(target), "B", "2")
    assert target.read_text() == "A=1\n"


def test_environment_changes_do_not_outlive_the_snippet(docs_sandbox: None) -> None:
    code = (
        "from dotenv.main import DotEnv\n"
        "import io, os\n"
        "DotEnv(None, stream=io.StringIO('QG_R3_SYNTHETIC=1\\n')).set_as_environment_variables()\n"
        "assert os.environ['QG_R3_SYNTHETIC'] == '1'\n"
        "os.environ['QG_R3_OTHER'] = 'x'\n"
    )
    assert _page(code) == []
    assert "QG_R3_SYNTHETIC" not in os.environ
    assert "QG_R3_OTHER" not in os.environ


def test_a_deleted_variable_comes_back_after_the_snippet(
    docs_sandbox: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QG_R3_KEEP", "kept")
    assert _page("import os\ndel os.environ['QG_R3_KEEP']\n") == []
    assert os.environ["QG_R3_KEEP"] == "kept"
