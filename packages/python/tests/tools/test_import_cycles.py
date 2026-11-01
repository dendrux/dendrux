"""tools/import_cycles.py: runtime import cycles, ratcheted by strongly connected component."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from tools import import_cycles as ic

if TYPE_CHECKING:
    from pathlib import Path


def make_pkg(root: Path, files: dict[str, str]) -> Path:
    src = root / "src" / "dendrux"
    for rel, text in {"__init__.py": "", **files}.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    return src


def graph(src: Path, include_lazy: bool = False) -> dict[str, set[str]]:
    return ic.build_graph(src, include_function_level=include_lazy)


def test_top_level_imports_make_edges_including_relative_ones(tmp_path: Path) -> None:
    src = make_pkg(
        tmp_path,
        {
            "a.py": "from dendrux import b\n",
            "b.py": "from . import c\nimport dendrux.d\n",
            "c.py": "",
            "d.py": "",
        },
    )
    g = graph(src)
    assert g["dendrux.a"] == {"dendrux.b"}
    assert g["dendrux.b"] == {"dendrux.c", "dendrux.d"}


def test_name_imports_from_a_module_point_at_the_module(tmp_path: Path) -> None:
    src = make_pkg(tmp_path, {"a.py": "from dendrux.b import thing\n", "b.py": "thing = 1\n"})
    assert graph(src)["dendrux.a"] == {"dendrux.b"}


def test_type_checking_and_function_level_imports_are_not_runtime_edges(tmp_path: Path) -> None:
    src = make_pkg(
        tmp_path,
        {
            "a.py": """
                from __future__ import annotations
                from typing import TYPE_CHECKING
                if TYPE_CHECKING:
                    from dendrux import b
                def f():
                    from dendrux import c
            """,
            "b.py": "",
            "c.py": "",
        },
    )
    assert graph(src)["dendrux.a"] == set()
    assert graph(src, include_lazy=True)["dendrux.a"] == {"dendrux.c"}


def test_imports_in_try_and_else_branches_are_runtime(tmp_path: Path) -> None:
    src = make_pkg(
        tmp_path,
        {
            "a.py": """
                import typing
                if typing.TYPE_CHECKING:
                    from dendrux import b
                else:
                    from dendrux import c
                try:
                    from dendrux import d
                except ImportError:
                    pass
            """,
            "b.py": "",
            "c.py": "",
            "d.py": "",
        },
    )
    assert graph(src)["dendrux.a"] == {"dendrux.c", "dendrux.d"}


def test_cycles_are_strongly_connected_components_of_two_or_more() -> None:
    g = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}, "e": {"e"}}
    assert ic.find_cycles(g) == [["a", "b", "c"]]


def test_a_new_cycle_fails() -> None:
    result = ic.compare([["x", "y"]], [["a", "b"]])
    assert result.failed
    assert "x" in result.new[0]


def test_a_module_joining_an_existing_cycle_fails() -> None:
    result = ic.compare([["a", "b", "c"]], [["a", "b"]])
    assert result.failed
    assert any("c" in item for item in result.new)


def test_two_cycles_merging_fails() -> None:
    assert ic.compare([["a", "b", "c", "d"]], [["a", "b"], ["c", "d"]]).failed


def test_a_cycle_that_shrinks_or_disappears_passes_and_is_an_improvement() -> None:
    result = ic.compare([["a", "b"]], [["a", "b", "c"], ["x", "y"]])
    assert not result.failed
    assert result.improved


def test_end_to_end(tmp_path: Path) -> None:
    src = make_pkg(tmp_path, {"a.py": "from dendrux import b\n", "b.py": "from dendrux import a\n"})
    baseline = tmp_path / "base.json"
    assert ic.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 2
    assert ic.main(["stamp", "--src", str(src), "--baseline", str(baseline)]) == 0
    assert ic.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 0
    (src / "c.py").write_text("from dendrux import a\n")
    (src / "b.py").write_text("from dendrux import a, c\n")  # c now joins the cycle
    (src / "a.py").write_text("from dendrux import b, c\n")
    assert ic.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 1
    assert ic.main(["report", "--src", str(src)]) == 0


def _pkg_cycle_tree(tmp_path: Path, a_import: str, init_import: str) -> Path:
    return make_pkg(
        tmp_path,
        {
            "a.py": f"{a_import}\na = 1\n",
            "pkg/__init__.py": f"{init_import}\n",
            "pkg/b.py": "b = 1\n",
        },
    )


def test_importing_a_submodule_runs_the_package_initializer(tmp_path: Path) -> None:
    src = _pkg_cycle_tree(tmp_path, "from dendrux.pkg.b import b", "from dendrux.a import a")
    g = graph(src)
    assert g["dendrux.a"] == {"dendrux.pkg", "dendrux.pkg.b"}
    assert ic.find_cycles(g) == [["dendrux.a", "dendrux.pkg"]]


def test_the_same_holds_for_relative_imports(tmp_path: Path) -> None:
    src = _pkg_cycle_tree(tmp_path, "from .pkg.b import b", "from ..a import a")
    assert ic.find_cycles(graph(src)) == [["dendrux.a", "dendrux.pkg"]]


def test_from_package_import_submodule_runs_the_package_initializer(tmp_path: Path) -> None:
    src = _pkg_cycle_tree(tmp_path, "from dendrux.pkg import b", "from dendrux.a import a")
    assert ic.find_cycles(graph(src)) == [["dendrux.a", "dendrux.pkg"]]
    src2 = _pkg_cycle_tree(tmp_path / "rel", "from .pkg import b", "from .. import a")
    assert ic.find_cycles(graph(src2)) == [["dendrux.a", "dendrux.pkg"]]


def test_import_pkg_dot_b_runs_the_package_initializer(tmp_path: Path) -> None:
    src = _pkg_cycle_tree(tmp_path, "import dendrux.pkg.b", "from dendrux.a import a")
    assert ic.find_cycles(graph(src)) == [["dendrux.a", "dendrux.pkg"]]


def test_a_module_inside_the_package_does_not_depend_on_its_own_initializer(
    tmp_path: Path,
) -> None:
    src = make_pkg(
        tmp_path,
        {
            "pkg/__init__.py": "from dendrux.pkg.x import x\n",
            "pkg/x.py": "from dendrux.pkg.y import y\nx = 1\n",
            "pkg/y.py": "y = 1\n",
        },
    )
    g = graph(src)
    assert g["dendrux.pkg.x"] == {"dendrux.pkg.y"}
    assert ic.find_cycles(g) == []


def test_the_root_package_initializer_is_not_an_edge(tmp_path: Path) -> None:
    src = make_pkg(tmp_path, {"__init__.py": "from dendrux.a import a\n", "a.py": "a = 1\n"})
    assert "dendrux" not in graph(src)["dendrux.a"]
    assert ic.find_cycles(graph(src)) == []


@pytest.mark.parametrize(
    "header",
    [
        "from typing import TYPE_CHECKING as TC\nif TC:\n    from . import b",
        "import typing as t\nif t.TYPE_CHECKING:\n    from . import b",
        "import typing\nif typing.TYPE_CHECKING:\n    from . import b",
        "from typing_extensions import TYPE_CHECKING as TC\nif TC:\n    from . import b",
    ],
)
def test_type_checking_aliases_are_not_runtime_edges(tmp_path: Path, header: str) -> None:
    src = make_pkg(tmp_path, {"a.py": header + "\n", "b.py": "from . import a\n"})
    assert graph(src)["dendrux.a"] == set()
    assert ic.find_cycles(graph(src)) == []


def test_an_unrelated_name_called_tc_is_a_runtime_condition(tmp_path: Path) -> None:
    src = make_pkg(
        tmp_path, {"a.py": "TC = True\nif TC:\n    from . import b\n", "b.py": "from . import a\n"}
    )
    assert ic.find_cycles(graph(src)) == [["dendrux.a", "dendrux.b"]]
