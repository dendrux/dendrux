"""TYPE_CHECKING does not excuse unexplained imports or missing runtime bindings."""

from __future__ import annotations

import pytest

from tests.tooling.test_move_manifest import manifest
from tools.move_check import check_move

HEAD = "from typing import TYPE_CHECKING\n"


def compare(old: str, after: dict[str, str], owners: dict[str, str], manifest_mode: bool):
    return check_move(
        old, after, move_manifest=manifest(old, after, owners) if manifest_mode else None
    )


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize("location", ["old", "new"])
def test_runtime_import_cannot_hide_under_type_checking(manifest_mode: bool, location: str) -> None:
    function = "def f(): return os.getcwd()\n"
    old = HEAD + "import os\n" + function
    after = {"old": "from new import f\n" if location == "new" else "", "new": ""}
    after[location] = HEAD + "if TYPE_CHECKING:\n import os\n" + function
    report = compare(old, after, {"f": location}, manifest_mode)
    assert not report.ok
    assert any("type-only-runtime" in p and "os" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize("location", ["old", "new"])
def test_unexplained_guarded_import_is_rejected(manifest_mode: bool, location: str) -> None:
    old = HEAD + "def f(): return 1\n"
    after = {"old": HEAD + "from new import f\n", "new": HEAD + "def f(): return 1\n"}
    after[location] += "if TYPE_CHECKING:\n import evil\n"
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert not report.ok
    assert any("extra-import" in p and "import evil" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize(
    "function",
    [
        "def f(callback=os.getcwd): return callback\n",
        "@os.decorator\ndef f(): return 1\n",
        "def f(): return lambda: os.getcwd()\n",
    ],
)
def test_runtime_headers_and_nested_functions_require_runtime_imports(
    manifest_mode: bool, function: str
) -> None:
    old = HEAD + "import os\n" + function
    after = {
        "old": "from new import f\n",
        "new": HEAD + "if TYPE_CHECKING:\n import os\n" + function,
    }
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert not report.ok
    assert any("type-only-runtime" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_annotations_and_a_shadowing_parameter_are_safe(manifest_mode: bool) -> None:
    future = "from __future__ import annotations\n"
    function = "def f(os: os.PathLike) -> os.PathLike: return os\n"
    old = future + HEAD + "import os\n" + function
    after = {
        "old": future + "from new import f\n",
        "new": future + HEAD + "if TYPE_CHECKING:\n import os\n" + function,
    }
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_existing_type_only_imports_remain_explained(manifest_mode: bool) -> None:
    future = "from __future__ import annotations\n"
    source = HEAD + (
        "if TYPE_CHECKING:\n from dependency import Shape\n"
        "def f(value: Shape) -> Shape: return value\n"
    )
    old = future + source
    after = {"old": future + "from new import f\n", "new": old}
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_runtime_binding_can_coexist_with_an_identical_guarded_import(manifest_mode: bool) -> None:
    old = HEAD + "import os\ndef f(): return os.getcwd()\n"
    after = {"old": "from new import f\n", "new": old + "if TYPE_CHECKING:\n import os\n"}
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize(
    "head,guard",
    [
        ("from typing import TYPE_CHECKING as TC\n", "TC"),
        ("import typing as t\n", "t.TYPE_CHECKING"),
    ],
)
def test_type_checking_aliases_cannot_hide_runtime_module_aliases(
    manifest_mode: bool, head: str, guard: str
) -> None:
    function = "def f(): return module.getcwd()\n"
    old = head + "import os as module\n" + function
    after = {
        "old": "from new import f\n",
        "new": head + f"if {guard}:\n import os as module\n" + function,
    }
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert not report.ok
    assert any("type-only-runtime" in p and "module" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        HEAD + "import os\ndef f():\n if TYPE_CHECKING:\n  import os\n return os.getcwd()\n",
        HEAD + "if TYPE_CHECKING:\n import os\nos: object\ndef f(): return os.getcwd()\n",
    ],
)
def test_an_outer_import_or_annotation_does_not_bind_a_guarded_local(
    manifest_mode: bool, source: str
) -> None:
    report = compare(
        source,
        {"old": source},
        {"f": "old", **({"os": "old"} if "\nos: object" in source else {})},
        manifest_mode,
    )
    assert not report.ok
    assert any("type-only-runtime" in p and "os" in p for p in report.problems), report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
def test_a_runtime_else_import_remains_available(manifest_mode: bool) -> None:
    source = HEAD + (
        "if TYPE_CHECKING:\n from dependency import Shape\nelse:\n"
        " import os\ndef f(): return os.getcwd()\n"
    )
    # Executable if statements remain inventoried unchanged by the ordinary checker.
    owners = {"<If>": "old", "f": "old"}
    report = compare(source, {"old": source}, owners, manifest_mode)
    assert report.ok, report.problems


@pytest.mark.parametrize("manifest_mode", [False, True])
@pytest.mark.parametrize("runtime_value", [False, True])
def test_annotated_assignment_values_are_runtime_reads(
    manifest_mode: bool, runtime_value: bool
) -> None:
    function = (
        "def f():\n value: os.PathLike = "
        + ("os.getcwd()" if runtime_value else "None")
        + "\n return value\n"
    )
    old = HEAD + "import os\n" + function
    after = {
        "old": "from new import f\n",
        "new": HEAD + "if TYPE_CHECKING:\n import os\n" + function,
    }
    report = compare(old, after, {"f": "new"}, manifest_mode)
    assert report.ok is not runtime_value, report.problems


@pytest.mark.parametrize("source_module", ["abc", "evil"])
def test_only_the_reviewed_scaffold_abstractmethod_import_is_explained(source_module: str) -> None:
    old = "class Store:\n def write(self, value: int = 1) -> int: return value\n"
    host = (
        "class _StoreHost:\n if TYPE_CHECKING:\n  @abstractmethod\n"
        "  def write(self, value: int = 1) -> int: ...\n"
    )
    after = {
        "old": old,
        "host": HEAD + f"if TYPE_CHECKING:\n from {source_module} import abstractmethod\n" + host,
    }
    report = check_move(
        old,
        after,
        move_manifest=manifest(
            old,
            after,
            {"Store": "old", "Store.write": "old"},
            scaffolding={"_StoreHost": {"module": "host", "source": host}},
        ),
    )
    assert report.ok is (source_module == "abc"), report.problems
    if source_module == "evil":
        assert any("extra-import" in p and "evil" in p for p in report.problems), report.problems
