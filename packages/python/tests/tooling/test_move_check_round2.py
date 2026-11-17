"""tools/move_check.py: each case was a reproduced false pass (class scope, guards, facades)."""

from __future__ import annotations

import textwrap
from typing import Any

import pytest

from tools import move_check as mc

FUT = "from __future__ import annotations\n"


def src(text: str) -> str:
    return FUT + textwrap.dedent(text)


def check(old: str, after: dict[str, str], **kw: Any) -> mc.MoveReport:
    return mc.check_move(src(old), {k: src(v) for k, v in after.items()}, **kw)


# ---- a statement may not change kind --------------------------------------------------


@pytest.mark.parametrize(
    "now",
    [
        "def C():\n    return 99\n",
        "async def C():\n    return 99\n",
        'C = print("changed")\n',
        "C = 1\n",
    ],
)
def test_a_class_that_becomes_anything_else_fails(now: str) -> None:
    report = check("class C:\n    x = 1\n", {"old": now})
    assert any("kind" in p or "changed" in p for p in report.problems), report.problems


def test_a_function_that_becomes_a_class_fails() -> None:
    assert check("def f():\n    return 1\n", {"old": "class f:\n    x = 1\n"}).problems


def test_a_class_that_moves_unchanged_still_passes() -> None:
    report = check(
        "class C:\n    x = 1\n", {"old": "from new import C\n", "new": "class C:\n    x = 1\n"}
    )
    assert report.problems == []


# ---- class-sensitive methods may not move into a mixin ---------------------------------

BASE_CLASS = "class C:\n    def f(self):\n        {body}\n    def g(self):\n        return 1\n"


def mixin_after(method: str, rest: str = "") -> dict[str, str]:
    return {
        "old": f"from new import M\nclass C(M):\n{rest or '    pass'}\n",
        "new": f"class M:\n{method}\n",
    }


def method_text(body: str, sig: str = "def f(self)", deco: str = "") -> str:
    return f"{deco}    {sig}:\n        {body}\n"


@pytest.mark.parametrize(
    "body",
    [
        "return super().f()",
        "return __class__.__name__",
        "return self.__secret()",
        "self.__x = 1",
        "return [self.__y for _ in range(2)]",
    ],
)
def test_a_method_that_depends_on_its_class_may_not_move_to_a_mixin(body: str) -> None:
    old = f"class C:\n    def f(self):\n        {body}\n"
    after = mixin_after(f"    def f(self):\n        {body}")
    report = check(old, after, allow_new=())
    assert any("manual review" in p for p in report.problems), report.problems


def test_a_default_that_reads_a_class_body_name_may_not_move() -> None:
    old = "TOKEN = 2\nclass C:\n    TOKEN = 1\n    def f(self, x=TOKEN):\n        return x\n"
    after = {
        "old": "from new import M\nTOKEN = 2\nclass C(M):\n    TOKEN = 1\n",
        "new": "TOKEN = 2\nclass M:\n    def f(self, x=TOKEN):\n        return x\n",
    }
    assert any("manual review" in p for p in check(old, after).problems)


def test_a_decorator_or_annotation_that_reads_a_class_body_name_may_not_move() -> None:
    old = "class C:\n    deco = staticmethod\n    @deco\n    def f(): ...\n"
    after = {
        "old": "from new import M\nclass C(M):\n    deco = staticmethod\n",
        "new": "class M:\n    @deco\n    def f(): ...\n",
    }
    assert any("manual review" in p for p in check(old, after).problems)
    old2 = "class C:\n    T = int\n    def f(self, x: T): ...\n"
    after2 = {
        "old": "from new import M\nclass C(M):\n    T = int\n",
        "new": "class M:\n    def f(self, x: T): ...\n",
    }
    assert any("manual review" in p for p in check(old2, after2).problems)


def test_an_ordinary_method_still_moves_to_a_mixin() -> None:
    old = "class C:\n    x = 1\n    def f(self, n=3):\n        return self._x + n\n"
    after = {
        "old": "from new import M\nclass C(M):\n    x = 1\n",
        "new": "class M:\n    def f(self, n=3):\n        return self._x + n\n",
    }
    assert check(old, after).problems == []


def test_a_dunder_name_is_not_mangled_and_may_move() -> None:
    old = "class C:\n    x = 1\n    def f(self):\n        return self.__dict__\n"
    after = {
        "old": "from new import M\nclass C(M):\n    x = 1\n",
        "new": "class M:\n    def f(self):\n        return self.__dict__\n",
    }
    assert check(old, after).problems == []


def test_a_property_pair_moves_together() -> None:
    old = (
        "class C:\n    x = 1\n    @property\n    def p(self): ...\n"
        "    @p.setter\n    def p(self, v): ...\n"
    )
    after = {
        "old": "from new import M\nclass C(M):\n    x = 1\n",
        "new": (
            "class M:\n    @property\n    def p(self): ...\n"
            "    @p.setter\n    def p(self, v): ...\n"
        ),
    }
    assert check(old, after).problems == []


# ---- only typing.TYPE_CHECKING is a type-only guard ------------------------------------


def test_an_attribute_named_type_checking_on_a_call_is_executable() -> None:
    old = 'if record("old").TYPE_CHECKING:\n    import os\n'
    now = 'if record("new").TYPE_CHECKING:\n    import os\n'
    assert check(old, {"old": now}).problems


def test_a_rebound_type_checking_is_executable() -> None:
    head = "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\n"
    report = check(head + "if TYPE_CHECKING:\n    import os\n", {"old": head})
    assert report.problems


def test_an_unbound_type_checking_name_is_executable() -> None:
    assert check("if TYPE_CHECKING:\n    import os\n", {"old": "pass\n"}).problems


@pytest.mark.parametrize(
    "head,test",
    [
        ("from typing import TYPE_CHECKING\n", "TYPE_CHECKING"),
        ("from typing import TYPE_CHECKING as TC\n", "TC"),
        ("import typing\n", "typing.TYPE_CHECKING"),
        ("import typing as t\n", "t.TYPE_CHECKING"),
    ],
)
def test_the_real_typing_guard_is_still_ignored(head: str, test: str) -> None:
    old = f"{head}if {test}:\n    import os\n"
    assert check(old, {"old": head}).problems == []


# ---- a deleted old module still needs its facade ---------------------------------------


def test_deleting_the_old_module_fails_when_a_public_name_needs_a_facade() -> None:
    report = mc.check_move(
        src("def f():\n    return 1\n"),
        {"new.py": src("def f():\n    return 1\n")},
        old_key="old.py",
        require_reexports=True,
        modules={"old.py": "old", "new.py": "new"},
    )
    assert any("re-export" in p for p in report.problems), report.problems


def test_deleting_the_old_module_is_fine_without_the_option() -> None:
    report = mc.check_move(
        src("def f():\n    return 1\n"),
        {"new.py": src("def f():\n    return 1\n")},
        old_key="old.py",
        modules={"old.py": "old", "new.py": "new"},
    )
    assert report.problems == []


# ---- a movable constant is a literal ---------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        'r["a"]',
        "a + b",
        "[x for x in y]",
        "f()",
        "a if b else c",
        "-a",
        "a.b[0]",
        "{**d}",
        'f"{a}"',
    ],
)
def test_a_non_literal_constant_needs_the_allowance(value: str) -> None:
    old = f"v = {value}\n"
    after = {"old": "from new import v\n", "new": f"v = {value}\n"}
    assert any("immovable" in p for p in check(old, after).problems)
    assert check(old, after, allow_effects=["v"]).problems == []


@pytest.mark.parametrize(
    "value",
    ["1", "-1", "'s'", "None", "name", "a.b.c", "(1, 'a', x)", "[1, [2]]", "{1, 2}", "{'k': a.b}"],
)
def test_a_literal_constant_moves_without_the_allowance(value: str) -> None:
    old = f"v = {value}\n"
    after = {"old": "from new import v\n", "new": f"v = {value}\n"}
    assert check(old, after).problems == []
