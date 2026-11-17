"""tools/patch_codemod.py rewrites test patch targets to the module that now defines a name."""

from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from tools import patch_codemod as pc

if TYPE_CHECKING:
    from pathlib import Path

MAPPING = {
    "dendrux.runtime.runner": {
        "_emit_event": "dendrux.runtime.events",
        "_finalize": "dendrux.runtime.finalize",
    }
}


PATCH_IMPORT = "from unittest.mock import patch\n"


def rewrite(source: str, mapping: dict | None = None) -> str:
    return pc.rewrite_source(
        textwrap.dedent(source), MAPPING if mapping is None else mapping
    ).source


def test_a_monkeypatch_string_target_is_rewritten() -> None:
    out = rewrite("""\
        def test_x(monkeypatch):
            monkeypatch.setattr("dendrux.runtime.runner._emit_event", fake)
        """)
    assert 'monkeypatch.setattr("dendrux.runtime.events._emit_event", fake)' in out


def test_patch_string_targets_in_every_call_shape_are_rewritten() -> None:
    out = rewrite("""\
        from unittest import mock
        from unittest.mock import patch

        @patch("dendrux.runtime.runner._finalize")
        def test_a(m):
            with mock.patch('dendrux.runtime.runner._emit_event', new=fake):
                with patch("dendrux.runtime.runner._emit_event") as p:
                    pass
        """)
    assert '"dendrux.runtime.finalize._finalize"' in out
    assert "'dendrux.runtime.events._emit_event'" in out
    assert '"dendrux.runtime.events._emit_event"' in out
    assert "dendrux.runtime.runner" not in out


def test_a_string_target_below_the_name_keeps_its_tail() -> None:
    out = rewrite(PATCH_IMPORT + 'patch("dendrux.runtime.runner._finalize.helper")\n')
    assert out == PATCH_IMPORT + 'patch("dendrux.runtime.finalize._finalize.helper")\n'


def test_a_module_alias_target_is_rewritten_and_imports_the_new_module() -> None:
    out = rewrite("""\
        from dendrux.runtime import runner


        def test_x(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", fake)
            monkeypatch.setattr(runner, "kept", fake)
        """)
    assert 'monkeypatch.setattr(events, "_emit_event", fake)' in out
    assert 'monkeypatch.setattr(runner, "kept", fake)' in out
    assert "from dendrux.runtime import events\n" in out


def test_patch_object_and_import_as_alias_forms_are_rewritten() -> None:
    out = rewrite("""\
        import dendrux.runtime.runner as r
        from unittest.mock import patch


        def test_x():
            with patch.object(r, "_finalize") as m:
                pass
        """)
    assert 'patch.object(finalize, "_finalize")' in out
    assert "from dendrux.runtime import finalize\n" in out


def test_an_existing_alias_for_the_new_module_is_reused() -> None:
    out = rewrite("""\
        from dendrux.runtime import events as ev
        from dendrux.runtime import runner


        def test_x(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", fake)
        """)
    assert 'monkeypatch.setattr(ev, "_emit_event", fake)' in out
    assert out.count("import") == 2


def test_the_new_module_name_does_not_shadow_an_existing_name() -> None:
    out = rewrite("""\
        from dendrux.runtime import runner

        events = []


        def test_x(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", fake)
        """)
    assert "from dendrux.runtime import events as events_mod\n" in out
    assert 'monkeypatch.setattr(events_mod, "_emit_event", fake)' in out


def test_unrelated_text_is_never_touched() -> None:
    source = textwrap.dedent('''\
        from dendrux.runtime import runner

        NOTE = "dendrux.runtime.runner._emit_event"  # not a patch call


        def test_x(monkeypatch):
            """Mentions dendrux.runtime.runner._emit_event in prose."""
            monkeypatch.setattr("dendrux.runtime.other._emit_event", fake)
            monkeypatch.setattr("dendrux.runtime.runner.kept", fake)
            runner._emit_event()
            setattr(runner, "_emit_event", fake)
        ''')
    assert pc.rewrite_source(source, MAPPING).source == source


def test_a_non_ascii_line_keeps_columns_correct() -> None:
    out = rewrite(PATCH_IMPORT + 'x = "é"; patch("dendrux.runtime.runner._emit_event")\n')
    assert out == PATCH_IMPORT + 'x = "é"; patch("dendrux.runtime.events._emit_event")\n'


def test_the_cli_is_a_dry_run_by_default_and_writes_with_the_flag(tmp_path: Path, capsys) -> None:
    test = tmp_path / "test_a.py"
    test.write_text(PATCH_IMPORT + 'patch("dendrux.runtime.runner._emit_event")\n')
    mapping = tmp_path / "map.json"
    mapping.write_text('{"dendrux.runtime.runner": {"_emit_event": "dendrux.runtime.events"}}')
    assert pc.main(["--map", str(mapping), str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "-patch(" in out and "+patch(" in out and "dendrux.runtime.events" in out
    assert "runner._emit_event" in test.read_text()  # untouched
    assert pc.main(["--map", str(mapping), "--write", str(tmp_path)]) == 0
    assert test.read_text() == PATCH_IMPORT + 'patch("dendrux.runtime.events._emit_event")\n'


def test_a_bad_mapping_is_refused(tmp_path: Path) -> None:
    mapping = tmp_path / "map.json"
    mapping.write_text('["not", "a", "mapping"]')
    with pytest.raises(SystemExit):
        pc.main(["--map", str(mapping), str(tmp_path)])


# ---- patch calls are recognised by what they resolve to, not how they are spelled -------------


def test_unrelated_patch_and_setattr_methods_are_not_rewritten() -> None:
    source = textwrap.dedent("""\
        def test_x(client, registry):
            client.patch("dendrux.runtime.runner._emit_event", json={})
            registry.setattr("dendrux.runtime.runner._emit_event", 1)
        """)
    assert pc.rewrite_source(source, MAPPING).source == source


def test_an_aliased_patch_import_is_rewritten() -> None:
    out = rewrite("""\
        from unittest.mock import patch as p
        from dendrux.runtime import runner

        p("dendrux.runtime.runner._emit_event")
        p.object(runner, "_emit_event")
        """)
    assert 'p("dendrux.runtime.events._emit_event")' in out
    assert 'p.object(events, "_emit_event")' in out


def test_a_keyword_target_is_rewritten() -> None:
    out = rewrite(PATCH_IMPORT + 'patch(target="dendrux.runtime.runner._emit_event")\n')
    assert 'patch(target="dendrux.runtime.events._emit_event")' in out


def test_pytest_mock_receivers_are_recognised_by_fixture_name() -> None:
    out = rewrite("""\
        def test_x(mocker, other):
            mocker.patch("dendrux.runtime.runner._emit_event")
            other.patch("dendrux.runtime.runner._emit_event")
        """)
    assert out.count("dendrux.runtime.events._emit_event") == 1
    assert out.count("dendrux.runtime.runner._emit_event") == 1


def test_a_monkeypatch_context_object_is_recognised() -> None:
    out = rewrite("""\
        import pytest


        def test_x():
            with pytest.MonkeyPatch.context() as m:
                m.setattr("dendrux.runtime.runner._emit_event", 1)
        """)
    assert "dendrux.runtime.events._emit_event" in out


# ---- names are resolved in the scope of the call ----------------------------------------------


def test_a_shadowing_fixture_parameter_is_not_redirected() -> None:
    source = textwrap.dedent("""\
        from dendrux.runtime import runner


        def test_a(runner, monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", 1)
        """)
    result = pc.rewrite_source(source, MAPPING)
    assert result.source == source
    assert any("shadowed" in note for note in result.notes)


def test_a_local_import_in_another_function_is_not_reused() -> None:
    out = rewrite("""\
        from dendrux.runtime import runner


        def test_a(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", 1)


        def test_b():
            from dendrux.runtime import events

            return events
        """)
    assert out.index("from dendrux.runtime import events as events_mod") < out.index("def test_a")
    assert 'monkeypatch.setattr(events_mod, "_emit_event", 1)' in out


def test_a_generated_name_never_collides_with_an_existing_one() -> None:
    out = rewrite("""\
        from dendrux.runtime import runner

        events = 1
        events_mod = 2


        def test_x(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", 3)
        """)
    assert "from dendrux.runtime import events as events_mod2\n" in out
    assert 'monkeypatch.setattr(events_mod2, "_emit_event", 3)' in out


def test_a_name_used_as_a_parameter_elsewhere_is_not_generated() -> None:
    out = rewrite("""\
        from dendrux.runtime import runner


        def other(events):
            return events


        def test_x(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", 3)
        """)
    assert "import events as events_mod" in out


# ---- the new import lands where every use can see it ------------------------------------------


def test_a_generated_import_goes_into_the_leading_import_block(tmp_path: Path) -> None:
    pkg = tmp_path / "fakepkg_split" / "runtime"
    pkg.mkdir(parents=True)
    for init in (tmp_path / "fakepkg_split" / "__init__.py", pkg / "__init__.py"):
        init.write_text("")
    (pkg / "runner.py").write_text("_emit_event = 1\n")
    (pkg / "events.py").write_text("_emit_event = 1\n")
    source = textwrap.dedent('''\
        """Doc."""
        from unittest.mock import patch

        from fakepkg_split.runtime import runner


        @patch.object(runner, "_emit_event")
        def test_x(m):
            return m


        import os
        ''')
    mapping = {"fakepkg_split.runtime.runner": {"_emit_event": "fakepkg_split.runtime.events"}}
    out = pc.rewrite_source(source, mapping).source
    assert out.index("import events") < out.index("@patch.object")
    test = tmp_path / "rewritten_test.py"
    test.write_text(out)
    code = compile(out, str(test), "exec")
    old_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        exec(code, {"__name__": "rewritten"})  # noqa: S102 - decorators run, so a NameError would raise
    finally:
        sys.path[:] = old_path
        for name in [n for n in sys.modules if n.startswith("fakepkg_split")]:
            del sys.modules[name]
