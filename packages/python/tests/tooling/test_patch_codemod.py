"""tools/patch_codemod.py rewrites test patch targets to the module that now defines a name."""

from __future__ import annotations

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
    out = rewrite('patch("dendrux.runtime.runner._finalize.helper")\n')
    assert out == 'patch("dendrux.runtime.finalize._finalize.helper")\n'


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
    out = rewrite('x = "é"; patch("dendrux.runtime.runner._emit_event")\n')
    assert out == 'x = "é"; patch("dendrux.runtime.events._emit_event")\n'


def test_the_cli_is_a_dry_run_by_default_and_writes_with_the_flag(tmp_path: Path, capsys) -> None:
    test = tmp_path / "test_a.py"
    test.write_text('patch("dendrux.runtime.runner._emit_event")\n')
    mapping = tmp_path / "map.json"
    mapping.write_text('{"dendrux.runtime.runner": {"_emit_event": "dendrux.runtime.events"}}')
    assert pc.main(["--map", str(mapping), str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "-patch(" in out and "+patch(" in out and "dendrux.runtime.events" in out
    assert "runner._emit_event" in test.read_text()  # untouched
    assert pc.main(["--map", str(mapping), "--write", str(tmp_path)]) == 0
    assert test.read_text() == 'patch("dendrux.runtime.events._emit_event")\n'


def test_a_bad_mapping_is_refused(tmp_path: Path) -> None:
    mapping = tmp_path / "map.json"
    mapping.write_text('["not", "a", "mapping"]')
    with pytest.raises(SystemExit):
        pc.main(["--map", str(mapping), str(tmp_path)])
