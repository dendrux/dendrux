"""Release notes distinguish explicit pending scopes from trusted historical defaults."""

from pathlib import Path


def test_historical_read_default_is_explicit_in_release_notes():
    notes = (Path(__file__).resolve().parents[2] / "CHANGELOG.md").read_text()
    release = notes.split("## 0.5.0", 1)[1].split("## 0.4.", 1)[0]
    assert "default to `UNSCOPED` (all tenants)" in release
    assert "`get_pending` and `follow` require it" in release
    assert "Reads are tenant-scoped" not in release
