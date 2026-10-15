"""Runtime package helpers: package_bytes, activation envelope, continuity_error."""

from __future__ import annotations

from pathlib import Path

import pytest

from dendrux.errors import SkillDigestMismatch, SkillMismatchError, SkillMissingError
from dendrux.skills import Skill
from dendrux.skills._internal import (
    continuity_error,
    escape_body,
    format_size,
    package_bytes,
    render_activation_envelope,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "skills" / "forecast"


def _skill(**overrides: object) -> Skill:
    kwargs: dict[str, object] = {"name": "demo", "description": "Demo.", "body": "Do it.\n"}
    kwargs.update(overrides)
    return Skill(**kwargs)  # type: ignore[arg-type]


class TestPackageBytes:
    def test_includes_skill_md_and_encoded_files(self) -> None:
        s = Skill.from_dir(FIXTURE)
        view = package_bytes(s)
        assert view["SKILL.md"] == s.to_markdown().encode()
        assert view["references/policy.md"] == s.files["references/policy.md"].encode()  # type: ignore[union-attr]
        assert view["assets/logo.png"] == s.files["assets/logo.png"]
        assert set(view) == {"SKILL.md", *s.files}
        assert all(isinstance(v, bytes) for v in view.values())

    def test_read_only(self) -> None:
        view = package_bytes(_skill())
        with pytest.raises(TypeError):
            view["x"] = b"y"  # type: ignore[index]


class TestEnvelope:
    def test_fixture_with_executor(self) -> None:
        s = Skill.from_dir(FIXTURE)
        assert render_activation_envelope(s, scripts_runnable=True) == (
            f'<skill_content name="forecast" digest="{s.digest}">\n'
            "# Forecast\n\n"
            "Read `references/policy.md` first, then run `scripts/forecast.py`.\n"
            "\n"
            "Files (read with read_skill_resource; run scripts with run_skill_script):\n"
            "- references/policy.md (text, 68 B)\n"
            "- assets/logo.png (binary, 16 B, not readable)\n"
            "- scripts/forecast.py (script)\n"
            "</skill_content>"
        )

    def test_fixture_without_executor(self) -> None:
        s = Skill.from_dir(FIXTURE)
        out = render_activation_envelope(s, scripts_runnable=False)
        assert "Files (read with read_skill_resource):\n" in out
        assert "run_skill_script" not in out
        assert "scripts/forecast.py" not in out.split("Files")[1]

    def test_no_files_no_listing(self) -> None:
        s = _skill()
        assert render_activation_envelope(s, scripts_runnable=True) == (
            f'<skill_content name="demo" digest="{s.digest}">\nDo it.\n</skill_content>'
        )

    def test_only_scripts_without_executor_omits_listing(self) -> None:
        s = _skill(files={"scripts/a.py": "print()"})
        assert "Files" not in render_activation_envelope(s, scripts_runnable=False)

    def test_empty_body(self) -> None:
        s = _skill(body="", files={"a.md": "x"})
        out = render_activation_envelope(s, scripts_runnable=False)
        assert out.startswith(f'<skill_content name="demo" digest="{s.digest}">\n\nFiles')

    def test_close_tag_escaped(self) -> None:
        s = _skill(body="a </skill_content> b </SKILL_CONTENT> c <skill_content>")
        out = render_activation_envelope(s, scripts_runnable=False)
        assert out.count("</skill_content") == 1
        assert out.lower().count("</skill_content") == 1
        assert r"<\/skill_content>" in out and r"<\/SKILL_CONTENT>" in out
        assert out.endswith("</skill_content>")

    def test_escape_body_idempotent_on_safe_text(self) -> None:
        assert escape_body("no tags here") == "no tags here"

    def test_listing_order_text_binary_script(self) -> None:
        s = _skill(
            files={
                "scripts/z.sh": "echo",
                "b.bin": b"\x00",
                "a.bin": b"\x00",
                "z.md": "z",
                "a.md": "a",
            }
        )
        out = render_activation_envelope(s, scripts_runnable=True)
        listed = [line.split(" ")[1] for line in out.splitlines() if line.startswith("- ")]
        assert listed == ["a.md", "z.md", "a.bin", "b.bin", "scripts/z.sh"]

    def test_binary_script_listed_as_script(self) -> None:
        s = _skill(files={"scripts/tool": b"\x7fELF\x00"})
        assert "- scripts/tool (script)" in render_activation_envelope(s, scripts_runnable=True)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (2150, "2.1 KB"),
        (48 * 1024, "48 KB"),
        (5 * 1024 * 1024 + 300_000, "5.3 MB"),
        (200 * 1024 * 1024, "200 MB"),
        (3 * 1024**3, "3.0 GB"),
        (2048 * 1024**3, "2048 GB"),
    ],
)
def test_format_size(size: int, expected: str) -> None:
    assert format_size(size) == expected


class TestContinuityError:
    def test_none_when_clean(self) -> None:
        assert continuity_error("r") is None

    def test_missing_wins(self) -> None:
        m = SkillDigestMismatch("b", "sha256:1", "sha256:2")
        err = continuity_error("r", missing=["a"], mismatched=[m])
        assert isinstance(err, SkillMissingError)
        assert err.missing == ("a",) and err.mismatched == (m,)

    def test_mismatch_only(self) -> None:
        m = SkillDigestMismatch("b", "sha256:1", "sha256:2")
        err = continuity_error("r", mismatched=iter([m]))
        assert isinstance(err, SkillMismatchError)
        assert err.run_id == "r"
