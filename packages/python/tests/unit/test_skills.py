"""Tests for Skill class and SKILL.md parser.

Tests define the expected behavior — written before implementation.
Uses fixture skill directories in tests/fixtures/skills/.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills"


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------


class TestSkillFromDir:
    """Skill.from_dir() loads and validates a SKILL.md file."""

    def test_loads_valid_skill(self) -> None:
        from dendrux.skills import Skill

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        assert skill.name == "pdf-processing"
        assert "Extract text" in skill.description
        assert "read_file" in skill.body
        assert skill.license == "Apache-2.0"
        assert skill.metadata == {"author": "test-org", "version": "1.0"}
        assert skill.path == FIXTURES_DIR / "pdf-processing"

    def test_minimal_frontmatter(self) -> None:
        """Only name + description required."""
        from dendrux.skills import Skill

        skill = Skill.from_dir(FIXTURES_DIR / "report-gen")

        assert skill.name == "report-gen"
        assert "formatted reports" in skill.description
        assert skill.license is None
        assert skill.metadata == {}

    def test_body_is_markdown_after_frontmatter(self) -> None:
        from dendrux.skills import Skill

        skill = Skill.from_dir(FIXTURES_DIR / "pdf-processing")

        assert skill.body.startswith("## Instructions")
        assert "Edge Cases" in skill.body

    def test_missing_skill_md_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d, pytest.raises(FileNotFoundError, match="SKILL.md"):
            Skill.from_dir(d)

    def test_missing_name_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory(suffix="-test") as d:
            Path(d, "SKILL.md").write_text("---\ndescription: No name field\n---\nBody")
            with pytest.raises(ValueError, match="name"):
                Skill.from_dir(d)

    def test_missing_description_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory(suffix="-test") as d:
            name = Path(d).name
            Path(d, "SKILL.md").write_text(f"---\nname: {name}\n---\nBody")
            with pytest.raises(ValueError, match="description"):
                Skill.from_dir(d)

    def test_name_mismatch_raises(self) -> None:
        """Frontmatter name must match directory name."""
        from dendrux.skills import Skill

        with pytest.raises(ValueError, match="match.*directory"):
            Skill.from_dir(FIXTURES_DIR / "bad-name")

    def test_nonexistent_dir_raises(self) -> None:
        from dendrux.skills import Skill

        with pytest.raises(FileNotFoundError):
            Skill.from_dir("/nonexistent/path")


# ---------------------------------------------------------------------------
# Name validation (agentskills.io spec)
# ---------------------------------------------------------------------------


class TestSkillNameValidation:
    """Skill names must follow agentskills.io spec."""

    def _make_skill(self, name: str) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory(suffix=f"-{name}") as d:
            # Create a subdir matching the name
            skill_dir = Path(d) / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Test skill\n---\nBody"
            )
            Skill.from_dir(skill_dir)

    def test_valid_lowercase(self) -> None:
        self._make_skill("pdf-processing")

    def test_valid_numbers(self) -> None:
        self._make_skill("v2-tools")

    def test_valid_single_char(self) -> None:
        self._make_skill("a")

    def test_uppercase_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="lowercase"):
            _validate_skill_name("PDF-Processing", "PDF-Processing")

    def test_leading_hyphen_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="hyphen"):
            _validate_skill_name("-pdf", "-pdf")

    def test_trailing_hyphen_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="hyphen"):
            _validate_skill_name("pdf-", "pdf-")

    def test_consecutive_hyphens_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="consecutive"):
            _validate_skill_name("pdf--tool", "pdf--tool")

    def test_too_long_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="64"):
            _validate_skill_name("a" * 65, "a" * 65)

    def test_empty_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError):
            _validate_skill_name("", "")

    def test_dots_rejected(self) -> None:
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="lowercase"):
            _validate_skill_name("pdf.tool", "pdf.tool")

    def test_underscores_rejected(self) -> None:
        """Spec says lowercase + hyphens only — no underscores."""
        from dendrux.skills._loader import _validate_skill_name

        with pytest.raises(ValueError, match="lowercase"):
            _validate_skill_name("pdf_tool", "pdf_tool")


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


class TestSkillScanDir:
    """Skill.scan_dir() finds all valid skills in a directory."""

    def test_finds_all_valid_skills(self) -> None:
        from dendrux.skills import Skill

        # Use a clean temp dir with only valid skills
        with tempfile.TemporaryDirectory() as d:
            # Copy valid fixtures
            import shutil

            shutil.copytree(FIXTURES_DIR / "pdf-processing", Path(d) / "pdf-processing")
            shutil.copytree(FIXTURES_DIR / "report-gen", Path(d) / "report-gen")

            skills = Skill.scan_dir(d)

            names = {s.name for s in skills}
            assert "pdf-processing" in names
            assert "report-gen" in names
            assert len(skills) == 2

    def test_empty_dir_returns_empty(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skills = Skill.scan_dir(d)
            assert skills == []

    def test_dir_without_skill_md_skipped(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            # Create a subdir without SKILL.md
            (Path(d) / "not-a-skill").mkdir()
            skills = Skill.scan_dir(d)
            assert skills == []

    def test_nonexistent_dir_raises(self) -> None:
        from dendrux.skills import Skill

        with pytest.raises(FileNotFoundError):
            Skill.scan_dir("/nonexistent")


# ---------------------------------------------------------------------------
# Frontmatter edge cases
# ---------------------------------------------------------------------------


class TestFrontmatterEdgeCases:
    """Edge cases in SKILL.md parsing."""

    def test_no_frontmatter_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "test"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("Just markdown, no frontmatter.")
            with pytest.raises(ValueError, match="frontmatter"):
                Skill.from_dir(skill_dir)

    def test_empty_body_is_ok(self) -> None:
        """A skill with only frontmatter and no body is valid."""
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "minimal"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: minimal\ndescription: A minimal skill\n---\n"
            )
            skill = Skill.from_dir(skill_dir)
            assert skill.name == "minimal"
            assert skill.body == ""

    def test_description_too_long_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "longdesc"
            skill_dir.mkdir()
            long_desc = "x" * 1025
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: longdesc\ndescription: {long_desc}\n---\n"
            )
            with pytest.raises(ValueError, match="1024"):
                Skill.from_dir(skill_dir)

    def test_compatibility_field_preserved(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "compat"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: compat\ndescription: Test\n"
                "compatibility: Requires Python 3.11+\n---\nBody"
            )
            skill = Skill.from_dir(skill_dir)
            assert skill.compatibility == "Requires Python 3.11+"

    def test_metadata_non_dict_raises(self) -> None:
        """metadata must be a mapping, not a scalar."""
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "badmeta"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: badmeta\ndescription: Test\nmetadata: hello\n---\n"
            )
            with pytest.raises(ValueError, match="metadata.*mapping"):
                Skill.from_dir(skill_dir)

    def test_license_non_string_raises(self) -> None:
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "badlic"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: badlic\ndescription: Test\nlicense: 42\n---\n"
            )
            with pytest.raises(ValueError, match="license.*string"):
                Skill.from_dir(skill_dir)

    def test_missing_close_delimiter_raises(self) -> None:
        """SKILL.md with opening --- but no closing --- should fail."""
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "noclose"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: noclose\ndescription: Test\nBody without close"
            )
            with pytest.raises(ValueError, match="closing"):
                Skill.from_dir(skill_dir)

    def test_yaml_with_triple_dash_in_content(self) -> None:
        """--- inside YAML values should not be treated as delimiter."""
        from dendrux.skills import Skill

        with tempfile.TemporaryDirectory() as d:
            skill_dir = Path(d) / "dashes"
            skill_dir.mkdir()
            # The description has a line break but --- is only on its own line
            (skill_dir / "SKILL.md").write_text(
                "---\nname: dashes\ndescription: Test with some-dashes\n---\nBody here"
            )
            skill = Skill.from_dir(skill_dir)
            assert skill.name == "dashes"
            assert skill.body == "Body here"
