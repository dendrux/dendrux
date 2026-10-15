"""Skill.from_dir: file loading, symlink rejection, hidden files, text/binary split."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dendrux.errors import SkillValidationError
from dendrux.skills import Skill, _fs

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURE = Path(__file__).parent.parent / "fixtures" / "skills" / "forecast"

SKILL_MD = "---\nname: demo\ndescription: Demo skill.\n---\n\nBody\n"

FD_SUPPORTED = _fs._FD_MODE


@pytest.fixture(autouse=True, params=["fd", "path"])
def traversal_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every test with descriptor traversal and with the path-based fallback."""
    if request.param == "fd" and not FD_SUPPORTED:
        pytest.skip("platform lacks dir_fd support")
    monkeypatch.setattr(_fs, "_FD_MODE", request.param == "fd")
    return str(request.param)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return root


class TestLoading:
    def test_fixture_files(self) -> None:
        s = Skill.from_dir(FIXTURE, revision="git-abc")
        assert set(s.files) == {"assets/logo.png", "references/policy.md", "scripts/forecast.py"}
        assert isinstance(s.files["references/policy.md"], str)
        assert "Café" in s.files["references/policy.md"]  # type: ignore[operator]
        assert isinstance(s.files["assets/logo.png"], bytes)
        assert s.revision == "git-abc"
        assert s.path == FIXTURE
        assert s.allowed_tools == ("read_skill_resource", "run_skill_script")

    def test_hidden_entries_skipped(self) -> None:
        s = Skill.from_dir(FIXTURE)
        assert not any(p.startswith(".") or "/." in p for p in s.files)

    def test_utf8_and_binary_split(self, skill_dir: Path) -> None:
        (skill_dir / "utf8.txt").write_bytes("héllo\r\n".encode())
        (skill_dir / "nul.txt").write_bytes(b"abc\x00def")
        (skill_dir / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
        (skill_dir / "empty.txt").write_bytes(b"")
        files = Skill.from_dir(skill_dir).files
        assert files["utf8.txt"] == "héllo\r\n"
        assert files["nul.txt"] == b"abc\x00def"
        assert files["latin1.txt"] == b"caf\xe9"
        assert files["empty.txt"] == ""

    def test_nested_skill_md_is_a_file(self, skill_dir: Path) -> None:
        (skill_dir / "sub").mkdir()
        (skill_dir / "sub" / "SKILL.md").write_text("nested")
        assert Skill.from_dir(skill_dir).files == {"sub/SKILL.md": "nested"}

    def test_empty_dirs_ignored(self, skill_dir: Path) -> None:
        (skill_dir / "empty" / "deeper").mkdir(parents=True)
        assert Skill.from_dir(skill_dir).files == {}

    def test_accepts_str_path(self, skill_dir: Path) -> None:
        assert Skill.from_dir(str(skill_dir)).name == "demo"

    def test_round_trip_through_markdown(self) -> None:
        s = Skill.from_dir(FIXTURE)
        back = Skill.from_markdown(s.to_markdown(), files=s.files)
        assert back.digest == s.digest

    def test_invalid_utf8_skill_md(self, skill_dir: Path) -> None:
        (skill_dir / "SKILL.md").write_bytes(b"---\nname: demo\ndescription: \xff\n---\n")
        with pytest.raises(SkillValidationError, match="UTF-8"):
            Skill.from_dir(skill_dir)

    def test_validation_error_prefixed_with_location(self, skill_dir: Path) -> None:
        (skill_dir / "SKILL.md").write_text("---\nname: demo\ndescription: d\nextra: 1\n---\n")
        with pytest.raises(SkillValidationError) as info:
            Skill.from_dir(skill_dir)
        assert str(skill_dir / "SKILL.md") in str(info.value)
        assert info.value.field == "frontmatter"

    def test_invalid_file_name_rejected(self, skill_dir: Path) -> None:
        (skill_dir / 'bad"name.txt').write_text("x")
        with pytest.raises(SkillValidationError) as info:
            Skill.from_dir(skill_dir)
        assert info.value.field == "files"

    def test_name_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "other"
        root.mkdir()
        (root / "SKILL.md").write_text(SKILL_MD)
        with pytest.raises(SkillValidationError, match="does not match directory") as info:
            Skill.from_dir(root)
        assert info.value.field == "name"

    def test_bad_name_reported_before_directory_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "Bad_Name"
        root.mkdir()
        (root / "SKILL.md").write_text("---\nname: Bad_Name\ndescription: d\n---\n")
        with pytest.raises(SkillValidationError, match="lowercase"):
            Skill.from_dir(root)


class TestMissing:
    def test_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Skill.from_dir(tmp_path / "nope")

    def test_missing_skill_md(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            Skill.from_dir(tmp_path)

    def test_skill_md_is_directory(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").mkdir()
        with pytest.raises(FileNotFoundError):
            Skill.from_dir(tmp_path)


needs_symlinks = pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")


@needs_symlinks
class TestSymlinks:
    def test_root_symlink(self, skill_dir: Path, tmp_path: Path) -> None:
        link = tmp_path / "links" / "demo"
        link.parent.mkdir()
        link.symlink_to(skill_dir, target_is_directory=True)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(link)

    def test_skill_md_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "real.md"
        real.write_text(SKILL_MD)
        root = tmp_path / "demo"
        root.mkdir()
        (root / "SKILL.md").symlink_to(real)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(root)

    def test_nested_file_symlink(self, skill_dir: Path, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "leak.txt").symlink_to(secret)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    def test_nested_dir_symlink(self, skill_dir: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "x.txt").write_text("x")
        (skill_dir / "linked").symlink_to(outside, target_is_directory=True)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    def test_hidden_symlink_rejected(self, skill_dir: Path, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        (skill_dir / ".hidden-link").symlink_to(secret)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    def test_dangling_symlink_rejected(self, skill_dir: Path) -> None:
        (skill_dir / "dangling").symlink_to(skill_dir / "does-not-exist")
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    def test_symlink_ancestor_of_root_is_allowed(self, skill_dir: Path, tmp_path: Path) -> None:
        alias = tmp_path / "alias"
        alias.symlink_to(skill_dir.parent, target_is_directory=True)
        assert Skill.from_dir(alias / "demo").name == "demo"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
def test_special_file_rejected(skill_dir: Path) -> None:
    os.mkfifo(skill_dir / "pipe")
    with pytest.raises(SkillValidationError, match="regular"):
        Skill.from_dir(skill_dir)


class TestScanDir:
    def test_scan_includes_files(self, tmp_path: Path) -> None:
        shutil.copytree(FIXTURE, tmp_path / "forecast")
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "loose.txt").write_text("x")
        skills = Skill.scan_dir(tmp_path)
        assert [s.name for s in skills] == ["forecast"]
        assert "references/policy.md" in skills[0].files

    def test_scan_sorted_by_name(self, tmp_path: Path) -> None:
        for name in ("zeta", "alpha"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n")
        assert [s.name for s in Skill.scan_dir(tmp_path)] == ["alpha", "zeta"]


class TestReadRegularFile:
    """Open-time guards for files swapped between enumeration and read."""

    @needs_symlinks
    def test_symlink_swapped_in_is_refused(self, tmp_path: Path) -> None:
        from dendrux.skills._fs import _read_regular_file

        target = tmp_path / "target.txt"
        target.write_text("x")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        if not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("platform has no O_NOFOLLOW")
        with pytest.raises(SkillValidationError, match="symlink"):
            _read_regular_file(link)

    def test_other_os_errors_propagate(self, tmp_path: Path) -> None:
        from dendrux.skills._fs import _read_regular_file

        with pytest.raises(FileNotFoundError):
            _read_regular_file(tmp_path / "gone.txt")

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
    def test_fifo_swapped_in_is_refused_without_blocking(self, tmp_path: Path) -> None:
        from dendrux.skills._fs import _read_regular_file

        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(SkillValidationError, match="regular"):
            _read_regular_file(fifo)


def test_legacy_loader_import_path() -> None:
    from dendrux.skills._loader import Skill as LegacySkill

    assert LegacySkill is Skill


class TestSwapRace:
    """An entry checked during enumeration is replaced by a symlink before it is opened."""

    @pytest.fixture
    def outside(self, tmp_path: Path) -> Path:
        out = tmp_path / "outside"
        out.mkdir()
        (out / "secret.txt").write_text("EXTERNAL SECRET")
        return out

    def _swap_before_open(
        self, monkeypatch: pytest.MonkeyPatch, victim: str, replace: Callable[[Path], None]
    ) -> None:
        real_open_entry = _fs._open_entry

        def racing_open_entry(dir_fd: int, name: str, *, directory: bool, display: Path) -> int:
            if name == victim:
                replace(display)
            return real_open_entry(dir_fd, name, directory=directory, display=display)

        monkeypatch.setattr(_fs, "_open_entry", racing_open_entry)

    @needs_symlinks
    def test_directory_swapped_for_symlink(
        self,
        traversal_mode: str,
        skill_dir: Path,
        outside: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if traversal_mode != "fd":
            pytest.skip("path-based fallback cannot close this race (documented)")
        nested = skill_dir / "nested"
        nested.mkdir()
        (nested / "secret.txt").write_text("inside")

        def replace(display: Path) -> None:
            nested.rename(skill_dir.parent / "moved-away")
            nested.symlink_to(outside, target_is_directory=True)

        self._swap_before_open(monkeypatch, "nested", replace)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    @needs_symlinks
    def test_file_swapped_for_symlink(
        self,
        traversal_mode: str,
        skill_dir: Path,
        outside: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if traversal_mode != "fd":
            pytest.skip("path-based fallback has its own O_NOFOLLOW test")
        (skill_dir / "notes.txt").write_text("inside")

        def replace(display: Path) -> None:
            (skill_dir / "notes.txt").unlink()
            (skill_dir / "notes.txt").symlink_to(outside / "secret.txt")

        self._swap_before_open(monkeypatch, "notes.txt", replace)
        with pytest.raises(SkillValidationError, match="symlink"):
            Skill.from_dir(skill_dir)

    def test_directory_swapped_for_file(
        self, traversal_mode: str, skill_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if traversal_mode != "fd":
            pytest.skip("descriptor traversal only")
        nested = skill_dir / "nested"
        nested.mkdir()

        def replace(display: Path) -> None:
            nested.rmdir()
            nested.write_text("now a file")

        self._swap_before_open(monkeypatch, "nested", replace)
        with pytest.raises(SkillValidationError, match="changed"):
            Skill.from_dir(skill_dir)

    def test_file_swapped_for_directory(
        self, traversal_mode: str, skill_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if traversal_mode != "fd":
            pytest.skip("descriptor traversal only")
        (skill_dir / "notes.txt").write_text("inside")

        def replace(display: Path) -> None:
            (skill_dir / "notes.txt").unlink()
            (skill_dir / "notes.txt").mkdir()

        self._swap_before_open(monkeypatch, "notes.txt", replace)
        with pytest.raises(SkillValidationError, match="regular"):
            Skill.from_dir(skill_dir)


def test_no_descriptor_leak(skill_dir: Path) -> None:
    (skill_dir / "a" / "b").mkdir(parents=True)
    (skill_dir / "a" / "b" / "f.txt").write_text("x")
    (skill_dir / "a" / "bad\x01.txt").write_text("x")
    fd_dir = Path("/dev/fd")
    if not fd_dir.is_dir():
        pytest.skip("cannot count open descriptors")
    before = len(os.listdir(fd_dir))
    for _ in range(20):
        with pytest.raises(SkillValidationError):
            Skill.from_dir(skill_dir)
    assert len(os.listdir(fd_dir)) == before
