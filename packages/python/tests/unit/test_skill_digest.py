"""Skill digest (spec §2.5): golden value, equivalence, two-process stability."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dendrux.skills import Skill

FIXTURE = Path(__file__).parent.parent / "fixtures" / "skills" / "forecast"
SRC_DIR = Path(__file__).resolve().parents[2] / "src"

# Pinned digest of tests/fixtures/skills/forecast. If this changes, every
# paused run created before the change would fail resume verification.
GOLDEN = "sha256:abb32a555723c3fe11aa724289e8af661a8eabd5dfd6f3a1d8171fadfec1621a"


def _reference_digest(header: dict[str, object], files: dict[str, str | bytes]) -> str:
    """Independent transcription of the spec §2.5 algorithm."""
    out = b"dendrux-skill-v1\n"
    out += json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    out += b"\n"
    for path in sorted(files):
        value = files[path]
        content = value.encode("utf-8") if isinstance(value, str) else value
        out += path.encode("utf-8") + b"\0" + (b"t" if isinstance(value, str) else b"b") + b"\0"
        out += len(content).to_bytes(8, "big") + content
    return "sha256:" + hashlib.sha256(out).hexdigest()


def _fixture_equivalent() -> Skill:
    """The fixture rebuilt by hand, without touching the filesystem."""
    return Skill(
        name="forecast",
        description=("Build a revenue forecast from monthly actuals. Use for planning questions."),
        body="# Forecast\n\nRead `references/policy.md` first, then run `scripts/forecast.py`.\n",
        license="MIT",
        allowed_tools=("read_skill_resource", "run_skill_script"),
        metadata={"owner": "finance", "version": "2"},
        files={
            "references/policy.md": (
                "Policy: use a 12-month window.\nCafé rules apply — no exceptions.\n"
            ),
            "scripts/forecast.py": 'import sys\n\nprint("forecast", *sys.argv[1:])\n',
            "assets/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        },
    )


class TestDigest:
    def test_format(self) -> None:
        d = Skill(name="a", description="d").digest
        assert d.startswith("sha256:")
        assert len(d) == len("sha256:") + 64

    def test_golden_fixture(self) -> None:
        skill = Skill.from_dir(FIXTURE)
        assert skill.digest == GOLDEN

    def test_matches_reference_algorithm(self) -> None:
        skill = Skill.from_dir(FIXTURE)
        header = {
            "name": skill.name,
            "description": skill.description,
            "body": skill.body,
            "license": skill.license,
            "compatibility": skill.compatibility,
            "allowed_tools": list(skill.allowed_tools),
            "metadata": dict(skill.metadata),
        }
        assert skill.digest == _reference_digest(header, dict(skill.files))

    def test_equivalent_construction_paths(self) -> None:
        from_dir = Skill.from_dir(FIXTURE)
        direct = _fixture_equivalent()
        from_md = Skill.from_markdown(
            (FIXTURE / "SKILL.md").read_text(encoding="utf-8"), files=dict(direct.files)
        )
        round_trip = Skill.from_markdown(direct.to_markdown(), files=dict(direct.files))
        assert from_dir.digest == direct.digest == from_md.digest == round_trip.digest

    def test_excludes_revision_and_path(self) -> None:
        a = Skill(name="a", description="d")
        b = Skill(name="a", description="d", revision="v9", path=Path("/x/a"))
        assert a.digest == b.digest

    def test_file_order_irrelevant(self) -> None:
        a = Skill(name="a", description="d", files={"x": "1", "y": "2"})
        b = Skill(name="a", description="d", files={"y": "2", "x": "1"})
        assert a.digest == b.digest

    def test_sensitive_to_every_content_field(self) -> None:
        base = dict(name="a", description="d", body="b", license="MIT", compatibility="c")
        ref = Skill(**base, metadata={"k": "v"}, allowed_tools=("t",), files={"f": "1"}).digest
        variants = [
            Skill(
                **{**base, "body": "b2"},
                metadata={"k": "v"},
                allowed_tools=("t",),
                files={"f": "1"},
            ),
            Skill(
                **{**base, "license": "X"},
                metadata={"k": "v"},
                allowed_tools=("t",),
                files={"f": "1"},
            ),
            Skill(
                **{**base, "compatibility": "z"},
                metadata={"k": "v"},
                allowed_tools=("t",),
                files={"f": "1"},
            ),
            Skill(**base, metadata={"k": "w"}, allowed_tools=("t",), files={"f": "1"}),
            Skill(**base, metadata={"k": "v"}, allowed_tools=("u",), files={"f": "1"}),
            Skill(**base, metadata={"k": "v"}, allowed_tools=("t",), files={"f": "2"}),
            Skill(**base, metadata={"k": "v"}, allowed_tools=("t",), files={"f": b"1"}),
            Skill(**base, metadata={"k": "v"}, allowed_tools=("t",), files={"g": "1"}),
        ]
        digests = {v.digest for v in variants}
        assert ref not in digests
        assert len(digests) == len(variants)

    def test_file_boundaries_unambiguous(self) -> None:
        a = Skill(name="a", description="d", files={"x": "ab", "y": "c"})
        b = Skill(name="a", description="d", files={"x": "a", "y": "bc"})
        assert a.digest != b.digest

    def test_stable_across_processes(self) -> None:
        code = (
            "import sys; from dendrux.skills import Skill; "
            "print(Skill.from_dir(sys.argv[1]).digest)"
        )
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONHASHSEED": "random"}
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code, str(FIXTURE)],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(2)
        }
        assert outputs == {Skill.from_dir(FIXTURE).digest}
