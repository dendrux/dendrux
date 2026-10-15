"""Skill value model: construction, validation, immutability, markdown round-trip."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import warnings
from pathlib import Path
from types import MappingProxyType

import pytest

from dendrux.errors import SkillValidationError
from dendrux.skills import Skill


def _skill(**overrides: object) -> Skill:
    kwargs: dict[str, object] = {"name": "triage", "description": "Triage tickets."}
    kwargs.update(overrides)
    return Skill(**kwargs)  # type: ignore[arg-type]


class TestDirectConstruction:
    def test_minimal(self) -> None:
        s = Skill(name="triage", description="Triage tickets.")
        assert s.body == ""
        assert s.license is None and s.compatibility is None
        assert s.metadata == {} and s.files == {}
        assert s.allowed_tools == () and s.revision is None and s.path is None

    def test_positional_name_description_body(self) -> None:
        s = Skill("triage", "Triage tickets.", "Do it.")
        assert s.body == "Do it."

    def test_optional_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            Skill("triage", "d", "b", "MIT")  # type: ignore[misc]

    def test_legacy_construction_with_path(self) -> None:
        s = Skill(name="helper", description="d", body="b", path=Path("."))
        assert s.path == Path(".")

    def test_path_str_coerced(self) -> None:
        assert _skill(path="some/dir").path == Path("some/dir")

    def test_full(self) -> None:
        s = _skill(
            body="Body\n",
            license="MIT",
            compatibility="Python 3.11+",
            metadata={"b": "2", "a": "1"},
            allowed_tools=["Read", "Bash(git:*)"],
            files={"references/x.md": "text", "assets/y.bin": b"\x00\x01"},
            revision="v1",
        )
        assert s.allowed_tools == ("Read", "Bash(git:*)")
        assert list(s.metadata) == ["a", "b"]
        assert s.files["assets/y.bin"] == b"\x00\x01"

    def test_body_stored_exactly(self) -> None:
        body = "\n\n  indented\n\n"
        assert _skill(body=body).body == body


class TestValidation:
    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            ({"name": 123}, "name"),
            ({"description": ""}, "description"),
            ({"description": "   "}, "description"),
            ({"description": "x" * 1025}, "description"),
            ({"description": None}, "description"),
            ({"body": None}, "body"),
            ({"body": "bad \ud800 surrogate"}, "body"),
            ({"license": 3}, "license"),
            ({"license": ""}, "license"),
            ({"compatibility": "x" * 501}, "compatibility"),
            ({"metadata": "x"}, "metadata"),
            ({"metadata": {"a": 1}}, "metadata"),
            ({"metadata": {1: "a"}}, "metadata"),
            ({"allowed_tools": "Read Bash"}, "allowed_tools"),
            ({"allowed_tools": ["Read", ""]}, "allowed_tools"),
            ({"allowed_tools": ["two words"]}, "allowed_tools"),
            ({"allowed_tools": [1]}, "allowed_tools"),
            ({"allowed_tools": 5}, "allowed_tools"),
            ({"files": ["a"]}, "files"),
            ({"files": {"../escape": "x"}}, "files"),
            ({"files": {"SKILL.md": "x"}}, "files"),
            ({"files": {"a.txt": 3}}, "files"),
            ({"files": {"a.txt": bytearray(b"x")}}, "files"),
            ({"revision": ""}, "revision"),
            ({"revision": "x" * 257}, "revision"),
            ({"revision": "v1\n"}, "revision"),
            ({"revision": "v1\x85"}, "revision"),
            ({"revision": 1}, "revision"),
            ({"path": 1}, "path"),
        ],
    )
    def test_invalid_field(self, overrides: dict[str, object], field: str) -> None:
        with pytest.raises(SkillValidationError) as info:
            _skill(**overrides)
        assert info.value.field == field
        assert isinstance(info.value, ValueError)

    def test_limits_accepted(self) -> None:
        _skill(description="x" * 1024, compatibility="x" * 500, revision="r" * 256)

    def test_description_measured_after_strip(self) -> None:
        _skill(description="  " + "x" * 1024 + "  ")

    def test_replace_revalidates(self) -> None:
        s = _skill()
        with pytest.raises(SkillValidationError):
            dataclasses.replace(s, name="Bad")


class TestImmutability:
    def test_frozen(self) -> None:
        s = _skill()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "other"  # type: ignore[misc]

    def test_mappings_are_read_only_copies(self) -> None:
        meta = {"a": "1"}
        files: dict[str, str | bytes] = {"x.md": "t"}
        s = _skill(metadata=meta, files=files)
        meta["a"] = "changed"
        files["y.md"] = "new"
        assert s.metadata == {"a": "1"}
        assert "y.md" not in s.files
        assert isinstance(s.metadata, MappingProxyType)
        assert isinstance(s.files, MappingProxyType)
        with pytest.raises(TypeError):
            s.files["z.md"] = "nope"  # type: ignore[index]

    def test_files_sorted(self) -> None:
        s = _skill(files={"b.md": "b", "a.md": "a"})
        assert list(s.files) == ["a.md", "b.md"]


class TestEqualityAndHash:
    def test_equal_and_hashable(self) -> None:
        a = _skill(files={"x.md": "t"}, metadata={"k": "v"})
        b = _skill(files={"x.md": "t"}, metadata={"k": "v"})
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_eq_covers_revision_and_path_but_digest_does_not(self) -> None:
        a = _skill(revision="r1")
        b = _skill(revision="r2", path=Path("x"))
        assert a != b
        assert a.digest == b.digest
        assert hash(a) == hash(b)

    def test_text_vs_binary_differs(self) -> None:
        assert _skill(files={"a": "x"}) != _skill(files={"a": b"x"})


class TestCopyAndPickle:
    def test_pickle_round_trip(self) -> None:
        s = _skill(files={"a.bin": b"\x00"}, metadata={"k": "v"}, revision="r", path=Path("p"))
        restored = pickle.loads(pickle.dumps(s))
        assert restored == s
        assert restored.digest == s.digest

    def test_deepcopy_and_copy(self) -> None:
        s = _skill(files={"a.md": "x"})
        assert copy.deepcopy(s) == s
        assert copy.copy(s) == s


class TestSizeWarning:
    def test_large_package_warns_once(self) -> None:
        big = "x" * (5 * 1024 * 1024)
        with pytest.warns(UserWarning, match="5 MiB") as record:
            _skill(body="y", files={"big.txt": big})
        assert len([w for w in record if issubclass(w.category, UserWarning)]) == 1

    def test_at_limit_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _skill(files={"big.bin": b"x" * (5 * 1024 * 1024)})

    def test_package_size_counts_utf8_bytes(self) -> None:
        assert _skill(body="é", files={"a": "é", "b": b"xyz"}).package_size == 2 + 2 + 3


class TestFromMarkdown:
    def test_basic(self) -> None:
        text = (
            "---\nname: triage\ndescription: Triage tickets.\nlicense: MIT\n"
            "compatibility: Py3\nallowed-tools: Read  Bash(git:*)\n"
            "metadata:\n  author: me\n---\n\n# Title\nBody\n"
        )
        s = Skill.from_markdown(text, files={"a.md": "x"}, revision="r1")
        assert s.body == "# Title\nBody\n"
        assert s.allowed_tools == ("Read", "Bash(git:*)")
        assert s.metadata == {"author": "me"}
        assert s.files == {"a.md": "x"}
        assert s.revision == "r1"
        assert s.path is None

    @pytest.mark.parametrize(
        ("text", "body"),
        [
            ("---\nname: a\ndescription: d\n---\nBody", "Body"),
            ("---\nname: a\ndescription: d\n---\n\n\nBody", "\nBody"),
            ("---\nname: a\ndescription: d\n---", ""),
            ("---\nname: a\ndescription: d\n---\n", ""),
            ("---\r\nname: a\r\ndescription: d\r\n---\r\n\r\nBody\r\n", "Body\r\n"),
            ("﻿---\nname: a\ndescription: d\n---\n\nBody", "Body"),
            ("---\nname: a\ndescription: d\n---  \n\nBody\n---\nmore", "Body\n---\nmore"),
        ],
    )
    def test_body_extraction(self, text: str, body: str) -> None:
        assert Skill.from_markdown(text).body == body

    @pytest.mark.parametrize(
        ("text", "match", "field"),
        [
            ("no frontmatter", "frontmatter", "frontmatter"),
            ("", "frontmatter", "frontmatter"),
            ("---\nname: a\ndescription: d\n", "closing", "frontmatter"),
            ("---\n---\nbody", "mapping", "frontmatter"),
            ("---\n- a\n- b\n---\n", "mapping", "frontmatter"),
            ("---\nname: [unclosed\n---\n", "YAML", "frontmatter"),
            ("---\nname: a\ndescription: d\nversion: 1\n---\n", "Unknown", "frontmatter"),
            ("---\nname: a\nname: b\ndescription: d\n---\n", "duplicate", "frontmatter"),
            ("---\ndescription: d\n---\n", "name", "name"),
            ("---\nname: a\n---\n", "description", "description"),
            ("---\nname: 123\ndescription: d\n---\n", "string", "name"),
            ("---\nname: a\ndescription: yes\n---\n", "string", "description"),
            ("---\nname: a\ndescription: 2026-10-15\n---\n", "string", "description"),
            ("---\nname: a\ndescription: d\nlicense: 42\n---\n", "license.*string", "license"),
            ("---\nname: a\ndescription: d\ncompatibility: 1.5\n---\n", "string", "compatibility"),
            (
                "---\nname: a\ndescription: d\nmetadata: hello\n---\n",
                "metadata.*mapping",
                "metadata",
            ),
            ("---\nname: a\ndescription: d\nmetadata:\n  v: 1.0\n---\n", "string", "metadata"),
            (
                "---\nname: a\ndescription: d\nallowed-tools: [Read]\n---\n",
                "string",
                "allowed_tools",
            ),
        ],
    )
    def test_strict_errors(self, text: str, match: str, field: str) -> None:
        with pytest.raises(SkillValidationError, match=match) as info:
            Skill.from_markdown(text)
        assert info.value.field == field

    def test_non_string_input(self) -> None:
        with pytest.raises(SkillValidationError):
            Skill.from_markdown(b"---\n")  # type: ignore[arg-type]

    def test_optional_null_values_treated_as_absent(self) -> None:
        s = Skill.from_markdown("---\nname: a\ndescription: d\nlicense:\nmetadata:\n---\n")
        assert s.license is None and s.metadata == {}

    def test_indented_dashes_do_not_close(self) -> None:
        text = "---\nname: a\ndescription: |\n  one\n  ---\n  two\n---\nBody"
        s = Skill.from_markdown(text)
        assert s.description == "one\n---\ntwo\n"
        assert s.body == "Body"


class TestToMarkdown:
    def test_canonical_form(self) -> None:
        s = _skill(
            body="Body\n",
            license="MIT",
            compatibility="Py3",
            allowed_tools=("Read", "Write"),
            metadata={"z": "1", "a": "2"},
        )
        assert s.to_markdown() == (
            "---\n"
            "name: triage\n"
            "description: Triage tickets.\n"
            "license: MIT\n"
            "compatibility: Py3\n"
            "allowed-tools: Read Write\n"
            "metadata:\n"
            "  a: '2'\n"
            "  z: '1'\n"
            "---\n"
            "\n"
            "Body\n"
        )

    def test_omits_empty(self) -> None:
        assert _skill().to_markdown() == "---\nname: triage\ndescription: Triage tickets.\n---\n\n"

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"body": "\n\nleading newlines"},
            {"body": "---\nlooks like frontmatter\n---\n"},
            {"body": "trailing spaces   \n\n"},
            {"description": "yes"},
            {"description": "multi\nline\n---\ndescription with: colon"},
            {"description": "  padded  "},
            {"description": "unicode é 漢字   sep"},
            {"description": "'quoted' \"double\" # not a comment"},
            {"license": "null"},
            {"compatibility": "1.0"},
            {"metadata": {"": "empty key", "k: v": "- item", "num": "007"}},
            {"allowed_tools": ("Bash(git:*)", "Read")},
            {"description": "x" * 1024},
        ],
    )
    def test_round_trip(self, overrides: dict[str, object]) -> None:
        s = _skill(files={"a.md": "x"}, revision="r", path=Path("p"), **overrides)
        back = Skill.from_markdown(s.to_markdown())
        assert back == dataclasses.replace(s, files={}, revision=None, path=None)
        assert back.to_markdown() == s.to_markdown()
