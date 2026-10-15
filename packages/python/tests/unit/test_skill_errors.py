"""Skill error types in dendrux.errors."""

from __future__ import annotations

import dataclasses

import pytest

from dendrux.errors import (
    SkillContinuityError,
    SkillDigestMismatch,
    SkillMismatchError,
    SkillMissingError,
    SkillValidationError,
)


class TestSkillValidationError:
    def test_is_value_error_with_field(self) -> None:
        err = SkillValidationError("bad name", field="name")
        assert isinstance(err, ValueError)
        assert err.field == "name"
        assert str(err) == "bad name"

    def test_field_defaults_to_none(self) -> None:
        assert SkillValidationError("x").field is None


class TestSkillDigestMismatch:
    def test_frozen(self) -> None:
        m = SkillDigestMismatch("a", "sha256:1", "sha256:2")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.name = "b"  # type: ignore[misc]


class TestSkillContinuityError:
    def test_hierarchy(self) -> None:
        assert issubclass(SkillContinuityError, RuntimeError)
        assert issubclass(SkillMissingError, SkillContinuityError)
        assert issubclass(SkillMismatchError, SkillContinuityError)

    def test_missing_sorted_and_mismatches_reported(self) -> None:
        err = SkillMissingError(
            "run-1",
            missing=["zeta", "alpha"],
            mismatched=[
                SkillDigestMismatch("y", "sha256:e1", "sha256:a1"),
                SkillDigestMismatch("b", "sha256:e2", "sha256:a2"),
            ],
        )
        assert err.run_id == "run-1"
        assert err.missing == ("alpha", "zeta")
        assert [m.name for m in err.mismatched] == ["b", "y"]
        msg = str(err)
        assert "run-1" in msg
        assert "alpha, zeta" in msg
        assert "sha256:e2" in msg and "sha256:a2" in msg

    def test_mismatch_only(self) -> None:
        err = SkillMismatchError(
            "run-2", mismatched=(SkillDigestMismatch("a", "sha256:1", "sha256:2"),)
        )
        assert err.missing == ()
        assert "changed skills" in str(err)
        assert "missing skills" not in str(err)

    def test_empty_still_has_message(self) -> None:
        assert "run-3" in str(SkillContinuityError("run-3"))

    def test_top_level_reexports(self) -> None:
        import dendrux

        for name in (
            "SkillContinuityError",
            "SkillDigestMismatch",
            "SkillMismatchError",
            "SkillMissingError",
            "SkillValidationError",
        ):
            assert getattr(dendrux, name) is getattr(__import__("dendrux.errors").errors, name)
