"""SkillExecutor contract types and validate_executor."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from types import MappingProxyType

import pytest

from dendrux.skills import SkillExecutionRequest, SkillExecutionResult, SkillExecutor
from dendrux.skills._executor import validate_executor


class EchoExecutor(SkillExecutor):
    async def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        return SkillExecutionResult(stdout=" ".join(request.args))


def _request(**overrides: object) -> SkillExecutionRequest:
    kwargs: dict[str, object] = {
        "run_id": "run-1",
        "tool_call_id": "call-1",
        "skill_name": "forecast",
        "revision": None,
        "digest": "sha256:" + "0" * 64,
        "script_path": "scripts/forecast.py",
        "args": ("--months", "12"),
        "files": MappingProxyType({"SKILL.md": b"---\n"}),
    }
    kwargs.update(overrides)
    return SkillExecutionRequest(**kwargs)  # type: ignore[arg-type]


class TestTypes:
    def test_request_frozen_and_repr_hides_files(self) -> None:
        req = _request()
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.run_id = "x"  # type: ignore[misc]
        assert "SKILL.md" not in repr(req)

    def test_args_normalized_to_tuple_copy(self) -> None:
        args = ["one"]
        req = _request(args=args)
        args.append("two")
        assert req.args == ("one",)

    def test_frozen_view_kept_without_copy(self) -> None:
        view = MappingProxyType({"SKILL.md": b"x"})
        assert _request(files=view).files is view

    def test_plain_dict_files_copied_and_frozen(self) -> None:
        files = {"SKILL.md": b"before"}
        req = _request(files=files)
        files["SKILL.md"] = b"after"
        assert req.files["SKILL.md"] == b"before"
        with pytest.raises(TypeError):
            req.files["x"] = b"y"  # type: ignore[index]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"run_id": 1},
            {"tool_call_id": None},
            {"skill_name": b"x"},
            {"revision": 3},
            {"digest": None},
            {"script_path": 1},
            {"args": "one two"},
            {"args": [1]},
            {"args": None},
            {"files": [("SKILL.md", b"x")]},
            {"files": {"SKILL.md": "text"}},
            {"files": MappingProxyType({"SKILL.md": bytearray(b"x")})},
            {"files": {1: b"x"}},
        ],
    )
    def test_request_types_checked(self, overrides: dict[str, object]) -> None:
        with pytest.raises(TypeError):
            _request(**overrides)

    def test_result_defaults(self) -> None:
        r = SkillExecutionResult()
        assert (r.stdout, r.stderr, r.exit_code) == ("", "", 0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"stdout": b"x"}, {"stderr": None}, {"exit_code": "1"}, {"exit_code": True}],
    )
    def test_result_type_checked(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(TypeError):
            SkillExecutionResult(**kwargs)  # type: ignore[arg-type]

    async def test_base_execute_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await SkillExecutor().execute(_request())

    async def test_subclass_executes(self) -> None:
        result = await EchoExecutor().execute(_request())
        assert result.stdout == "--months 12"

    def test_default_timeout(self) -> None:
        assert SkillExecutor.timeout_seconds == 300.0


class TestValidateExecutor:
    def test_none_accepted(self) -> None:
        validate_executor(None)

    @pytest.mark.parametrize("value", [10, 0.5, Fraction(1, 2), 1e9])
    def test_valid_timeouts(self, value: object) -> None:
        ex = EchoExecutor()
        ex.timeout_seconds = value  # type: ignore[assignment]
        validate_executor(ex)

    def test_not_an_executor(self) -> None:
        class Duck:
            timeout_seconds = 1.0

            async def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
                return SkillExecutionResult()

        with pytest.raises(TypeError, match="SkillExecutor"):
            validate_executor(Duck())

    def test_sync_execute_rejected(self) -> None:
        class SyncExecutor(SkillExecutor):
            def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:  # type: ignore[override]
                return SkillExecutionResult()

        with pytest.raises(TypeError, match="async"):
            validate_executor(SyncExecutor())

    @pytest.mark.parametrize(
        "value", [0, -1, math.inf, -math.inf, math.nan, True, False, "10", None, 1j]
    )
    def test_invalid_timeouts(self, value: object) -> None:
        ex = EchoExecutor()
        ex.timeout_seconds = value  # type: ignore[assignment]
        with pytest.raises(ValueError, match="timeout_seconds"):
            validate_executor(ex)

    def test_class_level_timeout_override(self) -> None:
        class Slow(EchoExecutor):
            timeout_seconds = 0.0

        with pytest.raises(ValueError):
            validate_executor(Slow())


def test_top_level_exports() -> None:
    import dendrux
    import dendrux.skills

    for name in ("Skill", "SkillExecutor", "SkillExecutionRequest", "SkillExecutionResult"):
        assert getattr(dendrux, name) is getattr(dendrux.skills, name)
        assert name in dendrux.__all__
        assert name in dendrux.skills.__all__
