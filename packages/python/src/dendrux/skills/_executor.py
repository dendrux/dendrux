"""Script executor contract for skills."""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class SkillExecutionRequest:
    """One ``run_skill_script`` call handed to a :class:`SkillExecutor`.

    ``files`` is a read-only view of the whole package, including
    ``SKILL.md``; text files are UTF-8 encoded. A ``MappingProxyType`` is
    kept as is (shared across calls, not copied); any other mapping is
    copied and frozen. ``args`` is normalized to a tuple. Wrong field
    types raise ``TypeError``.
    """

    run_id: str
    tool_call_id: str
    skill_name: str
    revision: str | None
    digest: str
    script_path: str
    args: tuple[str, ...]
    files: Mapping[str, bytes] = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("run_id", "tool_call_id", "skill_name", "digest", "script_path"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"SkillExecutionRequest.{name} must be a str.")
        if self.revision is not None and not isinstance(self.revision, str):
            raise TypeError("SkillExecutionRequest.revision must be a str or None.")
        object.__setattr__(self, "args", _freeze_args(self.args))
        object.__setattr__(self, "files", _freeze_files(self.files))


@dataclass(frozen=True)
class SkillExecutionResult:
    """What a script run produced. A non-zero ``exit_code`` is reported as a tool error."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("SkillExecutionResult stdout and stderr must be str.")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("SkillExecutionResult exit_code must be an int.")


def _freeze_args(args: object) -> tuple[str, ...]:
    if isinstance(args, str | bytes) or not isinstance(args, Iterable):
        raise TypeError("SkillExecutionRequest.args must be a sequence of str.")
    frozen = tuple(args)
    if not all(isinstance(a, str) for a in frozen):
        raise TypeError("SkillExecutionRequest.args items must be str.")
    return frozen


def _freeze_files(files: object) -> Mapping[str, bytes]:
    if not isinstance(files, Mapping):
        raise TypeError("SkillExecutionRequest.files must be a Mapping[str, bytes].")
    for key, value in files.items():
        if not isinstance(key, str) or not isinstance(value, bytes):
            raise TypeError("SkillExecutionRequest.files must map str paths to bytes.")
    if isinstance(files, MappingProxyType):
        return files
    return MappingProxyType(dict(files))


class SkillExecutor:
    """Runs skill scripts. Subclass it and implement :meth:`execute`.

    One executor serves every skill on an agent; route on
    ``request.skill_name`` if needed. Dendrux enforces ``timeout_seconds``
    around each call and propagates task cancellation into it.
    """

    timeout_seconds: float = 300.0

    async def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        """Run ``request.script_path`` from the package with ``request.args``."""
        raise NotImplementedError


def validate_executor(executor: object) -> None:
    """Validate an ``Agent(skill_executor=...)`` value. ``None`` is accepted.

    Raises:
        TypeError: If ``executor`` is not a :class:`SkillExecutor` or its
            ``execute`` is not an ``async def``.
        ValueError: If ``timeout_seconds`` is not a finite real number > 0
            (``bool`` is rejected).
    """
    if executor is None:
        return
    if not isinstance(executor, SkillExecutor):
        raise TypeError(
            f"skill_executor must be a SkillExecutor instance, got {type(executor).__name__}."
        )
    if not inspect.iscoroutinefunction(executor.execute):
        raise TypeError(f"{type(executor).__name__}.execute must be an async method (async def).")
    timeout = executor.timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, numbers.Real)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            f"{type(executor).__name__}.timeout_seconds must be a finite number > 0, "
            f"got {timeout!r}."
        )
