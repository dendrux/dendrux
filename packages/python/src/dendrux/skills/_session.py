"""Run-local skill identity, activation and execution. No state lives on Agent."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from dendrux.errors import SkillDigestMismatch
from dendrux.skills import Skill, SkillExecutionRequest, SkillExecutionResult
from dendrux.skills._internal import (
    continuity_error,
    is_script,
    package_bytes,
    render_activation_envelope,
    validate_file_path,
)
from dendrux.types import Message, Role, ToolCall, ToolDef, ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Sequence

    from dendrux.agent import Agent
    from dendrux.loops.base import Loop
    from dendrux.runtime.state import StateStore

NAMES = frozenset({"use_skill", "read_skill_resource", "run_skill_script"})
META_KEY = "dendrux.skills"
_current: ContextVar[SkillSession | None] = ContextVar("skill_session", default=None)


def current(agent: Agent | None = None) -> SkillSession | None:
    session = _current.get()
    return session if session is not None and (agent is None or session.agent is agent) else None


def set_session(session: SkillSession | None) -> Token[SkillSession | None]:
    """Set a session for a runtime-owned execution scope."""
    return _current.set(session)


def reset_session(token: Token[SkillSession | None]) -> None:
    _current.reset(token)


class SkillSession:
    def __init__(
        self, agent: Agent, *, skills: list[Skill], delivery: str, explicit: Sequence[str]
    ):
        self.agent = agent
        self.available = {s.name: s for s in skills}
        self.delivery = delivery
        self.explicit = list(dict.fromkeys(explicit))
        self.activated = set(self.explicit)
        self.executor = agent._skill_executor
        self.packages = {s.name: package_bytes(s) for s in skills}
        self.deny = agent._deny
        self.denied = frozenset(agent._deny_skills)
        self.configured = bool(agent._skills_dir or agent._explicit_skills)
        self.run_id = ""
        self.requested: set[str] = set()
        self.sink: Callable[[str, dict[str, Any], str | None], Awaitable[None]] | None = None

    @classmethod
    def prepare(
        cls, agent: Agent, *, loop: Loop | None = None, explicit: Sequence[str] | None = None
    ) -> SkillSession:
        from dendrux.loops.single import SingleCall

        agent._ensure_skills_loaded()
        skills = list(agent._loaded_skills or [])
        agent._validate_skill_contents(skills, loop=loop)
        names = {s.name for s in skills}
        requested = list(dict.fromkeys(explicit or []))
        invalid = [n for n in requested if n not in names]
        if invalid:
            raise ValueError(f"Unknown or denied skills: {invalid}")
        return cls(
            agent,
            skills=skills,
            delivery="inlined"
            if isinstance(agent._effective_loop(loop), SingleCall)
            else "progressive",
            explicit=requested,
        )

    @contextmanager
    def bind(self) -> Iterator[SkillSession]:
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "delivery": self.delivery,
            "skills": [
                {"name": s.name, "revision": s.revision, "digest": s.digest}
                for s in sorted(self.available.values(), key=lambda s: s.name)
            ],
            "explicit": self.explicit,
        }

    def snapshot(self) -> dict[str, Any]:
        """Serialize only versioned activation state; configuration stays on Agent."""
        return {"v": 1, "activated": sorted(self.activated)}

    def restore_snapshot(self, payload: dict[str, Any]) -> None:
        """Restore known activations and count unknown names without logging them."""
        import logging

        if payload.get("v") != 1 or not isinstance(payload.get("activated"), list):
            raise ValueError("Unsupported skills session snapshot")
        names = {name for name in payload["activated"] if isinstance(name, str)}
        self.activated = names.intersection(self.available)
        unknown = len(names - self.available.keys())
        if unknown:
            logging.getLogger(__name__).warning(
                "Unknown snapshot skills ignored",
                extra={"run_id": self.run_id, "code": "skills_session_unknown", "count": unknown},
            )

    def fingerprint(self) -> dict[str, Any] | None:
        if not self.configured:
            return None
        return {
            "delivery": self.delivery,
            "explicit": self.explicit,
            "skills": sorted((s.name, s.digest) for s in self.available.values()),
        }

    def envelopes(self) -> list[str]:
        if self.delivery == "inlined":
            return []
        return [
            render_activation_envelope(
                self.available[n], scripts_runnable=self.executor is not None
            )
            for n in self.explicit
        ]

    def restore(self, history: list[Message]) -> None:
        calls = {
            tc.id: tc.params.get("name")
            for m in history
            if m.role == Role.ASSISTANT
            for tc in m.tool_calls or []
            if tc.name == "use_skill"
        }
        for m in history:
            if m.role == Role.TOOL and not m.meta.get("is_error") and m.call_id in calls:
                name = calls[m.call_id]
                if isinstance(name, str) and name in self.available:
                    self.activated.add(name)

    async def emit(self, event: str, data: dict[str, Any], call_id: str | None = None) -> None:
        if self.sink:
            await self.sink(event, data, call_id)

    async def start(self) -> None:
        if not self.configured:
            return
        await self.emit(
            "skill.catalog_resolved",
            {
                "delivery": self.delivery,
                "skills": self.manifest()["skills"],
                "denied": sorted(self.denied),
            },
        )
        names = sorted(self.available) if self.delivery == "inlined" else self.explicit
        for name in names:
            await self.emit(
                "skill.activated",
                {
                    "name": name,
                    "digest": self.available[name].digest,
                    "origin": "explicit" if name in self.explicit else "inlined",
                },
            )

    def tool_defs(self) -> list[ToolDef]:
        if self.delivery == "inlined":
            return []
        names = sorted(self.explicit if "use_skill" in self.deny else self.available)
        if not names:
            return []
        definitions = []

        def add(name: str, props: dict[str, Any], parallel: bool = True) -> None:
            definitions.append(
                ToolDef(
                    name=name,
                    description={
                        "use_skill": "Activate a skill and receive its instructions.",
                        "read_skill_resource": "Read a text resource from an active skill.",
                        "run_skill_script": "Execute a script from an active skill.",
                    }[name],
                    parameters={"type": "object", "properties": props, "required": list(props)},
                    parallel=parallel,
                    meta={"dendrux.internal": True},
                )
            )

        if "use_skill" not in self.deny:
            add("use_skill", {"name": {"type": "string", "enum": names}}, False)
        props = {"skill": {"type": "string", "enum": names}, "path": {"type": "string"}}
        if any(
            isinstance(v, str) and not is_script(p)
            for n in names
            for p, v in self.available[n].files.items()
        ):
            add("read_skill_resource", props)
        if self.executor and any(is_script(p) for n in names for p in self.available[n].files):
            add(
                "run_skill_script",
                {
                    **props,
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 4096},
                        "maxItems": 64,
                    },
                },
                False,
            )
        return definitions

    async def preflight(
        self,
        tc: ToolCall,
        *,
        requested: bool = False,
        anticipated: set[str] | None = None,
        approved: bool = False,
    ) -> ToolResult | None:
        params = tc.params
        name = params.get("name" if tc.name == "use_skill" else "skill")
        if not isinstance(name, str) or name not in self.available:
            if tc.name == "use_skill":
                await self.emit(
                    "skill.activation_failed",
                    {
                        "name": str(name)[:256],
                        "reason": "denied"
                        if isinstance(name, str) and name in self.denied
                        else "not_found",
                    },
                    tc.id,
                )
            return self.failure(tc, f"Skill '{name}' not found or denied.")
        if tc.name == "use_skill":
            return None
        if name not in self.activated and name not in (anticipated or set()):
            return self.failure(tc, f"Skill '{name}' is not active in this run.")
        path = params.get("path")
        if not isinstance(path, str):
            return self.failure(tc, "path must be a string.")
        try:
            validate_file_path(path)
        except ValueError as exc:
            return self.failure(tc, str(exc))
        files = self.available[name].files
        if path not in files:
            return self.failure(tc, f"Unknown skill resource: {path}")
        if tc.name == "read_skill_resource":
            if is_script(path) or not isinstance(files[path], str):
                return self.failure(tc, "Scripts and binary resources cannot be read.")
        else:
            if not is_script(path):
                return self.failure(tc, "Only scripts/ files can be executed.")
            args = params.get("args")
            if (
                not isinstance(args, list)
                or len(args) > 64
                or any(not isinstance(a, str) or len(a) > 4096 for a in args)
            ):
                return self.failure(
                    tc, "args must contain at most 64 strings of at most 4096 characters."
                )
            if self.executor is None:
                if approved:
                    await self.emit(
                        "skill.script_failed",
                        {"name": name, "path": path, "error_type": "executor_unavailable"},
                        tc.id,
                    )
                return self.failure(tc, "Skill executor unavailable.")
            if requested and tc.id not in self.requested:
                self.requested.add(tc.id)
                await self.emit(
                    "skill.script_requested", {"name": name, "path": path, "argc": len(args)}, tc.id
                )
        return None

    @staticmethod
    def failure(tc: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=json.dumps({"error": message}),
            success=False,
            error=message,
        )

    async def execute(self, tc: ToolCall) -> ToolResult:
        failure = await self.preflight(tc)
        if failure:
            return failure
        p = tc.params
        name = p.get("name" if tc.name == "use_skill" else "skill")
        assert isinstance(name, str)
        skill = self.available[name]
        if tc.name == "use_skill":
            if name in self.activated:
                value = f"Skill '{name}' is already active."
            else:
                self.activated.add(name)
                value = render_activation_envelope(
                    skill, scripts_runnable=self.executor is not None
                )
                await self.emit(
                    "skill.activated",
                    {"name": name, "digest": skill.digest, "origin": "model"},
                    tc.id,
                )
            return ToolResult(name=tc.name, call_id=tc.id, payload=json.dumps(value))
        path = p["path"]
        if tc.name == "read_skill_resource":
            resource = skill.files[path]
            assert isinstance(resource, str)
            value = resource
            truncated = len(value) > 50000
            text = value[:50000] + (
                f"\n[truncated: showing 50000 of {len(value)} chars]" if truncated else ""
            )
            await self.emit(
                "skill.resource_read",
                {"name": name, "path": path, "chars": len(value), "truncated": truncated},
                tc.id,
            )
            return ToolResult(name=tc.name, call_id=tc.id, payload=json.dumps(text))
        assert self.executor is not None
        request = SkillExecutionRequest(
            run_id=self.run_id,
            tool_call_id=tc.id,
            skill_name=name,
            revision=skill.revision,
            digest=skill.digest,
            script_path=path,
            args=tuple(p["args"]),
            files=self.packages[name],
        )
        start = time.monotonic()
        deadline = asyncio.timeout(self.executor.timeout_seconds)
        try:
            async with deadline:
                result = await self.executor.execute(request)
        except Exception as exc:
            error_type = "timeout" if deadline.expired() else type(exc).__name__
            await self.emit(
                "skill.script_failed", {"name": name, "path": path, "error_type": error_type}, tc.id
            )
            return self.failure(tc, f"{error_type}: {exc}")
        if (
            not isinstance(result, SkillExecutionResult)
            or not isinstance(result.stdout, str)
            or not isinstance(result.stderr, str)
            or not isinstance(result.exit_code, int)
            or isinstance(result.exit_code, bool)
        ):
            await self.emit(
                "skill.script_failed",
                {"name": name, "path": path, "error_type": "invalid_result"},
                tc.id,
            )
            return self.failure(tc, "invalid_result: executor must return SkillExecutionResult.")
        truncated = len(result.stdout) > 20000 or len(result.stderr) > 20000
        duration = int((time.monotonic() - start) * 1000)
        await self.emit(
            "skill.script_completed",
            {
                "name": name,
                "path": path,
                "exit_code": result.exit_code,
                "duration_ms": duration,
                "stdout_chars": len(result.stdout),
                "stderr_chars": len(result.stderr),
                "truncated": truncated,
            },
            tc.id,
        )
        return ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=json.dumps(
                {
                    "exit_code": result.exit_code,
                    "stdout": result.stdout[:20000],
                    "stderr": result.stderr[:20000],
                    "truncated": truncated,
                }
            ),
            success=result.exit_code == 0,
            error=f"Script exited with code {result.exit_code}" if result.exit_code else None,
            duration_ms=duration,
        )


def _validate_manifest(manifest: Any, run_id: str) -> None:
    """Reject corrupt durable state before claiming, without echoing its contents."""
    valid = (
        isinstance(manifest, dict)
        and type(manifest.get("version")) is int
        and manifest["version"] == 1
        and manifest.get("delivery") in ("progressive", "inlined")
        and isinstance(manifest.get("skills"), list)
        and isinstance(manifest.get("explicit"), list)
    )
    if valid:
        entries = manifest["skills"]
        valid = all(
            isinstance(e, dict)
            and isinstance(e.get("name"), str)
            and isinstance(e.get("digest"), str)
            for e in entries
        )
        if valid:
            names = [e["name"] for e in entries]
            valid = len(names) == len(set(names)) and all(
                isinstance(n, str) and n in names for n in manifest["explicit"]
            )
    if not valid:
        raise ValueError(f"Invalid skill manifest for run '{run_id}'.")


async def continuation(
    agent: Agent, store: StateStore, run_id: str, *, loop: Loop | None = None
) -> SkillSession:
    """Read and verify before any claim; pin the manifest's available set."""
    session = SkillSession.prepare(agent, loop=loop)
    record = await store.get_run(run_id)
    manifest = (record.meta or {}).get(META_KEY) if record else None
    if manifest is not None:
        _validate_manifest(manifest, run_id)
        missing = []
        mismatched = []
        for entry in manifest["skills"]:
            skill = session.available.get(entry["name"])
            if skill is None:
                missing.append(entry["name"])
            elif skill.digest != entry["digest"]:
                mismatched.append(SkillDigestMismatch(entry["name"], entry["digest"], skill.digest))
        error = continuity_error(run_id, missing=missing, mismatched=mismatched)
        if error:
            raise error
        session = SkillSession(
            agent,
            skills=[session.available[e["name"]] for e in manifest["skills"]],
            delivery=manifest["delivery"],
            explicit=manifest["explicit"],
        )
        session.configured = True
    session.run_id = run_id
    pause = await store.get_pause_state(run_id)
    if pause:
        from dendrux.types import PauseState

        session.restore(PauseState.from_dict(pause).history)
    return session


class SkillTool:
    """Internal callable tagged with its session; executor receives the real call id."""

    def __init__(self, session: SkillSession, name: str):
        self.session = session
        self.name = name

    async def __call__(self, **params: Any) -> ToolResult:
        return await self.session.execute(ToolCall(name=self.name, params=params))
