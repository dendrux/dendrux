"""Agent — the primary interface for Dendrux.

Describes the agent's identity, capabilities, and limits, and provides
runtime methods (run, resume) that delegate to the runner.

Two creation styles:
    # Constructor — primary API
    agent = Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[add, multiply],
        prompt="You are a math assistant.",
    )

    # Subclass — for reusable agent types
    class AuditAgent(Agent):
        tools = [readRange, finishAudit]
        prompt = "You are a workbook auditor."
        max_iterations = 15

    agent = AuditAgent(provider=AnthropicProvider(model="..."))
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from dendrux._sentinel import _UnsetType
from dendrux.tool import get_tool_def, is_tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel
    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.guardrails._protocol import Guardrail
    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.mcp._server import MCPServer
    from dendrux.runtime.state import StateStore
    from dendrux.skills._loader import Skill
    from dendrux.tools import ToolLookups
    from dendrux.types import Budget, RunResult, RunStream, ToolDef, ToolResult

_agent_logger = logging.getLogger(__name__)

# Safety limit to prevent runaway LLM costs. Can be overridden per-agent
# once the worker/config layer ships (Sprint 6).
MAX_ITERATIONS_CEILING = 200

_UNSET = _UnsetType()


def _validate_max_delegation_depth(value: int | None | _UnsetType) -> None:
    """Validate max_delegation_depth — shared between constructor and _validate()."""
    if isinstance(value, _UnsetType):
        return
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError(
            f"max_delegation_depth must be a non-negative integer or None, got {value!r}"
        )


class Agent:
    """Dendrux agent — definition and runtime facade.

    Holds the agent's identity, capabilities, limits, and runtime config
    (provider, persistence, redaction). Provides run() and resume() as
    instance methods that delegate to the runner.

    Attributes:
        name: Agent identifier. Auto-derived from class name if subclassed.
        prompt: System prompt for the agent.
        tools: List of @tool-decorated functions this agent can use.
        max_iterations: Maximum ReAct loop iterations before stopping.
    """

    name: str = ""
    prompt: str = ""
    tools: list[Callable[..., Any]] = []
    max_iterations: int = 10
    max_delegation_depth: int | None | _UnsetType = _UNSET

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "model" in cls.__dict__:
            warnings.warn(
                f"Class-level 'model' on {cls.__name__} is deprecated. "
                "Pass provider= to __init__() instead. The model is set on the provider.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Remove so the base class property takes over
            delattr(cls, "model")

    @overload
    def __init__(
        self,
        *,
        provider: LLMProvider,
        prompt: str,
        name: str = ...,
        tools: list[Callable[..., Any]] = ...,
        tool_sources: list[MCPServer] | None = ...,
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        output_type: type[BaseModel] | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        redact: Callable[[str], str] | None = ...,
        deny: list[str] | None = ...,
        require_approval: list[str] | None = ...,
        budget: Budget | None = ...,
        guardrails: list[Guardrail] | None = ...,
        skills_dir: str | Path | None = ...,
        skills: list[Skill] | None = ...,
        deny_skills: list[str] | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        provider: LLMProvider | None = ...,
        name: str = ...,
        prompt: str = ...,
        tools: list[Callable[..., Any]] = ...,
        tool_sources: list[MCPServer] | None = ...,
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        output_type: type[BaseModel] | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        redact: Callable[[str], str] | None = ...,
        deny: list[str] | None = ...,
        require_approval: list[str] | None = ...,
        budget: Budget | None = ...,
        guardrails: list[Guardrail] | None = ...,
        skills_dir: str | Path | None = ...,
        skills: list[Skill] | None = ...,
        deny_skills: list[str] | None = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        name: str | _UnsetType = _UNSET,
        prompt: str | _UnsetType = _UNSET,
        tools: list[Callable[..., Any]] | _UnsetType = _UNSET,
        tool_sources: list[MCPServer] | None = None,
        max_iterations: int | _UnsetType = _UNSET,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        loop: Loop | None = None,
        output_type: type[BaseModel] | None = None,
        provider: LLMProvider | None | _UnsetType = _UNSET,
        database_url: str | None = None,
        database_options: dict[str, Any] | None = None,
        state_store: StateStore | None = None,
        redact: Callable[[str], str] | None = None,
        deny: list[str] | None = None,
        require_approval: list[str] | None = None,
        budget: Budget | None = None,
        guardrails: list[Guardrail] | None = None,
        skills_dir: str | Path | None = None,
        skills: list[Skill] | None = None,
        deny_skills: list[str] | None = None,
    ) -> None:
        # --- Subclass guard: block class-level provider ---
        from dendrux.llm.base import LLMProvider as _LLMBase

        cls_provider = self.__class__.__dict__.get("provider")
        if isinstance(cls_provider, _LLMBase):
            raise ValueError(
                "provider must be passed to __init__(), not set as a class attribute. "
                "Class-level provider creates a live HTTP client at class definition time."
            )

        # --- Identity ---
        if not isinstance(name, _UnsetType):
            self.name = name
        elif not self.name:
            self.name = type(self).__name__

        if not isinstance(prompt, _UnsetType):
            self.prompt = prompt
        if not isinstance(tools, _UnsetType):
            self.tools = list(tools)
        else:
            self.tools = list(self.__class__.tools)
        if not isinstance(max_iterations, _UnsetType):
            self.max_iterations = max_iterations
        if not isinstance(max_delegation_depth, _UnsetType):
            _validate_max_delegation_depth(max_delegation_depth)
            self.max_delegation_depth = max_delegation_depth

        # --- Loop ---
        self._loop: Loop | None = loop
        self._output_type: type[BaseModel] | None = output_type

        # --- Provider ---
        self._provider: LLMProvider | None = None if isinstance(provider, _UnsetType) else provider

        # --- Persistence ---
        if database_url is not None and state_store is not None:
            raise ValueError(
                "database_url and state_store are mutually exclusive. Pass one or the other."
            )
        if database_options is not None and database_url is None:
            raise ValueError("database_options requires database_url.")

        self._database_url = database_url
        self._database_options = database_options or {}
        self._state_store = state_store
        self._redact = redact
        self._deny: frozenset[str] = frozenset(deny) if deny else frozenset()
        self._require_approval: frozenset[str] = (
            frozenset(require_approval) if require_approval else frozenset()
        )
        self._budget: Budget | None = budget
        self._guardrails: list[Guardrail] | None = guardrails if guardrails else None
        self._lazy_store: StateStore | None = None
        self._private_engine: AsyncEngine | None = None

        # --- MCP tool sources ---
        self._tool_sources: list[MCPServer] = list(tool_sources) if tool_sources else []
        if self._tool_sources:
            from dendrux.mcp._server import MCPServer as _MCPServer

            seen_names: set[str] = set()
            for i, src in enumerate(self._tool_sources):
                if not isinstance(src, _MCPServer):
                    raise ValueError(
                        f"Agent '{self.name}' tool_sources[{i}] is {type(src).__name__}, "
                        f"not an MCPServer instance. Use MCPServer(name, url=... | command=[...])."
                    )
                if src.name in seen_names:
                    raise ValueError(
                        f"Agent '{self.name}' has duplicate tool_sources name '{src.name}'. "
                        f"Each MCP source must have a unique name."
                    )
                seen_names.add(src.name)
        self._discovered_tool_defs: list[ToolDef] | None = None
        self._mcp_executors: dict[str, Callable[..., Any]] | None = None
        self._discovery_lock = asyncio.Lock()

        # --- Skills ---
        if skills_dir is not None and not isinstance(skills_dir, (str, Path)):
            raise ValueError(
                f"Agent '{self.name}' skills_dir must be a string or Path, "
                f"got {type(skills_dir).__name__}."
            )
        self._skills_dir: Path | None = Path(skills_dir) if skills_dir is not None else None

        self._explicit_skills: list[Skill] | None = None
        if skills is not None:
            from dendrux.skills._loader import Skill as _Skill

            for i, s in enumerate(skills):
                if not isinstance(s, _Skill):
                    raise ValueError(
                        f"Agent '{self.name}' skills[{i}] is {type(s).__name__}, "
                        f"not a Skill instance. Use Skill.from_dir(path)."
                    )
            self._explicit_skills = list(skills)

        self._deny_skills: frozenset[str] = frozenset(deny_skills) if deny_skills else frozenset()
        self._loaded_skills: list[Skill] | None = None
        self._denied_skill_names: list[str] | None = None

        self._validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier — reads from provider."""
        if self._provider is None:
            return ""
        return self._provider.model

    @property
    def loop(self) -> Loop | None:
        """The execution loop, or None (runner defaults to ReActLoop)."""
        return self._loop

    @property
    def output_type(self) -> type[BaseModel] | None:
        """The default output type for structured output, or None."""
        return self._output_type

    @property
    def deny(self) -> frozenset[str]:
        """Tool names denied by policy. Empty frozenset if none."""
        return self._deny

    @property
    def require_approval(self) -> frozenset[str]:
        """Tool names requiring human approval. Empty frozenset if none."""
        return self._require_approval

    @property
    def budget(self) -> Budget | None:
        """Advisory token budget, or None."""
        return self._budget

    @property
    def guardrails(self) -> list[Guardrail] | None:
        """Content guardrails, or None."""
        return self._guardrails

    @property
    def provider(self) -> LLMProvider | None:
        """The LLM provider, or None if not configured."""
        return self._provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_provider(self) -> LLMProvider:
        """Return the provider or raise if not configured."""
        if self._provider is None:
            raise ValueError("Agent requires a provider. Pass provider= to the constructor.")
        return self._provider

    def _resolve_run_max_delegation_depth(
        self, requested: int | None | _UnsetType
    ) -> int | None | _UnsetType:
        """Resolve max_delegation_depth for a run/stream call.

        Returns the value to pass to the runner, or _UNSET if the runner
        should handle parent/default logic itself.

        Precedence: explicit run kwarg → agent default → _UNSET (runner decides).
        """
        if not isinstance(requested, _UnsetType):
            return requested
        if not isinstance(self.max_delegation_depth, _UnsetType):
            return self.max_delegation_depth
        return _UNSET

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate agent configuration at creation time.

        Governance name validation (deny/require_approval names exist in
        tools) is deferred to discovery time when tool_sources are present,
        because MCP tool names aren't known until the server is connected.
        """
        if not self.prompt:
            raise ValueError(
                f"Agent '{self.name}' requires a prompt. "
                f"Set prompt as a class attribute or pass it to the constructor."
            )

        # Validate max_delegation_depth (catches subclass defaults like -1)
        _validate_max_delegation_depth(self.max_delegation_depth)

        # SingleCall cannot have tools or tool_sources
        from dendrux.loops.single import SingleCall

        if isinstance(self._loop, SingleCall) and self.tools:
            tool_names = [getattr(fn, "__name__", str(fn)) for fn in self.tools]
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has {len(self.tools)} "
                f"tool(s): {tool_names}. SingleCall agents must have zero tools. "
                f"Either remove the tools or use ReActLoop."
            )

        if isinstance(self._loop, SingleCall) and self._tool_sources:
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has "
                f"{len(self._tool_sources)} tool_source(s). "
                f"SingleCall agents cannot have tool_sources."
            )

        if isinstance(self._loop, SingleCall) and (self._skills_dir or self._explicit_skills):
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has skills. "
                f"SingleCall agents cannot have skills (use_skill requires tool calling)."
            )

        # output_type requires SingleCall in v1
        if self._output_type is not None and not isinstance(self._loop, SingleCall):
            raise ValueError(
                f"Agent '{self.name}' has output_type={self._output_type.__name__} "
                f"but does not use SingleCall loop. Structured output is only "
                f"supported with SingleCall in this version. Either set "
                f"loop=SingleCall() or remove output_type."
            )

        for fn in self.tools:
            if not is_tool(fn):
                raise ValueError(
                    f"Agent '{self.name}' has a non-tool in its tools list: "
                    f"'{getattr(fn, '__name__', fn)}'. Decorate it with @tool()."
                )

        # Reserve 'use_skill' name when skills are configured
        has_skills = bool(self._skills_dir or self._explicit_skills)
        if has_skills:
            for fn in self.tools:
                if getattr(fn, "__name__", None) == "use_skill":
                    raise ValueError(
                        f"Agent '{self.name}' has a tool named 'use_skill' but also "
                        f"has skills configured. 'use_skill' is reserved as the "
                        f"skill activation tool when skills are present."
                    )

        if self.max_iterations < 1:
            raise ValueError(
                f"Agent '{self.name}' max_iterations must be >= 1, got {self.max_iterations}."
            )
        if self.max_iterations > MAX_ITERATIONS_CEILING:
            raise ValueError(
                f"Agent '{self.name}' max_iterations cannot exceed "
                f"{MAX_ITERATIONS_CEILING}, got {self.max_iterations}."
            )

        has_mcp = bool(self._tool_sources)

        # --- Shared tool name set for deny/require_approval validation ---
        known_tools: frozenset[str] = frozenset()
        if self.tools and (self._deny or self._require_approval):
            known_tools = frozenset(str(getattr(fn, "__name__", fn)) for fn in self.tools)

        # --- deny= validation ---
        if self._deny:
            # deny requires tools or tool_sources
            if not self.tools and not has_mcp:
                raise ValueError(f"Agent '{self.name}' has deny={sorted(self._deny)} but no tools.")

            # deny + SingleCall is not supported
            if isinstance(self._loop, SingleCall):
                raise ValueError(
                    f"Agent '{self.name}' has deny={sorted(self._deny)} but uses "
                    f"SingleCall loop. deny is not supported with SingleCall."
                )

            # Name existence check — skip if tool_sources present (names
            # may refer to MCP tools not yet discovered). Validated at
            # discovery time in _validate_governance_names().
            if not has_mcp:
                unknown = self._deny - known_tools
                if unknown:
                    raise ValueError(
                        f"Agent '{self.name}' deny contains unknown tool(s): "
                        f"{sorted(unknown)}. Available tools: {sorted(known_tools)}."
                    )

        # --- require_approval= validation ---
        if self._require_approval:
            # require_approval requires tools or tool_sources
            if not self.tools and not has_mcp:
                raise ValueError(
                    f"Agent '{self.name}' has require_approval="
                    f"{sorted(self._require_approval)} but no tools."
                )

            # require_approval + SingleCall is not supported
            if isinstance(self._loop, SingleCall):
                raise ValueError(
                    f"Agent '{self.name}' has require_approval="
                    f"{sorted(self._require_approval)} but uses SingleCall loop. "
                    f"require_approval is not supported with SingleCall."
                )

            # Name existence check — skip if tool_sources present (names
            # may refer to MCP tools not yet discovered). Full check at
            # discovery time in _validate_governance_names().
            if not has_mcp:
                unknown_approval = self._require_approval - known_tools
                if unknown_approval:
                    raise ValueError(
                        f"Agent '{self.name}' require_approval contains unknown tool(s): "
                        f"{sorted(unknown_approval)}. Available tools: {sorted(known_tools)}."
                    )

            # Target check on LOCAL tools — always runs, even with
            # tool_sources. Catches client-tool approval errors early
            # without waiting for MCP discovery.
            if self.tools:
                from dendrux.tool import get_tool_def as _get_tool_def
                from dendrux.types import ToolTarget as _ToolTarget

                for fn in self.tools:
                    td = _get_tool_def(fn)
                    if td.name in self._require_approval and td.target != _ToolTarget.SERVER:
                        raise ValueError(
                            f"Agent '{self.name}' require_approval contains "
                            f"'{td.name}' which has target={td.target.value}. "
                            f"require_approval only supports server tools."
                        )

            # deny and require_approval must not overlap
            overlap = self._deny & self._require_approval
            if overlap:
                raise ValueError(
                    f"Agent '{self.name}' has tools in both deny and "
                    f"require_approval: {sorted(overlap)}. "
                    f"A tool cannot be both denied and require approval."
                )

    def _validate_governance_names(self, all_tool_defs: dict[str, ToolDef]) -> None:
        """Validate deny/require_approval names against full tool set.

        Called at discovery time after MCP tools are discovered. This is
        the deferred half of governance validation — catches typos in
        deny/require_approval that refer to MCP tool names AND enforces
        the require_approval server-tools-only invariant on the merged set.
        """
        from dendrux.types import ToolTarget as _ToolTarget

        all_names = frozenset(all_tool_defs.keys())

        if self._deny:
            unknown = self._deny - all_names
            if unknown:
                raise ValueError(
                    f"Agent '{self.name}' deny contains unknown tool(s): "
                    f"{sorted(unknown)}. Available tools: {sorted(all_names)}."
                )

        if self._require_approval:
            unknown_approval = self._require_approval - all_names
            if unknown_approval:
                raise ValueError(
                    f"Agent '{self.name}' require_approval contains unknown tool(s): "
                    f"{sorted(unknown_approval)}. Available tools: {sorted(all_names)}."
                )

            # require_approval tools must be server tools (including MCP + local)
            for name in self._require_approval:
                td = all_tool_defs.get(name)
                if td is not None and td.target != _ToolTarget.SERVER:
                    raise ValueError(
                        f"Agent '{self.name}' require_approval contains "
                        f"'{name}' which has target={td.target.value}. "
                        f"require_approval only supports server tools."
                    )

    # ------------------------------------------------------------------
    # Persistence (lazy init)
    # ------------------------------------------------------------------

    async def _resolve_state_store(self) -> StateStore | None:
        """Lazily resolve the state store from config.

        Resolution order:
          1. Explicit state_store from constructor → returned as-is
          2. database_url from constructor → private engine (agent-owned)
          3. DENDRUX_DATABASE_URL env var → shared global engine
          4. None (ephemeral mode)

        Ownership rule:
          - Explicit database_url creates a private engine disposed by close().
          - Env-var path uses the shared singleton from get_engine() — NOT
            owned by this agent, NOT disposed by close().
        """
        if self._state_store is not None:
            return self._state_store

        if self._lazy_store is not None:
            return self._lazy_store

        from dendrux.runtime.state import SQLAlchemyStateStore

        if self._database_url is not None:
            engine = await self._create_private_engine(self._database_url)
            self._private_engine = engine
            self._lazy_store = SQLAlchemyStateStore(engine)
            return self._lazy_store

        env_url = os.environ.get("DENDRUX_DATABASE_URL")
        if env_url is not None:
            from dendrux.db.session import get_engine

            engine = await get_engine(env_url)
            self._lazy_store = SQLAlchemyStateStore(engine)
            return self._lazy_store

        return None

    async def _create_private_engine(self, url: str) -> AsyncEngine:
        """Create an agent-owned engine from an explicit database_url.

        Unlike the global get_engine() singleton, this creates a fresh engine
        per agent so multiple agents with different URLs can coexist.
        database_options are applied here.
        """
        from pathlib import Path

        from sqlalchemy.ext.asyncio import create_async_engine

        from dendrux.db.models import Base

        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        engine = create_async_engine(
            url,
            echo=False,
            connect_args=connect_args,
            **self._database_options,
        )

        # Auto-create tables for SQLite (zero-config promise)
        if url.startswith("sqlite"):
            db_path = url.split("///", 1)[-1] if "///" in url else None
            if db_path:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        return engine

    # ------------------------------------------------------------------
    # Runtime methods
    # ------------------------------------------------------------------

    @overload
    async def run(
        self,
        user_input: str,
        *,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        max_delegation_depth: int | None,
        idempotency_key: str | None = ...,
        output_type: type[BaseModel] | None = ...,
        **kwargs: Any,
    ) -> RunResult: ...

    @overload
    async def run(
        self,
        user_input: str,
        *,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        idempotency_key: str | None = ...,
        output_type: type[BaseModel] | None = ...,
        **kwargs: Any,
    ) -> RunResult: ...

    async def run(
        self,
        user_input: str,
        *,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        idempotency_key: str | None = None,
        output_type: type[BaseModel] | None | _UnsetType = _UNSET,
        **kwargs: Any,
    ) -> RunResult:
        """Start a new agent run.

        Delegates to runner.run() with this agent's provider, persistence,
        and redaction config.

        Args:
            user_input: The user's input to process.
            tenant_id: Optional tenant ID for multi-tenant isolation.
            metadata: Optional developer linking data (thread_id, user_id, etc.).
            notifier: Optional notifier for lifecycle events (e.g. ConsoleNotifier
                for terminal output, custom notifiers for Slack/Telegram/etc.).
                Composed with PersistenceRecorder internally if persistence is enabled.
            max_delegation_depth: Maximum allowed delegation depth for the run
                tree. Default 10. None means unbounded. Child runs inherit
                this limit automatically.
            idempotency_key: Optional key for duplicate run prevention. If a run
                with this key already exists and completed, the cached result is
                returned. If still active, RunAlreadyActiveError is raised. If the
                same key is reused with different input, IdempotencyConflictError
                is raised. Requires persistence (database_url, state_store, or
                DENDRUX_DATABASE_URL).
            **kwargs: Forwarded to the LLM provider (temperature, max_tokens, etc.).

        Returns:
            RunResult with status, answer, steps, and usage stats.

        Raises:
            ValueError: If no provider is configured, or if idempotency_key
                is provided without persistence.
            RunAlreadyActiveError: If idempotency_key matches an active run.
            IdempotencyConflictError: If idempotency_key is reused with
                different request parameters.
            DelegationDepthExceededError: If this run would exceed the inherited
                max_delegation_depth from a parent run.
        """
        _validate_max_delegation_depth(max_delegation_depth)
        provider = self._require_provider()

        # Resolve output_type: run-level overrides agent default
        resolved_output_type = (
            self._output_type if isinstance(output_type, _UnsetType) else output_type
        )

        # Validate output_type + loop compatibility at call time
        if resolved_output_type is not None:
            from dendrux.loops.single import SingleCall

            effective_loop = self._loop
            if effective_loop is None:
                # Runner will default to ReActLoop — that's not SingleCall
                raise ValueError(
                    f"output_type={resolved_output_type.__name__} requires SingleCall loop. "
                    f"Set loop=SingleCall() on the Agent or remove output_type."
                )
            if not isinstance(effective_loop, SingleCall):
                raise ValueError(
                    f"output_type={resolved_output_type.__name__} is only supported with "
                    f"SingleCall loop, not {type(effective_loop).__name__}."
                )

        store = await self._resolve_state_store()

        if idempotency_key is not None and store is None:
            raise ValueError(
                "idempotency_key requires persistence (database_url, state_store, "
                "or DENDRUX_DATABASE_URL). Idempotency cannot work without durable state."
            )

        from dendrux.runtime.runner import run as runner_run

        run_kwargs: dict[str, Any] = {}
        resolved_depth = self._resolve_run_max_delegation_depth(max_delegation_depth)
        if not isinstance(resolved_depth, _UnsetType):
            run_kwargs["max_delegation_depth"] = resolved_depth

        return await runner_run(
            self,
            provider=provider,
            user_input=user_input,
            state_store=store,
            tenant_id=tenant_id,
            metadata=metadata,
            redact=self._redact,
            extra_notifier=notifier,
            idempotency_key=idempotency_key,
            output_type=resolved_output_type,
            **run_kwargs,
            **kwargs,
        )

    async def retry(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Retry a terminal run with approximate prior context.

        Creates a fresh run seeded with the original run's conversation
        history from persisted traces. The LLM sees the prior conversation
        and can continue from context. This is NOT resume — it's a new
        run with approximate context, fresh counters, fresh usage.

        Works for any terminal run (error, cancelled, max_iterations,
        success). Source run must have used a retry-capable loop (not
        SingleCall). The retry agent can differ from the original.

        Args:
            run_id: The terminal run to retry from.
            tenant_id: Optional tenant ID for the retry run.
            metadata: Optional developer metadata for the retry run.
            notifier: Optional notifier for lifecycle events.
            **kwargs: Forwarded to the LLM provider.

        Returns:
            RunResult from the retry run.

        Raises:
            ValueError: If no provider/persistence configured, if the
                source run is not terminal, or if the source run was
                SingleCall.
        """
        provider = self._require_provider()
        store = await self._resolve_state_store()

        if store is None:
            raise ValueError(
                "retry() requires persistence (database_url, state_store, "
                "or DENDRUX_DATABASE_URL). Retry reads traces from the original run."
            )

        from dendrux.runtime.runner import retry as runner_retry

        return await runner_retry(
            run_id,
            agent=self,
            provider=provider,
            state_store=store,
            tenant_id=tenant_id,
            metadata=metadata,
            redact=self._redact,
            extra_notifier=notifier,
            **kwargs,
        )

    async def resume(
        self,
        run_id: str,
        *,
        tool_results: list[ToolResult] | None = None,
        user_input: str | None = None,
        notifier: LoopNotifier | None = None,
    ) -> RunResult:
        """Resume a paused run.

        Two modes:
          - tool_results provided -> store + resume with tool results
          - user_input provided -> resume with clarification answer

        Args:
            run_id: The paused run's ID.
            tool_results: Results for pending tool calls (for client tool runs).
            user_input: Clarification answer (for human-in-the-loop runs).
            notifier: Optional additional loop notifier (e.g. TransportNotifier
                for SSE streaming). Composed with PersistenceRecorder internally.

        Returns:
            RunResult with updated status.

        Raises:
            ValueError: If provider not set, persistence not configured,
                both args provided, or neither provided.
        """
        # Validate args before any I/O
        provider = self._require_provider()
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume().")

        store = await self._resolve_state_store()
        if store is None:
            raise ValueError(
                "resume() requires persistence. "
                "Pass database_url or state_store to the constructor."
            )

        from dendrux.runtime.runner import resume as runner_resume
        from dendrux.runtime.runner import resume_with_input

        if tool_results is not None:
            return await runner_resume(
                run_id,
                tool_results,
                state_store=store,
                agent=self,
                provider=provider,
                redact=self._redact,
                extra_notifier=notifier,
            )
        if user_input is not None:
            return await resume_with_input(
                run_id,
                user_input,
                state_store=store,
                agent=self,
                provider=provider,
                redact=self._redact,
                extra_notifier=notifier,
            )
        # No-arg resume — approve path. Runner validates status is
        # WAITING_APPROVAL and executes pending tools.
        return await runner_resume(
            run_id,
            None,
            state_store=store,
            agent=self,
            provider=provider,
            redact=self._redact,
            extra_notifier=notifier,
        )

    @overload
    def stream(
        self,
        user_input: str,
        *,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        max_delegation_depth: int | None,
        **kwargs: Any,
    ) -> RunStream: ...

    @overload
    def stream(
        self,
        user_input: str,
        *,
        tenant_id: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        notifier: LoopNotifier | None = ...,
        **kwargs: Any,
    ) -> RunStream: ...

    def stream(
        self,
        user_input: str,
        *,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        notifier: LoopNotifier | None = None,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        **kwargs: Any,
    ) -> RunStream:
        """Stream an agent run as RunEvents.

        Same parameters as run(). Returns a RunStream immediately —
        no ``await`` needed. The run_id is available before iteration.
        Async setup (DB row, notifiers) runs lazily on first iteration.

        Usage:
            # Full event stream
            async for event in agent.stream("analyze revenue"):
                print(event.type, event.text or "")

            # Just the text
            async for chunk in agent.stream("analyze revenue").text():
                print(chunk, end="")

            # With explicit cleanup (recommended)
            async with agent.stream("analyze revenue") as stream:
                async for event in stream:
                    ...

        After any terminal event (RUN_COMPLETED, RUN_PAUSED, RUN_ERROR),
        the stream ends. If the consumer breaks early, the run is
        cancelled via CAS-guarded cleanup.

        Args:
            user_input: The user's input to process.
            tenant_id: Optional tenant ID for multi-tenant isolation.
            metadata: Optional developer linking data (thread_id, user_id, etc.).
            notifier: Optional notifier for lifecycle events.
            max_delegation_depth: Maximum allowed delegation depth for the run
                tree. Default 10. None means unbounded.
            **kwargs: Forwarded to the LLM provider (temperature, max_tokens, etc.).

        Returns:
            RunStream — async iterable of RunEvent objects.

        Raises:
            ValueError: If no provider is configured.
        """
        _validate_max_delegation_depth(max_delegation_depth)

        if self._guardrails:
            raise ValueError(
                "guardrails are not supported with stream() in this version. "
                "Guardrails require the full response before scanning. "
                "Use agent.run() instead."
            )

        provider = self._require_provider()

        if self._output_type is not None:
            raise NotImplementedError(
                "Structured output streaming is not supported in this version. "
                "Use agent.run() for structured output, or create a separate "
                "Agent without output_type for streaming."
            )

        from dendrux.runtime.runner import run_stream as runner_run_stream

        stream_kwargs: dict[str, Any] = {}
        resolved_depth = self._resolve_run_max_delegation_depth(max_delegation_depth)
        if not isinstance(resolved_depth, _UnsetType):
            stream_kwargs["max_delegation_depth"] = resolved_depth

        return runner_run_stream(
            self,
            provider=provider,
            user_input=user_input,
            state_store_resolver=self._resolve_state_store,
            tenant_id=tenant_id,
            metadata=metadata,
            redact=self._redact,
            extra_notifier=notifier,
            **stream_kwargs,
            **kwargs,
        )

    def resume_stream(
        self,
        run_id: str,
        *,
        tool_results: list[ToolResult] | None = None,
        user_input: str | None = None,
        notifier: LoopNotifier | None = None,
    ) -> RunStream:
        """Stream a resumed run as RunEvents.

        Same parameters as resume(). Returns a RunStream immediately —
        no ``await`` needed. Async setup (store resolution, pause state
        load, claim, history build) runs lazily on first iteration.

        First event is RUN_RESUMED (not RUN_STARTED), carrying the
        existing run_id for correlation.

        Usage:
            async with agent.resume_stream(run_id, tool_results=results) as stream:
                async for event in stream:
                    if event.type == RunEventType.TEXT_DELTA:
                        print(event.text, end="")

        Args:
            run_id: The paused run's ID.
            tool_results: Results for pending tool calls (client tool runs).
            user_input: Clarification answer (human-in-the-loop runs).
            notifier: Optional additional loop notifier.

        Returns:
            RunStream — async iterable of RunEvent objects.

        Raises:
            ValueError: If provider not set, both args provided, or neither.
        """
        if self._guardrails:
            raise ValueError(
                "guardrails are not supported with resume_stream() in this version. "
                "Use agent.resume() instead."
            )
        provider = self._require_provider()
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume_stream().")

        from dendrux.runtime.runner import resume_stream as runner_resume_stream

        return runner_resume_stream(
            run_id,
            agent=self,
            provider=provider,
            state_store_resolver=self._resolve_state_store,
            tool_results=tool_results,
            user_input=user_input,
            redact=self._redact,
            extra_notifier=notifier,
        )

    # ------------------------------------------------------------------
    # MCP discovery
    # ------------------------------------------------------------------

    async def _ensure_discovered(self) -> None:
        """Lazy MCP tool discovery — called internally by get_tool_lookups().

        Uses asyncio.Lock with double-check pattern: fast path (already
        discovered) skips the lock. Lock protects one-time discovery
        against concurrent agent.run() calls.

        On any failure (connection, discovery, governance validation),
        all successfully opened sources are closed and caches stay unset
        so the next call retries.
        """
        if self._discovered_tool_defs is not None:
            return  # fast path

        if not self._tool_sources:
            # No MCP sources — set empty cache so fast path works
            self._discovered_tool_defs = []
            self._mcp_executors = {}
            return

        async with self._discovery_lock:
            if self._discovered_tool_defs is not None:
                return  # lost race, another caller finished

            discovered_defs: list[ToolDef] = []
            executors: dict[str, Callable[..., Any]] = {}
            opened_sources: list[MCPServer] = []

            # Hoist local tool info — computed once, not per-tool
            local_defs = {get_tool_def(fn).name: get_tool_def(fn) for fn in self.tools}

            try:
                for source in self._tool_sources:
                    tool_defs = await source._discover()
                    opened_sources.append(source)
                    for td in tool_defs:
                        # Collision check against reserved name
                        if td.name == "use_skill" and (self._skills_dir or self._explicit_skills):
                            raise ValueError(
                                f"MCP tool '{td.name}' from source "
                                f"'{td.meta.get('source_name')}' collides with "
                                f"the reserved 'use_skill' meta-tool."
                            )
                        # Collision check against local tools
                        if td.name in local_defs:
                            raise ValueError(
                                f"MCP tool '{td.name}' from source "
                                f"'{td.meta.get('source_name')}' collides with "
                                f"a local tool of the same name."
                            )
                        # Collision check against other MCP tools
                        if td.name in executors:
                            raise ValueError(
                                f"MCP tool '{td.name}' collides with a tool from "
                                f"another MCP source."
                            )
                        discovered_defs.append(td)
                        executors[td.name] = source._create_executor(td.meta["mcp_tool_name"])

                # Build merged name→ToolDef map for full governance validation
                all_tool_defs = dict(local_defs)
                for td in discovered_defs:
                    all_tool_defs[td.name] = td
                self._validate_governance_names(all_tool_defs)

            except BaseException:
                # Multi-source cleanup: close all opened sources
                for src in opened_sources:
                    try:
                        await src.close()
                    except Exception:
                        _agent_logger.warning(
                            "Failed to close MCP source '%s' during cleanup",
                            src.name,
                            exc_info=True,
                        )
                raise

            # Only commit cache after ALL sources discovered + validated.
            # Sort at storage so developer-visible order is alphabetical
            # everywhere, not just at the cache-facing getters.
            self._discovered_tool_defs = sorted(discovered_defs, key=lambda td: td.name)
            self._mcp_executors = executors

    async def get_tool_lookups(self) -> ToolLookups:
        """Build ToolLookups for local + MCP + use_skill tools.

        Calls _ensure_discovered() and _ensure_skills_loaded() internally.
        Safe to call multiple times (discovery/loading is cached).
        """
        await self._ensure_discovered()
        self._ensure_skills_loaded()
        from dendrux.tools import build_tool_lookups
        from dendrux.types import ToolTarget as _ToolTarget

        lookups = build_tool_lookups(
            self.tools,
            mcp_executors=self._mcp_executors,
            mcp_tool_defs=self._discovered_tool_defs,
        )

        # Inject use_skill meta-tool if skills are loaded
        if self._loaded_skills:
            td = self._get_use_skill_tool_def()
            lookups.fn[td.name] = self._execute_use_skill
            lookups.target[td.name] = _ToolTarget.SERVER
            lookups.timeout[td.name] = td.timeout_seconds
            lookups.explicit_timeout[td.name] = td.has_explicit_timeout
            lookups.max_calls[td.name] = td.max_calls_per_run
            lookups.parallel[td.name] = td.parallel

        return lookups

    def get_all_tool_defs(self) -> list[ToolDef]:
        """Local + discovered MCP + use_skill tool defs, sorted by name.

        Sorted output gives cache-stable prefixes to provider tool arrays.
        """
        self._ensure_skills_loaded()
        local = [get_tool_def(fn) for fn in self.tools]
        discovered = self._discovered_tool_defs or []
        all_defs = local + discovered
        if self._loaded_skills:
            all_defs.append(self._get_use_skill_tool_def())
        return sorted(all_defs, key=lambda td: td.name)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def _ensure_skills_loaded(self) -> None:
        """Lazy skill loading — scans skills_dir and merges explicit skills.

        Called at run start. Cached after first call. Cleared by refresh().
        Filters out deny_skills matches.
        """
        if self._loaded_skills is not None:
            return  # already loaded

        from dendrux.skills._loader import Skill as _Skill

        all_skills: list[Skill] = []

        # Scan skills_dir
        if self._skills_dir is not None:
            all_skills.extend(_Skill.scan_dir(self._skills_dir))

        # Add explicit skills
        if self._explicit_skills:
            all_skills.extend(self._explicit_skills)

        # Check for duplicate names
        seen: set[str] = set()
        for s in all_skills:
            if s.name in seen:
                raise ValueError(
                    f"Agent '{self.name}' has duplicate skill name '{s.name}'. "
                    f"Each skill must have a unique name."
                )
            seen.add(s.name)

        # Apply deny_skills filter
        loaded: list[Skill] = []
        denied: list[str] = []
        for s in all_skills:
            if s.name in self._deny_skills:
                denied.append(s.name)
            else:
                loaded.append(s)

        self._loaded_skills = loaded
        self._denied_skill_names = denied

    def get_system_prompt(self) -> str:
        """Compose system prompt with skill catalog.

        Progressive disclosure: includes only skill names and descriptions,
        NOT full bodies. The LLM activates skills by calling the use_skill
        tool, which returns the full instructions.

        Used by loops, runner events, and LLM semantic evidence.
        Single source of truth for the complete system prompt.
        """
        self._ensure_skills_loaded()
        if not self._loaded_skills:
            return self.prompt

        parts = [
            self.prompt,
            "\n\n## Available Skills",
            "\nYou have access to the following skills. To use a skill, "
            "call the `use_skill` tool with the skill name. The skill's "
            "detailed instructions will be returned to you.\n",
        ]
        for skill in sorted(self._loaded_skills, key=lambda s: s.name):
            parts.append(f"- **{skill.name}**: {skill.description}")

        return "\n".join(parts)

    async def _execute_use_skill(self, name: str) -> str:
        """Execute the use_skill meta-tool — returns skill body.

        Called by the loop when the LLM calls use_skill(name=...).
        Returns the full Markdown instructions for the skill.
        """
        if not self._loaded_skills:
            return f"Skill '{name}' not found. No skills are available."

        for skill in self._loaded_skills:
            if skill.name == name:
                return skill.body

        available = ", ".join(s.name for s in self._loaded_skills)
        return f"Skill '{name}' not found. Available skills: {available}"

    def _get_use_skill_tool_def(self) -> ToolDef:
        """Create the ToolDef for the auto-injected use_skill tool."""
        from dendrux.types import ToolDef as _ToolDef

        return _ToolDef(
            name="use_skill",
            description=(
                "Activate a skill to get detailed instructions for a task. "
                "Call this with the skill name before performing a skill-related task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to activate.",
                    }
                },
                "required": ["name"],
            },
            meta={"dendrux.internal": True},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Re-discover all dynamic capabilities.

        Clears MCP discovery cache and skill cache. Next run()
        re-discovers MCP tools and re-scans skills.

        IMPORTANT: Call only when all runs are terminal (success, error,
        cancelled). Paused runs are still active. Calling refresh()
        with active or paused runs closes MCP sessions under active
        executors and invalidates the frozen skill context.
        """
        # Close MCP sessions
        for source in self._tool_sources:
            try:
                await source.close()
            except Exception:
                _agent_logger.warning(
                    "Failed to close MCP source '%s' during refresh",
                    source.name,
                    exc_info=True,
                )

        # Clear all caches
        self._discovered_tool_defs = None
        self._mcp_executors = None
        self._loaded_skills = None
        self._denied_skill_names = None

    async def close(self) -> None:
        """Close MCP sources, provider, and agent-owned engine.

        Only disposes the private engine created from an explicit database_url.
        The shared global engine (from DENDRUX_DATABASE_URL env var) is NOT
        owned by this agent and is left untouched.
        """
        for source in self._tool_sources:
            try:
                await source.close()
            except Exception:
                _agent_logger.warning("Failed to close MCP source '%s'", source.name, exc_info=True)
        # Clear all caches so stale executors/skills are not reused.
        self._discovered_tool_defs = None
        self._mcp_executors = None
        self._loaded_skills = None
        self._denied_skill_names = None
        if self._provider is not None:
            await self._provider.close()
        if self._private_engine is not None:
            await self._private_engine.dispose()
            self._private_engine = None
            self._lazy_store = None

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_tool_defs(self) -> list[ToolDef]:
        """Get ToolDef for each LOCAL tool registered on this agent.

        Includes the auto-injected use_skill tool when skills are loaded.
        Does NOT include MCP tools. Use get_all_tool_defs() for the
        full set (local + discovered + use_skill).

        Output is sorted by ToolDef.name for cache-stable prefixes across
        runs. self.tools storage preserves caller order.
        """
        self._ensure_skills_loaded()
        defs = [get_tool_def(fn) for fn in self.tools]
        if self._loaded_skills:
            defs.append(self._get_use_skill_tool_def())
        return sorted(defs, key=lambda td: td.name)

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, max_iterations={self.max_iterations})"
        )
