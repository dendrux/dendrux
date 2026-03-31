"""Agent — the primary interface for Dendrite.

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

import os
import warnings
from typing import TYPE_CHECKING, Any

from dendrite.tool import get_tool_def, is_tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrite.llm.base import LLMProvider
    from dendrite.runtime.state import StateStore
    from dendrite.types import RunResult, ToolDef, ToolResult

# Safety limit to prevent runaway LLM costs. Can be overridden per-agent
# once the worker/config layer ships (Sprint 6).
MAX_ITERATIONS_CEILING = 200

# Sentinel to detect "not provided" vs explicitly set to a value
_UNSET: Any = object()


class Agent:
    """Dendrite agent — definition and runtime facade.

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

    def __init__(
        self,
        *,
        name: str = _UNSET,
        prompt: str = _UNSET,
        tools: list[Callable[..., Any]] = _UNSET,
        max_iterations: int = _UNSET,
        provider: LLMProvider = _UNSET,
        database_url: str | None = None,
        database_options: dict[str, Any] | None = None,
        state_store: StateStore | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        # --- Subclass guard: block class-level provider ---
        from dendrite.llm.base import LLMProvider as _LLMBase

        cls_provider = self.__class__.__dict__.get("provider")
        if isinstance(cls_provider, _LLMBase):
            raise ValueError(
                "provider must be passed to __init__(), not set as a class attribute. "
                "Class-level provider creates a live HTTP client at class definition time."
            )

        # --- Identity ---
        if name is not _UNSET:
            self.name = name
        elif not self.name:
            self.name = type(self).__name__

        if prompt is not _UNSET:
            self.prompt = prompt
        if tools is not _UNSET:
            self.tools = list(tools)
        else:
            self.tools = list(self.__class__.tools)
        if max_iterations is not _UNSET:
            self.max_iterations = max_iterations

        # --- Provider ---
        self._provider: LLMProvider | None = provider if provider is not _UNSET else None

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
        self._lazy_store: StateStore | None = None
        self._private_engine: AsyncEngine | None = None

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
    def provider(self) -> LLMProvider | None:
        """The LLM provider, or None if not configured."""
        return self._provider

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate agent configuration at creation time."""
        if not self.prompt:
            raise ValueError(
                f"Agent '{self.name}' requires a prompt. "
                f"Set prompt as a class attribute or pass it to the constructor."
            )

        for fn in self.tools:
            if not is_tool(fn):
                raise ValueError(
                    f"Agent '{self.name}' has a non-tool in its tools list: "
                    f"'{getattr(fn, '__name__', fn)}'. Decorate it with @tool()."
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

    # ------------------------------------------------------------------
    # Persistence (lazy init)
    # ------------------------------------------------------------------

    async def _resolve_state_store(self) -> StateStore | None:
        """Lazily resolve the state store from config.

        Resolution order:
          1. Explicit state_store from constructor → returned as-is
          2. database_url from constructor → private engine (agent-owned)
          3. DENDRITE_DATABASE_URL env var → shared global engine
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

        from dendrite.runtime.state import SQLAlchemyStateStore

        if self._database_url is not None:
            engine = await self._create_private_engine(self._database_url)
            self._private_engine = engine
            self._lazy_store = SQLAlchemyStateStore(engine)
            return self._lazy_store

        env_url = os.environ.get("DENDRITE_DATABASE_URL")
        if env_url is not None:
            from dendrite.db.session import get_engine

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

        from dendrite.db.models import Base

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

    async def run(
        self,
        user_input: str,
        *,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        observer: Any | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Start a new agent run.

        Delegates to runner.run() with this agent's provider, persistence,
        and redaction config.

        Args:
            user_input: The user's input to process.
            tenant_id: Optional tenant ID for multi-tenant isolation.
            metadata: Optional developer linking data (thread_id, user_id, etc.).
            observer: Optional observer for lifecycle events (e.g. ConsoleObserver
                for terminal output, custom observers for Slack/Telegram/etc.).
                Composed with PersistenceObserver internally if persistence is enabled.
            **kwargs: Forwarded to the LLM provider (temperature, max_tokens, etc.).

        Returns:
            RunResult with status, answer, steps, and usage stats.

        Raises:
            ValueError: If no provider is configured.
        """
        if self._provider is None:
            raise ValueError("Agent requires a provider. Pass provider= to the constructor.")

        store = await self._resolve_state_store()

        from dendrite.runtime.runner import run as runner_run

        return await runner_run(
            self,
            provider=self._provider,
            user_input=user_input,
            state_store=store,
            tenant_id=tenant_id,
            metadata=metadata,
            redact=self._redact,
            extra_observer=observer,
        )

    async def resume(
        self,
        run_id: str,
        *,
        tool_results: list[ToolResult] | None = None,
        user_input: str | None = None,
        observer: Any | None = None,
    ) -> RunResult:
        """Resume a paused run.

        Two modes:
          - tool_results provided -> store + resume with tool results
          - user_input provided -> resume with clarification answer

        Args:
            run_id: The paused run's ID.
            tool_results: Results for pending tool calls (for client tool runs).
            user_input: Clarification answer (for human-in-the-loop runs).
            observer: Optional additional loop observer (e.g. TransportObserver
                for SSE streaming). Composed with PersistenceObserver internally.

        Returns:
            RunResult with updated status.

        Raises:
            ValueError: If provider not set, persistence not configured,
                both args provided, or neither provided.
        """
        # Validate args before any I/O
        if self._provider is None:
            raise ValueError("Agent requires a provider. Pass provider= to the constructor.")
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume().")
        if tool_results is None and user_input is None:
            raise ValueError(
                "resume() requires either tool_results or user_input. "
                "No-arg resume (bridge path) is not yet supported."
            )

        store = await self._resolve_state_store()
        if store is None:
            raise ValueError(
                "resume() requires persistence. "
                "Pass database_url or state_store to the constructor."
            )

        from dendrite.runtime.runner import resume as runner_resume
        from dendrite.runtime.runner import resume_with_input

        if tool_results is not None:
            return await runner_resume(
                run_id,
                tool_results,
                state_store=store,
                agent=self,
                provider=self._provider,
                redact=self._redact,
                extra_observer=observer,
            )
        # user_input path (guarded by validation above)
        return await resume_with_input(
            run_id,
            user_input,  # type: ignore[arg-type]
            state_store=store,
            agent=self,
            provider=self._provider,
            redact=self._redact,
            extra_observer=observer,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the provider and dispose agent-owned engine.

        Only disposes the private engine created from an explicit database_url.
        The shared global engine (from DENDRITE_DATABASE_URL env var) is NOT
        owned by this agent and is left untouched.
        """
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
        """Get ToolDef for each tool registered on this agent."""
        return [get_tool_def(fn) for fn in self.tools]

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, max_iterations={self.max_iterations})"
        )
