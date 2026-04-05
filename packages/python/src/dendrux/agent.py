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

import os
import warnings
from typing import TYPE_CHECKING, Any, overload

from dendrux._sentinel import _UnsetType
from dendrux.tool import get_tool_def, is_tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from dendrux.llm.base import LLMProvider
    from dendrux.loops.base import Loop, LoopNotifier
    from dendrux.runtime.state import StateStore
    from dendrux.types import RunResult, RunStream, ToolDef, ToolResult

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
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        redact: Callable[[str], str] | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        provider: LLMProvider | None = ...,
        name: str = ...,
        prompt: str = ...,
        tools: list[Callable[..., Any]] = ...,
        max_iterations: int = ...,
        max_delegation_depth: int | None = ...,
        loop: Loop | None = ...,
        database_url: str | None = ...,
        database_options: dict[str, Any] | None = ...,
        state_store: StateStore | None = ...,
        redact: Callable[[str], str] | None = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        name: str | _UnsetType = _UNSET,
        prompt: str | _UnsetType = _UNSET,
        tools: list[Callable[..., Any]] | _UnsetType = _UNSET,
        max_iterations: int | _UnsetType = _UNSET,
        max_delegation_depth: int | None | _UnsetType = _UNSET,
        loop: Loop | None = None,
        provider: LLMProvider | None | _UnsetType = _UNSET,
        database_url: str | None = None,
        database_options: dict[str, Any] | None = None,
        state_store: StateStore | None = None,
        redact: Callable[[str], str] | None = None,
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
    def loop(self) -> Loop | None:
        """The execution loop, or None (runner defaults to ReActLoop)."""
        return self._loop

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
        """Validate agent configuration at creation time."""
        if not self.prompt:
            raise ValueError(
                f"Agent '{self.name}' requires a prompt. "
                f"Set prompt as a class attribute or pass it to the constructor."
            )

        # Validate max_delegation_depth (catches subclass defaults like -1)
        _validate_max_delegation_depth(self.max_delegation_depth)

        # SingleCall cannot have tools
        from dendrux.loops.single import SingleCall

        if isinstance(self._loop, SingleCall) and self.tools:
            tool_names = [getattr(fn, "__name__", str(fn)) for fn in self.tools]
            raise ValueError(
                f"Agent '{self.name}' uses SingleCall loop but has {len(self.tools)} "
                f"tool(s): {tool_names}. SingleCall agents must have zero tools. "
                f"Either remove the tools or use ReActLoop."
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
        # user_input path (guarded by validation above)
        return await resume_with_input(
            run_id,
            user_input,  # type: ignore[arg-type]
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
        provider = self._require_provider()

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
        provider = self._require_provider()
        if tool_results is not None and user_input is not None:
            raise ValueError("Cannot provide both tool_results and user_input to resume_stream().")
        if tool_results is None and user_input is None:
            raise ValueError("resume_stream() requires either tool_results or user_input.")

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
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the provider and dispose agent-owned engine.

        Only disposes the private engine created from an explicit database_url.
        The shared global engine (from DENDRUX_DATABASE_URL env var) is NOT
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
