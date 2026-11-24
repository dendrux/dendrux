"""Side-effect counters for fan-out tests (0.8 spec §16).

A fan-out test must prove that work is not repeated: a finished child's tools
and provider calls keep their counts after a sibling's approval, a restart or a
reclaim. Final state alone cannot show that, so tests count effects here and
assert the counts.

One :class:`FanoutEffects` recorder counts three things:

- a tool effect per ``(child, tool)`` — :meth:`FanoutEffects.counting` wraps a
  tool function, or call :meth:`FanoutEffects.record_tool` inside one;
- a provider call per child label — :meth:`FanoutEffects.provider` wraps any
  provider;
- a ``tool_result_transform`` invocation per ``(tool name, call id)`` —
  :meth:`FanoutEffects.transform`.

Every effect is counted when it *starts*, so an attempt that later fails, is
cancelled or dies with its process still counts.

Counters are external to the code under test. With no ``path`` they live in the
recorder; with a ``path`` every effect is appended to that file with one
unbuffered write, so the counts survive ``os._exit`` and are shared by every
process that records to the same path (crash-cut and multi-process tests).

This module imports no 0.8 fan-out module. Later tasks import it read-only.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from collections import Counter
from contextlib import aclosing
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from dendrux.llm.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping
    from pathlib import Path

    from dendrux.llm.base import ImageCapabilitySource, ImageSupport
    from dendrux.types import (
        LLMResponse,
        ProviderCapabilities,
        StreamEvent,
        ToolResultContext,
        ToolResultTransform,
    )

_F = TypeVar("_F", bound="Callable[..., Any]")

_TOOL = "tool"
_PROVIDER = "provider"
_TRANSFORM = "transform"
_Effect = tuple[str, str, str]  # (kind, owner, name)


@dataclass(frozen=True)
class EffectCounts:
    """An immutable snapshot of every counter.

    ``tools`` is keyed by ``(child, tool)``, ``provider`` by child label and
    ``transforms`` by ``(tool name, call id)``.
    """

    tools: Mapping[tuple[str, str], int]
    provider: Mapping[str, int]
    transforms: Mapping[tuple[str, str], int]

    def __post_init__(self) -> None:
        for name in ("tools", "provider", "transforms"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def tool(self, child: str, tool: str) -> int:
        """How many times ``child`` started ``tool``."""
        return self.tools.get((child, tool), 0)

    def provider_calls(self, child: str) -> int:
        """How many provider calls were started under the label ``child``."""
        return self.provider.get(child, 0)

    def transform_calls(self, call_id: str | None = None) -> int:
        """Transform invocations for one call id, or for all calls when None."""
        return sum(
            count
            for (_, seen_call), count in self.transforms.items()
            if call_id is None or seen_call == call_id
        )

    def for_child(self, child: str) -> EffectCounts:
        """Only ``child``'s tool and provider counts (transforms belong to the parent)."""
        return EffectCounts(
            tools={key: n for key, n in self.tools.items() if key[0] == child},
            provider={key: n for key, n in self.provider.items() if key == child},
            transforms={},
        )


class FanoutEffects:
    """Records and counts the side effects of one test's workflow."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._memory: list[_Effect] = []

    # -- recording --------------------------------------------------------

    def record_tool(self, child: str, tool: str) -> None:
        """Count one start of ``tool`` by ``child``."""
        self._record((_TOOL, child, tool))

    def record_provider(self, child: str) -> None:
        """Count one provider call started under the label ``child``."""
        self._record((_PROVIDER, child, ""))

    def counting(self, child: str, *, name: str | None = None) -> Callable[[_F], _F]:
        """Decorate a sync or async tool function so each call counts for ``child``.

        Apply it under ``@tool``; the wrapper keeps the function's name,
        docstring and signature, so the generated schema is unchanged.
        """

        def decorate(fn: _F) -> _F:
            label = name or fn.__name__
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def counted_async(*args: Any, **kwargs: Any) -> Any:
                    self.record_tool(child, label)
                    return await fn(*args, **kwargs)

                return counted_async  # type: ignore[return-value]

            @functools.wraps(fn)
            def counted(*args: Any, **kwargs: Any) -> Any:
                self.record_tool(child, label)
                return fn(*args, **kwargs)

            return counted  # type: ignore[return-value]

        return decorate

    def provider(self, child: str, inner: LLMProvider) -> CountingProvider:
        """Wrap ``inner`` so each ``complete`` / ``complete_stream`` counts for ``child``."""
        return CountingProvider(self, child, inner)

    def transform(self, inner: ToolResultTransform | None = None) -> ToolResultTransform:
        """A ``tool_result_transform`` that counts each invocation.

        It returns what ``inner`` returns (awaitables pass through), or the
        result unchanged when there is no ``inner``.
        """

        def counted(ctx: ToolResultContext) -> Any:
            self._record((_TRANSFORM, ctx.tool_name, ctx.call_id))
            return ctx.result if inner is None else inner(ctx)

        return counted

    # -- reading ----------------------------------------------------------

    def counts(self) -> EffectCounts:
        """A snapshot of every counter, read from the external log."""
        totals = Counter(self._effects())
        return EffectCounts(
            tools={(o, n): c for (kind, o, n), c in totals.items() if kind == _TOOL},
            provider={o: c for (kind, o, _), c in totals.items() if kind == _PROVIDER},
            transforms={(o, n): c for (kind, o, n), c in totals.items() if kind == _TRANSFORM},
        )

    def assert_unchanged(self, before: EffectCounts, child: str) -> None:
        """Fail unless ``child``'s tool and provider counts still equal ``before``'s."""
        was, now = before.for_child(child), self.counts().for_child(child)
        assert was == now, (
            f"side effects of finished child {child!r} changed: "
            f"tools {dict(was.tools)} -> {dict(now.tools)}, "
            f"provider {dict(was.provider)} -> {dict(now.provider)}"
        )

    # -- storage ----------------------------------------------------------

    def _record(self, effect: _Effect) -> None:
        if self._path is None:
            self._memory.append(effect)
            return
        # One O_APPEND write per effect: atomic for a short line, unbuffered, so
        # it is on disk even if the process is killed on the next statement.
        line = (json.dumps(effect) + "\n").encode()
        fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    def _effects(self) -> list[_Effect]:
        if self._path is None:
            return list(self._memory)
        if not self._path.exists():
            return []
        try:
            return [_decode(line) for line in self._path.read_text().splitlines() if line]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unreadable effect log: {self._path}") from exc


def _decode(line: str) -> _Effect:
    kind, owner, name = json.loads(line)
    return (str(kind), str(owner), str(name))


class CountingProvider(LLMProvider):
    """Delegates to ``inner`` and counts each call under one child label.

    A call counts when it starts (for a stream: when it is first advanced). The
    inner stream is opened, advanced and closed in the caller's task.
    """

    def __init__(self, effects: FanoutEffects, child: str, inner: LLMProvider) -> None:
        self._effects = effects
        self.child = child
        self.inner = inner
        self.tool_result_images = inner.tool_result_images

    @property
    def capabilities(self) -> ProviderCapabilities:  # type: ignore[override]
        return self.inner.capabilities

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def thinking_enabled(self) -> bool:
        return self.inner.thinking_enabled

    @property
    def reports_cost(self) -> bool:
        return self.inner.reports_cost

    async def image_input_support(self, model: str) -> ImageSupport:
        return await self.inner.image_input_support(model)

    def image_capability_source(self) -> ImageCapabilitySource:
        return self.inner.image_capability_source()

    def many_image_rule(self, model: str) -> tuple[int, int] | None:
        return self.inner.many_image_rule(model)

    async def close(self) -> None:
        await self.inner.close()

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self._effects.record_provider(self.child)
        return await self.inner.complete(messages, tools, **kwargs)

    async def complete_stream(
        self, messages: Any, tools: Any = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        self._effects.record_provider(self.child)
        async with aclosing(self.inner.complete_stream(messages, tools, **kwargs)) as events:
            async for event in events:
                yield event
