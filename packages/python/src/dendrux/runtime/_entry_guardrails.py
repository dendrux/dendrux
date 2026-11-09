"""Project fresh governed text before deriving or recording entry evidence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from dendrux.context_blocks import ContextBlock, fold_context
from dendrux.guardrails import GuardrailEngine
from dendrux.guardrails._engine import _bind_entry_engine
from dendrux.types import ContentPart, Message, Role, TextPart

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.agent import Agent
    from dendrux.chat import ChatMessage
    from dendrux.skills._session import SkillSession


@dataclass
class _EntryProjection:
    engine: GuardrailEngine | None
    error: str | None = None
    findings: list[Any] = field(default_factory=list)
    location: str = "message 1 (role=user)"

    @contextmanager
    def bind(self, run_id: str) -> Iterator[None]:
        with _bind_entry_engine(run_id, self.engine):
            yield

    async def project(self, message: Message, index: int) -> Message:
        if self.engine is None:
            return message
        projected, findings, error = await self.engine.project_incoming_message(
            message, index=index
        )
        if error is None:
            # Allowed reversible PII retains its original storage/identity contract.
            return message
        if self.error is None:
            self.error, self.findings = error, findings
            self.location = f"message {index} (role={message.role.value})"
        return projected

    async def text_items(
        self, text: str, items: tuple[Any, ...], role: Role, index: int
    ) -> tuple[str, tuple[Any, ...]]:
        original = Message(role, text)
        projected = await self.project(original, index)
        if projected is original:
            return text, items
        items = tuple(projected.content if isinstance(item, str) else item for item in items)
        return (
            "\n\n".join(item for item in items if isinstance(item, str))
            if items
            else projected.content,
            items,
        )

    async def history(self, history: list[ChatMessage] | None) -> list[ChatMessage] | None:
        if self.engine is None or not history:
            return history
        result = []
        for index, message in enumerate(history, 1):
            text, items = await self.text_items(
                message.content, message.parts or (), Role(message.role), index
            )
            result.append(replace(message, content=text, parts=items or None))
        return result

    async def context(self, context: list[ContextBlock] | None) -> list[ContextBlock] | None:
        if self.engine is None or not context:
            return context
        result = []
        for block in context:
            message = Message(Role.USER, block.content)
            projected = await self.project(message, 1)
            result.append(
                block
                if projected is message
                else ContextBlock(projected.content, placement=block.placement)
            )
        return result

    async def prepare(
        self,
        history: list[Message],
        context: list[ContextBlock] | None,
        text: str,
        parts: tuple[ContentPart, ...] | None,
        session: SkillSession | None,
    ) -> tuple[
        list[Message] | None,
        list[Message],
        list[ContextBlock] | None,
        str,
        tuple[ContentPart, ...] | None,
    ]:
        folded = fold_context(history, context, text, session=session, user_parts=parts)
        messages = folded if folded is not None else [Message(Role.USER, text)]
        previous = self.error, self.findings, self.location
        self.error = None
        broad = await self.rendered(messages, list(history), context, text, parts)
        if self.error is None:
            self.error, self.findings, self.location = previous
            return folded, history, context, text, parts
        rejected = self.error, self.findings, self.location
        self.error = None
        history = [await self.project(message, i + 1) for i, message in enumerate(history)]
        context = await self.context(context)
        own = await self.project(Message(Role.USER, text, parts=parts), len(history) + 1)
        if self.error is None:
            # A wrapper-only rejection has no safe per-piece attribution.
            self.error, self.findings, self.location = rejected
            projected, history, context, text, parts = broad
        else:
            text, parts = own.content, own.parts
            refolded = fold_context(history, context, text, session=session, user_parts=parts)
            projected, history, context, text, parts = await self.rendered(
                refolded, history, context, text, parts
            )
        return projected if folded is not None else None, history, context, text, parts

    async def rendered(
        self,
        messages: list[Message] | None,
        history: list[Message],
        context: list[ContextBlock] | None,
        text: str,
        parts: tuple[ContentPart, ...] | None,
    ) -> tuple[
        list[Message] | None,
        list[Message],
        list[ContextBlock] | None,
        str,
        tuple[ContentPart, ...] | None,
    ]:
        if self.engine is None or messages is None:
            return messages, history, context, text, parts
        projected = []
        for index, message in enumerate(messages):
            clean = await self.project(message, index + 1)
            projected.append(clean)
            if clean is message:
                continue
            placeholder = clean.content
            if index < len(history):
                history[index] = _replace_text(history[index], placeholder)
            else:
                parts = _replace_parts(parts, placeholder)
                text = _text_projection(parts, placeholder)
            context = _replace_context(context, index, len(messages), placeholder)
        return projected, history, context, text, parts


def _replace_context(
    context: list[ContextBlock] | None, index: int, length: int, placeholder: str
) -> list[ContextBlock] | None:
    return [
        ContextBlock(placeholder, placement=block.placement)
        if (block.placement == "stable" and index == 0)
        or (block.placement == "dynamic" and index == length - 1)
        else block
        for block in context or []
    ] or None


def _text_projection(parts: tuple[ContentPart, ...] | None, placeholder: str) -> str:
    return "\n\n".join(p.text for p in parts if isinstance(p, TextPart)) if parts else placeholder


def _replace_parts(
    parts: tuple[ContentPart, ...] | None, placeholder: str
) -> tuple[ContentPart, ...] | None:
    return (
        tuple(TextPart(placeholder) if isinstance(p, TextPart) else p for p in parts)
        if parts
        else parts
    )


def _replace_text(message: Message, placeholder: str) -> Message:
    parts = _replace_parts(message.parts, placeholder)
    return replace(
        message, content=_text_projection(parts, placeholder), parts=parts, meta={}, source=None
    )


def _entry_projection(agent: Agent, run_id: str) -> _EntryProjection:
    engine = GuardrailEngine(agent.guardrails, run_id=run_id) if agent.guardrails else None
    if engine is not None:
        # Legacy loops retain their complete safe audit samples; format 2 samples at its boundary.
        engine._retain_allowed_findings = True
    return _EntryProjection(engine)
