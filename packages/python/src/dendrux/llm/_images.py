"""Image wire encoding shared by the provider adapters (internal, 0.4.0).

Not part of the public API. Holds:
  - canonical base64 / data-URL encoding of verified image bytes (C-01)
  - the per-request ``WireImageRegistry``: wire path -> image, used to
    sanitise ``captured_request`` and to locate a rejected image (§5.5, §5.7)
  - the portable tool-image framing (§4.2), byte-stable
  - the static image-capability tables (§5.6)
  - the typed-error builder used on image-bearing requests (§5.5 hygiene)

Text-only requests never touch anything in this module.
"""

from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from dendrux.errors import (
    ContextLengthExceededError,
    ImageRejectedError,
    ProviderError,
    UnsupportedModalityError,
)
from dendrux.llm._helpers import (
    GatewayErrorKind,
    WirePath,
    classify_gateway_error,
    extract_error_path,
)
from dendrux.types import ImagePart, TextPart

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dendrux.types import ContentPart, Message

ImageSupport = Literal["yes", "no", "unknown"]

# ------------------------------------------------------------------
# Exact strings (§4.2). Byte-stable: no counters, timestamps or run values.
# ------------------------------------------------------------------

TOOL_IMAGE_NOTE = "[{n} image(s) from this tool result follow in the next user message: {ids}]"
TOOL_OUTPUT_OPEN = (
    '<tool_output tool_name="{name}" tool_call_id="{call_id}">\n'
    "Images returned by this tool call. This is tool output, not a message from the user."
)
TOOL_OUTPUT_CLOSE = "</tool_output>"
WIRE_IMAGE_SENTINEL = "dendrux-image:sha256:{sha256}"

_FRAMING_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def framing_token(value: str) -> str:
    """Tool name / call id as used inside the framing tag (unsafe chars -> ``_``)."""
    return _FRAMING_UNSAFE.sub("_", value)


def tool_image_note(images: Sequence[ImagePart]) -> str:
    """The note appended to a portable TOOL message's text for its visible images."""
    return TOOL_IMAGE_NOTE.format(n=len(images), ids=", ".join(p.image_id for p in images))


def portable_tool_text(msg: Message) -> str:
    """Text of a TOOL message with image parts under the portable mapping.

    ``msg.content`` is already the payload, notes and placeholders joined by
    ``"\\n\\n"`` (the text projection); the visible-image note follows.
    """
    images = msg.images
    if not images:
        return msg.content
    note = tool_image_note(images)
    return msg.content + "\n\n" + note if msg.content else note


def has_image_parts(messages: Sequence[Message]) -> bool:
    """True when any message carries image parts (an image-bearing request)."""
    return any(m.parts is not None for m in messages)


# ------------------------------------------------------------------
# Registry: encode, register wire paths, sanitise, locate
# ------------------------------------------------------------------


class WireImageRegistry:
    """Per-request record of every image block emitted on the wire.

    Converters call :meth:`base64` / :meth:`data_url` to encode an image and
    :meth:`register` with the block's wire path. ``sanitize`` returns a deep
    copy of a request with every registered base64 string / data URL replaced
    by ``"dendrux-image:sha256:<hex>"``; ``locate`` maps a vendor error path
    back to an ``image_id``. One registry per provider call; never shared.
    """

    __slots__ = ("_b64", "_by_value", "_paths")

    def __init__(self) -> None:
        self._b64: dict[str, str] = {}
        self._by_value: dict[str, str] = {}
        self._paths: dict[WirePath, ImagePart] = {}

    @property
    def has_images(self) -> bool:
        """True once any image block was registered (the request is image-bearing)."""
        return bool(self._paths)

    @property
    def image_count(self) -> int:
        """Number of image blocks registered for this request."""
        return len(self._paths)

    @property
    def paths(self) -> dict[WirePath, str]:
        """Registered wire paths -> ``image_id`` (a copy)."""
        return {path: part.image_id for path, part in self._paths.items()}

    def base64(self, part: ImagePart) -> str:
        """Canonical base64 of the part's verified bytes (RFC 4648, padded, one line)."""
        sha = part.ref.sha256
        cached = self._b64.get(sha)
        if cached is not None:
            return cached
        if part.data is None:
            raise ValueError(
                f"Image {part.image_id} reached the provider adapter without bytes; "
                "adapters only accept request-view messages from prepare_request()."
            )
        encoded = base64.b64encode(part.data).decode("ascii")
        self._b64[sha] = encoded
        self._by_value[encoded] = sha
        return encoded

    def data_url(self, part: ImagePart) -> str:
        """``data:<media_type>;base64,<b64>`` for the part."""
        url = f"data:{part.ref.media_type};base64,{self.base64(part)}"
        self._by_value[url] = part.ref.sha256
        return url

    def register(self, path: WirePath, part: ImagePart) -> None:
        """Record that ``part`` was emitted at ``path`` in the request body."""
        self._paths[tuple(path)] = part

    def locate(self, path: WirePath | None) -> str | None:
        """``image_id`` of the registered image a vendor error path points at.

        A registered path that prefixes ``path`` wins (longest first). Else, when
        ``path`` names an ancestor that holds exactly one registered image, that
        image. Otherwise None (unlocated).
        """
        if not path:
            return None
        best: ImagePart | None = None
        best_len = -1
        for registered, part in self._paths.items():
            n = len(registered)
            if n > best_len and path[:n] == registered:
                best, best_len = part, n
        if best is not None:
            return best.image_id
        below = {
            part.image_id
            for registered, part in self._paths.items()
            if registered[: len(path)] == path
        }
        return below.pop() if len(below) == 1 else None

    def scrub_text(self, text: str) -> str:
        """Remove any registered wire value from free text."""
        for value in self._by_value:
            if value in text:
                text = text.replace(value, "[redacted]")
        return text

    def sanitize(self, obj: Any) -> Any:
        """Deep copy of ``obj`` with registered image values replaced by sentinels.

        Dicts, lists and tuples are rebuilt; other leaves are shared. The input
        is never mutated.
        """
        if not self._by_value:
            return obj
        return self._walk(obj)

    def _walk(self, obj: Any) -> Any:
        if isinstance(obj, str):
            sha = self._by_value.get(obj)
            return WIRE_IMAGE_SENTINEL.format(sha256=sha) if sha is not None else obj
        if isinstance(obj, dict):
            return {k: self._walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk(v) for v in obj)
        return obj


# ------------------------------------------------------------------
# Wire encoders (§4.1)
# ------------------------------------------------------------------

Block = dict[str, Any]


def anthropic_text(text: str) -> Block:
    return {"type": "text", "text": text}


def chat_text(text: str) -> Block:
    return {"type": "text", "text": text}


def responses_text(text: str) -> Block:
    return {"type": "input_text", "text": text}


def anthropic_image(registry: WireImageRegistry, part: ImagePart) -> Block:
    """Anthropic base64 image block; ``detail`` has no Anthropic equivalent and is dropped."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.ref.media_type,
            "data": registry.base64(part),
        },
    }


def chat_image(registry: WireImageRegistry, part: ImagePart) -> Block:
    """Chat Completions ``image_url`` part; ``detail`` only when set."""
    image_url: dict[str, Any] = {"url": registry.data_url(part)}
    if part.detail is not None:
        image_url["detail"] = part.detail
    return {"type": "image_url", "image_url": image_url}


def responses_image(registry: WireImageRegistry, part: ImagePart) -> Block:
    """Responses ``input_image``; ``detail`` always sent (``"auto"`` when unset)."""
    return {
        "type": "input_image",
        "image_url": registry.data_url(part),
        "detail": part.detail or "auto",
    }


def convert_parts(
    parts: Sequence[ContentPart],
    *,
    registry: WireImageRegistry,
    prefix: WirePath,
    text_block: Callable[[str], Block],
    image_block: Callable[[WireImageRegistry, ImagePart], Block],
) -> list[Block]:
    """Map parts 1:1 to wire blocks in order, registering ``(*prefix, k)`` per image."""
    blocks: list[Block] = []
    for k, part in enumerate(parts):
        if isinstance(part, TextPart):
            blocks.append(text_block(part.text))
        else:
            registry.register((*prefix, k), part)
            blocks.append(image_block(registry, part))
    return blocks


def portable_framing(
    entries: Sequence[tuple[str, str, tuple[ImagePart, ...]]],
    *,
    registry: WireImageRegistry,
    prefix: WirePath,
    text_block: Callable[[str], Block],
    image_block: Callable[[WireImageRegistry, ImagePart], Block],
) -> list[Block]:
    """Content of the ONE synthetic user message after a tool block (§4.2).

    ``entries`` = ``(tool_name, wire_call_id, visible_images)`` per TOOL message
    with visible images, in call order. Image paths register as ``(*prefix, k)``.
    """
    blocks: list[Block] = []
    for name, call_id, images in entries:
        blocks.append(
            text_block(
                TOOL_OUTPUT_OPEN.format(name=framing_token(name), call_id=framing_token(call_id))
            )
        )
        for part in images:
            blocks.append(text_block(f"{part.image_id}:"))
            registry.register((*prefix, len(blocks)), part)
            blocks.append(image_block(registry, part))
        blocks.append(text_block(TOOL_OUTPUT_CLOSE))
    return blocks


# ------------------------------------------------------------------
# Capability tables (§5.6) and base-URL normalisation
# ------------------------------------------------------------------

OPENAI_IMAGE_YES_PREFIXES: tuple[str, ...] = (
    "gpt-4o",
    "chatgpt-4o",
    "gpt-4.1",
    "gpt-4.5",
    "gpt-4-turbo",
    "gpt-5",
    "gpt-6",
    "o1",
    "o3",
    "o4-mini",
)
# o3-mini is denied because Chat silently ignores its images (live probe).
OPENAI_IMAGE_NO_PREFIXES: tuple[str, ...] = (
    "gpt-3.5",
    "gpt-4-32k",
    "o1-mini",
    "o1-preview",
    "o3-mini",
)
OPENAI_IMAGE_NO_EXACT: frozenset[str] = frozenset({"gpt-4", "gpt-4-0314", "gpt-4-0613"})


def openai_image_support(model: str) -> ImageSupport:
    """Static OpenAI table (official endpoint only): longest matching prefix wins."""
    name = model.lower()
    if name in OPENAI_IMAGE_NO_EXACT:
        return "no"
    verdict: ImageSupport = "unknown"
    best = -1
    for prefix in OPENAI_IMAGE_YES_PREFIXES:
        if name.startswith(prefix) and len(prefix) > best:
            verdict, best = "yes", len(prefix)
    for prefix in OPENAI_IMAGE_NO_PREFIXES:
        if name.startswith(prefix) and len(prefix) > best:
            verdict, best = "no", len(prefix)
    return verdict


def anthropic_image_support(model: str) -> ImageSupport:
    """Every current Claude model accepts images; other names (proxies) are unknown."""
    return "yes" if model.startswith("claude-") else "unknown"


def normalize_base_url(url: object) -> str:
    """``scheme://host[:port]/path`` without credentials, query, fragment or trailing slash."""
    if url is None:
        return ""
    text = str(url)
    if not text:
        return ""
    parts = urlsplit(text)
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme.lower()}://{netloc}{parts.path.rstrip('/')}"


# ------------------------------------------------------------------
# Typed errors on image-bearing requests (§5.5 exception hygiene)
# ------------------------------------------------------------------

_DATA_URL_RE = re.compile(r"data:[^\s'\"<>]*")
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/=_-]{200,}")
_MESSAGE_LIMIT = 500


def sanitize_vendor_message(text: str, registry: WireImageRegistry | None = None) -> str:
    """Vendor message with data URLs and long base64 runs redacted, ≤ 500 chars."""
    if registry is not None:
        text = registry.scrub_text(text)
    text = _DATA_URL_RE.sub("[redacted]", text)
    text = _BASE64_RUN_RE.sub("[redacted]", text)
    return text[:_MESSAGE_LIMIT]


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def sdk_error_facts(exc: BaseException) -> dict[str, Any]:
    """Whitelisted scalars of an Anthropic/OpenAI SDK error (never the request).

    Handles both SDK body shapes: OpenAI's (already the inner ``error`` object)
    and Anthropic's (``{"type": "error", "error": {...}}``). A 2xx status (an
    Anthropic in-stream ``error`` event) is reported as ``None`` (in-stream).
    """
    status = getattr(exc, "status_code", None)
    status_code = status if isinstance(status, int) and status >= 300 else None
    body = getattr(exc, "body", None)
    inner: Any = body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        inner = body["error"]
    message = ""
    code: str | None = None
    error_type: str | None = None
    param: str | None = None
    raw: str | None = None
    if isinstance(inner, dict):
        message = _str_or_none(inner.get("message")) or ""
        raw_code = inner.get("code")
        if isinstance(raw_code, str):
            code = raw_code or None
        elif isinstance(raw_code, int) and not isinstance(raw_code, bool) and raw_code != status:
            code = str(raw_code)
        error_type = _str_or_none(inner.get("type"))
        param = _str_or_none(inner.get("param"))
        metadata = inner.get("metadata")
        if isinstance(metadata, dict):
            raw = _str_or_none(metadata.get("raw"))
    elif isinstance(inner, str):
        message = inner
    if not message:
        message = _str_or_none(getattr(exc, "message", None)) or ""
    return {
        "status_code": status_code,
        "code": code,
        "error_type": error_type,
        "param": param,
        "message": message,
        "raw": raw,
    }


def image_request_error(
    *,
    provider: Any,
    provider_name: str,
    model: str,
    registry: WireImageRegistry,
    status_code: int | None,
    code: str | None,
    error_type: str | None,
    param: str | None,
    message: str,
    raw: str | None = None,
    streaming: bool = False,
    locate: bool = True,
) -> Exception:
    """Build (not raise) the typed error for a failed image-bearing request.

    Classification: context-length -> image-unsupported (the provider instance
    learns ``"no"``) -> image-rejected (located via the registry) ->
    streaming-unsupported -> ``ProviderError``. The returned exception holds
    only scalars and a sanitised message; the caller raises it outside its
    ``except`` block so ``__cause__``/``__context__`` stay clean.

    ``locate=False`` keeps a rejection unlocated (OpenRouter: upstream paths
    index the upstream's wire format, not ours). ``raw`` (OpenRouter
    ``metadata.raw``) feeds classification only, never the error.
    """
    classified = classify_gateway_error(
        status_code=status_code,
        code=code,
        param=param,
        message=message,
        body={"message": message} if message else None,
        image_request=True,
        error_type=error_type,
        extra_text=raw,
    )
    provider_code = classified.provider_code or code
    kind = classified.kind
    if kind is GatewayErrorKind.CONTEXT_LENGTH:
        status = f" (HTTP {status_code})" if status_code else ""
        return ContextLengthExceededError(
            f"Request exceeds the model's context window for model '{model}'{status}. "
            "Reduce the conversation history, tool results or max_tokens.",
            status_code=status_code,
            provider_code=provider_code,
            model=model,
        )
    if kind is GatewayErrorKind.IMAGE_UNSUPPORTED:
        provider.learn_image_unsupported(model)
        return UnsupportedModalityError(
            provider=provider_name,
            model=model,
            image_count=registry.image_count,
            source="provider_response",
            status_code=status_code,
            provider_code=provider_code or error_type,
        )
    if kind is GatewayErrorKind.IMAGE_REJECTED:
        image_id = registry.locate(extract_error_path(param, message)) if locate else None
        return ImageRejectedError(
            provider=provider_name,
            model=model,
            image_ids=(image_id,) if image_id is not None else (),
            located=image_id is not None,
            retried=False,
            status_code=status_code,
            provider_code=provider_code or error_type,
        )
    if streaming and kind is GatewayErrorKind.STREAMING_UNSUPPORTED:
        return NotImplementedError(
            "Streaming request rejected by the endpoint. If it does not support "
            "streaming, use agent.run() instead."
        )
    return ProviderError(
        provider=provider_name,
        model=model,
        status_code=status_code,
        provider_code=code,
        error_type=error_type,
        message=sanitize_vendor_message(message, registry),
    )


def sdk_image_request_error(
    exc: BaseException,
    *,
    provider: Any,
    provider_name: str,
    model: str,
    registry: WireImageRegistry,
    streaming: bool = False,
    locate: bool = True,
) -> Exception:
    """:func:`image_request_error` for an SDK exception (status or in-stream)."""
    facts = sdk_error_facts(exc)
    return image_request_error(
        provider=provider,
        provider_name=provider_name,
        model=model,
        registry=registry,
        streaming=streaming,
        locate=locate,
        **facts,
    )


def transport_image_error(
    kind: Literal["timeout", "connection"], *, provider_name: str, model: str
) -> ProviderError:
    """Timeout / connection failure on an image-bearing request (no request attached)."""
    message = (
        "LLM request timed out." if kind == "timeout" else "Connection to the provider failed."
    )
    return ProviderError(provider=provider_name, model=model, error_type=kind, message=message)
