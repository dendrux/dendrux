"""Image configuration: :class:`ImagePolicy`, drop and URL policies, hook request types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from dendrux.types import ImageRef

SUPPORTED_MEDIA_TYPES: tuple[str, ...] = ("image/png", "image/jpeg", "image/gif", "image/webp")
Origin: TypeAlias = Literal["user", "tool", "client_tool", "mcp"]


@dataclass(frozen=True)
class ImageIngestRequest:
    """What an ``ImagePolicy.ingest_hook`` receives for one image.

    Return new bytes (at most ``max_output_bytes``) or ``None`` to keep ``data``.
    """

    data: bytes = field(repr=False)
    declared_media_type: str | None
    origin: Origin
    tool_name: str | None
    max_output_bytes: int


@dataclass(frozen=True)
class ImageDescribeRequest:
    """What a describe callable (``ImagePolicy.on_unsupported``) receives."""

    image_id: str
    ref: ImageRef
    origin: Literal["user", "tool"]
    tool_name: str | None
    data: bytes = field(repr=False)


DescribeImage: TypeAlias = Callable[[ImageDescribeRequest], "str | Awaitable[str]"]
IngestHook: TypeAlias = Callable[[ImageIngestRequest], "bytes | None | Awaitable[bytes | None]"]
UrlFetcher: TypeAlias = Callable[[str, int], Awaitable[bytes]]


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


@dataclass(frozen=True)
class ImageDropPolicy:
    """Opt-in "drop old images" request view.

    The model sees between ``keep_recent`` and ``keep_recent + interval - 1``
    eligible images; older ones are replaced by a placeholder, sticky per run.
    """

    keep_recent: int = 3
    interval: int = 25
    scope: Literal["tool_results", "all"] = "tool_results"

    def __post_init__(self) -> None:
        _positive_int("ImageDropPolicy.keep_recent", self.keep_recent)
        _positive_int("ImageDropPolicy.interval", self.interval)
        if self.scope not in ("tool_results", "all"):
            raise ValueError(
                f"ImageDropPolicy.scope must be 'tool_results' or 'all', got {self.scope!r}"
            )


@dataclass(frozen=True)
class UrlPolicy:
    """How ``Image.from_url`` sources are fetched (once, at ingest)."""

    fetch: bool = True
    allowed_schemes: tuple[str, ...] = ("https",)
    allowed_ports: tuple[int, ...] = (443,)
    allowed_hosts: tuple[str, ...] | None = None
    allow_private_networks: bool = False
    max_redirects: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 10.0
    total_timeout: float = 20.0
    fetcher: UrlFetcher | None = None

    def __post_init__(self) -> None:
        for name in ("allowed_schemes", "allowed_ports"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"UrlPolicy.{name} must be a sequence")
            object.__setattr__(self, name, tuple(value))
        if self.allowed_hosts is not None:
            if isinstance(self.allowed_hosts, (str, bytes)):
                raise ValueError("UrlPolicy.allowed_hosts must be a sequence of hosts")
            hosts = tuple(h.strip().lower().rstrip(".") for h in self.allowed_hosts)
            if any(not h or h == "*." for h in hosts):
                raise ValueError("UrlPolicy.allowed_hosts entries must be non-empty")
            object.__setattr__(self, "allowed_hosts", hosts)
        schemes = tuple(s.lower() for s in self.allowed_schemes)
        if not schemes or any(s not in ("http", "https") for s in schemes):
            raise ValueError("UrlPolicy.allowed_schemes must be a non-empty subset of http/https")
        object.__setattr__(self, "allowed_schemes", schemes)
        if not self.allowed_ports:
            raise ValueError("UrlPolicy.allowed_ports must not be empty")
        for port in self.allowed_ports:
            if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
                raise ValueError(f"UrlPolicy.allowed_ports has an invalid port {port!r}")
        if "http" in schemes and 80 not in self.allowed_ports:
            raise ValueError("UrlPolicy: allowing 'http' requires port 80 in allowed_ports")
        if isinstance(self.max_redirects, bool) or not isinstance(self.max_redirects, int):
            raise ValueError("UrlPolicy.max_redirects must be an int")
        if self.max_redirects < 0:
            raise ValueError("UrlPolicy.max_redirects must be >= 0")
        _positive_number("UrlPolicy.connect_timeout", self.connect_timeout)
        _positive_number("UrlPolicy.read_timeout", self.read_timeout)
        _positive_number("UrlPolicy.total_timeout", self.total_timeout)
        if self.fetcher is not None and not callable(self.fetcher):
            raise ValueError("UrlPolicy.fetcher must be callable")


@dataclass(frozen=True)
class ImagePolicy:
    """Caps, capability handling, drop view, URL fetching and ingest hook for images.

    Defaults are the vendor limits. ``ingest_hook`` is the only byte transform
    Dendrux runs; it runs once at ingest and requires ``ingest_hook_version``.
    """

    max_image_bytes: int = 5 * 1024 * 1024
    max_source_bytes: int = 25 * 1024 * 1024
    max_long_side: int = 8000
    max_pixels: int = 8000 * 8000
    max_images_per_message: int = 20
    max_images_per_request: int = 100
    max_request_image_bytes: int = 20 * 1024 * 1024
    max_estimated_image_tokens_per_request: int | None = None
    allowed_media_types: tuple[str, ...] = SUPPORTED_MEDIA_TYPES
    allow_animated: bool = False
    on_unsupported: Literal["error", "placeholder"] | DescribeImage = "error"
    unknown_capability: Literal["send", "error"] = "send"
    capability_overrides: Mapping[str, Literal["yes", "no"]] = field(default_factory=dict)
    drop: ImageDropPolicy | None = None
    urls: UrlPolicy = field(default_factory=UrlPolicy)
    ingest_hook: IngestHook | None = None
    ingest_hook_version: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_image_bytes",
            "max_source_bytes",
            "max_long_side",
            "max_pixels",
            "max_images_per_message",
            "max_images_per_request",
            "max_request_image_bytes",
        ):
            _positive_int(f"ImagePolicy.{name}", getattr(self, name))
        if self.max_estimated_image_tokens_per_request is not None:
            _positive_int(
                "ImagePolicy.max_estimated_image_tokens_per_request",
                self.max_estimated_image_tokens_per_request,
            )
        if isinstance(self.allowed_media_types, (str, bytes)):
            raise ValueError("ImagePolicy.allowed_media_types must be a sequence")
        types_ = tuple(t.lower() for t in self.allowed_media_types)
        if not types_ or any(t not in SUPPORTED_MEDIA_TYPES for t in types_):
            raise ValueError(
                "ImagePolicy.allowed_media_types must be a non-empty subset of "
                f"{SUPPORTED_MEDIA_TYPES}"
            )
        object.__setattr__(self, "allowed_media_types", types_)
        if not isinstance(self.allow_animated, bool):
            raise ValueError("ImagePolicy.allow_animated must be a bool")
        if self.on_unsupported not in ("error", "placeholder") and not callable(
            self.on_unsupported
        ):
            raise ValueError(
                "ImagePolicy.on_unsupported must be 'error', 'placeholder' or a describe callable"
            )
        if self.unknown_capability not in ("send", "error"):
            raise ValueError("ImagePolicy.unknown_capability must be 'send' or 'error'")
        for model, value in dict(self.capability_overrides).items():
            if not isinstance(model, str) or value not in ("yes", "no"):
                raise ValueError("ImagePolicy.capability_overrides maps model ids to 'yes' or 'no'")
        if self.drop is not None:
            if not isinstance(self.drop, ImageDropPolicy):
                raise ValueError("ImagePolicy.drop must be an ImageDropPolicy or None")
            visible_max = self.drop.keep_recent + self.drop.interval - 1
            if visible_max > self.max_images_per_request:
                raise ValueError(
                    "ImagePolicy.drop: keep_recent + interval - 1 must be <= "
                    f"max_images_per_request ({visible_max} > {self.max_images_per_request})"
                )
        if not isinstance(self.urls, UrlPolicy):
            raise ValueError("ImagePolicy.urls must be a UrlPolicy")
        if self.ingest_hook is not None:
            if not callable(self.ingest_hook):
                raise ValueError("ImagePolicy.ingest_hook must be callable")
            version = self.ingest_hook_version
            if (
                not isinstance(version, str)
                or not version.strip()
                or version != version.strip()
                or len(version) > 100
            ):
                raise ValueError(
                    "ImagePolicy.ingest_hook requires ingest_hook_version: a non-empty "
                    "string of at most 100 characters without surrounding whitespace"
                )
