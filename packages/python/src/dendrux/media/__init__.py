"""Image input for Dendrux: inputs, policies, blob storage and the ingest pipeline.

Public names: :class:`Image`, :class:`ToolOutput`, :class:`ImagePolicy`,
:class:`ImageDropPolicy`, :class:`UrlPolicy`, :class:`BlobStore`,
:class:`InMemoryBlobStore`, :class:`BlobKey` and the records returned by
``agent.blobs``. ``dendrux.media`` imports ``dendrux.types``, never the reverse.
"""

from dendrux.media._catalog import BlobCatalog, BlobEdge, BlobResolver
from dendrux.media._ids import IMAGE_ID_RE, PLACEHOLDERS, image_id, placeholder
from dendrux.media._image import Detail, Image, ImageInput, ImageSource, ToolOutput, UserContent
from dendrux.media._ingest import ingest
from dendrux.media._policy import (
    SUPPORTED_MEDIA_TYPES,
    DescribeImage,
    ImageDescribeRequest,
    ImageDropPolicy,
    ImageIngestRequest,
    ImagePolicy,
    IngestHook,
    Origin,
    UrlFetcher,
    UrlPolicy,
)
from dendrux.media._prepared import (
    GovernanceEvent,
    ImageStats,
    ImageViewDelta,
    PreparedRequest,
    SlotInfo,
)
from dendrux.media._runtime import ImageRuntime, tenant_scope_for
from dendrux.media._sniff import ImageHeader, inspect_image, sniff_media_type, verify_blob_bytes
from dendrux.media._store import (
    BlobAccessor,
    BlobInfo,
    BlobKey,
    BlobReference,
    BlobState,
    BlobStorage,
    BlobStore,
    GcResult,
    InMemoryBlobStore,
)
from dendrux.media._tokens import estimate_image_tokens
from dendrux.types import ImagePart, ImageRef, TextPart

__all__ = [
    "IMAGE_ID_RE",
    "PLACEHOLDERS",
    "SUPPORTED_MEDIA_TYPES",
    "BlobAccessor",
    "BlobCatalog",
    "BlobEdge",
    "BlobInfo",
    "BlobKey",
    "BlobReference",
    "BlobResolver",
    "BlobState",
    "BlobStorage",
    "BlobStore",
    "DescribeImage",
    "Detail",
    "GcResult",
    "GovernanceEvent",
    "Image",
    "ImageDescribeRequest",
    "ImageDropPolicy",
    "ImageHeader",
    "ImageIngestRequest",
    "ImageInput",
    "ImagePart",
    "ImagePolicy",
    "ImageRef",
    "ImageRuntime",
    "ImageSource",
    "ImageStats",
    "ImageViewDelta",
    "InMemoryBlobStore",
    "IngestHook",
    "Origin",
    "PreparedRequest",
    "SlotInfo",
    "TextPart",
    "ToolOutput",
    "UrlFetcher",
    "UrlPolicy",
    "UserContent",
    "estimate_image_tokens",
    "image_id",
    "ingest",
    "inspect_image",
    "placeholder",
    "sniff_media_type",
    "tenant_scope_for",
    "verify_blob_bytes",
]
