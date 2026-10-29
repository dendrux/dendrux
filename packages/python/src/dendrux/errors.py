"""Public exception hierarchy for developers writing their own HTTP routes.

Methods like :meth:`Agent.submit_tool_results`, :meth:`Agent.submit_input`,
:meth:`Agent.submit_approval`, and :meth:`Agent.cancel_run` raise these
exceptions on precondition failures. Developers catch them and map to
HTTP status codes (or CLI prompts, or log lines) in their own code —
Dendrux never forces a status-code contract.

Suggested HTTP mappings:

    RunNotFoundError            -> 404
    RunNotPausedError           -> 409
    PauseStatusMismatchError    -> 409
    RunAlreadyClaimedError      -> 409
    RunAlreadyTerminalError     -> 409
    InvalidToolResultError      -> 400
    ReplayRunError              -> 400 (404 when reason == "not_found")
    PersistenceNotConfiguredError -> 500
    SchemaNotInitializedError   -> 500
    SkillMissingError           -> 409
    ContextLengthExceededError  -> 413 or 400
    SkillMismatchError          -> 409
    SqliteSchemaSyncError       -> 500 (startup: legacy SQLite file needs migration)
    SchemaOutOfDateError        -> 500 (startup: run `dendrux db migrate`)
    ImageError                  -> 400 (base of the next three)
    InvalidImageError           -> 400
    ImageLimitError             -> 413
    ImageFetchError             -> 400 (502 when reason is "http_status" or "timeout")
    UnsupportedModalityError    -> 400
    ImageRejectedError          -> 422
    ProviderError               -> 502 (base of the provider errors below)
    ProviderAuthError           -> 502
    ProviderRateLimitError      -> 503 (``retry_after``)
    ProviderTimeoutError        -> 504
    ProviderUnavailableError    -> 503
    ProviderStreamIncompleteError -> 502
    ModelRefusalError           -> 422
    ContentFilteredError        -> 422
    BudgetConfigurationError    -> 500 (configuration)
    TenantScopeRequiredError    -> 500 (configuration: pass tenant_id)
    NoteLimitExceededError      -> 429 or 409 (too many notes on this run)
    GovernanceMismatchError     -> 409
    BlobNotFoundError           -> 404
    BlobIntegrityError          -> 500
    ImagePersistenceUnsupportedError -> 500
    BlobContentionError         -> 503 (transient; safe to retry)

Image errors never carry image bytes, base64, data URLs, file paths or URL
paths/queries/credentials, and are raised with no ``__cause__`` and no
``__context__``.

``SkillValidationError`` is raised while building a :class:`~dendrux.skills.Skill`
(a configuration error, not a request error).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, get_args

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from dendrux.types import RunStatus


def _rebuild_error(cls: type[BaseException], kwargs: dict[str, object]) -> BaseException:
    return cls(**kwargs)


def _rebuild_context_length_error(
    cls: type[BaseException], message: str, kwargs: dict[str, object]
) -> BaseException:
    return cls(message, **kwargs)


class _KwargsReduceMixin:
    """Pickle/copy support for errors whose ``__init__`` takes named fields.

    ``_init_fields`` lists the constructor parameters, each stored as an
    attribute of the same name; the rebuilt error has no traceback or chain.
    """

    _init_fields: tuple[str, ...] = ()

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            _rebuild_error,
            (type(self), {name: getattr(self, name) for name in self._init_fields}),
        )


class RunNotFoundError(_KwargsReduceMixin, LookupError):
    """Raised when ``run_id`` does not exist in the store."""

    _init_fields = ("run_id",)

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run '{run_id}' not found.")


class StoreCapabilityError(TypeError):
    """A declared store capability does not match the implementation."""

    def __init__(self, store: str, capability: str, method: str, missing: tuple[str, ...]) -> None:
        self.store = store
        self.capability = capability
        self.method = method
        self.missing = missing
        super().__init__(
            f"StateStore {store} capability '{capability}': {method} lacks {missing!r}."
        )


class RunControlError(RuntimeError):
    """Base for conflicts while controlling a persisted run."""


def _restore_logged_control_error(cls: Any, args: tuple[Any, ...], state: dict[str, Any]) -> Any:
    """Deserialize an existing failure without firing its construction log again."""
    error = cls.__new__(cls)
    RuntimeError.__init__(error, *args)
    error.__dict__.update(state)
    return error


class RunNotReclaimableError(_KwargsReduceMixin, RunControlError):
    """A run does not satisfy the ordered recovery preconditions.

    ``reason`` is one of ``legacy_run``, ``store_capability``, ``not_running``,
    ``lease_fresh``, ``already_reclaimed``, ``reclaim_limit``, ``agent_mismatch``,
    ``skill_mismatch``, ``loop_unsupported``, ``history_too_large``, ``cancelled``
    (cancellation won), ``loop_unknown`` (missing or unrecognized checkpoint loop),
    or ``governance_changed`` (the recovery governance fingerprint differs).
    """

    _init_fields = ("run_id", "reason")

    def __reduce__(self) -> Any:
        return _restore_logged_control_error, (type(self), self.args, self.__dict__)

    def __init__(self, run_id: str, reason: str) -> None:
        self.run_id = run_id
        self.reason = reason
        logging.getLogger(__name__).warning(
            "Run cannot be reclaimed",
            extra={"run_id": run_id, "code": "run_not_reclaimable", "reason": reason},
        )
        super().__init__(f"Run '{run_id}' cannot be reclaimed ({reason}).")


class RunCancelledError(_KwargsReduceMixin, RunControlError):
    """A delegated run was refused by a cancelled or terminal parent."""

    _init_fields = ("run_id", "parent_run_id")

    def __reduce__(self) -> Any:
        return _restore_logged_control_error, (type(self), self.args, self.__dict__)

    def __init__(self, run_id: str, parent_run_id: str) -> None:
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        logging.getLogger(__name__).warning(
            "Child admission refused", extra={"run_id": run_id, "code": "run_cancelled"}
        )
        super().__init__(f"Run '{run_id}' cannot start under parent '{parent_run_id}'.")


class NotifierUnavailableError(RuntimeError):
    """The configured notifier could not become ready."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        logging.getLogger(__name__).warning(
            "Notifier unavailable", extra={"run_id": None, "code": "notifier_unavailable"}
        )
        super().__init__(f"Notifier unavailable ({reason}).")


class RunNotPausedError(RunControlError):
    """Raised when a submit method is called on a run that is not paused.

    The run is either still executing, already terminal, or in a status
    that cannot accept a resume signal. Use :meth:`Agent.cancel_run` to
    stop an active run; terminal runs cannot be resumed.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(f"Run '{run_id}' is not paused (current status: {current_status.value}).")


class PauseStatusMismatchError(_KwargsReduceMixin, RunControlError):
    """Raised when a submit targets the wrong pause.

    ``reason="status"``: the run is paused, but not for the kind of signal
    this method delivers (for example :meth:`Agent.submit_tool_results` on a
    ``WAITING_HUMAN_INPUT`` run). ``reason="pause_id"``: the run is paused for
    the right kind of signal, but not the pause the caller expected
    (``expected_pause_id`` differs from ``current_pause_id``; a stale submit).
    """

    _init_fields = (
        "run_id",
        "current_status",
        "expected_status",
        "expected_pause_id",
        "current_pause_id",
        "reason",
    )

    def __init__(
        self,
        run_id: str,
        current_status: RunStatus,
        expected_status: RunStatus,
        *,
        expected_pause_id: str | None = None,
        current_pause_id: str | None = None,
        reason: Literal["status", "pause_id"] = "status",
    ) -> None:
        if reason not in ("status", "pause_id"):
            raise ValueError(f"Unknown PauseStatusMismatchError reason {reason!r}.")
        self.run_id = run_id
        self.current_status = current_status
        self.expected_status = expected_status
        self.expected_pause_id = expected_pause_id
        self.current_pause_id = current_pause_id
        self.reason: Literal["status", "pause_id"] = reason
        if reason == "pause_id":
            super().__init__(
                f"Run '{run_id}' is paused at a different generation; the submit is stale."
            )
        else:
            super().__init__(
                f"Run '{run_id}' is {current_status.value}, "
                f"but this method requires {expected_status.value}."
            )


class RunAlreadyClaimedError(RunControlError):
    """Raised when a concurrent submit has already claimed this pause.

    Two submits raced; one won the CAS transition from paused to
    running. The loser receives this error. The run continues under
    the winning caller — retries should poll for completion, not
    re-submit.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Run '{run_id}' was claimed by another submit. Poll for completion instead."
        )


class RunAlreadyTerminalError(RunControlError):
    """Raised by submit methods when the run has already reached a terminal state.

    :meth:`Agent.cancel_run` does **not** raise this — cancellation on
    a terminal run is a silent no-op.
    """

    def __init__(self, run_id: str, current_status: RunStatus) -> None:
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run '{run_id}' is already terminal ({current_status.value}); cannot resume."
        )


class InvalidToolResultError(ValueError):
    """Raised when submitted tool results do not match the pending set.

    Every result's ``tool_call_id`` must correspond to a pending call
    on the paused run; unknown or duplicate ids cause this error and no
    partial claim is made.
    """

    def __init__(self, run_id: str, detail: str) -> None:
        self.run_id = run_id
        self.detail = detail
        super().__init__(f"Invalid tool results for run '{run_id}': {detail}")


class PersistenceError(RuntimeError):
    """A database operation failed; SQL, parameters and driver text are omitted."""

    code = "persistence_error"
    safe_message = "Database operation failed (persistence_error)."

    def __init__(self, *, retryable: bool = False, code: str = "persistence_error") -> None:
        self.retryable = retryable
        self.code = code
        self.safe_message = f"Database operation failed ({code})."
        super().__init__(self.safe_message)


class PersistenceNotConfiguredError(RuntimeError):
    """Raised when an Agent method requires persistence but none is configured.

    Resume-family methods need a ``database_url`` or ``state_store`` on
    the agent. This is a configuration bug, not a runtime condition —
    it will raise on every call until the agent is reconfigured.
    """

    def __init__(self) -> None:
        super().__init__(
            "This operation requires persistence. Pass database_url or state_store "
            "to the Agent constructor."
        )


class SchemaNotInitializedError(RuntimeError):
    """Raised when the configured database has no Dendrux tables yet.

    SQLite auto-creates its schema on first use, so this never fires
    there. Postgres (and other backends) require an explicit migration
    step before the first run. The message points at the CLI commands
    that set the schema up.
    """

    def __init__(self) -> None:
        super().__init__(
            "Dendrux tables not found in the configured database. "
            "Run migrations before the first run:\n\n"
            "    export DENDRUX_DATABASE_URL=<your-database-url>\n"
            "    dendrux db migrate\n\n"
            "SQLite auto-creates its schema; Postgres and other backends "
            "require this step. Check status with 'dendrux db status'."
        )


class ReplayRunError(ValueError):
    """Raised when a run passed via ``agent.run(runs=...)`` cannot be replayed.

    ``reason`` is one of ``"not_found"``, ``"tenant_mismatch"``,
    ``"not_terminal"``, ``"duplicate"``. Raised before any run is created
    or any model is called.
    """

    def __init__(self, run_id: str, reason: str, detail: str) -> None:
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"Cannot replay run '{run_id}': {detail}")


class SkillValidationError(ValueError):
    """Raised when a skill's content or metadata is invalid.

    ``field`` names the offending field (``"name"``, ``"files"``,
    ``"frontmatter"``, ...) when one applies.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class SkillDigestMismatch:
    """One skill whose content changed between run creation and continuation."""

    name: str
    expected_digest: str
    actual_digest: str


class SkillContinuityError(RuntimeError):
    """Raised when a paused run's skills are not what the run was created with.

    Base class of :class:`SkillMissingError` and :class:`SkillMismatchError`.
    The run is left in its paused status. ``missing`` and ``mismatched``
    are sorted by skill name.
    """

    def __init__(
        self,
        run_id: str,
        *,
        missing: Iterable[str] = (),
        mismatched: Iterable[SkillDigestMismatch] = (),
    ) -> None:
        self.run_id = run_id
        self.missing: tuple[str, ...] = tuple(sorted(missing))
        self.mismatched: tuple[SkillDigestMismatch, ...] = tuple(
            sorted(mismatched, key=lambda m: m.name)
        )
        parts: list[str] = []
        if self.missing:
            parts.append("missing skills: " + ", ".join(self.missing))
        if self.mismatched:
            parts.append(
                "changed skills: "
                + ", ".join(
                    f"{m.name} (expected {m.expected_digest}, got {m.actual_digest})"
                    for m in self.mismatched
                )
            )
        detail = "; ".join(parts) if parts else "skills do not match"
        super().__init__(
            f"Run '{run_id}' cannot continue: {detail}. "
            "Attach the same skill content the run was created with."
        )


class SkillMissingError(SkillContinuityError):
    """Raised when a skill the run was created with is not attached to the agent.

    Any digest mismatches found in the same check are reported in ``mismatched``.
    """


class SkillMismatchError(SkillContinuityError):
    """Raised when every skill is attached but some have different content."""


class SqliteSchemaSyncError(RuntimeError):
    """A SQLite database cannot be upgraded in place to the current schema.

    Raised at startup when an older Dendrux SQLite file is missing a column
    that SQLite cannot add (NOT NULL without a constant default), when an
    index the schema needs collides with an unrelated object, or when an
    ``ALTER`` fails. The message names the table, column or index. Run
    ``dendrux db migrate`` against the database, or recreate the file.
    """


class SchemaOutOfDateError(RuntimeError):
    """A non-SQLite database is missing tables, columns or indexes this Dendrux version needs.

    Raised once per engine, before the first query, when the database was
    migrated by an older Dendrux and ``dendrux db migrate`` has not been run
    since upgrading. The check is read-only: it compares the live tables and
    columns with the models, ignores anything extra in the database, and does
    not depend on Alembic's version table. Run ``dendrux db migrate`` with
    ``DENDRUX_DATABASE_URL`` pointing at the database, then restart. If the
    schema is managed some other way, set ``DENDRUX_SKIP_SCHEMA_CHECK=1`` to
    skip the check.

    Attributes:
        missing_tables: Names of model tables absent from the database.
        missing_columns: ``"table.column"`` entries for columns absent from
            tables that do exist.
        missing_indexes: Missing or invalid ``"table.index (index)"`` entries.
    """

    def __init__(
        self,
        missing_tables: Iterable[str] = (),
        missing_columns: Iterable[str] = (),
        missing_indexes: Iterable[str] = (),
    ) -> None:
        self.missing_tables: tuple[str, ...] = tuple(missing_tables)
        self.missing_columns: tuple[str, ...] = tuple(missing_columns)
        self.missing_indexes: tuple[str, ...] = tuple(missing_indexes)
        missing = [
            *self.missing_columns,
            *self.missing_indexes,
            *(f"table {t}" for t in self.missing_tables),
        ]
        super().__init__(
            f"The database schema is behind this version of dendrux (missing: "
            f"{', '.join(missing)}). Run `dendrux db migrate` with DENDRUX_DATABASE_URL "
            f"pointing at this database, then restart. If you manage the schema yourself "
            f"and want to skip this check, set DENDRUX_SKIP_SCHEMA_CHECK=1."
        )

    @property
    def missing(self) -> tuple[str, ...]:
        """Every missing object: columns, tables by name, then indexes."""
        return (*self.missing_columns, *self.missing_tables, *self.missing_indexes)


# ---------------------------------------------------------------------------
# Images (0.4.0)
# ---------------------------------------------------------------------------


class ImageError(ValueError):
    """Base class of image input errors (bad input -> HTTP 400 family).

    Subclasses :class:`ValueError` so existing "bad input" handlers keep working.
    """


class InvalidImageError(_KwargsReduceMixin, ImageError):
    """An image input failed validation at ingest.

    ``reason`` is one of ``"unsupported_type"``, ``"type_mismatch"``,
    ``"corrupt"``, ``"animated"``, ``"empty"``, ``"purged"``,
    ``"descriptor_version"``, ``"descriptor_mismatch"``, ``"bad_base64"``,
    ``"path"``. ``index`` is the position of the offending input when known.
    """

    _init_fields = ("reason", "index", "media_type", "byte_length", "detail")

    def __init__(
        self,
        reason: str,
        *,
        index: int | None = None,
        media_type: str | None = None,
        byte_length: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.index = index
        self.media_type = media_type
        self.byte_length = byte_length
        msg = f"Invalid image ({reason})"
        if index is not None:
            msg += f" at index {index}"
        extras = []
        if media_type is not None:
            extras.append(f"media_type={media_type}")
        if byte_length is not None:
            extras.append(f"bytes={byte_length}")
        if extras:
            msg += " [" + ", ".join(extras) + "]"
        if detail:
            msg += f": {detail}"
        super().__init__(msg + ".")


class ImageLimitError(_KwargsReduceMixin, ImageError):
    """An image or request exceeded an :class:`~dendrux.media.ImagePolicy` cap.

    ``scope`` is one of ``"image_bytes"``, ``"source_bytes"``, ``"image_side"``,
    ``"image_pixels"``, ``"message_count"``, ``"request_count"``,
    ``"request_bytes"``, ``"estimated_tokens"``, ``"many_image_dimension"``,
    or ``"store_bytes"`` (an :class:`~dendrux.media.InMemoryBlobStore` is full
    even after one garbage-collection pass).
    """

    _init_fields = ("scope", "limit", "actual", "index")

    def __init__(
        self,
        scope: str,
        *,
        limit: int,
        actual: int,
        index: int | None = None,
    ) -> None:
        self.scope = scope
        self.limit = limit
        self.actual = actual
        self.index = index
        where = f" at index {index}" if index is not None else ""
        super().__init__(f"Image limit exceeded ({scope}){where}: {actual} > {limit}.")


class ImageFetchError(_KwargsReduceMixin, ImageError):
    """Fetching an ``Image.from_url`` source failed or was refused.

    ``reason`` is one of ``"scheme"``, ``"port"``, ``"credentials"``,
    ``"host_not_allowed"``, ``"blocked_address"``, ``"dns"``, ``"redirect"``,
    ``"timeout"``, ``"too_large"``, ``"http_status"``, ``"content_encoding"``,
    ``"disabled"``, ``"connect"``. Only the host is ever reported, never the
    URL path, query or credentials.
    """

    _init_fields = ("reason", "host", "status_code")

    def __init__(
        self,
        reason: str,
        *,
        host: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.reason = reason
        self.host = host
        self.status_code = status_code
        msg = f"Image fetch failed ({reason})"
        if host:
            msg += f" for host {host!r}"
        if status_code is not None:
            msg += f" with HTTP status {status_code}"
        super().__init__(msg + ".")


class UnsupportedModalityError(_KwargsReduceMixin, RuntimeError):
    """The target model does not accept image input.

    ``source`` says how that was decided: ``"capability"`` (static table or
    override), ``"catalog"`` (provider model catalogue), ``"adapter"`` (the
    provider adapter cannot convert images) or ``"provider_response"`` (the
    vendor refused the request).
    """

    _init_fields = (
        "provider",
        "model",
        "image_count",
        "origin",
        "source",
        "status_code",
        "provider_code",
        "modality",
    )

    def __init__(
        self,
        *,
        provider: str | None,
        model: str | None,
        image_count: int = 0,
        origin: str | None = None,
        source: str = "capability",
        status_code: int | None = None,
        provider_code: str | None = None,
        modality: str = "image",
    ) -> None:
        self.provider = provider
        self.model = model
        self.modality = modality
        self.image_count = image_count
        self.origin = origin
        self.source = source
        self.status_code = status_code
        self.provider_code = provider_code
        super().__init__(
            f"Model {model!r} on provider {provider!r} does not accept {modality} input "
            f"({image_count} {modality}(s), source={source})."
        )


class ImageRejectedError(_KwargsReduceMixin, RuntimeError):
    """The provider rejected a specific image in a request.

    ``image_ids`` lists the rejected images when the provider error could be
    located (``located=True``); ``retried`` is True when the call had already
    been retried once with the image replaced by a placeholder.
    """

    _init_fields = (
        "provider",
        "model",
        "image_ids",
        "located",
        "retried",
        "status_code",
        "provider_code",
    )

    def __init__(
        self,
        *,
        provider: str | None,
        model: str | None,
        image_ids: tuple[str, ...] = (),
        located: bool = False,
        retried: bool = False,
        status_code: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.image_ids = tuple(image_ids)
        self.located = located
        self.retried = retried
        self.status_code = status_code
        self.provider_code = provider_code
        ids = ", ".join(self.image_ids) if self.image_ids else "unlocated"
        super().__init__(
            f"Provider {provider!r} rejected image(s) for model {model!r}: {ids}"
            f"{' (after retry)' if retried else ''}."
        )


ProviderErrorCategory: TypeAlias = Literal[
    "auth",
    "credits",
    "rate_limit",
    "rejected",
    "unavailable",
    "timeout",
    "context_length",
    "refusal",
    "content_filter",
    "stream_incomplete",
    "unknown",
]
PROVIDER_ERROR_CATEGORIES: tuple[str, ...] = get_args(ProviderErrorCategory)

TimeoutReason: TypeAlias = Literal[
    "connect",
    "call_deadline",
    "stream_idle",
    "stream_total",
    "deadline_during_backoff",
]
TIMEOUT_REASONS: tuple[str, ...] = get_args(TimeoutReason)

# A provider code is echoed into ``safe_message`` only when it looks like a code.
_SAFE_PROVIDER_CODE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")


def _safe_status_code(value: Any) -> int | None:
    return value if type(value) is int and 100 <= value <= 599 else None


def _safe_identity(value: str | None) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= 255:
        return None
    return value if re.fullmatch(r"[A-Za-z0-9_.:/@+\-]+", value) else None


def _provider_code_mapper(provider: str | None, raw: str | None) -> str | None:
    return raw


class ProviderError(_KwargsReduceMixin, RuntimeError):
    """A provider request failed.

    ``category`` classifies the failure (see :data:`PROVIDER_ERROR_CATEGORIES`)
    and ``retryable`` says whether repeating the same request may succeed.
    Both default per subclass. ``safe_message`` is a fixed template built only
    from these fields (never vendor text), safe to persist, log and show.

    On image-bearing requests ``message`` is the vendor message truncated to
    500 characters with base64 runs and data URLs replaced by ``[redacted]``.
    """

    _init_fields: tuple[str, ...] = (
        "provider",
        "model",
        "status_code",
        "provider_code",
        "error_type",
        "message",
        "category",
        "retryable",
        "timeout_reason",
    )
    default_category: ClassVar[ProviderErrorCategory] = "unknown"
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
        error_type: str | None = None,
        message: str = "",
        category: ProviderErrorCategory | None = None,
        retryable: bool | None = None,
        timeout_reason: TimeoutReason | None = None,
    ) -> None:
        self._set_provider_fields(
            provider=provider,
            model=model,
            status_code=status_code,
            provider_code=provider_code,
            error_type=error_type,
            message=message[:500],
            category=category,
            retryable=retryable,
            timeout_reason=timeout_reason,
        )
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Provider {provider!r} error{status}: {self.message}")

    def _set_provider_fields(
        self,
        *,
        provider: str | None,
        model: str | None,
        status_code: int | None,
        provider_code: str | None,
        error_type: str | None,
        message: str,
        category: ProviderErrorCategory | None,
        retryable: bool | None,
        timeout_reason: TimeoutReason | None,
    ) -> None:
        resolved = self.default_category if category is None else category
        if resolved not in PROVIDER_ERROR_CATEGORIES:
            raise ValueError(f"Unknown provider error category {resolved!r}.")
        if timeout_reason is not None and timeout_reason not in TIMEOUT_REASONS:
            raise ValueError(f"Unknown provider timeout reason {timeout_reason!r}.")
        self.provider = _safe_identity(provider)
        self.model = _safe_identity(model)
        self.status_code = _safe_status_code(status_code)
        self.provider_code_raw = (
            provider_code
            if isinstance(provider_code, str) and _SAFE_PROVIDER_CODE.fullmatch(provider_code)
            else None
        )
        self.error_type = _safe_identity(error_type)
        self.message = message
        self.category: ProviderErrorCategory = resolved
        self.retryable: bool = self.default_retryable if retryable is None else bool(retryable)
        self.timeout_reason: TimeoutReason | None = timeout_reason

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        fields = {name: getattr(self, name) for name in self._init_fields}
        fields["provider_code"] = self.provider_code_raw
        return _rebuild_error, (type(self), fields)

    @property
    def provider_code(self) -> str | None:
        """The safe mapped code; the regex-validated vendor code remains private."""
        return _provider_code_mapper(self.provider, self.provider_code_raw)

    @property
    def safe_provider_code(self) -> str | None:
        """The same mapped code used by safe result projections."""
        return self.provider_code

    @property
    def safe_message(self) -> str:
        """``"{provider} {category} (HTTP {status}, code {provider_code})"``; no vendor text."""
        status = self.status_code if self.status_code is not None else "n/a"
        shown = self.safe_provider_code or "n/a"
        return f"{self.provider or 'provider'} {self.category} (HTTP {status}, code {shown})"


class ProviderAuthError(ProviderError):
    """The provider rejected the credentials (HTTP 401/403)."""

    default_category = "auth"


class ProviderRateLimitError(ProviderError):
    """The provider rate-limited the request (HTTP 429).

    ``retry_after`` is the server's suggested wait in seconds, when it sent one.
    """

    _init_fields = (*ProviderError._init_fields, "retry_after")
    default_category = "rate_limit"
    default_retryable = True

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
        error_type: str | None = None,
        message: str = "",
        category: ProviderErrorCategory | None = None,
        retryable: bool | None = None,
        timeout_reason: TimeoutReason | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            provider=provider,
            model=model,
            status_code=status_code,
            provider_code=provider_code,
            error_type=error_type,
            message=message,
            category=category,
            retryable=retryable,
            timeout_reason=timeout_reason,
        )


class ProviderTimeoutError(ProviderError, TimeoutError):
    """A provider call hit a deadline. Also a builtin :class:`TimeoutError`."""

    default_category = "timeout"


class ProviderUnavailableError(ProviderError, ConnectionError):
    """The provider could not be reached or is overloaded (connection, 5xx).

    Also a builtin :class:`ConnectionError`.
    """

    default_category = "unavailable"
    default_retryable = True


class ProviderResponseError(ProviderError):
    """A provider response cannot be accepted within the local response contract."""

    default_category = "rejected"


class ProviderStreamIncompleteError(ProviderError):
    """A response stream ended without its terminal event."""

    default_category = "stream_incomplete"


class ModelRefusalError(ProviderError):
    """The model refused to answer.

    ``refusal_text`` is an in-process attribute only: it is never persisted,
    emitted, logged or carried through pickling. ``copy.copy`` and
    ``copy.deepcopy`` go through the same reduce path and drop it too.
    """

    default_category = "refusal"

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
        error_type: str | None = None,
        message: str = "",
        category: ProviderErrorCategory | None = None,
        retryable: bool | None = None,
        timeout_reason: TimeoutReason | None = None,
        refusal_text: str | None = None,
    ) -> None:
        self.refusal_text = refusal_text
        super().__init__(
            provider=provider,
            model=model,
            status_code=status_code,
            provider_code=provider_code,
            error_type=error_type,
            message=message,
            category=category,
            retryable=retryable,
            timeout_reason=timeout_reason,
        )


class ContentFilteredError(ProviderError):
    """The provider's content filter stopped the request or the response."""

    default_category = "content_filter"


class ContextLengthExceededError(ProviderError):
    """Raised when an LLM endpoint rejects a request as too large for the model's context window.

    Raised by the OpenAI-compatible providers (and therefore
    ``OpenRouterProvider``) for both batch and streaming calls. The message
    never contains prompt content; the original SDK exception is chained as
    ``__cause__`` for debugging. ``str(exc)`` is ``message`` unchanged.

    Attributes:
        status_code: HTTP status the endpoint returned (``None`` for an
            in-stream error event).
        provider_code: Provider error code when one was sent
            (for example ``"context_length_exceeded"``), else ``None``.
        model: Model name the request targeted, when known.
    """

    _init_fields = (
        "message",
        "status_code",
        "provider_code",
        "model",
        "provider",
        "error_type",
        "category",
        "retryable",
        "timeout_reason",
    )
    default_category = "context_length"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        error_type: str | None = None,
        category: ProviderErrorCategory | None = None,
        retryable: bool | None = None,
        timeout_reason: TimeoutReason | None = None,
    ) -> None:
        self._set_provider_fields(
            provider=provider,
            model=model,
            status_code=status_code,
            provider_code=provider_code,
            error_type=error_type,
            message=message,
            category=category,
            retryable=retryable,
            timeout_reason=timeout_reason,
        )
        RuntimeError.__init__(self, message)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        kwargs = {name: getattr(self, name) for name in self._init_fields if name != "message"}
        return (_rebuild_context_length_error, (type(self), self.message, kwargs))


class BlobNotFoundError(_KwargsReduceMixin, LookupError):
    """The image does not exist for this tenant (or this run has no reference to it).

    Absence and a tenant mismatch are reported identically, so the error is
    never an existence oracle.
    """

    _init_fields = ("image_id",)

    def __init__(self, image_id: str | None = None) -> None:
        self.image_id = image_id
        what = f"Image {image_id!r}" if image_id else "Image"
        super().__init__(f"{what} not found.")


class BlobIntegrityError(_KwargsReduceMixin, RuntimeError):
    """Stored image bytes or a persisted descriptor failed verification.

    ``check`` is one of ``"length"``, ``"sha256"``, ``"media_type"``,
    ``"dimensions"``, ``"descriptor"``.
    """

    _init_fields = ("image_id", "check")

    def __init__(self, image_id: str | None, check: str) -> None:
        self.image_id = image_id
        self.check = check
        super().__init__(f"Image {image_id!r} failed integrity check ({check}).")


class ImagePersistenceUnsupportedError(_KwargsReduceMixin, RuntimeError):
    """The configured store, blob store or loop cannot handle images.

    ``reason`` is one of ``"state_store"`` (custom ``StateStore`` without the
    blob catalog), ``"blob_store_not_configured"`` (an external blob row with
    no ``blob_store``) or ``"loop"`` (a loop that does not accept
    ``image_runtime``).
    """

    _init_fields = ("reason", "store_type")

    def __init__(self, reason: str, *, store_type: str | None = None) -> None:
        self.reason = reason
        self.store_type = store_type
        hint = {
            "state_store": "the configured state store has no blob catalog; "
            "use the built-in SQLAlchemy store",
            "blob_store_not_configured": "an image is stored externally but no "
            "blob_store is configured",
            "loop": "the loop does not accept image_runtime",
        }.get(reason, reason)
        store = f" ({store_type})" if store_type else ""
        super().__init__(f"Images are not supported here{store}: {hint}.")


class BlobContentionError(_KwargsReduceMixin, RuntimeError):
    """Storing an image lost too many races with concurrent purge, GC or ingest.

    Transient: retrying the same call is safe.
    """

    _init_fields = ("image_id",)

    def __init__(self, image_id: str | None = None) -> None:
        self.image_id = image_id
        what = f"image {image_id!r}" if image_id else "an image"
        super().__init__(f"Could not store {what} due to concurrent changes; retry the call.")


# ---------------------------------------------------------------------------
# Trust (0.5.0)
# ---------------------------------------------------------------------------


class BudgetConfigurationError(_KwargsReduceMixin, ValueError):
    """A budget cannot be safely evaluated from pricing or bounded live accounting.

    ``reason="unpriced_model"`` means neither the provider nor ``PriceTable``
    prices the effective model. ``reason="subtree_limit"`` means the configured
    node cap prevented complete root-inclusive accounting.
    """

    _init_fields = ("model", "reason")

    def __init__(self, *, model: str | None = None, reason: str = "unpriced_model") -> None:
        self.model = model
        self.reason = reason
        if reason == "subtree_limit":
            super().__init__("Budget subtree accounting limit reached; raise max_subtree_nodes.")
            return
        super().__init__(
            f"Budget max_cost_usd cannot be enforced for model {model!r} ({reason}): add a "
            "PriceTable entry for the model you request, or use a provider that reports cost."
        )


class TenantScopeRequiredError(_KwargsReduceMixin, ValueError):
    """An entry point was called without ``tenant_id`` on an ``Agent(strict_tenancy=True)``.

    ``operation`` names the method (``"run"``, ``"submit_input"``, ...).
    """

    _init_fields = ("operation",)

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"{operation}() requires an explicit tenant_id= because the agent has "
            "strict_tenancy=True."
        )


class NoteLimitExceededError(_KwargsReduceMixin, ValueError):
    """``Agent.submit_note`` would exceed ``Limits.max_notes_per_run`` for this run."""

    _init_fields = ("run_id", "limit")

    def __init__(self, run_id: str, *, limit: int) -> None:
        self.run_id = run_id
        self.limit = limit
        super().__init__(f"Run '{run_id}' already has the maximum of {limit} notes.")


class GovernanceMismatchError(_KwargsReduceMixin, RuntimeError):
    """A paused run was resumed with governance different from what it started with.

    Raised only with ``Agent(on_governance_change="raise")``. ``expected_digest``
    and ``actual_digest`` are opaque digests; ``changed`` names the parts that
    differ (for example ``"guardrails"``, ``"deny"``). The run stays paused.
    """

    _init_fields = ("run_id", "expected_digest", "actual_digest", "changed")

    def __init__(
        self,
        run_id: str,
        *,
        expected_digest: str | None = None,
        actual_digest: str | None = None,
        changed: Iterable[str] = (),
    ) -> None:
        self.run_id = run_id
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest
        self.changed: tuple[str, ...] = tuple(changed)
        what = f" ({', '.join(self.changed)})" if self.changed else ""
        super().__init__(
            f"Run '{run_id}' was started with different governance{what}; "
            "resume it with the same guardrails, hooks, deny and approval settings."
        )


class RunFencedError(_KwargsReduceMixin, RuntimeError):
    """Internal: this worker lost its claim on the run (a newer claim epoch exists).

    Not public API. The runner stops without writing further state.
    """

    _init_fields = ("run_id", "claim_epoch")

    def __init__(self, run_id: str, *, claim_epoch: int | None = None) -> None:
        self.run_id = run_id
        self.claim_epoch = claim_epoch
        super().__init__(f"Run '{run_id}' is fenced: claim epoch {claim_epoch} is stale.")


__all__ = [
    "NotifierUnavailableError",
    "StoreCapabilityError",
    "RunCancelledError",
    "RunNotReclaimableError",
    "RunControlError",
    "BlobContentionError",
    "BlobIntegrityError",
    "BlobNotFoundError",
    "BudgetConfigurationError",
    "ContentFilteredError",
    "ContextLengthExceededError",
    "GovernanceMismatchError",
    "ImageError",
    "ImageFetchError",
    "ImageLimitError",
    "ImagePersistenceUnsupportedError",
    "ImageRejectedError",
    "InvalidImageError",
    "InvalidToolResultError",
    "ModelRefusalError",
    "NoteLimitExceededError",
    "PROVIDER_ERROR_CATEGORIES",
    "PauseStatusMismatchError",
    "PersistenceNotConfiguredError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderErrorCategory",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderStreamIncompleteError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ReplayRunError",
    "RunAlreadyClaimedError",
    "RunAlreadyTerminalError",
    "RunNotFoundError",
    "RunNotPausedError",
    "SchemaNotInitializedError",
    "SchemaOutOfDateError",
    "SkillContinuityError",
    "SkillDigestMismatch",
    "SkillMismatchError",
    "SkillMissingError",
    "SkillValidationError",
    "SqliteSchemaSyncError",
    "TIMEOUT_REASONS",
    "TenantScopeRequiredError",
    "TimeoutReason",
    "UnsupportedModalityError",
]


def provider_error_in(error: BaseException) -> ProviderError | None:
    """Find a typed provider failure without formatting worker-group messages."""
    pending = [error]
    seen: set[int] = set()
    for _ in range(1024):
        if not pending:
            return None
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ProviderError):
            return current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions[:1024])
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    # A diagnostic graph too large to inspect is never safe to format.
    return ProviderError()


def _database_failure_in(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    for _ in range(1024):
        if not pending:
            return False
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, PersistenceError) or any(
            base.__module__.startswith("sqlalchemy.exc") for base in type(current).__mro__
        ):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions[:1024])
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return True


def safe_exception_info(
    error: BaseException | None = None,
) -> bool | tuple[type[BaseException], BaseException, TracebackType | None]:
    """Keep ordinary stacks and secondary failures without provider chains.

    The original exception and its chain remain untouched for callers. Custom
    provider errors must use ProviderError to identify vendor-bearing chains.
    """
    import copy
    import sys

    error = error if error is not None else sys.exception()
    if error is not None and _database_failure_in(error):
        safe = PersistenceError()
        return type(safe), safe, None
    if error is None or provider_error_in(error) is None:
        return True
    if isinstance(error, (ProviderError, BaseExceptionGroup)):
        return False
    try:
        own = copy.copy(error)
        if not isinstance(own, BaseException) or own is error:
            raise TypeError("exception copy retained its original identity")
    except Exception:
        own = RuntimeError(f"{type(error).__name__}: {error}")
    BaseException.__setattr__(own, "__cause__", None)
    BaseException.__setattr__(own, "__context__", None)
    BaseException.__setattr__(own, "__suppress_context__", True)
    BaseException.__setattr__(own, "__traceback__", error.__traceback__)
    return type(own), own, own.__traceback__
