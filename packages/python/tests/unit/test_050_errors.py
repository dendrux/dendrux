"""0.5.0 contracts: the generalised ProviderError family and the new errors.

T0 adds names, fields and defaults only; provider mapping lands later.
"""

from __future__ import annotations

import copy
import pickle

import pytest

import dendrux
from dendrux import errors
from dendrux.errors import (
    PROVIDER_ERROR_CATEGORIES,
    BudgetConfigurationError,
    ContentFilteredError,
    ContextLengthExceededError,
    GovernanceMismatchError,
    ModelRefusalError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderStreamIncompleteError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RunFencedError,
    RunNotFoundError,
    TenantScopeRequiredError,
)

_COMMON = {
    "provider": "anthropic",
    "model": "claude-x",
    "status_code": 429,
    "provider_code": "rate_limit_error",
    "error_type": "rate_limit_error",
    "message": "vendor text",
}


def _roundtrip(exc: BaseException) -> BaseException:
    return pickle.loads(pickle.dumps(exc))


class TestProviderErrorBase:
    def test_existing_image_path_constructor_still_works(self) -> None:
        exc = ProviderError(
            provider="openai",
            model="gpt-x",
            status_code=400,
            provider_code="bad",
            error_type="invalid_request_error",
            message="x" * 600,
        )
        assert exc.error_type == "invalid_request_error"
        assert exc.message == "x" * 500
        assert str(exc) == f"Provider 'openai' error (HTTP 400): {'x' * 500}"
        assert isinstance(exc, RuntimeError)

    def test_transport_style_constructor_without_status(self) -> None:
        exc = ProviderError(provider="openai", model="m", error_type="timeout", message="t")
        assert str(exc) == "Provider 'openai' error: t"

    def test_defaults(self) -> None:
        exc = ProviderError()
        assert exc.provider is None and exc.model is None
        assert exc.category == "unknown"
        assert exc.retryable is False

    @pytest.mark.parametrize("category", ["credits", "rejected", "unknown"])
    def test_base_class_carries_other_categories(self, category: str) -> None:
        exc = ProviderError(provider="p", category=category)  # type: ignore[arg-type]
        assert exc.category == category

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(ValueError, match="category"):
            ProviderError(category="nope")  # type: ignore[arg-type]

    def test_categories_listed(self) -> None:
        assert set(PROVIDER_ERROR_CATEGORIES) == {
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
        }

    def test_safe_message_is_fixed_template(self) -> None:
        exc = ProviderError(**_COMMON, category="rate_limit")  # type: ignore[arg-type]
        assert exc.safe_message == "anthropic rate_limit (HTTP 429, code rate_limit_error)"
        assert "vendor text" not in exc.safe_message

    def test_safe_message_without_status_or_code(self) -> None:
        assert ProviderError().safe_message == "provider unknown (HTTP n/a, code n/a)"

    def test_safe_message_never_echoes_a_non_code_provider_code(self) -> None:
        exc = ProviderError(provider="p", provider_code="Invalid value: 'secret@x.com'")
        assert "secret" not in exc.safe_message
        assert exc.safe_message.endswith("code n/a)")

    def test_retryable_override(self) -> None:
        assert ProviderError(retryable=True).retryable is True
        assert ProviderRateLimitError(retryable=False).retryable is False


_SUBCLASSES: list[tuple[type[ProviderError], str, bool, tuple[type[BaseException], ...]]] = [
    (ProviderAuthError, "auth", False, ()),
    (ProviderRateLimitError, "rate_limit", True, ()),
    (ProviderTimeoutError, "timeout", False, (TimeoutError, OSError)),
    (ProviderUnavailableError, "unavailable", True, (ConnectionError, OSError)),
    (ProviderStreamIncompleteError, "stream_incomplete", False, ()),
    (ModelRefusalError, "refusal", False, ()),
    (ContentFilteredError, "content_filter", False, ()),
]


class TestProviderSubclasses:
    @pytest.mark.parametrize(("cls", "category", "retryable", "builtins"), _SUBCLASSES)
    def test_defaults_and_hierarchy(
        self,
        cls: type[ProviderError],
        category: str,
        retryable: bool,
        builtins: tuple[type[BaseException], ...],
    ) -> None:
        exc = cls(**_COMMON)  # type: ignore[arg-type]
        assert exc.category == category
        assert exc.retryable is retryable
        assert isinstance(exc, ProviderError)
        assert isinstance(exc, RuntimeError)
        for base in builtins:
            assert isinstance(exc, base)
        assert exc.safe_message.startswith(f"anthropic {category} ")

    @pytest.mark.parametrize(("cls", "category", "retryable", "builtins"), _SUBCLASSES)
    def test_pickle_and_copy_roundtrip(
        self,
        cls: type[ProviderError],
        category: str,
        retryable: bool,
        builtins: tuple[type[BaseException], ...],
    ) -> None:
        exc = cls(**_COMMON, retryable=not retryable)  # type: ignore[arg-type]
        for clone in (_roundtrip(exc), copy.copy(exc), copy.deepcopy(exc)):
            assert type(clone) is cls
            assert str(clone) == str(exc)
            for name in ("provider", "model", "status_code", "provider_code", "error_type"):
                assert getattr(clone, name) == getattr(exc, name)
            assert clone.message == exc.message  # type: ignore[attr-defined]
            assert clone.category == category  # type: ignore[attr-defined]
            assert clone.retryable is (not retryable)  # type: ignore[attr-defined]

    def test_timeout_is_caught_by_builtin_handlers(self) -> None:
        with pytest.raises(TimeoutError):
            raise ProviderTimeoutError(provider="p")
        with pytest.raises(ConnectionError):
            raise ProviderUnavailableError(provider="p")

    def test_rate_limit_retry_after(self) -> None:
        exc = ProviderRateLimitError(provider="p", retry_after=2.5)
        assert exc.retry_after == 2.5
        assert ProviderRateLimitError().retry_after is None
        assert _roundtrip(exc).retry_after == 2.5  # type: ignore[attr-defined]

    def test_refusal_text_is_in_process_only(self) -> None:
        exc = ModelRefusalError(provider="p", refusal_text="I can't help with that")
        assert exc.refusal_text == "I can't help with that"
        assert "help" not in str(exc)
        assert "help" not in exc.safe_message
        assert _roundtrip(exc).refusal_text is None  # type: ignore[attr-defined]


class TestContextLengthExceeded:
    def test_positional_message_constructor_kept(self) -> None:
        exc = ContextLengthExceededError(
            "Request exceeds the model's context window.",
            status_code=400,
            provider_code="context_length_exceeded",
            model="gpt-x",
        )
        assert str(exc) == "Request exceeds the model's context window."
        assert exc.args == ("Request exceeds the model's context window.",)
        assert exc.message == "Request exceeds the model's context window."
        assert exc.status_code == 400
        assert exc.provider_code == "context_length_exceeded"
        assert exc.model == "gpt-x"
        assert exc.provider is None

    def test_is_a_provider_error(self) -> None:
        exc = ContextLengthExceededError("too big")
        assert isinstance(exc, ProviderError)
        assert isinstance(exc, RuntimeError)
        assert exc.category == "context_length"
        assert exc.retryable is False

    def test_message_is_not_truncated(self) -> None:
        assert ContextLengthExceededError("m" * 900).message == "m" * 900

    def test_pickle_roundtrip(self) -> None:
        exc = ContextLengthExceededError(
            "too big", status_code=413, provider_code="ctx", model="m", provider="openai"
        )
        clone = _roundtrip(exc)
        assert type(clone) is ContextLengthExceededError
        assert str(clone) == "too big"
        assert clone.args == ("too big",)
        assert (clone.status_code, clone.provider_code, clone.model, clone.provider) == (  # type: ignore[attr-defined]
            413,
            "ctx",
            "m",
            "openai",
        )
        assert copy.deepcopy(exc).message == "too big"


class TestNewNonProviderErrors:
    def test_budget_configuration_error(self) -> None:
        exc = BudgetConfigurationError(model="gpt-5-2026-01-01")
        assert isinstance(exc, ValueError)
        assert exc.model == "gpt-5-2026-01-01"
        assert exc.reason == "unpriced_model"
        assert "PriceTable" in str(exc)
        clone = _roundtrip(exc)
        assert (clone.model, clone.reason, str(clone)) == (exc.model, exc.reason, str(exc))  # type: ignore[attr-defined]

    def test_tenant_scope_required_error(self) -> None:
        exc = TenantScopeRequiredError("submit_input")
        assert isinstance(exc, ValueError)
        assert exc.operation == "submit_input"
        assert "submit_input()" in str(exc)
        clone = _roundtrip(exc)
        assert clone.operation == "submit_input"  # type: ignore[attr-defined]
        assert str(clone) == str(exc)

    def test_governance_mismatch_error(self) -> None:
        exc = GovernanceMismatchError(
            "r1", expected_digest="aa", actual_digest="bb", changed=["guardrails", "deny"]
        )
        assert isinstance(exc, RuntimeError)
        assert exc.changed == ("guardrails", "deny")
        clone = _roundtrip(exc)
        assert (clone.run_id, clone.expected_digest, clone.actual_digest, clone.changed) == (  # type: ignore[attr-defined]
            "r1",
            "aa",
            "bb",
            ("guardrails", "deny"),
        )
        assert str(clone) == str(exc)

    def test_run_fenced_error_is_internal(self) -> None:
        exc = RunFencedError("r1", claim_epoch=3)
        assert isinstance(exc, RuntimeError)
        assert exc.claim_epoch == 3
        clone = _roundtrip(exc)
        assert (clone.run_id, clone.claim_epoch) == ("r1", 3)  # type: ignore[attr-defined]
        assert "RunFencedError" not in errors.__all__
        assert "RunFencedError" not in dendrux.__all__


class TestRunNotFoundErrorAlias:
    def test_store_alias_is_the_canonical_class(self) -> None:
        from dendrux.store import RunNotFoundError as StoreRunNotFoundError

        assert StoreRunNotFoundError is RunNotFoundError
        exc = StoreRunNotFoundError("r1")
        assert isinstance(exc, LookupError)
        assert exc.run_id == "r1"
        assert str(exc) == "Run 'r1' not found."

    async def test_store_paths_raise_the_canonical_class(self, tmp_path) -> None:
        from dendrux.store import RunStore

        async with RunStore.from_database_url(f"sqlite+aiosqlite:///{tmp_path / 'r.db'}") as store:
            with pytest.raises(RunNotFoundError) as info:
                await store.get_subtree("missing")
            assert info.value.run_id == "missing"
            with pytest.raises(RunNotFoundError):
                async for _ in store.stream_events("missing"):
                    pass


class TestFixRoundErrors:
    @pytest.mark.parametrize(
        "reason",
        ["connect", "call_deadline", "stream_idle", "stream_total", "deadline_during_backoff"],
    )
    def test_timeout_reason_on_base_and_subclasses(self, reason: str) -> None:
        from dendrux.errors import TIMEOUT_REASONS

        assert reason in TIMEOUT_REASONS
        for exc in (
            ProviderError(timeout_reason=reason),  # type: ignore[arg-type]
            ProviderRateLimitError(timeout_reason=reason),  # type: ignore[arg-type]
            ProviderTimeoutError(timeout_reason=reason),  # type: ignore[arg-type]
            ModelRefusalError(timeout_reason=reason),  # type: ignore[arg-type]
            ContextLengthExceededError("x", timeout_reason=reason),  # type: ignore[arg-type]
        ):
            assert exc.timeout_reason == reason
            assert _roundtrip(exc).timeout_reason == reason  # type: ignore[attr-defined]

    def test_deadline_during_backoff_keeps_rate_limit_category(self) -> None:
        exc = ProviderRateLimitError(provider="p", timeout_reason="deadline_during_backoff")
        assert exc.category == "rate_limit"

    def test_timeout_reason_defaults_none_and_is_validated(self) -> None:
        assert ProviderError().timeout_reason is None
        with pytest.raises(ValueError, match="timeout reason"):
            ProviderError(timeout_reason="whenever")  # type: ignore[arg-type]

    def test_safe_provider_code(self) -> None:
        assert ProviderError(provider_code="rate_limit_error").safe_provider_code == (
            "rate_limit_error"
        )
        assert ProviderError(provider_code="free text leak here").safe_provider_code is None
        assert ProviderError().safe_provider_code is None

    def test_run_not_found_unpickles_with_one_message(self) -> None:
        clone = _roundtrip(RunNotFoundError("r1"))
        assert str(clone) == "Run 'r1' not found."
        assert clone.run_id == "r1"  # type: ignore[attr-defined]

    def test_pause_status_mismatch_status_form_unchanged(self) -> None:
        from dendrux.errors import PauseStatusMismatchError
        from dendrux.types import RunStatus

        exc = PauseStatusMismatchError(
            "r1", RunStatus.WAITING_HUMAN_INPUT, RunStatus.WAITING_CLIENT_TOOL
        )
        assert exc.reason == "status"
        assert exc.expected_pause_id is None and exc.current_pause_id is None
        assert (
            str(exc)
            == "Run 'r1' is waiting_human_input, but this method requires waiting_client_tool."
        )
        clone = _roundtrip(exc)
        assert str(clone) == str(exc)
        assert clone.current_status is RunStatus.WAITING_HUMAN_INPUT  # type: ignore[attr-defined]

    def test_pause_status_mismatch_pause_id_form(self) -> None:
        from dendrux.errors import PauseStatusMismatchError
        from dendrux.types import RunStatus

        exc = PauseStatusMismatchError(
            "r1",
            RunStatus.WAITING_CLIENT_TOOL,
            RunStatus.WAITING_CLIENT_TOOL,
            expected_pause_id="p-old",
            current_pause_id="p-new",
            reason="pause_id",
        )
        assert "stale" in str(exc)
        assert "waiting_client_tool, but" not in str(exc)
        clone = _roundtrip(exc)
        assert (clone.reason, clone.expected_pause_id, clone.current_pause_id) == (  # type: ignore[attr-defined]
            "pause_id",
            "p-old",
            "p-new",
        )
        with pytest.raises(ValueError, match="reason"):
            PauseStatusMismatchError(
                "r1",
                RunStatus.PAUSED,
                RunStatus.PAUSED,
                reason="other",  # type: ignore[arg-type]
            )

    def test_note_limit_exceeded_error(self) -> None:
        from dendrux.errors import NoteLimitExceededError

        exc = NoteLimitExceededError("r1", limit=1000)
        assert isinstance(exc, ValueError)
        assert (exc.run_id, exc.limit) == ("r1", 1000)
        clone = _roundtrip(exc)
        assert (clone.run_id, clone.limit, str(clone)) == ("r1", 1000, str(exc))  # type: ignore[attr-defined]

    def test_idempotency_conflict_without_run_id(self) -> None:
        from dendrux.types import IdempotencyConflictError

        exc = IdempotencyConflictError(None, "k1")
        assert exc.run_id is None
        assert "for run" not in str(exc)
        assert "None" not in str(exc)
        clone = _roundtrip(exc)
        assert (clone.run_id, clone.idempotency_key, str(clone)) == (None, "k1", str(exc))  # type: ignore[attr-defined]
        with_id = _roundtrip(IdempotencyConflictError(run_id="r1", idempotency_key="k1"))
        assert "for run r1" in str(with_id)
