"""The helper object contract tests receive: a fresh store plus the author's fixtures."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest

from dendrux.testing.store_kit._declaration import (
    CLOCK,
    FACTORY,
    FAULT_INJECTOR,
    REOPEN,
    SECOND_INSTANCE,
    Declaration,
    validate_declaration,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.runtime.state import ClaimResult
    from dendrux.testing.store_kit._protocols import (
        Clock,
        FaultInjector,
        FaultKind,
        FaultPoint,
    )

#: The marker contract tests carry (see ``contracts/_marks.py``).
MARKER = "dendrux_kit"
USER_PROPERTY_STORE = "dendrux_kit_store"
USER_PROPERTY_CATEGORY = "dendrux_kit_category"


class CertificationFailure(pytest.fail.Exception):  # type: ignore[misc,name-defined]
    """A store does not meet what it declared (reported as a failed test, not an error)."""

    def __init__(self, message: str) -> None:
        super().__init__(f"CERTIFICATION FAILURE: {message}", pytrace=False)


async def maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def fixture_or_none(request: pytest.FixtureRequest, name: str) -> Any | None:
    """The author's fixture value, or ``None`` when no fixture of that name exists.

    Call from a synchronous fixture only: pytest-asyncio cannot set up an async fixture from
    inside a running loop.
    """
    try:
        return request.getfixturevalue(name)
    except pytest.FixtureLookupError:
        return None


@dataclass
class Faulted:
    """What happened to a call made with a fault armed."""

    raised: BaseException | None
    result: Any
    fired: int

    def expect_fired(self, point: str) -> None:
        """Fail certification when the requested cut never happened (a no-op injector)."""
        if self.fired < 1:
            raise CertificationFailure(
                f"the fault injector did not fire the requested {point} cut: "
                "dendrux_fault_injector must make the next write transaction fail"
            )


class Kit:
    """One fresh store, its declaration and helpers for the author's other fixtures."""

    def __init__(
        self,
        provided: dict[str, Any],
        store: Any,
        declaration: Declaration,
        *,
        needs_format_2: bool,
    ) -> None:
        self._provided = provided
        self.store = store
        self.declaration = declaration
        self._needs_format_2 = needs_format_2
        self.format: int | None = None

    # -- the author's fixtures -------------------------------------------------------------

    async def reopen(self, store: Any | None = None) -> Any:
        derive = self._provided.get(REOPEN)
        if derive is None:
            raise CertificationFailure(f"{REOPEN} is required but not provided")
        return await maybe_await(derive(store if store is not None else self.store))

    async def second(self, store: Any | None = None) -> Any:
        derive = self._provided.get(SECOND_INSTANCE)
        if derive is None:
            raise CertificationFailure(f"{SECOND_INSTANCE} is required but not provided")
        return await maybe_await(derive(store if store is not None else self.store))

    async def advance(self, delta: timedelta) -> None:
        clock: Clock | None = self._provided.get(CLOCK)
        if clock is None:
            raise CertificationFailure(f"{CLOCK} is required but not provided")
        await maybe_await(clock.advance(delta))

    @property
    def has_clock(self) -> bool:
        return self._provided.get(CLOCK) is not None

    @property
    def injector(self) -> FaultInjector:
        found: FaultInjector | None = self._provided.get(FAULT_INJECTOR)
        if found is None:
            pytest.skip(f"{FAULT_INJECTOR} not provided: crash recovery is not certified")
        return found

    def require_format_2(self) -> None:
        """The rest of this test needs format-2 runs (skips at the first run created otherwise)."""
        self._needs_format_2 = True

    def disconnect_forms(self, store: Any | None = None) -> tuple[str, ...]:
        """The lost-connection forms the injector can produce for this store."""
        target = store if store is not None else self.store
        lister = getattr(self.injector, "disconnect_forms", None)
        return tuple(lister(target)) if lister is not None else ("invalidated",)

    async def age(self) -> timedelta:
        """Make what exists so far old; return a sweep threshold that spares what comes next.

        The world moves two hours forward through ``dendrux_clock`` and the threshold is one
        hour. Without the clock, time-based contracts cannot be run, so they skip as unverified.
        """
        if not self.has_clock:
            pytest.skip("unverified: time-based sweeps and leases: dendrux_clock not provided")
        await self.advance(timedelta(hours=2))
        return timedelta(hours=1)

    async def now(self) -> datetime:
        """The store's present, as the clock sees it (wall clock when the clock has no ``now``)."""
        clock = self._provided.get(CLOCK)
        reader = getattr(clock, "now", None)
        if reader is None:
            return datetime.now(UTC)
        return cast("datetime", await maybe_await(reader()))

    async def faulted(
        self,
        point: FaultPoint,
        call: Callable[[], Awaitable[Any]],
        *,
        kind: FaultKind = "error",
        after: int = 0,
        sticky: bool = False,
        form: str = "invalidated",
        store: Any | None = None,
    ) -> Faulted:
        """Run ``call`` with a fault armed on the store, then disarm and report what happened."""
        target = store if store is not None else self.store
        injector = self.injector
        injector.arm(target, point, kind=kind, after=after, sticky=sticky, form=form)
        raised: BaseException | None = None
        result: Any = None
        try:
            result = await call()
        except asyncio.CancelledError as exc:
            if kind != "cancel":
                raise
            raised = exc
        except Exception as exc:
            raised = exc
        finally:
            injector.disarm(target)
        return Faulted(raised, result, int(injector.fired(target)))

    # -- runs ------------------------------------------------------------------------------

    async def new_run(self, store: Any | None = None, **kwargs: Any) -> str:
        """Create a running run and return its id. Skips when format 2 is needed but absent."""
        from dendrux.types import generate_ulid

        target = store if store is not None else self.store
        run_id = kwargs.pop("run_id", None) or generate_ulid()
        kwargs.setdefault("input_data", {"input": "kit"})
        kwargs.setdefault("model", "kit-model")
        created = await target.create_run(run_id, kwargs.pop("agent_name", "KitAgent"), **kwargs)
        self.format = created.storage_format
        if self._needs_format_2 and created.storage_format != 2:
            pytest.skip("store creates storage format 1 here: format-2 contracts do not apply")
        return str(created.run_id)

    async def paused_run(
        self, store: Any | None = None, *, status: str = "waiting_client_tool", **kwargs: Any
    ) -> str:
        """A run paused at ``status`` with minimal pause data."""
        target = store if store is not None else self.store
        run_id = await self.new_run(target, **kwargs)
        await target.pause_run(
            run_id,
            status=status,
            pause_data={"pause_id": "kit-pause", "pending_tool_calls": []},
        )
        return run_id

    async def claimed_run(self, store: Any | None = None, **kwargs: Any) -> tuple[str, ClaimResult]:
        """A run claimed once after a pause: its epoch is one past the creation epoch."""
        target = store if store is not None else self.store
        run_id = await self.paused_run(target, **kwargs)
        claim = await target.claim_paused_run(run_id, expected_status="waiting_client_tool")
        assert claim is not False, "a fresh pause must be claimable"
        return run_id, claim

    async def epoch(self, run_id: str, store: Any | None = None) -> int:
        record = await (store if store is not None else self.store).get_run(run_id)
        assert record is not None
        return int(record.claim_epoch)


def check_declaration(store: Any, declaration: Declaration) -> None:
    """Fail certification when a declared promise lacks the methods that keep it."""
    from dendrux.errors import StoreCapabilityError

    try:
        validate_declaration(store)
    except StoreCapabilityError as exc:
        if exc.capability == "deployment":
            raise CertificationFailure(
                f"{declaration.store_name} declares deployment={declaration.deployment!r}; "
                "it must be single_process, single_host or multi_server"
            ) from None
        raise CertificationFailure(
            f"{declaration.store_name} declares {exc.capability} but {exc.method} lacks "
            f"{exc.missing!r}"
        ) from None


__all__ = ["FACTORY", "CertificationFailure", "Faulted", "Kit", "check_declaration", "maybe_await"]
