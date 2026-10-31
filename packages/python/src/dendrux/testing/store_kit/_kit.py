"""The helper object contract tests receive: a fresh store plus the author's fixtures."""

from __future__ import annotations

import asyncio
import inspect
from datetime import timedelta
from typing import TYPE_CHECKING, Any

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

    def arm(
        self, point: FaultPoint, *, kind: FaultKind = "error", store: Any | None = None
    ) -> None:
        self.injector.arm(store if store is not None else self.store, point, kind=kind)

    def disarm(self, store: Any | None = None) -> None:
        injector = self._provided.get(FAULT_INJECTOR)
        if injector is not None:
            injector.disarm(store if store is not None else self.store)

    async def age(self) -> timedelta:
        """Make what exists so far old; return a sweep threshold that spares what comes next.

        With ``dendrux_clock`` the world moves two hours forward and the threshold is one hour.
        Without it the kit waits a fraction of a second of real time.
        """
        if self.has_clock:
            await self.advance(timedelta(hours=2))
            return timedelta(hours=1)
        await asyncio.sleep(0.4)
        return timedelta(seconds=0.2)

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


def open_declaration(store: Any) -> Declaration:
    """Read and check the declaration; skip nothing here, fail what cannot be certified."""
    from dendrux.errors import StoreCapabilityError

    declaration = Declaration.of(store)
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
    return declaration


__all__ = ["FACTORY", "CertificationFailure", "Kit", "maybe_await", "open_declaration"]
