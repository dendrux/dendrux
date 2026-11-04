"""The helper object contract tests receive: a fresh store plus the author's fixtures."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NoReturn, cast

import pytest

from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux.testing.store_kit._declaration import (
    CLOCK,
    FACTORY,
    FAULT_INJECTOR,
    FORMAT_2_FLAGS,
    LIVE_FACTORY,
    REOPEN,
    SECOND_INSTANCE,
    Declaration,
    validate_declaration,
)
from dendrux.testing.store_kit._live import RecordingLiveUpdates, RunView
from dendrux.types import RunStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

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
USER_PROPERTY_NEEDS = "dendrux_kit_needs"
#: Prefix of a skip the store caused (see ``store_skip``); the report parses it.
STORE_SKIP = "store-controlled"


def store_skip(reason: str, *capabilities: str) -> NoReturn:
    """Skip because of something the store controls, which leaves a promise unchecked.

    The report treats this differently from an ordinary skip: when the store declares one of
    ``capabilities``, that capability is NOT CERTIFIED, and the reason is listed.
    """
    pytest.skip(f"{STORE_SKIP}[{','.join(capabilities)}]: {reason}")


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


@dataclass
class LiveStore:
    """A store that announces through a ``RecordingLiveUpdates``, and who reads it back."""

    store: Any
    updates: RecordingLiveUpdates
    #: A second instance (its own connection) when one is available, else the store itself.
    reader: Any
    kit: Kit
    #: Whether ``reader`` is independent of the writing store. If not, what it sees may include
    #: the store's staged, uncommitted state, so visibility after commit cannot be certified.
    independent: bool = False

    async def new_run(self, **kwargs: Any) -> str:
        return await self.kit.new_run(self.store, **kwargs)

    def visibility_checked(self) -> None:
        """End a visibility contract: skip as unverified unless an independent reader looked."""
        if not self.independent:
            pytest.skip(
                "unverified: live-update visibility on a second connection: no independent "
                f"reader is available ({SECOND_INSTANCE} is missing or cannot make one)"
            )

    async def settle(self) -> None:
        """Send whatever is pending and forget it: what follows starts from a clean record."""
        await self.updates.flush()
        self.updates.clear()

    def seen_when_announced(self, run_id: str) -> RunView | None:
        """What the reader saw of this run the first time it was announced; ``None`` if never."""
        for batch in self.updates.batches:
            if run_id in batch.ids:
                return batch.seen.get(run_id)
        return None

    async def events(self, run_id: str) -> int:
        return len(await self.reader.get_run_events(run_id))


async def _view(reader: Any, run_id: str) -> RunView:
    record = await reader.get_run(run_id)
    if record is None:
        return RunView(False, None, 0)
    return RunView(True, str(record.status), len(await reader.get_run_events(run_id)))


class Kit:
    """One fresh store, its declaration and helpers for the author's other fixtures."""

    def __init__(
        self,
        provided: dict[str, Any],
        store: Any,
        declaration: Declaration,
        *,
        needs_format_2: bool,
        guards: tuple[str, ...] = (),
    ) -> None:
        self._provided = provided
        self.store = store
        self.declaration = declaration
        self._needs_format_2 = needs_format_2
        self._guards = guards
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

    @asynccontextmanager
    async def live_store(self) -> AsyncIterator[LiveStore]:
        """A fresh store announcing into a recorder that probes a second connection.

        The recorder asks the reader what it sees of each announced run at the moment the
        announcement is sent: an announcement sent before its commit is visible is caught there.
        """
        factory = self._provided.get(LIVE_FACTORY)
        if factory is None:
            pytest.skip(f"{LIVE_FACTORY} not provided: live updates are not tested")
        updates = RecordingLiveUpdates()
        store = await maybe_await(factory(updates))
        reader, independent = store, False
        if self._provided.get(SECOND_INSTANCE) is not None:
            try:
                reader, independent = await self.second(store), True
            except Exception:
                if self.declaration.needs_second_instance:
                    raise  # this store promised a second instance; failing to make one is a failure
        updates.probe = lambda ids: _probe(reader, ids)
        try:
            yield LiveStore(store, updates, reader, self, independent)
        finally:
            try:
                closer = getattr(store, "close", None)
                if closer is not None:
                    await maybe_await(closer())
            finally:
                await updates.close()

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
        if self._needs_format_2 and created.storage_format != STORAGE_FORMAT_CHECKPOINTED:
            if self.declaration.creates_format_2:
                store_skip(
                    "it declares the four format-2 capabilities but creates storage-format-1 "
                    "runs, so the format-2 contracts cannot run (start the factory's schema "
                    "at format 2)",
                    *(self._guards or FORMAT_2_FLAGS),
                )
            pytest.skip("store creates storage format 1 here: format-2 contracts do not apply")
        return str(created.run_id)

    async def paused_run(
        self,
        store: Any | None = None,
        *,
        status: str = RunStatus.WAITING_CLIENT_TOOL.value,
        **kwargs: Any,
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
        claim = await target.claim_paused_run(
            run_id, expected_status=RunStatus.WAITING_CLIENT_TOOL.value
        )
        assert claim is not False, "a fresh pause must be claimable"
        return run_id, claim

    async def epoch(self, run_id: str, store: Any | None = None) -> int:
        record = await (store if store is not None else self.store).get_run(run_id)
        assert record is not None
        return int(record.claim_epoch)


async def _probe(reader: Any, ids: tuple[str, ...]) -> dict[str, RunView]:
    return {run_id: await _view(reader, run_id) for run_id in ids}


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


__all__ = [
    "FACTORY",
    "CertificationFailure",
    "Faulted",
    "Kit",
    "LiveStore",
    "check_declaration",
    "maybe_await",
    "store_skip",
]
