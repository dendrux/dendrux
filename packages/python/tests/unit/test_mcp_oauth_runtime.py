"""OAuth bindings on the managed runtime, over real SDK servers."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

pytest.importorskip("uvicorn")

from dendrux.mcp import (
    MCPAuthenticationError,
    MCPAuthKey,
    MCPAuthorizationRequired,
    MCPBindingConflictError,
    MCPCircuitOpened,
    MCPConnectionAborted,
    MCPConnectionFailed,
    MCPConnectionOpened,
    MCPConnectionOpening,
    MCPEvictionStarted,
    MCPOAuth,
    MCPOAuthClient,
    MCPOAuthFlowError,
    MCPOAuthMemoryStore,
    MCPOAuthRefreshed,
    MCPOAuthRefreshError,
    MCPOAuthRequired,
    MCPOAuthRevoked,
    MCPRuntime,
    MCPRuntimeEvent,
    MCPRuntimeShutdownCompleted,
    MCPSource,
    MCPStaleConnectionError,
)
from dendrux.mcp._runtime import _ViewToolSource
from tests.unit.mcp_oauth_harness import ASConfig, OAuthHarness, basic_header, free_port
from tests.unit.test_mcp_oauth_refresh import barrier_clients

CALLBACK = "http://127.0.0.1:9/callback"
CONNECTION = "demo-conn"
TENANT = "user-a"
KEY = MCPAuthKey(CONNECTION, tenant_key=TENANT)
LATER = 3700.0


class Recorder:
    def __init__(self) -> None:
        self.events: list[MCPRuntimeEvent] = []

    def on_event(self, event: MCPRuntimeEvent) -> None:
        self.events.append(event)

    def of(self, kind: type) -> list[Any]:
        return [event for event in self.events if isinstance(event, kind)]


def make_oauth(store: MCPOAuthMemoryStore, **kwargs: Any) -> MCPOAuth:
    kwargs.setdefault("client", MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK]))
    return MCPOAuth(store=store, **kwargs)


def shift(oauth: MCPOAuth, seconds: float) -> None:
    oauth._clock = lambda: time.time() + seconds


def source(harness: OAuthHarness, **kwargs: Any) -> MCPSource:
    return MCPSource.http("demo", harness.rs_url, **kwargs)


async def sign_in(
    harness: OAuthHarness, oauth: MCPOAuth, key: MCPAuthKey = KEY, **begin: Any
) -> None:
    url = await oauth.begin_authorization(key, source(harness), redirect_uri=CALLBACK, **begin)
    callback = await harness.browser(url)
    await oauth.complete_authorization(
        key=key, state=callback["state"], code=callback["code"], iss=callback["iss"]
    )


def bind(runtime: MCPRuntime, harness: OAuthHarness, oauth: MCPOAuth, **kwargs: Any):  # type: ignore[no-untyped-def]
    return runtime.bind(
        connection_key=CONNECTION, tenant_key=TENANT, source=source(harness, **kwargs), oauth=oauth
    )


async def executor_for(connection, tool: str = "echo"):  # type: ignore[no-untyped-def]
    view = _ViewToolSource(connection.tools())
    await view._discover()
    return view, view._create_executor(tool)


async def required(awaitable: Any) -> MCPAuthorizationRequired:
    with pytest.raises(MCPAuthorizationRequired) as caught:
        await awaitable
    return caught.value


# -- acceptance --------------------------------------------------------------------


async def test_two_runtimes_share_one_refresh_redemption() -> None:
    async with OAuthHarness(config=ASConfig(refresh_latency=0.2)) as harness:
        backend = MCPOAuthMemoryStore()
        setup = make_oauth(backend.client())
        await sign_in(harness, setup)
        first_store, second_store = barrier_clients(backend, 2)
        first, second = make_oauth(first_store), make_oauth(second_store)
        shift(first, LATER)
        shift(second, LATER)
        first_store.armed = second_store.armed = True
        async with MCPRuntime() as runtime_a, MCPRuntime() as runtime_b:
            conn_a = bind(runtime_a, harness, first)
            conn_b = bind(runtime_b, harness, second)
            discoveries = await asyncio.gather(conn_a.discover(), conn_b.discover())
            assert all(d.tools for d in discoveries)
            assert harness.counters["refresh"] == 1
            assert harness.counters["refresh_replayed"] == 0
            view, echo = await executor_for(conn_b)
            assert await echo(text="hi") == {"result": "hi"}
            await view.close()
        for oauth in (setup, first, second):
            await oauth.aclose()


async def test_fresh_runtime_refreshes_a_stored_expired_token() -> None:
    async with OAuthHarness() as harness:
        backend = MCPOAuthMemoryStore()
        setup = make_oauth(backend.client())
        await sign_in(harness, setup)
        authorizations = harness.counters["authorize"]
        worker = make_oauth(backend.client())
        shift(worker, LATER)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            await bind(runtime, harness, worker).discover()
        assert harness.counters["refresh"] == 1
        assert harness.counters["authorize"] == authorizations
        assert len(recorder.of(MCPOAuthRefreshed)) == 1
        for oauth in (setup, worker):
            await oauth.aclose()


async def test_step_up_flow_and_refresh_authorization() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            connection = bind(runtime, harness, oauth)
            view, admin = await executor_for(connection, "admin_echo")
            error = await required(admin(text="x"))
            assert (error.reason, error.scopes, error.status_code) == (
                "insufficient_scope",
                ("admin",),
                403,
            )
            assert error.key == KEY
            assert recorder.of(MCPOAuthRequired)[-1].scope_count == 1
            status = await oauth.status(KEY)
            union = [*(status.scope or "").split(), *error.scopes]
            await sign_in(harness, oauth, scopes=union)
            opened_before = len(recorder.of(MCPConnectionOpened))
            await runtime.refresh_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            assert await admin(text="x") == {"result": "admin:x"}
            assert len(recorder.of(MCPConnectionOpened)) == opened_before + 1
            await view.close()
        await oauth.aclose()


async def test_preregistered_basic_secret_on_the_managed_refresh_path() -> None:
    async with OAuthHarness() as harness:
        harness.preregister("pre", "pre-secret", "client_secret_basic", redirect_uris=[CALLBACK])
        client = MCPOAuthClient.preregistered(
            "pre",
            "pre-secret",
            issuer=harness.as_url,
            token_endpoint_auth_method="client_secret_basic",
        )
        oauth = make_oauth(MCPOAuthMemoryStore(), client=client)
        await sign_in(harness, oauth)
        shift(oauth, LATER)
        async with MCPRuntime() as runtime:
            await bind(runtime, harness, oauth).discover()
        assert harness.counters["refresh"] == 1
        headers = harness.provider.token_request_headers  # type: ignore[union-attr]
        assert headers and all(
            h["authorization"] == basic_header("pre", "pre-secret") for h in headers
        )
        await oauth.aclose()


async def test_preregistered_basic_secret_on_the_unmanaged_sdk_provider_path() -> None:
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )

    from dendrux.mcp._client import MCPClientAdapter

    async with OAuthHarness() as harness:
        harness.preregister("pre", "pre-secret", "client_secret_basic", redirect_uris=[CALLBACK])

        class Storage:
            def __init__(self) -> None:
                self.tokens: OAuthToken | None = None
                self.client = OAuthClientInformationFull(
                    client_id="pre",
                    client_secret="pre-secret",
                    token_endpoint_auth_method="client_secret_basic",
                    redirect_uris=[CALLBACK],  # type: ignore[list-item]
                    scope="mcp",
                )

            async def get_tokens(self) -> OAuthToken | None:
                return self.tokens

            async def set_tokens(self, tokens: OAuthToken) -> None:
                self.tokens = tokens

            async def get_client_info(self) -> OAuthClientInformationFull | None:
                return self.client

            async def set_client_info(self, info: OAuthClientInformationFull) -> None:
                self.client = info

        pending: dict[str, str | None] = {}

        async def redirect(url: str) -> None:
            pending.update(await harness.browser(url))

        async def callback() -> AuthorizationCodeResult:
            return AuthorizationCodeResult(
                code=pending["code"] or "", state=pending["state"], iss=pending["iss"]
            )

        provider = OAuthClientProvider(
            server_url=harness.rs_url,
            client_metadata=OAuthClientMetadata(redirect_uris=[CALLBACK]),  # type: ignore[list-item]
            storage=Storage(),
            redirect_handler=redirect,
            callback_handler=callback,
        )
        adapter = MCPClientAdapter(MCPSource.http("sdk", harness.rs_url, auth=provider))
        await adapter.connect()
        try:
            assert [tool.name for tool in await adapter.list_tools()][:1] == ["echo"]
        finally:
            await adapter.close()
        headers = harness.provider.token_request_headers  # type: ignore[union-attr]
        assert headers[-1]["authorization"] == basic_header("pre", "pre-secret")


# -- binding rules -----------------------------------------------------------------


async def test_bind_validation() -> None:
    oauth = make_oauth(MCPOAuthMemoryStore())
    runtime = MCPRuntime()
    for headers in ({"Authorization": "Bearer x"}, {"authorization": "x"}, {"AUTHORIZATION": "x"}):
        with pytest.raises(ValueError):
            runtime.bind(
                connection_key="c",
                source=MCPSource.http("d", "https://rs.example/mcp", headers=headers),
                oauth=oauth,
            )

    class Provider:
        async def get_auth(self) -> dict[str, str]:
            return {}

    with pytest.raises(ValueError):
        runtime.bind(
            connection_key="c",
            source=MCPSource.http("d", "https://rs.example/mcp"),
            credentials=Provider(),
            oauth=oauth,
        )
    with pytest.raises(ValueError):
        runtime.bind(connection_key="c", source=MCPSource.stdio("d", ["x"]), oauth=oauth)
    with pytest.raises(ValueError):
        runtime.bind(
            connection_key="c", source=MCPSource.http("d", "https://rs.example/a/../b"), oauth=oauth
        )
    with pytest.raises(ValueError):
        runtime.bind(
            connection_key="c",
            source=MCPSource.http("d", "https://rs.example/mcp", auth=("u", "p")),
            oauth=oauth,
        )
    with pytest.raises(ValueError):
        runtime.bind(
            connection_key="c",
            source=MCPSource.http("d", "https://rs.example/mcp"),
            oauth=object(),  # type: ignore[arg-type]
        )
    # reauthenticate_on_401 needs no application provider for OAuth bindings.
    runtime.bind(
        connection_key="c",
        source=MCPSource.http("d", "https://rs.example/mcp", reauthenticate_on_401=True),
        oauth=oauth,
    )
    await runtime.close()
    await oauth.aclose()


async def test_equivalent_bind_keeps_the_provider_and_other_oauth_conflicts_while_live() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        other = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        async with MCPRuntime() as runtime:
            first = bind(runtime, harness, oauth)
            provider = runtime._registrations[(TENANT, CONNECTION)].credentials
            second = bind(runtime, harness, oauth)
            assert runtime._registrations[(TENANT, CONNECTION)].credentials is provider
            assert first.generation == second.generation
            await second.discover()
            with pytest.raises(MCPBindingConflictError):
                bind(runtime, harness, other)
            await runtime.evict(connection_key=CONNECTION, tenant_key=TENANT)
            bind(runtime, harness, oauth)
            replaced = bind(runtime, harness, other)
            assert runtime._registrations[(TENANT, CONNECTION)].credentials is not provider
            assert replaced.generation != second.generation
            rebound = await runtime.rebind(
                connection_key=CONNECTION, tenant_key=TENANT, source=source(harness), oauth=oauth
            )
            assert rebound.generation != replaced.generation
        for instance in (oauth, other):
            await instance.aclose()


async def test_exactly_one_authorization_header_on_the_wire() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            view, echo = await executor_for(connection)
            await echo(text="hi")
            await view.close()
        wire = [headers for headers in harness.rs_auth_headers if headers]
        assert wire and all(len(headers) == 1 for headers in wire)
        assert all(headers[0].startswith("Bearer at-") for headers in wire)
        await oauth.aclose()


async def test_rotate_credentials_refuses_oauth_bindings() -> None:
    oauth = make_oauth(MCPOAuthMemoryStore())

    class Provider:
        async def get_auth(self) -> dict[str, str]:
            return {"Authorization": "Bearer x"}

    async with MCPRuntime() as runtime:
        runtime.bind(
            connection_key="c", source=MCPSource.http("d", "https://rs.example/mcp"), oauth=oauth
        )
        with pytest.raises(ValueError):
            await runtime.rotate_credentials(connection_key="c", credentials=Provider())
        runtime.bind(connection_key="p", source=MCPSource.http("d", "https://rs.example/mcp"))
        with pytest.raises(MCPStaleConnectionError):
            await runtime.refresh_authorization(connection_key="p")
        with pytest.raises(MCPStaleConnectionError):
            await runtime.revoke_authorization(connection_key="p")
        with pytest.raises(MCPStaleConnectionError):
            await runtime.revoke_authorization(connection_key="missing")
        # No live transport: refreshing is a no-op that keeps the handle valid.
        await runtime.refresh_authorization(connection_key="c")
    await oauth.aclose()


# -- recovery ------------------------------------------------------------------------


async def test_call_path_refresh_then_401_on_reconnect_uses_one_callback() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            connection = bind(runtime, harness, oauth)
            view, echo = await executor_for(connection)
            harness.reject_all_tokens = True
            error = await required(echo(text="hi"))
            assert (error.reason, error.status_code) == ("revoked", 401)
            assert harness.counters["refresh"] == 1
            assert not recorder.of(MCPCircuitOpened)
            await view.close()
        await oauth.aclose()


async def test_call_path_401_refreshes_and_retries_once() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            view, echo = await executor_for(connection)
            harness.provider.expire_access_tokens()  # type: ignore[union-attr]
            assert await echo(text="again") == {"result": "again"}
            assert harness.counters["refresh"] == 1
            await view.close()
        await oauth.aclose()


async def test_discovery_time_403_is_translated() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        harness.forbid_list_tools = True
        recorder = Recorder()
        async with MCPRuntime(observer=recorder, circuit_failure_threshold=1) as runtime:
            connection = bind(runtime, harness, oauth)
            error = await required(connection.discover())
            assert (error.reason, error.scopes) == ("insufficient_scope", ("admin",))
            assert not recorder.of(MCPCircuitOpened)
            assert len(recorder.of(MCPOAuthRequired)) == 1
            assert recorder.of(MCPConnectionFailed)[-1].cause == "MCPAuthorizationRequired"
        await oauth.aclose()


async def test_step_up_elsewhere_lets_the_original_worker_recover() -> None:
    async with OAuthHarness() as harness:
        backend = MCPOAuthMemoryStore()
        worker = make_oauth(backend.client())
        await sign_in(harness, worker)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder, circuit_failure_threshold=1) as runtime:
            connection = bind(runtime, harness, worker)
            view, admin = await executor_for(connection, "admin_echo")
            elsewhere = make_oauth(backend.client())
            await sign_in(harness, elsewhere, scopes=["mcp", "admin"])
            assert await admin(text="x") == {"result": "admin:x"}
            entry = runtime._entries[(TENANT, CONNECTION)]
            assert not entry.refresh_failed
            assert not recorder.of(MCPCircuitOpened)
            assert not recorder.of(MCPOAuthRequired)
            await view.close()
            await elsewhere.aclose()
        await worker.aclose()


async def test_sign_in_verdicts_never_open_the_circuit_and_recover_after_sign_in() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        recorder = Recorder()
        async with MCPRuntime(observer=recorder, circuit_failure_threshold=1) as runtime:
            connection = bind(runtime, harness, oauth)
            for _ in range(3):
                assert (await required(connection.discover())).reason == "not_signed_in"
            assert not recorder.of(MCPCircuitOpened)
            assert len(recorder.of(MCPOAuthRequired)) == 3
            await sign_in(harness, oauth)
            assert (await connection.discover()).tools
        await oauth.aclose()


async def test_refresh_error_never_marks_refresh_failed_and_the_next_rejection_recovers() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder, circuit_failure_threshold=1) as runtime:
            connection = bind(runtime, harness, oauth)
            view, echo = await executor_for(connection)
            row = await store.load(KEY)
            assert row is not None and row.credential is not None
            from dataclasses import replace

            dead = replace(
                row,
                version=replace(row.version, revision=row.version.revision + 1),
                credential=replace(
                    row.credential, token_endpoint=f"http://127.0.0.1:{free_port()}/token"
                ),
            )
            assert await store.save(KEY, dead, expected=row.version)
            harness.provider.expire_access_tokens()  # type: ignore[union-attr]
            with pytest.raises(MCPOAuthRefreshError):
                await echo(text="x")
            entry = runtime._entries[(TENANT, CONNECTION)]
            assert not entry.refresh_failed
            assert not recorder.of(MCPCircuitOpened)
            current = await store.load(KEY)
            assert current is not None and current.credential is not None
            restored = replace(
                current,
                version=replace(current.version, revision=current.version.revision + 1),
                credential=replace(
                    current.credential, token_endpoint=row.credential.token_endpoint
                ),
            )
            assert await store.save(KEY, restored, expected=current.version)
            assert await echo(text="y") == {"result": "y"}
            await view.close()
        await oauth.aclose()


# -- lifecycle -----------------------------------------------------------------------


async def test_cancelled_waiters_never_cancel_the_refresh() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        shift(oauth, LATER)
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            waiters = [asyncio.create_task(connection.discover()) for _ in range(3)]
            await entered.wait()
            waiters[0].cancel()
            gate.set()
            results = await asyncio.gather(*waiters, return_exceptions=True)
            assert isinstance(results[0], asyncio.CancelledError)
            assert all(not isinstance(r, BaseException) for r in results[1:])
            assert harness.counters["refresh"] == 1
        await oauth.aclose()


async def test_all_waiters_cancelled_refresh_still_publishes() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        before = await store.load(KEY)
        shift(oauth, LATER)
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            waiters = [asyncio.create_task(connection.discover()) for _ in range(2)]
            await entered.wait()
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            gate.set()
        await oauth.aclose()
        after = await store.load(KEY)
        assert after is not None and after.status == "active"
        assert after.credential.publication_id != before.credential.publication_id  # type: ignore[union-attr]


@pytest.mark.parametrize("action", ["evict", "rebind", "close"])
async def test_teardown_during_refresh(action: str) -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        shift(oauth, LATER)
        recorder = Recorder()
        runtime = MCPRuntime(observer=recorder, shutdown_timeout=1.0)
        connection = bind(runtime, harness, oauth)
        task = asyncio.create_task(connection.discover())
        await entered.wait()
        if action == "evict":
            await runtime.evict(connection_key=CONNECTION, tenant_key=TENANT, mode="force")
        elif action == "rebind":
            await runtime.rebind(
                connection_key=CONNECTION,
                tenant_key=TENANT,
                source=source(harness),
                oauth=oauth,
                mode="force",
            )
        else:
            await runtime.close()
        with pytest.raises(BaseException):  # noqa: B017 - the specific teardown error varies
            await task
        gate.set()
        await oauth.aclose()
        row = await store.load(KEY)
        assert row is not None and row.status == "active"
        if action != "close":
            await runtime.close()
        openings = recorder.of(MCPConnectionOpening)
        terminals = [
            e
            for e in recorder.events
            if isinstance(e, (MCPConnectionOpened, MCPConnectionFailed, MCPConnectionAborted))
        ]
        assert len(openings) == len(terminals) == 1
        if action == "close":
            shutdown = recorder.of(MCPRuntimeShutdownCompleted)
            assert shutdown and recorder.events[-1] is shutdown[-1]
            assert not recorder.of(MCPOAuthRefreshed)


async def test_one_oauth_required_event_for_three_waiters() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            connection = bind(runtime, harness, oauth)
            results = await asyncio.gather(
                *(connection.discover() for _ in range(3)), return_exceptions=True
            )
            assert all(isinstance(r, MCPAuthorizationRequired) for r in results)
            assert len(recorder.of(MCPOAuthRequired)) == 1
        await oauth.aclose()


# -- revocation ------------------------------------------------------------------------


async def test_revoke_authorization_signs_out_and_evicts() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            connection = bind(runtime, harness, oauth)
            await connection.discover()
            revocation = await runtime.revoke_authorization(
                connection_key=CONNECTION, tenant_key=TENANT
            )
            assert (revocation.local, revocation.remote) == ("signed_out", "confirmed")
            assert [e.remote for e in recorder.of(MCPOAuthRevoked)] == ["confirmed"]
            assert (TENANT, CONNECTION) not in runtime._registrations
            with pytest.raises(MCPStaleConnectionError):
                await connection.discover()
            fresh = bind(runtime, harness, oauth)
            assert (await required(fresh.discover())).reason == "not_signed_in"
        await oauth.aclose()


async def test_revoke_with_no_live_entry() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        async with MCPRuntime() as runtime:
            bind(runtime, harness, oauth)
            revocation = await runtime.revoke_authorization(
                connection_key=CONNECTION, tenant_key=TENANT
            )
            assert revocation.local == "already_signed_out"
            assert (TENANT, CONNECTION) not in runtime._registrations
        await oauth.aclose()


async def test_concurrent_revokes_write_own_tombstones_and_share_one_eviction() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        harness.revocation_latency = 0.2
        recorder = Recorder()
        async with MCPRuntime(observer=recorder) as runtime:
            connection = bind(runtime, harness, oauth)
            await connection.discover()
            before = await store.load(KEY)
            first = asyncio.create_task(
                runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            )
            await asyncio.sleep(0.05)
            flow_url = await oauth.begin_authorization(KEY, source(harness), redirect_uri=CALLBACK)
            second = asyncio.create_task(
                runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            )
            results = await asyncio.gather(first, second)
            assert {r.local for r in results} == {"signed_out", "already_signed_out"}
            assert len(recorder.of(MCPEvictionStarted)) == 1
            assert len(recorder.of(MCPOAuthRevoked)) == 2
            after = await store.load(KEY)
            assert after is not None and before is not None
            assert after.version.generation != before.version.generation
            callback = await harness.browser(flow_url)
            with pytest.raises(MCPOAuthFlowError) as caught:
                await oauth.complete_authorization(
                    key=KEY, state=callback["state"], code=callback["code"], iss=callback["iss"]
                )
            assert caught.value.reason == "superseded"
        await oauth.aclose()


async def test_revoke_fences_new_leases_while_pending() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        harness.revocation_latency = 0.3
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            await connection.discover()
            view, echo = await executor_for(connection)
            task = asyncio.create_task(
                runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            )
            await asyncio.sleep(0.05)
            assert (await required(connection.discover())).reason == "revoked"
            assert (await required(echo(text="x"))).reason == "revoked"
            await view.close()
            await task
        await oauth.aclose()


@pytest.mark.parametrize("when", ["before_store_write", "during_remote_post", "between_phases"])
async def test_cancelled_revoke_caller_never_cancels_the_revoke(when: str) -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        harness.revocation_latency = 0.3 if when == "during_remote_post" else 0.0
        async with MCPRuntime(shutdown_timeout=2.0) as runtime:
            connection = bind(runtime, harness, oauth)
            await connection.discover()
            lease_view, _ = await executor_for(connection)
            task = asyncio.create_task(
                runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            )
            if when == "during_remote_post":
                await asyncio.sleep(0.1)
            elif when == "between_phases":
                # Phase 2 drains the lease held by this view.
                while (await store.load(KEY)).status != "signed_out":  # type: ignore[union-attr]
                    await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await lease_view.close()
            for _ in range(200):
                if not runtime._revoke_tasks and (TENANT, CONNECTION) not in runtime._registrations:
                    break
                await asyncio.sleep(0.01)
            assert not runtime._revoke_tasks
            assert (TENANT, CONNECTION) not in runtime._registrations
            assert (await store.load(KEY)).status == "signed_out"  # type: ignore[union-attr]
        assert harness.counters["revoke"] == 1
        await oauth.aclose()


async def test_close_awaits_a_pending_revoke_without_deadlock() -> None:
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        harness.revocation_latency = 0.3
        recorder = Recorder()
        runtime = MCPRuntime(observer=recorder, shutdown_timeout=1.0)
        connection = bind(runtime, harness, oauth)
        await connection.discover()
        task = asyncio.create_task(
            runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
        )
        await asyncio.sleep(0.05)
        await asyncio.wait_for(runtime.close(), timeout=5.0)
        assert task.done()
        assert task.result().remote == "confirmed"
        assert harness.counters["revoke"] == 1
        await oauth.aclose()


async def test_revoke_after_rebind_evicts_only_the_captured_registration() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        harness.revocation_latency = 0.3
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            await connection.discover()
            task = asyncio.create_task(
                runtime.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            )
            await asyncio.sleep(0.05)
            rebound = await runtime.rebind(
                connection_key=CONNECTION,
                tenant_key=TENANT,
                source=source(harness, call_timeout=60.0),
                oauth=oauth,
            )
            await task
            registration = runtime._registrations[(TENANT, CONNECTION)]
            assert registration.generation == rebound.generation
            assert not registration.revoking
            second = await runtime.revoke_authorization(
                connection_key=CONNECTION, tenant_key=TENANT
            )
            assert second.local == "already_signed_out"
            assert (TENANT, CONNECTION) not in runtime._registrations
        await oauth.aclose()


async def test_second_worker_works_until_its_next_401_then_revoked() -> None:
    async with OAuthHarness() as harness:
        backend = MCPOAuthMemoryStore()
        first = make_oauth(backend.client())
        second = make_oauth(backend.client())
        await sign_in(harness, first)
        async with MCPRuntime() as runtime_a, MCPRuntime() as runtime_b:
            conn_a = bind(runtime_a, harness, first)
            conn_b = bind(runtime_b, harness, second)
            await conn_a.discover()
            view, echo = await executor_for(conn_b)
            assert await echo(text="before") == {"result": "before"}
            await runtime_a.revoke_authorization(connection_key=CONNECTION, tenant_key=TENANT)
            error = await required(echo(text="after"))
            assert error.reason == "revoked"
            await view.close()
        for oauth in (first, second):
            await oauth.aclose()


async def test_remote_revocation_after_a_newer_sign_in_invalidates_it() -> None:
    """Documented: a late grant-wide revocation can kill a newer grant."""
    async with OAuthHarness() as harness:
        oauth = make_oauth(MCPOAuthMemoryStore())
        await sign_in(harness, oauth)
        old_families = set(harness.provider.families.values())  # type: ignore[union-attr]
        await sign_in(harness, oauth)
        new_families = set(harness.provider.families.values()) - old_families  # type: ignore[union-attr]
        harness.provider.revoked_families.update(new_families)  # type: ignore[union-attr]
        async with MCPRuntime() as runtime:
            connection = bind(runtime, harness, oauth)
            error = await required(connection.discover())
            assert error.reason == "revoked"
        await oauth.aclose()


async def test_non_oauth_401_stays_an_authentication_error() -> None:
    async with OAuthHarness() as harness, MCPRuntime() as runtime:
        connection = runtime.bind(
            connection_key="plain",
            source=MCPSource.http("demo", harness.rs_url, headers={"Authorization": "Bearer x"}),
        )
        with pytest.raises(MCPAuthenticationError) as caught:
            await connection.discover()
        assert not isinstance(caught.value, MCPAuthorizationRequired)
