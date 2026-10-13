"""Refresh serialization, outcome classification, and delivery verdicts."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("uvicorn")

from dendrux.mcp import MCPSource
from dendrux.mcp._challenge import challenge_from_headers
from dendrux.mcp._credentials import MCPAuthKey
from dendrux.mcp._errors import MCPAuthorizationRequired, MCPOAuthError, MCPOAuthRefreshError
from dendrux.mcp._oauth import (
    MCPOAuth,
    MCPOAuthClaim,
    MCPOAuthClient,
    MCPOAuthMemoryStore,
    MCPOAuthRecord,
)
from dendrux.mcp._observability import (
    MCPOAuthRefreshed,
    MCPOAuthRefreshFailed,
    MCPRuntimeEvent,
)
from tests.unit.mcp_oauth_harness import ASConfig, OAuthHarness, free_port

CALLBACK = "http://127.0.0.1:9/callback"
KEY = MCPAuthKey("conn", tenant_key="user")
LATER = 3700.0


def make_oauth(store: MCPOAuthMemoryStore, **kwargs: Any) -> MCPOAuth:
    return MCPOAuth(
        store=store,
        client=MCPOAuthClient.dynamic(client_name="t", redirect_uris=[CALLBACK]),
        **kwargs,
    )


def shift(oauth: MCPOAuth, seconds: float) -> None:
    oauth._clock = lambda: time.time() + seconds


async def sign_in(harness: OAuthHarness, oauth: MCPOAuth, key: MCPAuthKey = KEY) -> MCPSource:
    src = MCPSource.http("demo", harness.rs_url)
    url = await oauth.begin_authorization(key, src, redirect_uri=CALLBACK)
    callback = await harness.browser(url)
    await oauth.complete_authorization(
        key=key, state=callback["state"], code=callback["code"], iss=callback["iss"]
    )
    return src


class Events:
    def __init__(self) -> None:
        self.events: list[MCPRuntimeEvent] = []

    def __call__(self, event: MCPRuntimeEvent) -> None:
        self.events.append(event)

    def outcomes(self) -> list[str]:
        return [e.outcome for e in self.events if isinstance(e, MCPOAuthRefreshFailed)]

    def refreshed(self) -> int:
        return sum(isinstance(e, MCPOAuthRefreshed) for e in self.events)


def provider(oauth: MCPOAuth, src: MCPSource, key: MCPAuthKey = KEY) -> Any:
    credentials = oauth.credentials(key, src)
    events = Events()
    credentials._emit = events
    credentials.events = events
    return credentials


async def row(store: MCPOAuthMemoryStore, key: MCPAuthKey = KEY) -> MCPOAuthRecord:
    record = await store.load(key)
    assert record is not None
    return record


async def required(awaitable: Any) -> MCPAuthorizationRequired:
    with pytest.raises(MCPAuthorizationRequired) as caught:
        await awaitable
    return caught.value


class BarrierStore(MCPOAuthMemoryStore):
    """Holds each client's first load until every client has loaded."""

    barrier: asyncio.Barrier | None = None
    armed = False

    async def load(self, key):  # type: ignore[no-untyped-def]
        record = await super().load(key)
        if self.armed and self.barrier is not None:
            self.armed = False
            await self.barrier.wait()
        return record


def barrier_clients(backend: MCPOAuthMemoryStore, count: int) -> list[BarrierStore]:
    barrier = asyncio.Barrier(count)
    clients = []
    for _ in range(count):
        client = BarrierStore()
        client._backend = backend._backend
        client.barrier = barrier
        clients.append(client)
    return clients


# -- serialization -------------------------------------------------------------


async def test_two_instances_one_refresh_post_and_the_loser_delivers_the_winner() -> None:
    async with OAuthHarness(config=ASConfig(refresh_latency=0.2)) as harness:
        backend = MCPOAuthMemoryStore()
        setup = make_oauth(backend.client())
        src = await sign_in(harness, setup)
        first_store, second_store = barrier_clients(backend, 2)
        first, second = make_oauth(first_store), make_oauth(second_store)
        shift(first, LATER)
        shift(second, LATER)
        first_store.armed = second_store.armed = True
        a, b = provider(first, src), provider(second, src)
        results = await asyncio.gather(a.get_auth(), b.get_auth())
        assert results[0] == results[1]
        assert harness.counters["refresh"] == 1
        assert harness.counters["refresh_redeemed"] == 1
        assert harness.counters["refresh_replayed"] == 0
        assert a.events.refreshed() + b.events.refreshed() == 1
        current = await row(backend)
        assert current.status == "active" and current.refresh_claim is None
        assert results[0]["Authorization"] == current.credential.authorization  # type: ignore[union-attr]
        for oauth in (setup, first, second):
            await oauth.aclose()


async def test_one_provider_coalesces_concurrent_refreshes() -> None:
    async with OAuthHarness(config=ASConfig(refresh_latency=0.1)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        results = await asyncio.gather(*(credentials.get_auth() for _ in range(3)))
        assert len({r["Authorization"] for r in results}) == 1
        assert harness.counters["refresh"] == 1
        await oauth.aclose()


async def test_consume_then_drop_retires_without_replay() -> None:
    async with OAuthHarness(config=ASConfig(consume_then_drop=True)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store, oauth_timeout=0.5, claim_ttl=61)
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        with pytest.raises(MCPOAuthRefreshError):
            await credentials.get_auth()
        retired = await row(store)
        assert retired.status == "retired"
        assert retired.credential.refresh_token is None  # type: ignore[union-attr]
        assert retired.retired_claim_id is not None
        assert credentials.events.outcomes() == ["uncertain"]
        assert (await required(credentials.get_auth())).reason == "expired"
        assert harness.counters["refresh_redeemed"] == 1
        assert harness.counters["refresh_replayed"] == 0
        assert harness.counters["refresh"] == 1
        await oauth.aclose()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_consumed_then_any_error_status_retires_without_replay(status: int) -> None:
    config = ASConfig(consume_then_status=status, reuse_detection=True)
    async with OAuthHarness(config=config) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        with pytest.raises(MCPOAuthRefreshError):
            await credentials.get_auth()
        assert (await row(store)).status == "retired"
        assert credentials.events.outcomes() == ["uncertain"]
        assert (await required(credentials.get_auth())).reason == "expired"
        assert harness.counters["refresh_redeemed"] == 1
        assert harness.counters["refresh_replayed"] == 0
        assert not harness.provider.revoked_families  # type: ignore[union-attr]
        await oauth.aclose()


async def test_caller_cancelled_during_post_does_not_cancel_the_refresh() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        backend = MCPOAuthMemoryStore()
        oauth = make_oauth(backend.client())
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        before = await row(backend)
        task = asyncio.create_task(provider(oauth, src).get_auth())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        await oauth.aclose()
        observer = make_oauth(backend.client())
        shift(observer, LATER)
        after = await row(backend.client())
        assert after.status == "active"
        assert after.credential.publication_id != before.credential.publication_id  # type: ignore[union-attr]
        header = await provider(observer, src).get_auth()
        assert header["Authorization"] == after.credential.authorization  # type: ignore[union-attr]
        assert harness.counters["refresh"] == 1
        await observer.aclose()


async def test_aclose_cancelling_the_post_retires_the_refresh_token() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        backend = MCPOAuthMemoryStore()
        oauth = make_oauth(backend.client(), oauth_timeout=0.3, claim_ttl=61)
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        oauth._http._timeout = 30.0  # the per-request deadline must not fire first
        task = asyncio.create_task(provider(oauth, src).get_auth())
        await entered.wait()
        await oauth.aclose()
        with pytest.raises(asyncio.CancelledError):
            await task
        retired = await row(backend.client())
        assert retired.status == "retired"
        assert retired.credential.refresh_token is None  # type: ignore[union-attr]
        gate.set()


async def test_unsent_refresh_keeps_the_refresh_token() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        good = await row(store)
        assert good.credential is not None
        dead = replace(
            good,
            version=replace(good.version, revision=good.version.revision + 1),
            credential=replace(
                good.credential, token_endpoint=f"http://127.0.0.1:{free_port()}/token"
            ),
        )
        assert await store.save(KEY, dead, expected=good.version)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        with pytest.raises(MCPOAuthRefreshError):
            await credentials.get_auth()
        kept = await row(store)
        assert kept.status == "active" and kept.refresh_claim is None
        assert kept.credential.refresh_token == good.credential.refresh_token  # type: ignore[union-attr]
        assert credentials.events.outcomes() == ["unsent"]
        restored = replace(
            kept,
            version=replace(kept.version, revision=kept.version.revision + 1),
            credential=replace(kept.credential, token_endpoint=good.credential.token_endpoint),  # type: ignore[arg-type]
        )
        assert await store.save(KEY, restored, expected=kept.version)
        await credentials.get_auth()
        assert harness.counters["refresh"] == 1
        await oauth.aclose()


async def test_expired_foreign_claim_is_retired_without_a_post() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        current = await row(store)
        stale = replace(
            current,
            version=replace(current.version, revision=1),
            refresh_claim=MCPOAuthClaim(id="ghost", expires_at=time.time() - 1),
        )
        assert await store.save(KEY, stale, expected=current.version)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        assert (await required(credentials.get_auth())).reason == "expired"
        retired = await row(store)
        assert retired.status == "retired" and retired.retired_claim_id == "ghost"
        assert credentials.events.outcomes() == ["expired_claim"]
        assert harness.counters["refresh"] == 0
        await oauth.aclose()


async def test_late_success_publishes_with_claim_provenance() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        backend = MCPOAuthMemoryStore()
        slow = make_oauth(backend.client())
        src = await sign_in(harness, slow)
        shift(slow, LATER)
        task = asyncio.create_task(provider(slow, src).get_auth())
        await entered.wait()
        claimed = await row(backend)
        assert claimed.refresh_claim is not None
        reader = make_oauth(backend.client())
        shift(reader, LATER + slow.claim_ttl + 5)
        reader_credentials = provider(reader, src)
        assert (await required(reader_credentials.get_auth())).reason == "expired"
        retired = await row(backend)
        assert retired.status == "retired"
        assert retired.retired_claim_id == claimed.refresh_claim.id
        gate.set()
        header = await task
        published = await row(backend)
        assert published.status == "active"
        assert header["Authorization"] == published.credential.authorization  # type: ignore[union-attr]
        assert harness.counters["refresh_redeemed"] == 1
        for oauth in (slow, reader):
            await oauth.aclose()


async def test_late_success_after_another_generation_is_discarded() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    async with OAuthHarness(config=ASConfig(refresh_gate=gate, refresh_entered=entered)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        shift(oauth, LATER)
        task = asyncio.create_task(provider(oauth, src).get_auth())
        await entered.wait()
        other = make_oauth(store.client())
        await other.revoke(KEY)
        gate.set()
        error = await required(task)
        assert error.reason == "revoked"
        assert (await row(store)).status == "signed_out"
        for instance in (oauth, other):
            await instance.aclose()


async def test_retired_row_is_never_claimed_while_waiting() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store, refresh_skew=600.0)
        src = await sign_in(harness, oauth)
        current = await row(store)
        claimed = replace(
            current,
            version=replace(current.version, revision=1),
            refresh_claim=MCPOAuthClaim(id="other", expires_at=time.time() + 3600),
        )
        assert await store.save(KEY, claimed, expected=current.version)
        # Inside the skew window and old enough to refresh proactively.
        shift(oauth, 3600 - 300)
        credentials = provider(oauth, src)
        task = asyncio.create_task(credentials.get_auth())
        await asyncio.sleep(0.25)
        retired = replace(
            claimed,
            version=replace(claimed.version, revision=2),
            status="retired",
            refresh_claim=None,
            retired_claim_id="other",
            credential=replace(claimed.credential, refresh_token=None),  # type: ignore[arg-type]
        )
        assert await store.save(KEY, retired, expected=claimed.version)
        header = await task
        assert header["Authorization"] == current.credential.authorization  # type: ignore[union-attr]
        assert harness.counters["refresh"] == 0
        shift(oauth, LATER)
        assert (await required(credentials.get_auth())).reason == "expired"
        assert harness.counters["refresh"] == 0
        await oauth.aclose()


async def test_contention_raises_a_transient_refresh_error() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store, oauth_timeout=0.3, claim_ttl=61)
        src = await sign_in(harness, oauth)
        current = await row(store)
        held = replace(
            current,
            version=replace(current.version, revision=1),
            refresh_claim=MCPOAuthClaim(id="holder", expires_at=time.time() + 50_000),
        )
        assert await store.save(KEY, held, expected=current.version)
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        with pytest.raises(MCPOAuthRefreshError):
            await credentials.get_auth()
        assert credentials.events.outcomes() == ["contention"]
        assert harness.counters["refresh"] == 0
        await oauth.aclose()


async def test_refresh_dedup_by_publication_when_the_access_token_string_is_kept() -> None:
    config = ASConfig(keep_access_token_string=True, refresh_latency=0.1)
    async with OAuthHarness(config=config) as harness:
        backend = MCPOAuthMemoryStore()
        setup = make_oauth(backend.client())
        src = await sign_in(harness, setup)
        first_store, second_store = barrier_clients(backend, 2)
        first, second = make_oauth(first_store), make_oauth(second_store)
        shift(first, LATER)
        shift(second, LATER)
        first_store.armed = second_store.armed = True
        results = await asyncio.gather(
            provider(first, src).get_auth(), provider(second, src).get_auth()
        )
        assert results[0] == results[1]
        assert harness.counters["refresh"] == 1
        for oauth in (setup, first, second):
            await oauth.aclose()


async def test_recently_published_credential_is_not_refreshed_again() -> None:
    async with OAuthHarness(config=ASConfig(access_token_lifetime=30)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store, refresh_skew=60.0)
        src = await sign_in(harness, oauth)
        credentials = provider(oauth, src)
        first = await credentials.get_auth()
        second = await credentials.get_auth()
        assert first == second
        assert harness.counters["refresh"] == 0
        shift(oauth, 61)
        await credentials.get_auth()
        assert harness.counters["refresh"] == 1
        await oauth.aclose()


async def test_refresh_without_rotation_keeps_the_previous_refresh_token() -> None:
    async with OAuthHarness(config=ASConfig(rotate_refresh_tokens=False)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        before = await row(store)
        shift(oauth, LATER)
        await provider(oauth, src).get_auth()
        after = await row(store)
        assert after.status == "active"
        assert after.credential.refresh_token == before.credential.refresh_token  # type: ignore[union-attr]
        await oauth.aclose()


async def test_rejected_refresh_retires_and_reports_revoked() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        harness.provider.revoked_families.update(harness.provider.families.values())  # type: ignore[union-attr]
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        error = await required(credentials.get_auth())
        assert (error.reason, error.status_code) == ("revoked", 400)
        assert (await row(store)).status == "retired"
        assert credentials.events.outcomes() == ["rejected"]
        await oauth.aclose()


@pytest.mark.parametrize(
    ("status", "body", "outcome"),
    [
        (400, b'{"error": "invalid_target"}', "uncertain"),
        (400, b"not json", "uncertain"),
        (401, b'{"error": "invalid_client"}', "rejected"),
        (403, b'{"error": "invalid_grant"}', "uncertain"),
        (200, b"{}", "uncertain"),
        (200, b"garbage", "uncertain"),
        (201, b'{"access_token": "x"}', "uncertain"),
        (302, b"", "uncertain"),
    ],
)
async def test_refresh_outcome_table(status: int, body: bytes, outcome: str) -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        harness.config.token_status_override = status
        harness.config.token_body_override = body
        shift(oauth, LATER)
        credentials = provider(oauth, src)
        expected = MCPAuthorizationRequired if outcome == "rejected" else MCPOAuthRefreshError
        with pytest.raises(expected):
            await credentials.get_auth()
        assert credentials.events.outcomes() == [outcome]
        assert (await row(store)).status == "retired"
        await oauth.aclose()


async def test_expired_client_secret_at_refresh_is_expired() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        current = await row(store)
        assert current.credential is not None
        client = replace(
            current.credential.client,
            client_secret="s",
            client_secret_expires_at=time.time() + 10,
            token_endpoint_auth_method="client_secret_post",
        )
        edited = replace(
            current,
            version=replace(current.version, revision=1),
            credential=replace(current.credential, client=client),
        )
        assert await store.save(KEY, edited, expected=current.version)
        shift(oauth, LATER)
        assert (await required(provider(oauth, src).get_auth())).reason == "expired"
        assert (await row(store)).status == "retired"
        assert harness.counters["refresh"] == 0
        await oauth.aclose()


# -- verdicts ------------------------------------------------------------------------


async def test_not_signed_in_and_signed_out() -> None:
    store = MCPOAuthMemoryStore()
    oauth = make_oauth(store)
    credentials = oauth.credentials(KEY, MCPSource.http("d", "https://rs.example/mcp"))
    assert (await required(credentials.get_auth())).reason == "not_signed_in"
    await oauth.revoke(KEY)
    assert (await required(credentials.get_auth())).reason == "not_signed_in"
    await oauth.aclose()


async def test_resource_bound_delivery_runs_before_any_status_decision() -> None:
    async with OAuthHarness(config=ASConfig(issue_refresh_tokens=False)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        await sign_in(harness, oauth)
        assert (await row(store)).status == "retired"
        other = MCPSource.http("demo", harness.rs_url + "?tenant=B")
        assert (
            await required(oauth.credentials(KEY, other).get_auth())
        ).reason == "resource_mismatch"
        await oauth.aclose()


async def test_issuer_mismatch_and_missing_credential_are_never_delivered() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        current = await row(store)
        assert current.credential is not None
        forged = replace(
            current,
            version=replace(current.version, revision=1),
            credential=replace(
                current.credential,
                client=replace(current.credential.client, issuer="https://other.example"),
            ),
        )
        assert await store.save(KEY, forged, expected=current.version)
        credentials = provider(oauth, src)
        assert (await required(credentials.get_auth())).reason == "resource_mismatch"
        malformed = replace(forged, version=replace(forged.version, revision=2), credential=None)
        assert await store.save(KEY, malformed, expected=forged.version)
        assert (await required(credentials.get_auth())).reason == "resource_mismatch"
        await oauth.aclose()


async def test_refresh_loser_receiving_another_resource_gets_resource_mismatch() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store, oauth_timeout=2.0, claim_ttl=63)
        src = await sign_in(harness, oauth)
        current = await row(store)
        held = replace(
            current,
            version=replace(current.version, revision=1),
            refresh_claim=MCPOAuthClaim(id="winner", expires_at=time.time() + 50_000),
        )
        assert await store.save(KEY, held, expected=current.version)
        shift(oauth, LATER)
        task = asyncio.create_task(provider(oauth, src).get_auth())
        await asyncio.sleep(0.25)
        assert held.credential is not None
        moved = replace(
            held,
            version=replace(held.version, generation="new-generation", revision=0),
            refresh_claim=None,
            credential=replace(held.credential, resource=harness.rs_url + "?tenant=B"),
        )
        assert await store.save(KEY, moved, expected=held.version)
        assert (await required(task)).reason == "resource_mismatch"
        await oauth.aclose()


async def test_expired_access_token_is_never_delivered() -> None:
    async with OAuthHarness(config=ASConfig(issue_refresh_tokens=False)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        credentials = provider(oauth, src)
        assert "Authorization" in await credentials.get_auth()
        shift(oauth, LATER)
        assert (await required(credentials.get_auth())).reason == "expired"
        await oauth.aclose()


# -- rejection handling ---------------------------------------------------------------


async def test_handle_rejection_paths() -> None:
    async with OAuthHarness() as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        credentials = provider(oauth, src)
        header = (await credentials.get_auth())["Authorization"]

        await credentials.handle_rejection(401, None, "Bearer stale-value")
        assert harness.counters["refresh"] == 0

        challenge = challenge_from_headers(
            [
                'Bearer error="insufficient_scope", scope="admin extra", resource_metadata="https://x.example/m"'
            ]
        )
        error = await required(credentials.handle_rejection(403, challenge, header))
        assert error.reason == "insufficient_scope"
        assert error.scopes == ("admin", "extra")
        assert error.resource_metadata_url == "https://x.example/m"
        assert (await required(credentials.handle_rejection(403, None, header))).reason == "revoked"

        await credentials.handle_rejection(401, None, header)
        assert harness.counters["refresh"] == 1
        fresh = (await credentials.get_auth())["Authorization"]
        assert fresh != header

        await credentials.on_auth_rejected()
        assert harness.counters["refresh"] == 2

        await oauth.revoke(KEY)
        assert (await required(credentials.handle_rejection(401, None, fresh))).reason == "revoked"
        await oauth.aclose()


async def test_401_on_a_retired_credential_is_expired() -> None:
    async with OAuthHarness(config=ASConfig(issue_refresh_tokens=False)) as harness:
        store = MCPOAuthMemoryStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        credentials = provider(oauth, src)
        header = (await credentials.get_auth())["Authorization"]
        assert (await required(credentials.handle_rejection(401, None, header))).reason == "expired"
        await oauth.aclose()


class FailingCleanupStore(MCPOAuthMemoryStore):
    fail_retire = False

    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        if self.fail_retire and record.status == "retired":
            raise RuntimeError("SENTINEL-cleanup-failure")
        return await super().save(key, record, expected=expected)


async def test_failed_cleanup_write_is_logged_with_the_outcome_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with OAuthHarness() as harness:
        store = FailingCleanupStore()
        oauth = make_oauth(store)
        src = await sign_in(harness, oauth)
        harness.config.token_status_override = 500
        store.fail_retire = True
        shift(oauth, LATER)
        with caplog.at_level(logging.WARNING), pytest.raises(MCPOAuthRefreshError):
            await provider(oauth, src).get_auth()
        messages = [r.getMessage() for r in caplog.records if r.name.startswith("dendrux")]
        assert any("uncertain" in message for message in messages)
        assert all("SENTINEL" not in message for message in messages)
        # The claim survives the failed retire; the next reader reconciles it.
        assert (await row(store)).refresh_claim is not None
        await oauth.aclose()


async def test_closed_oauth_rejects_provider_operations() -> None:
    store = MCPOAuthMemoryStore()
    oauth = make_oauth(store)
    credentials = oauth.credentials(KEY, MCPSource.http("d", "https://rs.example/mcp"))
    await oauth.aclose()
    with pytest.raises(MCPOAuthError):
        await credentials.get_auth()
    assert repr(credentials) == "MCPOAuthCredentials()"
    with pytest.raises(ValueError):
        oauth.credentials(KEY, MCPSource.stdio("d", ["x"]))
    with pytest.raises(ValueError):
        oauth.credentials(KEY, MCPSource.http("d", "https://rs.example/a/../b"))
