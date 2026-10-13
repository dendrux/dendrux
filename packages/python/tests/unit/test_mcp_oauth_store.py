"""Record model, reference store, and the public store conformance check."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from dendrux.mcp._credentials import MCPAuthKey
from dendrux.mcp._oauth import (
    MCPOAuthClientRecord,
    MCPOAuthCredential,
    MCPOAuthFlow,
    MCPOAuthMemoryStore,
    MCPOAuthRecord,
    MCPOAuthStatus,
    MCPOAuthStore,
    MCPOAuthVersion,
    check_oauth_store,
)

SECRET = "SENTINEL-store-91c2"


def _client() -> MCPOAuthClientRecord:
    return MCPOAuthClientRecord(
        client_id="client",
        client_secret=SECRET,
        token_endpoint_auth_method="client_secret_post",
        source="dcr",
        issuer="https://as.example",
    )


def _record(generation: str = "g", revision: int = 0) -> MCPOAuthRecord:
    return MCPOAuthRecord(
        version=MCPOAuthVersion(generation, revision),
        status="active",
        credential=MCPOAuthCredential(
            access_token=SECRET,
            refresh_token=SECRET,
            expires_at=time.time() + 60,
            scope="mcp",
            issuer="https://as.example",
            resource="https://rs.example/mcp",
            token_endpoint="https://as.example/token",
            revocation_endpoint=None,
            client=_client(),
            publication_id="p",
            published_at=time.time(),
        ),
        refresh_claim=None,
        retired_claim_id=None,
        updated_at=time.time(),
    )


def test_reprs_never_show_secrets() -> None:
    record = _record()
    flow = MCPOAuthFlow(
        state=SECRET,
        key=MCPAuthKey("c"),
        resource="https://rs.example/mcp",
        code_verifier=SECRET,
        redirect_uri="https://app.example/cb",
        scope="mcp",
        issuer="https://as.example",
        authorization_endpoint="https://as.example/authorize",
        token_endpoint="https://as.example/token",
        revocation_endpoint=None,
        client=_client(),
        iss_required=False,
        expires_at=1.0,
        previous_generation=None,
    )
    status = MCPOAuthStatus(
        state="active",
        scope=SECRET,
        expires_at=1.0,
        issuer=f"https://{SECRET}.example",
        resource=f"https://{SECRET}.example",
    )
    for rendered in (repr(record), str(record), repr(flow), repr(status), repr(_client())):
        assert SECRET not in rendered
    assert repr(status) == "MCPOAuthStatus(state='active')"
    assert record.credential is not None
    assert record.credential.authorization == f"Bearer {SECRET}"


async def test_memory_store_passes_the_conformance_check() -> None:
    store = MCPOAuthMemoryStore()
    await check_oauth_store(store.client)
    assert isinstance(store, MCPOAuthStore)


async def test_memory_store_exact_version_semantics() -> None:
    store = MCPOAuthMemoryStore()
    key = MCPAuthKey("c", "t")
    first = _record()
    assert await store.save(key, first, expected=None)
    assert not await store.save(key, first, expected=None)
    second = replace(first, version=MCPOAuthVersion("g", 1))
    assert not await store.save(key, second, expected=MCPOAuthVersion("g", 5))
    assert not await store.save(key, second, expected=MCPOAuthVersion("h", 0))
    assert await store.client().save(key, second, expected=MCPOAuthVersion("g", 0))
    assert await store.load(key) == second
    assert await store.load(MCPAuthKey("c")) is None


class _InsertOverwrites(MCPOAuthMemoryStore):
    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        self._backend.rows[key] = record
        return True


class _IgnoresRevision(MCPOAuthMemoryStore):
    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        current = self._backend.rows.get(key)
        if expected is None:
            if current is not None:
                return False
        elif current is None or current.version.generation != expected.generation:
            return False
        self._backend.rows[key] = record
        return True


class _TakeIsNotOneShot(MCPOAuthMemoryStore):
    async def take_flow(self, state):  # type: ignore[no-untyped-def]
        return self._backend.flows.get(state)


class _ConcatenatedKeys(MCPOAuthMemoryStore):
    @staticmethod
    def _flat(key: MCPAuthKey) -> MCPAuthKey:
        return MCPAuthKey(f"{key.tenant_key}:{key.connection_key}")

    async def load(self, key):  # type: ignore[no-untyped-def]
        return await super().load(self._flat(key))

    async def save(self, key, record, *, expected):  # type: ignore[no-untyped-def]
        return await super().save(self._flat(key), record, expected=expected)


@pytest.mark.parametrize(
    ("store_type", "check"),
    [
        (_InsertOverwrites, "exactly one concurrent insert wins"),
        (_IgnoresRevision, "exactly one concurrent CAS wins"),
        (_TakeIsNotOneShot, "exactly one take_flow wins"),
        (_ConcatenatedKeys, "isolated"),
    ],
)
async def test_conformance_catches_broken_semantics(
    store_type: type[MCPOAuthMemoryStore], check: str
) -> None:
    backend = store_type()

    def factory() -> MCPOAuthMemoryStore:
        client = store_type()
        client._backend = backend._backend
        return client

    with pytest.raises(AssertionError) as caught:
        await check_oauth_store(factory)
    assert check in str(caught.value)


async def test_conformance_requires_a_shared_backend() -> None:
    with pytest.raises(AssertionError) as caught:
        await check_oauth_store(MCPOAuthMemoryStore)
    assert "read-after-write" in str(caught.value) or "insert" in str(caught.value)


async def test_conformance_requires_new_clients() -> None:
    store = MCPOAuthMemoryStore()
    with pytest.raises(AssertionError):
        await check_oauth_store(lambda: store)
