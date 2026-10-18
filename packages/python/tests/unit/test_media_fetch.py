"""SSRF-hardened URL fetching (U-10, S-02): no real network; stub resolver + network backend.

A local TLS server on 127.0.0.1 is used once to prove certificate/SNI checks
run against the original host name while the socket goes to the pinned IP.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import ipaddress
import ssl
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import httpcore
import pytest

from dendrux.errors import ImageFetchError
from dendrux.media import UrlPolicy
from dendrux.media._fetch import fetch_url, host_allowed, is_blocked_address

if TYPE_CHECKING:
    from pathlib import Path

PUBLIC = "93.184.216.34"
PUBLIC_2 = "151.101.1.1"
BODY = b"\x89PNG fake image body"

Responder = Callable[[str, dict[str, str]], Any]


def http_response(
    body: bytes = BODY,
    *,
    status: str = "200 OK",
    headers: dict[str, str] | None = None,
    content_length: bool = True,
) -> bytes:
    head = f"HTTP/1.1 {status}\r\nConnection: close\r\n"
    if content_length:
        head += f"Content-Length: {len(body)}\r\n"
    for k, v in (headers or {}).items():
        head += f"{k}: {v}\r\n"
    return (head + "\r\n").encode() + body


def chunked(body: bytes, size: int = 1024) -> bytes:
    out = b""
    for i in range(0, len(body), size):
        part = body[i : i + size]
        out += f"{len(part):x}\r\n".encode() + part + b"\r\n"
    return out + b"0\r\n\r\n"


class FakeStream(httpcore.AsyncNetworkStream):
    def __init__(self, backend: FakeBackend, ip: str) -> None:
        self.backend = backend
        self.ip = ip
        self.sent = b""
        self.pending: list[tuple[float, bytes]] | None = None
        self.sni: str | None = None

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.sent += buffer

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self.pending is None:
            head = self.sent.split(b"\r\n\r\n", 1)[0].decode()
            lines = head.split("\r\n")
            target = lines[0].split(" ")[1]
            headers = {
                k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:])
            }
            self.backend.requests.append(
                {"ip": self.ip, "sni": self.sni, "target": target, "headers": headers}
            )
            result = self.backend.responder(target, headers)
            self.pending = [(0.0, result)] if isinstance(result, bytes) else list(result)
        if not self.pending:
            return b""
        delay, data = self.pending[0]
        if delay:
            await asyncio.sleep(delay)
        chunk, rest = data[:max_bytes], data[max_bytes:]
        if rest:
            self.pending[0] = (0.0, rest)
        else:
            self.pending.pop(0)
        return chunk

    async def aclose(self) -> None:
        return None

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.sni = server_hostname
        self.backend.tls.append((server_hostname, ssl_context))
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class FakeBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, responder: Responder | None = None) -> None:
        self.responder: Responder = responder or (lambda target, headers: http_response())
        self.connects: list[tuple[str, int]] = []
        self.requests: list[dict[str, Any]] = []
        self.tls: list[tuple[str | None, ssl.SSLContext]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connects.append((host, port))
        return FakeStream(self, host)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def resolver_for(table: dict[str, list[str]]) -> Any:
    calls: list[str] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        if host not in table:
            raise OSError("no such host")
        return list(table[host])

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


async def fetch(
    url: str,
    *,
    backend: FakeBackend | None = None,
    table: dict[str, list[str]] | None = None,
    policy: UrlPolicy | None = None,
    max_bytes: int = 1024 * 1024,
    resolver: Any = None,
) -> bytes:
    return await fetch_url(
        url,
        policy=policy or UrlPolicy(),
        max_bytes=max_bytes,
        resolver=resolver or resolver_for(table or {"img.example.com": [PUBLIC]}),
        network_backend=backend or FakeBackend(),
    )


async def fetch_error(url: str, **kw: Any) -> ImageFetchError:
    with pytest.raises(ImageFetchError) as ei:
        await fetch(url, **kw)
    err = ei.value
    assert err.__cause__ is None and err.__context__ is None
    return err


# ---------------------------------------------------------------------------
# Happy path, pinning, headers
# ---------------------------------------------------------------------------


async def test_fetch_pins_validated_ip_and_keeps_host_for_tls() -> None:
    backend = FakeBackend()
    data = await fetch("https://img.example.com/a.png?sig=SECRET", backend=backend)
    assert data == BODY
    assert backend.connects == [(PUBLIC, 443)]
    assert backend.tls[0][0] == "img.example.com"
    ctx = backend.tls[0][1]
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True
    req = backend.requests[0]
    assert req["headers"]["host"] == "img.example.com"
    assert req["headers"]["accept-encoding"] == "identity"
    assert req["target"] == "/a.png?sig=SECRET"


async def test_disabled() -> None:
    err = await fetch_error("https://img.example.com/x", policy=UrlPolicy(fetch=False))
    assert err.reason == "disabled"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://img.example.com/x", "scheme"),
        ("ftp://img.example.com/x", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("not a url", "scheme"),
        ("https://img.example.com:8443/x", "port"),
        ("https://user:hunter2@img.example.com/x", "credentials"),
        ("https:///nohost", "host_not_allowed"),
    ],
)
async def test_url_checks(url: str, reason: str) -> None:
    err = await fetch_error(url)
    assert err.reason == reason
    assert "hunter2" not in str(err) and "/x" not in str(err) and "passwd" not in str(err)


async def test_http_when_allowed() -> None:
    backend = FakeBackend()
    policy = UrlPolicy(allowed_schemes=("http", "https"), allowed_ports=(80, 443))
    assert await fetch("http://img.example.com/x", backend=backend, policy=policy) == BODY
    assert backend.tls == [] and backend.connects == [(PUBLIC, 80)]


async def test_allowed_hosts() -> None:
    table = {"a.cdn.test": [PUBLIC], "cdn.test": [PUBLIC], "img.example.com": [PUBLIC]}
    policy = UrlPolicy(allowed_hosts=("*.cdn.test", "IMG.example.com."))
    assert await fetch("https://a.cdn.test/x", table=table, policy=policy) == BODY
    assert await fetch("https://img.example.com/x", table=table, policy=policy) == BODY
    err = await fetch_error("https://cdn.test/x", table=table, policy=policy)
    assert err.reason == "host_not_allowed" and err.host == "cdn.test"
    assert host_allowed("x.y.cdn.test", ("*.cdn.test",))
    assert not host_allowed("evilcdn.test", ("*.cdn.test",))


# ---------------------------------------------------------------------------
# Address validation (IPv4 / IPv6 / metadata / DNS)
# ---------------------------------------------------------------------------

BLOCKED_V4 = [
    "127.0.0.1",
    "10.1.2.3",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "224.0.0.1",
    "255.255.255.255",
    "192.0.2.10",
    "198.18.0.1",
    "240.0.0.1",
]
BLOCKED_V6 = [
    "::1",
    "::",
    "fe80::1",
    "fc00::1",
    "fd12:3456::1",
    "fec0::1",
    "ff02::1",
    "::ffff:127.0.0.1",
    "::ffff:8.8.8.8",
    "::7f00:1",
    "64:ff9b::7f00:1",
    "64:ff9b::808:808",
    "64:ff9b:1::1",
    "2002:7f00:1::1",
    "2002:a00:1::1",
    "2001:db8::1",
    "2001:0::1",
]


@pytest.mark.parametrize("ip", BLOCKED_V4 + BLOCKED_V6)
async def test_blocked_dns_answers(ip: str) -> None:
    backend = FakeBackend()
    err = await fetch_error(
        "https://img.example.com/x", backend=backend, table={"img.example.com": [ip]}
    )
    assert err.reason == "blocked_address" and err.host == "img.example.com"
    assert backend.connects == []
    assert is_blocked_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize("ip", ["8.8.8.8", PUBLIC, "2606:4700::1111", "2002:808:808::1"])
def test_public_addresses_allowed(ip: str) -> None:
    assert not is_blocked_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://[::ffff:127.0.0.1]/x",
        "https://[::ffff:7f00:1]/x",
        "https://169.254.169.254/latest/meta-data",
        "https://2130706433/x",
        "https://0x7f.1/x",
        "https://[fe80::1]/x",
    ],
)
async def test_literal_ip_urls_blocked_without_dns(url: str) -> None:
    resolver = resolver_for({})
    backend = FakeBackend()
    err = await fetch_error(url, backend=backend, resolver=resolver)
    assert err.reason == "blocked_address"
    assert resolver.calls == [] and backend.connects == []


async def test_octal_ipv4_refused_as_unparseable() -> None:
    # httpx refuses this form outright; it never reaches resolution or connect.
    resolver = resolver_for({})
    backend = FakeBackend()
    err = await fetch_error("https://0177.0.0.1/x", backend=backend, resolver=resolver)
    assert err.reason in ("scheme", "blocked_address")
    assert resolver.calls == [] and backend.connects == []


async def test_literal_public_ip_connects_directly() -> None:
    backend = FakeBackend()
    resolver = resolver_for({})
    assert await fetch(f"https://{PUBLIC}/x", backend=backend, resolver=resolver) == BODY
    assert backend.connects == [(PUBLIC, 443)] and resolver.calls == []


@pytest.mark.parametrize(
    "host", ["metadata.google.internal", "METADATA.google.internal.", "metadata", "instance-data"]
)
async def test_metadata_hostnames(host: str) -> None:
    resolver = resolver_for({host.lower().rstrip("."): [PUBLIC]})
    err = await fetch_error(f"https://{host}/computeMetadata/v1/", resolver=resolver)
    assert err.reason == "blocked_address" and resolver.calls == []


async def test_mixed_public_and_private_answers_rejected() -> None:
    backend = FakeBackend()
    err = await fetch_error(
        "https://img.example.com/x",
        backend=backend,
        table={"img.example.com": [PUBLIC, "10.0.0.5"]},
    )
    assert err.reason == "blocked_address" and backend.connects == []


async def test_dns_failures() -> None:
    assert (await fetch_error("https://nx.example.com/x")).reason == "dns"
    empty = await fetch_error("https://img.example.com/x", table={"img.example.com": []})
    assert empty.reason == "dns"
    junk = await fetch_error("https://img.example.com/x", table={"img.example.com": ["junk"]})
    assert junk.reason == "dns"


async def test_dns_rebinding_uses_the_validated_ip() -> None:
    answers = [[PUBLIC], ["127.0.0.1"]]
    calls: list[str] = []

    async def rebinding(host: str, port: int) -> list[str]:
        calls.append(host)
        return answers[min(len(calls) - 1, 1)]

    backend = FakeBackend()
    assert await fetch("https://img.example.com/x", backend=backend, resolver=rebinding) == BODY
    assert calls == ["img.example.com"]
    assert backend.connects == [(PUBLIC, 443)]


async def test_allow_private_networks() -> None:
    backend = FakeBackend()
    policy = UrlPolicy(allow_private_networks=True)
    data = await fetch(
        "https://intranet.local/x",
        backend=backend,
        table={"intranet.local": ["10.0.0.7"]},
        policy=policy,
    )
    assert data == BODY and backend.connects == [("10.0.0.7", 443)]


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


def redirect(location: str | None, status: str = "302 Found") -> bytes:
    headers = {"Location": location} if location is not None else {}
    return http_response(b"", status=status, headers=headers)


async def test_redirect_followed_and_each_hop_resolved() -> None:
    def responder(target: str, headers: dict[str, str]) -> bytes:
        if headers["host"] == "img.example.com":
            return redirect("https://cdn.example.net/final.png")
        if target == "/final.png":
            return redirect("/again", status="301 Moved Permanently")
        return http_response()

    backend = FakeBackend(responder)
    table = {"img.example.com": [PUBLIC], "cdn.example.net": [PUBLIC_2]}
    assert await fetch("https://img.example.com/x", backend=backend, table=table) == BODY
    assert [c[0] for c in backend.connects] == [PUBLIC, PUBLIC_2, PUBLIC_2]
    assert [r["target"] for r in backend.requests] == ["/x", "/final.png", "/again"]


@pytest.mark.parametrize(
    ("location", "table", "reason"),
    [
        (
            "https://internal.example.com/",
            {"internal.example.com": ["10.0.0.1"]},
            "blocked_address",
        ),
        ("https://169.254.169.254/latest/", {}, "blocked_address"),
        ("https://[::1]/", {}, "blocked_address"),
        ("http://img2.example.com/", {"img2.example.com": [PUBLIC]}, "scheme"),
        ("https://u:p@img2.example.com/", {"img2.example.com": [PUBLIC]}, "credentials"),
        ("https://img2.example.com:22/", {"img2.example.com": [PUBLIC]}, "port"),
        ("https://metadata.google.internal/", {}, "blocked_address"),
    ],
)
async def test_redirect_hop_rechecked(
    location: str, table: dict[str, list[str]], reason: str
) -> None:
    backend = FakeBackend(lambda target, headers: redirect(location))
    err = await fetch_error(
        "https://img.example.com/x",
        backend=backend,
        table={"img.example.com": [PUBLIC], **table},
    )
    assert err.reason == reason
    assert len(backend.requests) == 1


async def test_too_many_redirects() -> None:
    backend = FakeBackend(lambda target, headers: redirect("/loop"))
    err = await fetch_error(
        "https://img.example.com/x", backend=backend, policy=UrlPolicy(max_redirects=2)
    )
    assert err.reason == "redirect" and len(backend.requests) == 3
    none = await fetch_error(
        "https://img.example.com/x",
        backend=FakeBackend(lambda t, h: redirect("/a")),
        policy=UrlPolicy(max_redirects=0),
    )
    assert none.reason == "redirect"


async def test_redirect_without_location() -> None:
    err = await fetch_error(
        "https://img.example.com/x", backend=FakeBackend(lambda t, h: redirect(None))
    )
    assert err.reason == "http_status" and err.status_code == 302


# ---------------------------------------------------------------------------
# Response handling: status, encoding, size caps, timeouts
# ---------------------------------------------------------------------------


async def test_http_status() -> None:
    backend = FakeBackend(lambda t, h: http_response(b"nope", status="404 Not Found"))
    err = await fetch_error("https://img.example.com/x?token=abc", backend=backend)
    assert (err.reason, err.status_code, err.host) == ("http_status", 404, "img.example.com")
    assert "token" not in str(err)


async def test_content_encoding_rejected() -> None:
    backend = FakeBackend(lambda t, h: http_response(headers={"Content-Encoding": "gzip"}))
    assert (await fetch_error("https://img.example.com/x", backend=backend)).reason == (
        "content_encoding"
    )
    ok = FakeBackend(lambda t, h: http_response(headers={"Content-Encoding": "identity"}))
    assert await fetch("https://img.example.com/x", backend=ok) == BODY


async def test_declared_length_over_cap_rejected_early() -> None:
    backend = FakeBackend(lambda t, h: http_response(b"x" * 100))
    err = await fetch_error("https://img.example.com/x", backend=backend, max_bytes=50)
    assert err.reason == "too_large"


async def test_chunked_body_over_cap() -> None:
    body = b"x" * 10_000

    def responder(t: str, h: dict[str, str]) -> bytes:
        return http_response(
            chunked(body), headers={"Transfer-Encoding": "chunked"}, content_length=False
        )

    err = await fetch_error(
        "https://img.example.com/x", backend=FakeBackend(responder), max_bytes=4096
    )
    assert err.reason == "too_large"
    ok = await fetch("https://img.example.com/x", backend=FakeBackend(responder), max_bytes=10_000)
    assert ok == body


async def test_unbounded_body_without_length_capped() -> None:
    def responder(t: str, h: dict[str, str]) -> list[tuple[float, bytes]]:
        head = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n"
        return [(0.0, head)] + [(0.0, b"y" * 1000)] * 50

    err = await fetch_error(
        "https://img.example.com/x", backend=FakeBackend(responder), max_bytes=5000
    )
    assert err.reason == "too_large"


async def test_false_content_length_never_trusted_over_cap() -> None:
    # Declares 10 bytes but streams more: only the declared bytes are read (h11
    # framing), so the result is the truncated body, never more than the cap.
    def responder(t: str, h: dict[str, str]) -> bytes:
        return b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n" + b"z" * 5000

    data = await fetch("https://img.example.com/x", backend=FakeBackend(responder), max_bytes=20)
    assert data == b"z" * 10
    # Declares more than it sends: a framing error, reported without detail.
    short = FakeBackend(
        lambda t, h: b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nabc"
    )
    assert (await fetch_error("https://img.example.com/x", backend=short)).reason == "connect"


async def test_total_timeout() -> None:
    def slow(t: str, h: dict[str, str]) -> list[tuple[float, bytes]]:
        return [(0.0, b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n"), (0.2, b"abcd")] * 1

    policy = UrlPolicy(total_timeout=0.05, read_timeout=5.0)
    err = await fetch_error("https://img.example.com/x", backend=FakeBackend(slow), policy=policy)
    assert err.reason == "timeout"


async def test_read_timeout() -> None:
    class TimeoutStream(FakeStream):
        async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            raise httpcore.ReadTimeout("slow")

    class TimeoutBackend(FakeBackend):
        async def connect_tcp(self, host: str, port: int, *a: Any, **k: Any) -> Any:
            return TimeoutStream(self, host)

    assert (await fetch_error("https://img.example.com/x", backend=TimeoutBackend())).reason == (
        "timeout"
    )


async def test_connect_failure_tries_next_address_then_reports_connect() -> None:
    class Refusing(FakeBackend):
        async def connect_tcp(self, host: str, port: int, *a: Any, **k: Any) -> Any:
            self.connects.append((host, port))
            if host == PUBLIC:
                raise httpcore.ConnectError("refused")
            return FakeStream(self, host)

    backend = Refusing()
    table = {"img.example.com": [PUBLIC, PUBLIC_2]}
    assert await fetch("https://img.example.com/x", backend=backend, table=table) == BODY
    assert [c[0] for c in backend.connects] == [PUBLIC, PUBLIC_2]
    only = Refusing()
    err = await fetch_error("https://img.example.com/x", backend=only)
    assert err.reason == "connect"


# ---------------------------------------------------------------------------
# Environment is ignored; app fetcher
# ---------------------------------------------------------------------------


async def test_proxy_environment_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    hits: list[bytes] = []

    async def capture(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hits.append(await reader.read(100))
        writer.close()

    server = await asyncio.start_server(capture, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    proxy = f"http://127.0.0.1:{port}"
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ca.pem")
    try:
        backend = FakeBackend()
        assert await fetch("https://img.example.com/x", backend=backend) == BODY
        await asyncio.sleep(0.05)
    finally:
        server.close()
        await server.wait_closed()
    assert hits == []
    assert backend.connects == [(PUBLIC, 443)]
    assert "proxy-authorization" not in backend.requests[0]["headers"]


async def test_app_fetcher() -> None:
    seen: list[tuple[str, int]] = []

    async def fetcher(url: str, max_bytes: int) -> bytes:
        seen.append((url, max_bytes))
        return b"ok" * 5

    policy = UrlPolicy(fetcher=fetcher)
    assert await fetch_url("http://anything/x", policy=policy, max_bytes=100) == b"ok" * 5
    assert seen == [("http://anything/x", 100)]
    with pytest.raises(ImageFetchError) as ei:
        await fetch_url("https://h.example/x?q=1", policy=policy, max_bytes=3)
    assert ei.value.reason == "too_large" and ei.value.host == "h.example"
    with pytest.raises(ImageFetchError, match="disabled"):
        await fetch_url("https://h/x", policy=UrlPolicy(fetch=False, fetcher=fetcher), max_bytes=3)

    async def bad(url: str, max_bytes: int) -> Any:
        return "text"

    with pytest.raises(TypeError):
        await fetch_url("https://h/x", policy=UrlPolicy(fetcher=bad), max_bytes=10)


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        UrlPolicy(allowed_schemes=("http",), allowed_ports=(443,))
    with pytest.raises(ValueError):
        UrlPolicy(allowed_schemes=("gopher",))
    with pytest.raises(ValueError):
        UrlPolicy(allowed_ports=(0,))
    with pytest.raises(ValueError):
        UrlPolicy(allowed_hosts=("",))
    with pytest.raises(ValueError):
        UrlPolicy(allowed_hosts="example.com")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        UrlPolicy(max_redirects=-1)
    with pytest.raises(ValueError):
        UrlPolicy(total_timeout=0)
    assert UrlPolicy(allowed_schemes=["HTTPS"]).allowed_schemes == ("https",)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real TLS on loopback: SNI + certificate checked against the original host
# ---------------------------------------------------------------------------


def _make_certs(tmp: Path) -> tuple[Path, Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dendrux test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "good.test")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("good.test")]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_path, cert_path, key_path = tmp / "ca.pem", tmp / "leaf.pem", tmp / "leaf.key"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path


async def test_tls_certificate_checked_against_original_host(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    ca_path, cert_path, key_path = _make_certs(tmp_path)
    server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_ctx.load_cert_chain(cert_path, key_path)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(http_response())
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_ctx)
    port = server.sockets[0].getsockname()[1]
    client_ctx = ssl.create_default_context(cafile=str(ca_path))
    policy = UrlPolicy(allowed_ports=(port,), allow_private_networks=True)
    resolve = resolver_for({"good.test": ["127.0.0.1"], "other.test": ["127.0.0.1"]})
    try:
        data = await fetch_url(
            f"https://good.test:{port}/a.png",
            policy=policy,
            max_bytes=1000,
            resolver=resolve,
            ssl_context=client_ctx,
        )
        assert data == BODY
        with pytest.raises(ImageFetchError) as ei:
            await fetch_url(
                f"https://other.test:{port}/a.png",
                policy=policy,
                max_bytes=1000,
                resolver=resolve,
                ssl_context=client_ctx,
            )
        assert ei.value.reason == "connect" and ei.value.host == "other.test"
        assert ei.value.__context__ is None
    finally:
        server.close()
        await server.wait_closed()
