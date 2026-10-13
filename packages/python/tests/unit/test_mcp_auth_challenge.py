"""``WWW-Authenticate`` challenge capture on credential rejections.

The parsed challenge rides on ``MCPAuthenticationError.challenge`` at every
raise site (connect, discovery, tool call) and never enters the message.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from mcp.server.mcpserver import MCPServer as SDKServer

from dendrux.mcp import MCPAuthChallenge, MCPAuthenticationError, MCPSource
from dendrux.mcp._challenge import challenge_from_headers, parse_auth_challenge
from dendrux.mcp._client import MCPClientAdapter
from tests.unit.test_mcp_streamable_http_auth import _serve

SOURCE_URL = "https://mcp.example.com/mcp"
FULL_HEADER = (
    'Bearer realm="mcp", error="insufficient_scope", '
    'error_description="needs HEADER-DESCRIPTION", scope="files:read files:write", '
    'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
)


def _response(
    status: int,
    header: str | None,
    *,
    url: str = SOURCE_URL,
    body: dict[str, Any] | None = None,
) -> httpx2.Response:
    headers = {"www-authenticate": header} if header is not None else {}
    request = httpx2.Request("POST", url, json=body or {"method": "initialize"})
    return httpx2.Response(status, headers=headers, request=request)


def _rendered(error: BaseException) -> str:
    parts: list[str] = [repr(error), error.transport_detail or ""]
    node: BaseException | None = error
    while node is not None:
        parts.append(str(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse(*values: str) -> MCPAuthChallenge:
    return challenge_from_headers(values)


def test_full_challenge_is_parsed_without_free_text() -> None:
    challenge = parse_auth_challenge(_response(403, FULL_HEADER))
    assert challenge == MCPAuthChallenge(
        scheme="Bearer",
        error="insufficient_scope",
        scope=("files:read", "files:write"),
        resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
    )
    assert not hasattr(challenge, "error_description")
    assert "HEADER-DESCRIPTION" not in repr(challenge)


def test_unquoted_fields_and_bare_scheme() -> None:
    challenge = _parse("Bearer error=invalid_token, scope=read")
    assert challenge.scheme == "Bearer"
    assert challenge.error == "invalid_token"
    assert challenge.scope == ("read",)
    assert challenge.resource_metadata_url is None
    assert _parse("Bearer") == MCPAuthChallenge(scheme="Bearer")
    assert _parse("bearer error=invalid_token").error == "invalid_token"


def test_bearer_is_selected_among_several_challenges_in_one_header() -> None:
    challenge = _parse('Basic realm="x", Bearer error="insufficient_scope", scope="a b"')
    assert challenge == MCPAuthChallenge(
        scheme="Bearer", error="insufficient_scope", scope=("a", "b")
    )
    # Parameters of a preceding challenge never leak into the Bearer one.
    assert _parse('Basic error="from_basic", Bearer').error is None


def test_bearer_is_selected_across_several_headers() -> None:
    response = httpx2.Response(
        401,
        headers=[
            ("www-authenticate", 'Basic realm="x", error="from_basic"'),
            ("www-authenticate", 'Bearer error="invalid_token", scope="read"'),
        ],
        request=httpx2.Request("POST", SOURCE_URL),
    )
    challenge = parse_auth_challenge(response)
    assert challenge == MCPAuthChallenge(scheme="Bearer", error="invalid_token", scope=("read",))


def test_without_bearer_only_the_first_scheme_is_kept() -> None:
    assert _parse('Basic realm="x", error="nope"') == MCPAuthChallenge(scheme="Basic")
    assert _parse('Basic realm="x"', 'Digest realm="y"') == MCPAuthChallenge(scheme="Basic")


def test_parameter_names_inside_quoted_values_are_not_parameters() -> None:
    challenge = _parse('Bearer realm="x scope=admin error=bad", error="invalid_token"')
    assert challenge.scope == ()
    assert challenge.error == "invalid_token"


def test_quoted_string_unescaping() -> None:
    challenge = _parse(r'Bearer realm="a \"quoted\" b", scope="a\:b c", error="invalid\_token"')
    assert challenge.scope == ("a:b", "c")
    assert challenge.error == "invalid_token"


def test_first_occurrence_of_a_parameter_wins() -> None:
    assert _parse("Bearer error=invalid_token, error=insufficient_scope").error == "invalid_token"


@pytest.mark.parametrize(
    "header",
    [
        "Bearer error=invalid_token\U0001f4a5",
        'Bearer error="invalid_token"garbage',
        "Bearer error=invalid_token/extra",
        'Bearer error="invalid_token"=',
    ],
)
def test_error_with_a_malformed_suffix_is_discarded(header: str) -> None:
    assert _parse(header).error is None


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer scope=read/write", ()),
        ('Bearer scope="read"write', ()),
        ("Bearer scope=read\U0001f4a5, error=invalid_token", ()),
    ],
)
def test_scope_with_a_malformed_suffix_is_discarded(header: str, expected: tuple[str, ...]) -> None:
    challenge = _parse(header)
    assert challenge.scope == expected


def test_malformed_parameter_does_not_swallow_the_next_one() -> None:
    challenge = _parse("Bearer scope=read/write, error=invalid_token")
    assert challenge.scope == ()
    assert challenge.error == "invalid_token"


@pytest.mark.parametrize(
    "header",
    [
        "Bearer error=invalid_token",
        "Bearer error=invalid_token ",
        "Bearer error=invalid_token\t, scope=a",
        'Bearer error="invalid_token" , scope=a',
    ],
)
def test_parameter_followed_by_end_space_or_comma_is_kept(header: str) -> None:
    assert _parse(header).error == "invalid_token"


@pytest.mark.parametrize(
    "error",
    [
        "invalid_token\r\nX-Injected: 1",
        "<script>alert(1)</script>",
        "invalid token",
        "a" * 65,
        "",
        "inválido",
    ],
)
def test_error_with_unexpected_characters_is_dropped(error: str) -> None:
    value = error.replace("\\", "\\\\").replace('"', '\\"')
    assert _parse(f'Bearer error="{value}"').error is None


def test_scope_items_are_filtered_individually_and_bounded() -> None:
    long_item = "x" * 129
    raw = f'read "quoted" back\\slash {long_item} write  café ok:1'
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    challenge = _parse(f'Bearer scope="{escaped}"')
    assert challenge.scope == ("read", "write", "ok:1")
    many = " ".join(f"s{i}" for i in range(40))
    assert _parse(f'Bearer scope="{many}"').scope == tuple(f"s{i}" for i in range(32))
    assert _parse('Bearer scope=""').scope == ()


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.com/" + "a" * 3000,
        "javascript:alert(1)",
        "https://user:pass@mcp.example.com/prm",
        "https://token@mcp.example.com/prm",
        "https://mcp.example.com/prm#frag",
        "ftp://mcp.example.com/prm",
        "/relative/prm",
        "https:///nohost",
        "https://mcp.example.com:99999/prm",
        "https://mcp.example.com/a b",
        "https://mcp.exämple.com/prm",
    ],
)
def test_unsafe_resource_metadata_url_is_dropped(url: str) -> None:
    challenge = _parse(f'Bearer error="invalid_token", resource_metadata="{url}"')
    assert challenge.resource_metadata_url is None
    assert challenge.error == "invalid_token"


def test_safe_resource_metadata_url_is_kept() -> None:
    url = "http://127.0.0.1:8080/.well-known/oauth-protected-resource/mcp?x=1"
    assert _parse(f'Bearer resource_metadata="{url}"').resource_metadata_url == url


@pytest.mark.parametrize("scheme", ["B" * 33, "Bear!er"])
def test_scheme_is_validated(scheme: str) -> None:
    assert _parse(f"{scheme} realm=x").scheme is None


def test_missing_header_has_no_challenge() -> None:
    assert parse_auth_challenge(_response(401, None)) is None
    assert parse_auth_challenge(object()) is None


@pytest.mark.parametrize(
    "header",
    [
        '"=,,',
        'error=""',
        "=",
        '   , scope="" ',
        'Bearer error="unterminated',
        "Bearer error==",
        'Bearer error="trailing-escape\\',
    ],
)
def test_malformed_header_yields_empty_fields(header: str) -> None:
    challenge = parse_auth_challenge(_response(401, header))
    assert challenge is not None
    assert challenge.error is None
    assert challenge.scope == ()
    assert challenge.resource_metadata_url is None


def test_oversized_header_is_ignored() -> None:
    header = 'Bearer error="invalid_token", realm="' + "x" * 20000 + '"'
    assert _parse(header) == MCPAuthChallenge()


def test_parser_failure_is_never_an_exception() -> None:
    with patch("dendrux.mcp._challenge._parse_challenges", side_effect=ValueError("boom")):
        challenge = parse_auth_challenge(_response(401, FULL_HEADER))
    assert challenge == MCPAuthChallenge()


def test_unreadable_or_non_text_header_has_no_challenge() -> None:
    class RaisingHeaders:
        def get(self, name: str) -> str:
            raise RuntimeError("headers unavailable")

    class Fake:
        def __init__(self, headers: Any) -> None:
            self.headers = headers

    assert parse_auth_challenge(Fake(RaisingHeaders())) is None
    assert parse_auth_challenge(Fake({"www-authenticate": b"Bearer"})) is None
    assert parse_auth_challenge(Fake({"www-authenticate": ""})) is None
    assert parse_auth_challenge(Fake({"www-authenticate": "Bearer"})) == MCPAuthChallenge(
        scheme="Bearer"
    )


def test_repr_never_renders_server_chosen_values() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9tok"
    challenge = MCPAuthChallenge(
        scheme=token,
        error=token,
        scope=(token, "read"),
        resource_metadata_url=f"https://{token}.example/prm",
    )
    assert token not in repr(challenge)
    assert token not in str(challenge)
    assert repr(challenge) == (
        "MCPAuthChallenge(scheme='<other>', error='<other>', scope=<2 items>, "
        "resource_metadata_url='<set>')"
    )
    known = MCPAuthChallenge(scheme="Bearer", error="insufficient_scope")
    assert repr(known) == (
        "MCPAuthChallenge(scheme='Bearer', error='insufficient_scope', scope=<0 items>, "
        "resource_metadata_url=None)"
    )


async def test_response_without_a_request_is_not_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    # A bare httpx2 response raises RuntimeError on ``.request``.
    error = await _connect_with(
        adapter, httpx2.Response(401, headers={"www-authenticate": "Bearer"})
    )
    assert error.status_code == 401
    assert not error.request_rejected
    assert error.challenge == MCPAuthChallenge(scheme="Bearer")


def test_stdio_source_status_is_never_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.stdio("demo", ["echo"]))
    response = _response(401, "Bearer")
    failure = httpx2.HTTPStatusError("rejected", request=response.request, response=response)
    error = adapter._safe_error("connect to", failure)
    assert isinstance(error, MCPAuthenticationError)
    assert not error.request_rejected
    assert error.challenge == MCPAuthChallenge(scheme="Bearer")


class _StatusError(RuntimeError):
    """An error carrying a 401 status but no response object."""

    status_code = 401


@pytest.mark.parametrize(
    ("observed_url", "rejected_here"),
    [(SOURCE_URL, True), ("https://elsewhere.example/mcp", False)],
)
async def test_status_attribute_without_response_falls_back_to_observed_rejection(
    observed_url: str, rejected_here: bool
) -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    await adapter._observe_response(
        _response(401, 'Bearer error="invalid_token", scope="read"', url=observed_url)
    )
    error = adapter._safe_error("connect to", _StatusError("rejected"))
    assert isinstance(error, MCPAuthenticationError)
    assert error.status_code == 401
    assert error.challenge == MCPAuthChallenge(
        scheme="Bearer", error="invalid_token", scope=("read",)
    )
    assert error.request_rejected is rejected_here


async def test_observed_rejection_with_another_status_is_not_borrowed() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    await adapter._observe_response(_response(403, 'Bearer error="insufficient_scope"'))
    error = adapter._safe_error("connect to", _StatusError("rejected"))
    assert isinstance(error, MCPAuthenticationError)
    assert error.status_code == 401
    assert error.challenge is None
    assert not error.request_rejected


# ---------------------------------------------------------------------------
# Raise sites with an observed response
# ---------------------------------------------------------------------------


class _ObservedClient:
    adapter: MCPClientAdapter
    responses: tuple[httpx2.Response, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _ObservedClient:
        for response in type(self).responses:
            await type(self).adapter._observe_response(response)
        raise RuntimeError("Server returned an error response")

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def _connect_with(adapter: MCPClientAdapter, *responses: httpx2.Response) -> Any:
    _ObservedClient.adapter = adapter
    _ObservedClient.responses = responses
    with (
        patch("dendrux.mcp._client.Client", _ObservedClient),
        patch("dendrux.mcp._client.streamable_http_client", return_value=object()),
        pytest.raises(MCPAuthenticationError) as caught,
    ):
        await adapter.connect()
    return caught.value


@pytest.mark.parametrize("status", [401, 403])
async def test_connect_rejection_carries_the_challenge(status: int) -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(adapter, _response(status, FULL_HEADER))
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    assert "HEADER-DESCRIPTION" not in str(error)
    assert "insufficient_scope" not in str(error)
    assert "well-known" not in str(error)


async def test_connect_rejection_from_another_origin_is_not_request_rejected() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(
        adapter, _response(401, FULL_HEADER, url="https://elsewhere.example/mcp")
    )
    assert error.status_code == 401
    assert not error.request_rejected
    assert error.challenge is not None


async def test_connect_rejection_without_header_has_no_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    error = await _connect_with(adapter, _response(401, None))
    assert error.request_rejected
    assert error.challenge is None


async def test_status_in_exception_tree_uses_that_response() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))
    response = _response(401, 'Bearer error="invalid_token"')
    failure = httpx2.HTTPStatusError("rejected", request=response.request, response=response)
    error = adapter._safe_error("connect to", failure)
    assert isinstance(error, MCPAuthenticationError)
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "invalid_token"


async def test_discovery_rejection_carries_the_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))

    class Client:
        async def list_tools(self, **kwargs: Any) -> Any:
            await adapter._observe_response(_response(401, 'Bearer error="invalid_token"'))
            raise RuntimeError("Server returned an error response")

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.list_tools()
    assert caught.value.request_rejected
    assert caught.value.challenge is not None
    assert caught.value.challenge.error == "invalid_token"


async def test_tool_call_rejection_carries_the_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))

    async def call(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        body = {"method": "tools/call", "params": {"name": name}}
        await adapter._observe_response(_response(403, FULL_HEADER, body=body))
        raise RuntimeError("SDK rejected request")

    class Client:
        call_tool = staticmethod(call)

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.call_tool("echo", {})
    error = caught.value
    assert error.status_code == 403
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    assert "HEADER-DESCRIPTION" not in str(error)


# ---------------------------------------------------------------------------
# End to end through the real SDK Streamable HTTP stack
# ---------------------------------------------------------------------------


def _challenging_app(state: dict[str, Any]) -> Any:
    server = SDKServer(name="challenge-demo")

    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {state['token']}".encode():
                await send(
                    {
                        "type": "http.response.start",
                        "status": state["status"],
                        "headers": [(b"www-authenticate", state["challenge"].encode())],
                    }
                )
                await send({"type": "http.response.body", "body": b"rejected"})
                return
        await inner(scope, receive, send)

    return app


@pytest.mark.parametrize("status", [401, 403])
async def test_real_sdk_connect_rejection_carries_the_challenge(status: int) -> None:
    state = {"token": "good", "status": status, "challenge": FULL_HEADER}
    async with _serve(_challenging_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer WRONG-TOKEN"})
        )
        with pytest.raises(MCPAuthenticationError) as caught:
            await adapter.connect()
        await adapter.close()
    error = caught.value
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.scheme == "Bearer"
    assert error.challenge.error == "insufficient_scope"
    assert error.challenge.scope == ("files:read", "files:write")
    rendered = _rendered(error)
    assert "WRONG-TOKEN" not in rendered
    assert "HEADER-DESCRIPTION" not in str(error)


@pytest.mark.parametrize("status", [401, 403])
async def test_real_sdk_discovery_rejection_carries_the_challenge(status: int) -> None:
    state = {"token": "one", "status": status, "challenge": FULL_HEADER}
    async with _serve(_challenging_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer one"})
        )
        await adapter.connect()
        try:
            state["token"] = "two"
            with pytest.raises(MCPAuthenticationError) as caught:
                await adapter.list_tools()
        finally:
            await adapter.close()
    error = caught.value
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge == MCPAuthChallenge(
        scheme="Bearer",
        error="insufficient_scope",
        scope=("files:read", "files:write"),
        resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
    )


async def test_discovery_403_carries_the_challenge() -> None:
    adapter = MCPClientAdapter(MCPSource.http("demo", SOURCE_URL))

    class Client:
        async def list_tools(self, **kwargs: Any) -> Any:
            await adapter._observe_response(_response(403, FULL_HEADER))
            raise RuntimeError("Server returned an error response")

    adapter._client = Client()
    with pytest.raises(MCPAuthenticationError) as caught:
        await adapter.list_tools()
    assert caught.value.status_code == 403
    assert caught.value.challenge is not None
    assert caught.value.challenge.error == "insufficient_scope"
    assert caught.value.challenge.scope == ("files:read", "files:write")


@pytest.mark.parametrize(
    ("status", "header", "expected_error"),
    [
        (401, 'Bearer error="invalid_token", error_description="expired"', "invalid_token"),
        (403, FULL_HEADER, "insufficient_scope"),
    ],
)
async def test_real_sdk_tool_call_rejection_carries_the_challenge(
    status: int, header: str, expected_error: str
) -> None:
    state = {"token": "one", "status": status, "challenge": header}
    async with _serve(_challenging_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer one"})
        )
        await adapter.connect()
        try:
            state["token"] = "two"
            with pytest.raises(MCPAuthenticationError) as caught:
                await adapter.call_tool("echo", {"text": "hi"})
        finally:
            await adapter.close()
    error = caught.value
    assert error.status_code == status
    assert error.request_rejected
    assert error.challenge is not None
    assert error.challenge.error == expected_error
    assert expected_error not in str(error)


# ---------------------------------------------------------------------------
# No response header or body text escapes into messages, events, or logs
# ---------------------------------------------------------------------------

# Shaped like a bearer token so it passes every field validator: a hostile
# server echoing a token must still not reach any rendering.
SENTINEL = "SENTINELtok7f3a"
SENTINEL_HEADER = (
    f'{SENTINEL} realm="{SENTINEL}", Bearer realm="{SENTINEL}-realm", error="{SENTINEL}", '
    f'error_description="{SENTINEL}-description", scope="read {SENTINEL}", '
    f'resource_metadata="https://{SENTINEL}.example/prm"'
)


def _sentinel_app(state: dict[str, Any]) -> Any:
    server = SDKServer(name="sentinel-demo")

    @server.tool()
    def echo(text: str) -> str:
        return text

    inner = server.streamable_http_app()

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {state['token']}".encode():
                await send(
                    {
                        "type": "http.response.start",
                        "status": state["status"],
                        "headers": [
                            (b"www-authenticate", SENTINEL_HEADER.encode()),
                            (b"content-type", b"application/json"),
                        ],
                    }
                )
                body = f'{{"error": "{SENTINEL}-body", "detail": "{SENTINEL}"}}'
                await send({"type": "http.response.body", "body": body.encode()})
                return
        await inner(scope, receive, send)

    return app


def _exception_text(error: BaseException) -> str:
    parts: list[str] = []
    stack: list[BaseException] = [error]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        parts.extend([str(node), repr(node), str(getattr(node, "transport_detail", ""))])
        if isinstance(node, BaseExceptionGroup):
            stack.extend(node.exceptions)
        for linked in (node.__cause__, node.__context__):
            if linked is not None:
                stack.append(linked)
    return "\n".join(parts)


def _log_text(caplog: pytest.LogCaptureFixture) -> str:
    # httpcore2's DEBUG wire trace prints raw response (and request) headers
    # by design; it is a third-party transport log outside Dendrux's control,
    # so only its INFO-and-above records are checked. Every other logger,
    # Dendrux and the MCP SDK included, is checked at every level.
    records = [
        r for r in caplog.records if not r.name.startswith("httpcore2") or r.levelno >= logging.INFO
    ]
    assert any(r.name.startswith("dendrux") for r in records)
    return "\n".join(r.getMessage() for r in records)


def _assert_no_sentinel(error: BaseException, events: list[Any], log_text: str) -> None:
    assert SENTINEL not in _exception_text(error)
    challenge = getattr(error, "challenge", None)
    assert SENTINEL not in repr(challenge)
    assert SENTINEL not in str(challenge)
    for event in events:
        assert SENTINEL not in repr(event)
    assert SENTINEL not in log_text


@pytest.mark.parametrize("status", [401, 403])
async def test_rejection_text_never_escapes_at_connect_discovery_and_call(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    state = {"token": "good", "status": status}
    async with _serve(_sentinel_app(state)) as url:
        adapter = MCPClientAdapter(
            MCPSource.http("demo", url, headers={"Authorization": "Bearer stale"})
        )
        with pytest.raises(MCPAuthenticationError) as connect_error:
            await adapter.connect()
        await adapter.close()
        failures: list[BaseException] = [connect_error.value]
        # Discovery and call: the token is revoked after the handshake.
        for operation in ("list_tools", "call_tool"):
            state["token"] = "good"
            adapter = MCPClientAdapter(
                MCPSource.http("demo", url, headers={"Authorization": "Bearer good"})
            )
            await adapter.connect()
            state["token"] = "rotated"
            try:
                with pytest.raises(MCPAuthenticationError) as caught:
                    if operation == "list_tools":
                        await adapter.list_tools()
                    else:
                        await adapter.call_tool("echo", {"text": "hi"})
            finally:
                await adapter.close()
            failures.append(caught.value)
    for error in failures:
        assert isinstance(error, MCPAuthenticationError)
        assert error.status_code == status
        assert error.request_rejected
        # The token-shaped values pass validation and are kept as data, but
        # the free-text fields are gone and nothing renders them.
        assert error.challenge == MCPAuthChallenge(
            scheme="Bearer",
            error=SENTINEL,
            scope=("read", SENTINEL),
            resource_metadata_url=f"https://{SENTINEL}.example/prm",
        )
        _assert_no_sentinel(error, [], _log_text(caplog))


async def test_rejection_text_never_escapes_through_the_runtime(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dendrux.mcp import MCPRuntime
    from dendrux.mcp._runtime import _ViewToolSource

    class Provider:
        def __init__(self) -> None:
            self.token = "stale"

        async def get_auth(self) -> dict[str, str]:
            return {"Authorization": f"Bearer {self.token}"}

        async def on_auth_rejected(self) -> None:
            self.token = "still-stale"

    class Recorder:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def on_event(self, event: Any) -> None:
            self.events.append(event)

    caplog.set_level("DEBUG")
    recorder = Recorder()
    state = {"token": "good", "status": 401}
    async with _serve(_sentinel_app(state)) as url, MCPRuntime(observer=recorder) as runtime:
        # Connect-time recovery runs once and the second 401 surfaces.
        connection = runtime.bind(
            connection_key="recovering",
            credentials=Provider(),
            source=MCPSource.http("demo", url, reauthenticate_on_401=True),
        )
        with pytest.raises(MCPAuthenticationError) as connect_error:
            await connection.discover()
        assert connect_error.value.request_rejected

        # A 403 on a tool call through a managed view.
        good = runtime.bind(
            connection_key="calls",
            source=MCPSource.http("calls", url, headers={"Authorization": "Bearer good"}),
        )
        view = _ViewToolSource(good.tools())
        await view._discover()
        state["token"] = "rotated"
        state["status"] = 403
        try:
            with pytest.raises(MCPAuthenticationError) as call_error:
                await view._create_executor("echo")(text="hi")
        finally:
            await view.close()
    assert recorder.events
    for error in (connect_error.value, call_error.value):
        _assert_no_sentinel(error, recorder.events, _log_text(caplog))
    assert call_error.value.challenge is not None
    assert call_error.value.challenge.error == SENTINEL


# --------------------------------------------------------------------------
# OAuth bindings: which rejections recover, and how terminal ones translate
# --------------------------------------------------------------------------


def _rejection(status: int, header: str | None, *, request_rejected: bool = True):  # type: ignore[no-untyped-def]
    error = MCPAuthenticationError("rejected")
    error.status_code = status
    error.request_rejected = request_rejected
    error.challenge = challenge_from_headers([header]) if header is not None else None
    return error


@pytest.mark.parametrize(
    ("status", "header", "oauth", "reauth", "expected"),
    [
        (401, None, True, False, True),
        (401, None, False, True, True),
        (401, None, False, False, False),
        (403, 'Bearer error="insufficient_scope", scope="admin"', True, False, True),
        (403, 'Bearer error="insufficient_scope", scope="admin"', False, True, False),
        (403, 'Bearer error="invalid_token"', True, False, False),
        (403, None, True, False, False),
    ],
)
def test_recoverable_rejections(
    status: int, header: str | None, oauth: bool, reauth: bool, expected: bool
) -> None:
    from dendrux.mcp._runtime import _recoverable_rejection

    error = _rejection(status, header)
    assert _recoverable_rejection(error, oauth=oauth, reauthenticate_on_401=reauth) is expected
    error.request_rejected = False
    assert not _recoverable_rejection(error, oauth=oauth, reauthenticate_on_401=reauth)


def test_terminal_rejections_translate_with_challenge_attributes() -> None:
    from dendrux.mcp._runtime import _oauth_rejection_error

    identity = ("tenant", "conn")
    scoped = _oauth_rejection_error(
        identity,
        _rejection(
            403,
            'Bearer error="insufficient_scope", scope="admin write", '
            'resource_metadata="https://mcp.example.com/prm"',
        ),
    )
    assert scoped is not None
    assert (scoped.reason, scoped.scopes, scoped.status_code) == (
        "insufficient_scope",
        ("admin", "write"),
        403,
    )
    assert scoped.resource_metadata_url == "https://mcp.example.com/prm"
    assert scoped.key.connection_key == "conn" and scoped.key.tenant_key == "tenant"
    revoked = _oauth_rejection_error(identity, _rejection(401, 'Bearer scope="mcp"'))
    assert revoked is not None and (revoked.reason, revoked.scopes) == ("revoked", ("mcp",))
    assert _oauth_rejection_error(identity, _rejection(403, None)) is None
    assert _oauth_rejection_error(identity, _rejection(401, None, request_rejected=False)) is None


class _FakeAdapter:
    instances: list[_FakeAdapter] = []
    statuses: list[int] = []

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self.info = None
        self.instances.append(self)

    async def connect(self) -> None:
        if self.statuses:
            raise _rejection(self.statuses.pop(0), None)

    async def list_tools(self) -> list[Any]:
        from mcp.types import Tool

        return [Tool(name="echo", input_schema={"type": "object"})]

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("raised", ["credential", "authorization"])
async def test_application_provider_errors_pass_through_unchanged(raised: str) -> None:
    from dendrux.mcp import MCPAuthKey, MCPAuthorizationRequired, MCPCredentialError, MCPRuntime

    sentinel = (
        MCPCredentialError("application credential failure")
        if raised == "credential"
        else MCPAuthorizationRequired(MCPAuthKey("c"), "expired")
    )

    class Provider:
        async def get_auth(self) -> dict[str, str]:
            return {"Authorization": "Bearer one"}

        async def on_auth_rejected(self) -> None:
            raise sentinel

    _FakeAdapter.instances = []
    _FakeAdapter.statuses = [401]
    with patch("dendrux.mcp._runtime.MCPClientAdapter", _FakeAdapter):
        async with MCPRuntime() as runtime:
            connection = runtime.bind(
                connection_key="c",
                source=MCPSource.http("demo", SOURCE_URL, reauthenticate_on_401=True),
                credentials=Provider(),
            )
            with pytest.raises(type(sentinel)) as caught:
                await connection.discover()
            assert caught.value is sentinel
