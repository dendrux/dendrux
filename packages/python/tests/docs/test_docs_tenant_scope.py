"""Persisted Python examples must choose a tenant, including skipped fragments.

Persistence is page-wide: a database_url/state_store keyword in any Python block,
or a call that inherently needs persisted state, covers later continuation blocks.
A block with strict_tenancy=False is exempt; fragments in other blocks on that
page are still checked. Explicit None is a valid scope. Persistence is not inferred
from prose or environment variables.

Use AST calls when a block parses. For incomplete signatures/fragments, tokenize
instead: find dotted calls and balance delimiters to inspect only their own keyword
arguments. Tokens keep comments, strings and nested calls from hiding omissions.
This is a static audit, not receiver type inference or **kwargs data-flow analysis.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

from tests.docs._snippets import Snippet, extract_snippets

REPO = Path(__file__).resolve().parents[4]
# agent.py entry_checks call sites, get_result's required keyword, and reclaim's
# unconditional omission check; runtime.runner also exposes run_stream directly.
METHODS = frozenset(
    {
        "run",
        "stream",
        "run_stream",
        "retry",
        "resume",
        "resume_stream",
        "submit_tool_results",
        "submit_input",
        "submit_approval",
        "submit_resume",
        "cancel_run",
        "request_pause",
        "submit_note",
        "get_result",
        "reclaim",
    }
)
PERSISTED_ONLY = METHODS - {"run", "stream", "run_stream"}
# Unrelated APIs with the same method name, not exceptions for Agent snippets.
OTHER_CALLS = {"asyncio.run", "sandbox.commands.run"}
# Exact source line, not a whole block/page exemption; stale entries fail below.
LENIENT_CONTINUATION = "Continues the strict_tenancy=False agent built earlier on this page."
ALLOWLIST = {
    (
        "docs/architecture/multi-tenancy.mdx",
        'await agent.run("hi")                       # TenantScopeRequiredError',
    ): "Intentionally demonstrates TenantScopeRequiredError on omission.",
    (
        "docs/quickstart.mdx",
        'result = await agent.run("Please refund order 42.")',
    ): LENIENT_CONTINUATION,
    (
        "docs/quickstart.mdx",
        "result = await agent.submit_approval(run_id, approved=True, expected_pause_id=pause_id)",
    ): LENIENT_CONTINUATION,
    (
        "docs/architecture/runs.mdx",
        'result = await agent.run("Please refund order 42.")',
    ): LENIENT_CONTINUATION,
    (
        "docs/architecture/runs.mdx",
        "result = await agent.submit_approval(",
    ): LENIENT_CONTINUATION,
    (
        "docs/recipes/client-tools.mdx",
        'result = await agent.run("Add up Q3 sales from Sheet1!A1:A12 '
        'and tell me if it beats AAPL.")',
    ): LENIENT_CONTINUATION,
}


def _tokens(code: str) -> list[tokenize.TokenInfo]:
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type not in {
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                tokens.append(token)
    except (tokenize.TokenError, IndentationError):
        pass  # Audit tokens preceding an unfinished trailing fragment too.
    return tokens


def _fallback_calls(code: str) -> list[tuple[int, str, bool]]:
    tokens = _tokens(code)
    calls = []
    for i, token in enumerate(tokens):
        if token.string not in METHODS or i < 2 or tokens[i - 1].string != ".":
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].string != "(":
            continue
        start = i - 2
        while start >= 2 and tokens[start - 1].string == ".":
            start -= 2
        name = "".join(t.string for t in tokens[start : i + 1])
        if name in OTHER_CALLS:
            continue
        depth = 0
        scoped = False
        for j in range(i + 2, len(tokens)):
            value = tokens[j].string
            if value in {"(", "[", "{"}:
                depth += 1
            elif value in {")", "]", "}"}:
                if depth == 0:
                    break
                depth -= 1
            elif (
                depth == 0
                and value == "tenant_id"
                and j + 1 < len(tokens)
                and tokens[j + 1].string == "="
            ):
                scoped = True
        calls.append((tokens[start].start[0], token.string, scoped))
    return calls


def _calls(code: str) -> list[tuple[int, str, bool]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _fallback_calls(code)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in METHODS and ast.unparse(node.func) not in OTHER_CALLS:
            calls.append(
                (node.lineno, node.func.attr, any(k.arg == "tenant_id" for k in node.keywords))
            )
    return sorted(calls)


def _uses_persistence(snippets: list[Snippet]) -> bool:
    for snippet in snippets:
        tokens = _tokens(snippet.code)
        if any(
            token.string in {"database_url", "state_store"} and tokens[i + 1].string == "="
            for i, token in enumerate(tokens[:-1])
        ) or any(method in PERSISTED_ONLY for _, method, _ in _calls(snippet.code)):
            return True
    return False


def _missing_scope(snippets: list[Snippet]) -> list[tuple[str, int, str]]:
    if not _uses_persistence(snippets):
        return []
    return [
        (snippet.file, snippet.line + line - 1, snippet.code.splitlines()[line - 1].strip())
        for snippet in snippets
        if not _is_lenient_block(snippet.code)
        for line, _, scoped in _calls(snippet.code)
        if not scoped
    ]


def _is_lenient_block(code: str) -> bool:
    tokens = _tokens(code)
    return any(
        [token.string for token in tokens[i : i + 3]] == ["strict_tenancy", "=", "False"]
        for i in range(len(tokens) - 2)
    )


def test_persisted_docs_choose_tenant_scope() -> None:
    paths = sorted(path for path in (REPO / "docs").rglob("*") if path.suffix in {".mdx", ".md"})
    assert paths, "no docs found"
    seen = set()
    failures = []
    for path in paths:
        for file, line, source in _missing_scope(extract_snippets(path, REPO)):
            key = (file, source)
            if key in ALLOWLIST:
                seen.add(key)
            else:
                failures.append(f"{file}:{line}: missing tenant_id=: {source}")
    assert seen == ALLOWLIST.keys(), f"stale tenant-scope exceptions: {ALLOWLIST.keys() - seen}"
    assert not failures, "\n" + "\n".join(failures)


@pytest.mark.parametrize("method", sorted(METHODS - {"run_stream"}))
def test_detects_every_agent_entry_point_across_blocks(method: str) -> None:
    snippets = [
        Snippet("page.mdx", 2, "agent = Agent(state_store=store)\n", None),
        Snippet("page.mdx", 9, f'await agent.{method}("input")\n', "fragment"),
    ]
    assert _missing_scope(snippets) == [("page.mdx", 9, f'await agent.{method}("input")')]


@pytest.mark.parametrize("prefix", ["", "async def unfinished() -> Result\n"])
def test_keywords_belong_to_their_own_call(prefix: str) -> None:
    code = (
        prefix
        + """await agent.run(
    build_input(tenant_id="acme"),  # tenant_id="acme" is not an outer keyword
    metadata={"tenant_id": "acme"},
)
await agent.stream("tenant_id=ignored", tenant_id=None)
asyncio.run(main())
await sandbox.commands.run(command)
"""
    )
    calls = _calls(code)
    assert [(method, scoped) for _, method, scoped in calls] == [("run", False), ("stream", True)]
    assert calls[0][0] == (2 if prefix else 1)


def test_nonpersisted_examples_and_explicit_none_are_valid() -> None:
    assert not _missing_scope([Snippet("page.md", 1, 'await agent.run("hi")', None)])
    assert not _missing_scope(
        [
            Snippet(
                "page.md",
                1,
                'agent = Agent(database_url=url)\nawait agent.run("hi", tenant_id=None)',
                None,
            )
        ]
    )


def test_continuation_implies_persistence_without_constructor() -> None:
    assert _missing_scope([Snippet("page.mdx", 7, "await agent.submit_note(run_id, text)", None)])


def test_lenient_block_can_omit_tenant_scope() -> None:
    snippet = Snippet(
        "page.mdx",
        2,
        'agent = Agent(database_url=url, strict_tenancy=False)\nawait agent.run("hi")',
        None,
    )
    assert not _missing_scope([snippet])


def test_lenient_block_does_not_exempt_a_fragment_on_the_same_page() -> None:
    snippets = [
        Snippet(
            "page.mdx",
            2,
            'agent = Agent(state_store=store, strict_tenancy=False)\nawait agent.run("hi")',
            None,
        ),
        Snippet("page.mdx", 9, "await agent.submit_resume(run_id)", "fragment"),
    ]
    assert _missing_scope(snippets) == [("page.mdx", 9, "await agent.submit_resume(run_id)")]
