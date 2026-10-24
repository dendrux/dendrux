# dendrux docs source

The `.mdx` files in this directory are the source of truth for the public docs at [dendrux.dev/docs](https://dendrux.dev/docs).

The site (a separate Next.js app) pulls these files in at build time. Editing any file here and rebuilding the site updates the published docs.

## Structure

```
docs/
├── overview.mdx            → /docs
├── is-dendrux-for-you.mdx  → /docs/is-dendrux-for-you
├── quickstart.mdx          → /docs/quickstart
├── upgrading-to-0.5.0.mdx  → /docs/upgrading-to-0.5.0
├── architecture/           → /docs/architecture/*
│   ├── runs.mdx, state-persistence.mdx, event-ordering.mdx
│   ├── pause-resume.mdx, cancellation.mdx, inbox.mdx
│   ├── multi-tenancy.mdx
│   ├── governance.mdx, access-control.mdx, tool-policy.mdx, approval.mdx
│   ├── guardrails.mdx, pii-redaction.mdx, pii-restore.mdx
│   ├── budget.mdx, pricing.mdx
│   ├── loops.mdx, models.mdx, reasoning.mdx, prompt-cache.mdx
│   ├── mcp.mdx, skills.mdx
│   └── notifier.mdx, recorder.mdx
├── recipes/                → /docs/recipes/*
└── reference/              → /docs/reference/*
```

Adding a page here does not add it to the site's sidebar. The navigation is hand-curated in the docs site repo (`lib/docs-nav.ts`); add the entry there in the same change.

## Editing rules

- Each `.mdx` file starts with frontmatter (`title`, `description`).
- **Plain Markdown only.** The docs site provides no custom components: no `<Callout>`, no JSX tags, no `{...}` expressions in prose. A single unknown component fails the whole site deploy. For a note, use a blockquote with a bold lead: `> **Note.** text`. Anything with `<` or `{` that is not code goes in backticks or a fenced block.
- No HTML comments (`<!-- -->`): MDX rejects them. To leave an invisible reviewer note, use a Markdown reference definition on its own line, with the text in double quotes: `[//]: # "note text"` (no double quotes or backslashes inside; parentheses in parentheses-delimited titles break the build). Notes that start `VERIFY-0.5.0:` mark text that depends on code not yet verified; remove them once it is.
- Code blocks use triple backticks with a language tag (` ```python`, ` ```bash`). Shiki highlights them at build time.
- One H1 per page (the title). Use H2/H3 for sections.
- Do not name or compare with other agent frameworks.
- Describe the secure behaviour a release ships, not the weakness it replaced.
- Verify before you push: build the site against this directory (`DENDRUX_DOCS_PATH=<this dir> npm run build` in the docs site repo).

## Python snippets are tested

Every ` ```python` block in `docs/**/*.mdx` and the READMEs is run by `packages/python/tests/docs/` against a mock model (no network, no key). Blocks on a page run in order and share one namespace, like a reader following along, so a later block may use names an earlier one defined. Tools that a snippet defines are called by the mock once, which lets approval and client-tool pauses happen.

A block that cannot run on its own gets a reference note on the line above its fence (blank lines between are fine), with a required reason:

```text
[//]: # "docs-test: skip - needs a live MCP server"
```

Use it for signature listings, fragments that rely on setup shown elsewhere, placeholders such as `tools=[...]`, and snippets that need an optional package, a real file or a real service. Skipped blocks that parse as Python still have their `dendrux` imports checked, so a renamed or removed name fails the test either way. Run the checks with `pytest tests/docs` from `packages/python`.

## When you change a public API

If you rename, add, or remove anything in `dendrux/__init__.py`, update the corresponding page in `docs/reference/` (and the [API glossary](reference/api-glossary.mdx)) in the same PR. Reviewers will check.
