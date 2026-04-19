# dendrux docs source

The `.mdx` files in this directory are the source of truth for the public docs at [dendrux.dev/docs](https://dendrux.dev/docs).

The site (a separate Next.js app) pulls these files in at build time. Editing any file here and rebuilding the site updates the published docs.

## Structure

```
docs/
├── overview.mdx            → /docs
├── quickstart.mdx          → /docs/quickstart
├── architecture/           → /docs/architecture/*
│   ├── runs.mdx
│   ├── state-persistence.mdx
│   ├── event-ordering.mdx
│   ├── pause-resume.mdx
│   ├── cancellation.mdx
│   ├── governance.mdx
│   ├── pii-redaction.mdx
│   ├── access-control.mdx
│   ├── guardrails.mdx
│   ├── approval.mdx
│   ├── notifier.mdx
│   ├── recorder.mdx
│   └── loops.mdx
├── recipes/                → /docs/recipes/*
└── reference/              → /docs/reference/*
```

## Editing rules

- Each `.mdx` file starts with frontmatter (`title`, `description`).
- Use plain Markdown wherever possible. JSX components are available for callouts, diagrams, and tabs — see existing pages for examples.
- Code blocks use triple backticks with a language tag (` ```python`, ` ```bash`). Shiki highlights them at build time.
- One H1 per page (the title). Use H2/H3 for sections.

## When you change a public API

If you rename, add, or remove anything in `dendrux/__init__.py`, update the corresponding page in `docs/reference/` in the same PR. Reviewers will check.
