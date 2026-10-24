# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue. Use GitHub's
**Report a vulnerability** button on the repository's *Security* tab (a private security
advisory).

Include the affected version, what you did, what you expected, and what happened. A
minimal reproduction helps most. Do not include real credentials or customer data.

We aim to acknowledge a report within 3 working days and to tell you our plan within 10.
Fixes ship in a patch release, and we credit reporters who want credit.

## Supported versions

Security fixes go to the latest minor release. Dendrux is pre-1.0, so upgrade to the
latest release to receive them.

## What is in scope

Dendrux is a library: your application owns its server, network exposure, authentication
and deployment. In scope are flaws in the code we ship, such as:

- tenant isolation in `RunStore`, `StateStore` and `make_read_router`;
- the `dendrux dashboard` server and the `dendrux` CLI;
- secrets or personal data leaking into logs, events, errors or persisted rows;
- guardrail, approval, budget or cancellation checks that can be bypassed;
- the package build and release process.

Out of scope: missing authentication on an endpoint you mounted yourself, vulnerabilities
in a model provider or MCP server, and findings that need an attacker who already has
your database credentials.

## Running the dashboard and read router safely

- `dendrux dashboard` listens on `127.0.0.1` and needs no token there. To bind any other
  address you must set `DENDRUX_DASHBOARD_TOKEN`; prefer the environment variable to
  `--token`, which is visible in process listings. Serve it over TLS behind a proxy.
- Cross-origin browser access is off. Allow a specific origin with `--allow-origin`.
- The dashboard shows full prompts, tool results and provider payloads. Treat access to it
  like access to the database.
- `make_read_router` requires a tenant resolver and hides raw payloads unless you opt in
  with `include_payloads=True`.

## Supply chain

CI actions are pinned to commit SHAs and workflows run with read-only permissions.
Dependencies are scanned with `pip-audit` on every change and daily, and the code with
CodeQL. Releases are built in CI and published with PyPI trusted publishing after a
manual approval.
