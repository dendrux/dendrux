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

- tenant isolation in `RunStore` and `StateStore`;
- the `dendrux` CLI;
- secrets or personal data leaking into logs, events, errors or persisted rows;
- guardrail, approval, budget or cancellation checks that can be bypassed;
- the package build and release process.

Out of scope: missing authentication on an endpoint you mounted yourself, vulnerabilities
in a model provider or MCP server, and findings that need an attacker who already has
your database credentials.

## Serving library reads safely

Your application owns every HTTP route. Authenticate with headers, authorize the
actor's run ownership within a tenant, and pass `tenant_id` on every read/write.
Enforce approval roles on the server; client callbacks are UX, and tool results are
client assertions. Use contextual `RunStore.get_event_view` or `follow` for events,
and an allowlisted browser DTO. Pending client-tool arguments can contain real
personal data. Keep raw traces, provider bodies and PII mappings operator-only.
Bound live network sends and always close followers in `finally`; never put bearer
tokens in URLs or tie browser streams directly to `agent.stream()`.

## Supply chain

CI actions are pinned to commit SHAs and workflows run with read-only permissions.
Dependencies are scanned with `pip-audit` on every change and daily, and the code with
CodeQL. A release workflow builds and checks the distributions for every version tag.
Publishing through PyPI trusted publishing, behind a manual approval, is being enabled;
the owner steps are in `.github/RELEASING.md`, and until they are done releases are
uploaded manually.
