# Release setup (owner checklist)

`release.yml` builds and checks distributions for every `v*` tag. It publishes to PyPI
only after the steps below are done, **in this order**. The checks inside the workflow
file are not the security boundary, because workflow code comes from the tagged commit:
anyone with write access can tag a commit whose `release.yml` drops its own checks. The
boundary is the GitHub environment, the tag ruleset and the PyPI publisher configuration.

1. **Create the `pypi` environment** (Settings, Environments).
   - Add the owner as a **required reviewer**.
   - Under "Deployment branches and tags", choose "Selected branches and tags" and add
     the tag rule `v*` only.
2. **Add a tag ruleset** (Settings, Rules, Rulesets): target tags matching `v*`, and
   restrict creation, update and deletion to the owner (bypass list: the owner only).
3. **Protect `main`**: require pull requests, and require the CI, Security and CodeQL
   checks. Consider requiring code-owner review for `.github/`.
4. **Register the PyPI trusted publisher** (PyPI project `dendrux`, Publishing):
   repository `dendrux/dendrux`, workflow `release.yml`, environment **`pypi`**.
   The environment name is mandatory. A publisher registered without one lets any
   workflow run from the repository publish.
5. **Check variables.** Make sure no organisation-level variable named
   `PYPI_TRUSTED_PUBLISHING` exists (it would enable the job too).
6. **Only now** set the repository variable `PYPI_TRUSTED_PUBLISHING` to `true`.

Also recommended: enable private vulnerability reporting, secret scanning and push
protection; set the default `GITHUB_TOKEN` permission to read-only; allow only pinned
actions.

## Releasing

1. Bump the version sites, merge the release PR to `main`.
2. Tag the merge commit `vX.Y.Z` and push the tag. The build job checks the tag is on
   `main` and equals the `pyproject.toml` version, builds, and runs `twine check --strict`
   before uploading the artifact.
3. Read the build log, then approve the `pypi` deployment. The publish job re-checks that
   the distributions belong to the tag.
4. Verify the digests and install in a clean virtualenv, then announce.

Until step 6 above is done, a tag only builds and checks, and the manual upload flow
still works.
