# Quality gates

Dendrux is built mostly by AI agents, so mechanical gates in CI stop code quality from
silently getting worse. Every gate is a **ratchet**: today's problems are recorded in a
committed baseline, and CI fails only on a **new** problem or an existing one getting
**worse**. Baselines are small, sorted JSON keyed by `module:qualified_name` (no line
numbers), so unrelated edits do not change them.

Run everything with `make quality` (it is part of `make ci`). Run one gate with its own
target. Setup, once: `pip install -e ".[dev]"` brings in the pinned tools. Semgrep is not in
the dev extra because it pins `mcp` and OpenTelemetry versions that clash with the `mcp` and
`all` extras. Install it in its own environment and point to it:

```
python -m venv ~/.cache/dendrux-semgrep-venv
~/.cache/dendrux-semgrep-venv/bin/pip install -r tools/semgrep-requirements.txt
make quality SEMGREP=~/.cache/dendrux-semgrep-venv/bin/semgrep
```

## Toolchain parity

Every direct development tool and test-support dependency in the `dev` extra has an exact
version in `pyproject.toml`, including mypy, ruff, pytest and its plugins, coverage, griffe and
the quality checkers. CI installs that same extra. Keep ruff's pre-commit revision and the
API job's standalone griffe pin in step with it. Semgrep remains pinned separately in
`tools/semgrep-requirements.txt` because of its dependency conflicts.

From `packages/python`, run `make dev-sync` in your development venv to install the pinned
tools and upgrade runtime dependencies to the newest allowed versions, as a fresh CI
environment does. Then run `make test-model-install` if the pinned NLP model is absent.
`DEV_EXTRAS=dev,all,mysql` also installs the optional MySQL driver; the default is `dev,all`,
the same extras as the Python shard jobs. For a SQLAlchemy matrix run, apply the leg's
`pip install "sqlalchemy[asyncio]==2.0.*"` or `2.1.*` after syncing.

Published runtime requirements (including SQLAlchemy, provider SDKs and database drivers)
remain ranges. CI continues to resolve their newest allowed versions and tests both
SQLAlchemy minors. The direct tool pins are not a transitive dependency lockfile. A tool
upgrade is a deliberate change with checks on the supported Python/SQLAlchemy matrix;
baseline decreases are stamped only by the coordinator. Never restamp a baseline to hide
a regression.

| Target | Script | Fails when |
|---|---|---|
| `make cognitive` | `tools/cognitive_ratchet.py` | A new function scores above 15 (complexipy cognitive complexity), or a function already above 15 scores higher. Baseline: `cognitive_baseline.json` (only functions above 15 are recorded; overloads and setters of one name count at their worst). |
| `make duplicates` | `tools/duplicate_ratchet.py` | A new block of 10 or more near-identical lines appears across two files (pylint's similarity engine; imports, signatures, comments and docstrings ignored), or a known block grows. Baseline: `duplicate_baseline.json`, block sizes per file pair; a block matches within 15 percent (at least 2 lines). |
| `make deadcode` | `tools/vulture_ratchet.py` | vulture (confidence 80 or more) finds a new unused name. Intentional public API and hooks go in `vulture_whitelist.py` with a comment. Baseline: `vulture_baseline.json`. |
| `make cycles` | `tools/import_cycles.py` | A module import cycle appears, a module joins a baselined cycle, or two cycles merge. Only top-level runtime imports count: `if TYPE_CHECKING:` blocks and imports inside functions are ignored. `python tools/import_cycles.py report` also lists the cycles that function-level imports hide (information only). Baseline: `import_cycles_baseline.json`. |
| `make semgrep` | `tools/semgrep_ratchet.py` | A rule in `.semgrep/dendrux.yml` has a new match, or more matches in a function that already had some; or a rule stops passing its fixture. Baseline: `semgrep_baseline.json` (rule, module, function). |
| `make crap` | `tools/crap_ratchet.py` | A new function scores above 30; complexity rises and the function scores above 30; or edited code exceeds its own recorded CRAP baseline. An edited function may remain above 30 when neither complexity nor CRAP rises. Unchanged code retains the 0.5-point noise band and the warning allowance for rises up to 25 percent. Backend-only local coverage is handled explicitly below. Needs `coverage.json` from `make test`. Baseline: `crap_baseline.json`. |

## Reading a failure

Each failure line names the gate, the problem and where: `NEW` is a problem that was not in
the baseline, `WORSE` is a known one that grew. Fix the code. Do not stamp the baseline to
make a failure go away; `stamp` refuses to record new or worse entries.

- Cognitive: split the function, or return early to cut the nesting. Cognitive complexity
  counts nesting, so a deeply nested function costs far more than a flat one of the same size.
- Duplicates: move the shared lines into one helper. The report lists the file pair and size.
- Dead code: delete it. If it is public API or a hook the framework calls, whitelist it with
  a one-line reason.
- Import cycles: move the shared piece down a layer, or import it inside the function
  (and say why). The `report` view shows which function-level imports hold a cycle together.
- Semgrep: the rule message says what to do instead. A false positive is fixed in the rule
  (and its fixture), not silenced in code: `# nosemgrep` comments are ignored by the gate.
  A rule's `metadata.allow_in` (`module:Class.method`) exempts exactly that method.
- CRAP: add tests for the function, or make it simpler. A `CRAP warning` line (unchanged code,
  score up by at most 25 percent) does not fail the build: it is coverage variance between
  runs. Look at it if it keeps growing; a loss above 25 percent (for example deleted tests)
  fails. Edited code must not exceed its recorded score, compared at the baseline's two-decimal
  precision. The hard limit of 30 still applies to new functions and functions whose complexity rises.

## CRAP coverage and debt

An edited function above 30 passes when its complexity did not rise and its CRAP score is no
higher than its own recorded baseline. This permits a type-correctness edit to an existing
hotspot without requiring a large runtime refactor in the same task. It does not permit an
increase. Baseline updates also refuse increases, including those inside the check's noise
band; the coordinator alone records decreases.

The short reviewed MySQL-family module list is:

- `src/dendrux/db/_dialects/mysql.py`
- `src/dendrux/runtime/stores/sql/mysql.py`

Only those exact modules are advisory in a normal SQLite/Postgres local CRAP check. Their
violations print warnings when real MySQL/MariaDB coverage is absent. Mixed modules such as
`runtime/stores/sql/base.py` and `runtime/_blob_catalog.py` remain enforced. Missing or stale
source evidence always fails, even for a listed module. CI enforces every module, using
`--ci` (also selected automatically by nonempty `CI` or `GITHUB_ACTIONS`). The MySQL and
MariaDB jobs each upload a coverage artifact; the coverage jobs require both server jobs,
download both artifacts, and verify all four shard files plus both backend files before
combining. A missing backend cannot silently disappear from the measurement.

**Run when needed: any change under the MySQL-family list.** Install the optional `mysql`
extra, collect the normal base-suite coverage, then run:

```sh
make cov-mysql
make crap
```

`cov-mysql` is opt-in and is in neither `make ci` nor `make test-merge`. It requires the
already-running `dendrux-test-mysql` (127.0.0.1:53306) and `dendrux-test-mariadb`
(127.0.0.1:53307) containers, with the test credentials `root` / `dendrux_test`. It never
starts or stops containers; a stopped container or unavailable Docker produces a clear
error before testing. It runs `tests/db` on one server at a time, four workers, no benchmarks,
using private `dendrux_p5_b_cigreen_cov_<server>_test` prefixes (each fixture creates and drops
its own private database). It requires `.coverage`, `coverage.json` and its current source
digests from the base suite. Both backend suites must succeed before the combined report
is replaced. Coverage from the base suite is preserved and merged with both fresh reports.

The resulting `coverage.json.backends.json` binds both datasets to the combined report's
SHA-256. When this matching provenance is present, `make crap` enforces the listed modules
locally too. A new base-only report invalidates that provenance and restores advisory mode
until `cov-mysql` is rerun. Advisory mode never applies to baseline updates.

Every CRAP check prints the number of functions above 30, their total excess above 30, and
the top ten scores. It also warns about **touched but not reduced**: edited functions above
30 whose recorded score did not decrease. This warning does not change the no-worse hard
gate. The same summary is written next to the input as `coverage.json.crap.json`, including
the touched-function list, for merge-gate reporting. It is produced even when the check
fails, provided the coverage evidence is valid. CI uploads it as `crap-debt-<coverage-leg>`
even when the CRAP step fails.

## Stamping after an improvement

When a change fixes problems, the baselines are stale (the check prints "improved; run
`make quality-stamp`"). Run `make quality-stamp` (the CRAP part needs a fresh `coverage.json`
from `make test`) and commit the baseline diff with the fix. The diff must only remove or
lower entries.

**Baselines only shrink.** The coordinator records decreases; never stamp over a new or
worse entry. A local coverage warning is not permission to increase a baseline.

## Test tiers

Four tiers, from the loop you run while building to the gate the coordinator runs before a merge.
Normal local gates use SQLite and Postgres; MySQL and MariaDB run in GitHub CI and the explicit
`cov-mysql` target above. Run one heavy suite at a time, and at most 6 xdist workers
(`DENDRUX_TEST_WORKERS` or `TEST_WORKERS=<n>` can lower it;
a larger number or `auto` ends at 6).

| Tier | Command | What runs | Measured (11 cores, idle machine) |
|---|---|---|---|
| 1. Fast | `make test-fast` | Only the tests your change can affect (pytest-testmon), SQLite, no coverage, without the tests marked `slow`, plus ruff on the changed files | first run on a fresh checkout 182 s; nothing changed 4 s; a leaf function 5 s; `tools/_coerce.py` 48 s (951 tests); `runtime/runner.py` 89 s (2,098 tests) |
| 2. Area | `make test-area AREA=<name>` | One area, SQLite, no coverage, without `slow` | `runtime` 69 s (3,378 tests), `mcp` 32 s, `stores` 29 s |
| 3. Full | `make ci` (with `DENDRUX_TEST_PG_URL` set) | Lint, types, architecture, the whole suite on SQLite and the Postgres matrix with coverage, CRAP, the quality gates, the API check | 584 s (the test step 560 s); SQLite only, no Postgres: 253 s |
| 4. Merge gate | `make test-merge` | `make ci`, then the store kit on SQLite and Postgres (serial), then the benchmarks (serial) | `make ci` + kit + 55 s of benchmarks |

Other targets: `make test-pg` (the Postgres integration matrix on its own, 6 workers, 225 s; 248 s
with 4; 951 s in one process), `make test-kit`, `make bench` (benchmarks, always one process),
`make test-lists`, `make test-shard-proof`. Numbers are from the 2026-11-01 run on the shared
Mac; the raw logs, with the load average of every run, are in
`owner-docs/build-0.6.0/work/t1-fasttests/`.

**How `test-fast` decides.** testmon records which source blocks each test executes. After a
change it re-runs only the tests that touched a changed block, so a leaf edit runs one test and an
edit to the runner runs a couple of thousand. The data lives in `.pytest_cache/testmondata`
(per checkout; delete it to start over). It does not see non-Python files (fixtures, JSON, the
slow list), so run `make test-area` before you hand work back. Plain `-m` makes testmon select
nothing, which is why the target passes `--testmon-forceselect`. `SLOW=1 make test-fast` includes
the slow tests.

**Markers** (applied centrally by `tests/conftest.py`, no per-file edits):

- `area_<name>` from the test's directory, then its file name (`tests/_helpers/tiers.py`):
  runtime, stores, store, loops, llm, mcp, db, skills, testing, tools, guardrails, media,
  notifiers, cli, strategies, docs, tooling, live. A file that matches no name rule is `runtime`.
  `python tools/test_tiers.py areas` lists them.
- `slow` for the tests that took 2 s or more in one measured run, listed in
  `tests/_helpers/tier_lists/slow.txt` (65 tests). Regenerate after a big change:
  `pytest tests/ -n 6 --durations=0 > log; python tools/test_tiers.py slow-list log --write`.
  `slow` only keeps a test out of tiers 1 and 2; tiers 3 and 4 and CI run it. The fast loop
  never runs the tests marked `slow`, not even on its first run with no testmon data: that is
  deliberate, and the full `make ci` at hand-back is what covers them (`SLOW=1 make test-fast`
  includes them).
- `flaky` for the tests listed in `tests/_helpers/tier_lists/flaky.txt`, each with a reason.
  A flaky test is rerun once when it fails (pytest-rerunfailures) and still runs everywhere; it is
  never skipped. A flake gets a root-cause fix, not a list entry, unless the cause is the machine
  (the list holds the one benchmark-only timing test today).
- `benchmark`: a test that asserts how fast something is, or measures the server. It is excluded
  from every parallel run and always runs in one process (`-m benchmark` forces `-n 0`). A test
  with its own `@pytest.mark.benchmark` needs nothing more; others are listed with a reason in
  `tests/_helpers/tier_lists/benchmark.txt`. **Rule:** a test that compares a measured time with a
  number under 2.5 s is a benchmark. A bound that is only a generous hang tripwire (several
  times the expected time, or a window the test computed itself) goes in
  `wallclock_reviewed.txt` instead. `tests/tooling/test_test_tiers.py` scans every test for such
  comparisons and fails on an unlisted one (`python tools/test_tiers.py audit-wallclock`).
  GitHub CI runs them in their own blocking `benchmarks` job (Python 3.12, SQLAlchemy 2.1, the
  Postgres service, `pytest -m benchmark -n 0`, no coverage), so a test that moved out of the
  parallel shards is still checked on every pull request. Locally they run in `make bench` and
  `make test-merge`. The shards skip them (`-m "not benchmark"`) and the combined coverage and
  CRAP gate do not include them.

**Postgres.** With `DENDRUX_TEST_PG_URL` set and no `-n` given, pytest
runs in parallel (6 workers at most), one database per worker. Only `-m "not benchmark"`,
`"not benchmark and not slow"` and `"not slow and not benchmark"` switch that on; any other
expression (for example `"not benchmark or flaky"`) or none could select a benchmark, so it
stays in one process. A run that names single tests
(`path::test`) or the conformance suite also stays in one process (its Postgres cells need that). A template database for
the workers was measured and dropped: building the schema takes 0.05 s per worker.

**CI shards.** The GitHub test job runs four shards per Python/SQLAlchemy leg. A shard is a set of
test files from `python tools/test_tiers.py shard --count 4 --index N`: a deterministic split
balanced by the per-file seconds in `tests/_helpers/tier_lists/file_weights.json` (regenerate with
`python tools/test_tiers.py weights log --write`). Each shard writes its own coverage data; the
`python-coverage` job combines the four files of a leg, applies the 80 percent floor and runs the
CRAP ratchet on the whole. Lint, types and architecture run once per leg, on shard 1.
`make test-shard-proof` checks the shards are a disjoint cover of the test files.

## Notes

- CRAP scores depend on coverage. Run `make test` with `DENDRUX_TEST_PG_URL` set (a private
  database ending in `_test`) so the Postgres tests run as they do in CI; with them skipped,
  coverage is lower and the ratchet reports regressions that are not real.
- The tool versions are pinned exactly. A tool upgrade can change scores, so it is its own
  commit that re-stamps the baselines and shows the difference.
- The ratchets look at `src/dendrux` only and skip `db/migrations`.
- Limits: the cognitive gate allows an existing function at or under 15 to grow up to 15;
  the cycles gate cannot see a new edge that stays inside an existing cycle (it models
  package `__init__` execution, but not conditional or `importlib` imports); the Semgrep
  `str-exc-in-log` rule looks for `str(exc)` and f-strings inside log calls and `error_info`,
  not for `%s` with the exception.
