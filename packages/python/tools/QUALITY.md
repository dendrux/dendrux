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

| Target | Script | Fails when |
|---|---|---|
| `make cognitive` | `tools/cognitive_ratchet.py` | A new function scores above 15 (complexipy cognitive complexity), or a function already above 15 scores higher. Baseline: `cognitive_baseline.json` (only functions above 15 are recorded; overloads and setters of one name count at their worst). |
| `make duplicates` | `tools/duplicate_ratchet.py` | A new block of 10 or more near-identical lines appears across two files (pylint's similarity engine; imports, signatures, comments and docstrings ignored), or a known block grows. Baseline: `duplicate_baseline.json`, block sizes per file pair; a block matches within 15 percent (at least 2 lines). |
| `make deadcode` | `tools/vulture_ratchet.py` | vulture (confidence 80 or more) finds a new unused name. Intentional public API and hooks go in `vulture_whitelist.py` with a comment. Baseline: `vulture_baseline.json`. |
| `make cycles` | `tools/import_cycles.py` | A module import cycle appears, a module joins a baselined cycle, or two cycles merge. Only top-level runtime imports count: `if TYPE_CHECKING:` blocks and imports inside functions are ignored. `python tools/import_cycles.py report` also lists the cycles that function-level imports hide (information only). Baseline: `import_cycles_baseline.json`. |
| `make semgrep` | `tools/semgrep_ratchet.py` | A rule in `.semgrep/dendrux.yml` has a new match, or more matches in a function that already had some; or a rule stops passing its fixture. Baseline: `semgrep_baseline.json` (rule, module, function). |
| `make crap` | `tools/crap_ratchet.py` | A new function scores CRAP above 30, a changed function scores above 30, or an existing function's score rises (by more than 0.5). If the function's code is unchanged (same AST digest) a rise of up to 25 percent of its baseline score is only a warning; a bigger rise fails. Needs `coverage.json` from `make test`. Baseline: `crap_baseline.json`. |

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
  fails. Any rise on a function whose code changed fails.

## Stamping after an improvement

When a change fixes problems, the baselines are stale (the check prints "improved; run
`make quality-stamp`"). Run `make quality-stamp` (the CRAP part needs a fresh `coverage.json`
from `make test`) and commit the baseline diff with the fix. The diff must only remove or
lower entries.

**Baselines may only shrink unless the coordinator approves.** Never stamp over a new or
worse entry. If one is truly unavoidable, the coordinator approves it, and the commit uses
`stamp --allow-regress --reason "..."` (the CRAP ratchet records it as a waiver in its
baseline) and says why in the commit message.

## Test tiers

Four tiers, from the loop you run while building to the gate the coordinator runs before a merge.
Local runs use SQLite and Postgres only (MySQL and MariaDB run in GitHub CI), one heavy suite at a
time, and at most 6 xdist workers (`DENDRUX_TEST_WORKERS` or `TEST_WORKERS=<n>` can lower it;
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
- `benchmark` (existing): measures the server, so it is excluded from every parallel run and
  always runs in one process (`-m benchmark` forces `-n 0`).

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
