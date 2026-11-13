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

Tests that construct a MySQL-family engine or use a driver exception skip individually
when `asyncmy` is absent; pure error-classification tests still run. The dedicated MySQL
and MariaDB jobs install `[dev,mysql]`, require a successful driver import, and run those
contract/error test files alongside `tests/db`, so the optional cases remain enforced.

Published runtime requirements (including SQLAlchemy, provider SDKs and database drivers)
remain ranges. CI continues to resolve their newest allowed versions and tests both
SQLAlchemy minors. The direct tool pins are not a transitive dependency lockfile. A tool
upgrade is a deliberate change with checks on the supported Python/SQLAlchemy matrix;
baseline decreases are stamped only by the coordinator. Never restamp a baseline to hide
a regression.

Provider SDK floors are Anthropic `0.77.0` and OpenAI `1.97.2`, with upper bounds `<1.0`
and `<3.0`. `openrouter` inherits the OpenAI extra and `all` includes both SDKs.
When changing these floors, run the transport and provider tests at the proposed
versions, their preceding releases, installed versions and the newer SDK overlay.
Use isolated pip target directories so the shared development environment stays
unchanged.

| Target | Script | Fails when |
|---|---|---|
| `make cognitive` | `tools/cognitive_ratchet.py` | A new function scores above 15 (complexipy cognitive complexity), or a function already above 15 scores higher. Baseline: `cognitive_baseline.json` (only functions above 15 are recorded; overloads and setters of one name count at their worst). |
| `make size` | `tools/size_ratchet.py` | A new module exceeds 1,000 physical lines, a new function exceeds 80, or an existing offender grows. Baseline: `size_baseline.json`, measured lengths keyed by `module` and `module:qualname`. |
| `make ruff-ratchet` | `tools/ruff_ratchet.py` | A file gains a finding for a new rule, a new file has findings, or a file/rule count increases. Rules: PLC0415, PLR0911, PLR0912, PLR0915, BLE001. Baseline: `ruff_baseline.json`, counts keyed by `module:RULE`. |
| `make duplicates` | `tools/duplicate_ratchet.py` | A new block of 10 or more near-identical lines appears across two files (pylint's similarity engine; imports, signatures, comments and docstrings ignored), or a known block grows. Baseline: `duplicate_baseline.json`, block sizes per file pair; a block matches within 15 percent (at least 2 lines). |
| `make deadcode` | `tools/vulture_ratchet.py` | vulture (confidence 80 or more) finds a new unused name. Intentional public API and hooks go in `vulture_whitelist.py` with a comment. Baseline: `vulture_baseline.json`. |
| `make cycles` | `tools/import_cycles.py` | A module import cycle appears, a module joins a baselined cycle, or two cycles merge. Only top-level runtime imports count: `if TYPE_CHECKING:` blocks and imports inside functions are ignored. `python tools/import_cycles.py report` also lists the cycles that function-level imports hide (information only). Baseline: `import_cycles_baseline.json`. |
| `make semgrep` | `tools/semgrep_ratchet.py` | A rule in `.semgrep/dendrux.yml` has a new match, or more matches in a function that already had some; or a rule stops passing its fixture. Baseline: `semgrep_baseline.json` (rule, module, function). |
| `make crap` | `tools/crap_ratchet.py` | A new function scores above 30; complexity rises and the function scores above 30; or edited code exceeds its own recorded CRAP baseline. An edited function may remain above 30 when neither complexity nor CRAP rises. Unchanged code retains the 0.5-point noise band and the warning allowance for rises up to 25 percent. Backend-only local coverage is handled explicitly below. Needs `coverage.json` from `make test`. Baseline: `crap_baseline.json`. |
| `make stale-patches` | `tools/stale_patch_lint.py` | A test patches a name its module only re-exports and never reads at call time, so the patch silently does nothing. Zero allowed; a missing `stale_patch_baseline.json` means none allowed. Part of `make lint`. |

### Size and Ruff counts

Both gates scan only `src/dendrux` (including `testing`) and exclude migration
directories. Run from `packages/python`:

```sh
python tools/size_ratchet.py report
python tools/ruff_ratchet.py report
```

Each script accepts `check`, `stamp`, `report`, and `move`, plus `--src` and `--baseline`.
`report` prints measurements without reading or writing a baseline. `check` exits 0
on success, 1 on regressions, and 2 when its baseline is missing. Tool/configuration
errors also fail. Each invocation prints measured totals; reports also list individual
offenders. No coverage file is needed.

Size counts every physical file line, including comments and blank lines. A function
spans its first decorator (or `def` / `async def`) through its last AST line, inclusive;
comments and blank lines within that span count. Nested functions are included in the
outer span and measured independently as `outer.<locals>.inner`. Classes qualify
methods as `Class.method`. Overload definitions from `typing` or `typing_extensions`
(including import aliases) are excluded; implementations count. Repeated qualified
names, such as property getters/setters, keep the longest implementation. The exact
data-module allow list is `dendrux._codes`, `dendrux.types`; those modules are exempt
from size measurement. Other modules with those basenames are still checked.

Only sizes above the hard limits are recorded. Below-limit code can grow up to its
limit. Existing offenders may stay the same or shrink. Keys use names, not line
numbers. An implicit rename of an over-limit file/function fails as a new offender,
even if unchanged. For a reviewed pure move, the coordinator can explicitly transfer
its allowance with `move`, as described below. Do not delete or increase a baseline
entry to make the rename pass.

Ruff uses the pinned interpreter's `python -m ruff`, the project's `pyproject.toml`
and `--extend-select` for the five rules. Existing selections, ignores, per-file
ignores, thresholds and `noqa` remain in effect. Other diagnostics are left to
`make lint`. Only the five rules enter the baseline. Counts cannot move between
files or rules to offset a regression. Ruff's defaults are 6 returns, 12 branches
and 50 statements unless the project config overrides them.

Suppressions of those five rules must include a reason after the code list on the
same line, or in a standalone comment immediately above. For example,
`# noqa: BLE001 - notifier fan-out isolation` or an immediately preceding
`# Optional extra stays lazy to keep the base install importable.` Blank/punctuation-only
comments, another noqa directive, and code names alone are not reasons. The check
examines Python comment tokens, including blanket and file-level noqa, and ignores
string literals and migrations. Reviewers still judge whether the reason is valid
under AGENTS §6.2/§6.3; the tool checks its presence.

`check` reports unreasoned suppressions as `file:line: CODE noqa lacks reason`.
Reason debt is a separate shrink-only count per module/rule in `ruff_baseline.json`,
under `noqa-reason|module|RULE` keys. Counts cannot be offset across modules/rules;
new/increased debt fails, and `stamp` refuses it. Initial coordinator stamping may
record existing debt; subsequent fixes remove/lower those entries. Report prints the
exact locations even for existing debt. Today's source has zero unreasoned suppressions,
so no reason-debt entries are needed. Reason debt cannot be transferred with `move`.

The coordinator initializes the missing baselines with `stamp` (R5 of the rollout)
and stamps decreases once per merge (AGENTS §0 item 7). Builders leave generated
worktree baselines uncommitted. After initialization these two scripts refuse every
new/increased offender, including during `stamp`; there is no `--allow-regress`
override. `make quality-stamp` includes both scripts. Never commit baseline values
from a builder task.

#### Explicit allowance transfers (coordinator only)

`move` is for pure code moves and file splits. Reviewers check both the code move and
the explicit move list in the PR; the tools validate allowances and current tree
presence, not semantic equivalence. Ordinary `check` and `stamp` still fail on implicit
renames. Run with an existing baseline, using one or more `OLD_KEY NEW_KEY` pairs:

```sh
python tools/size_ratchet.py move dendrux.old:Box.run dendrux.new:Box.run
python tools/size_ratchet.py move dendrux.old dendrux.new_a dendrux.old dendrux.new_b
python tools/ruff_ratchet.py move dendrux.old:PLC0415 dendrux.new_a:PLC0415 dendrux.old:PLC0415 dendrux.new_b:PLC0415
```

Size keys are exactly the baseline's module or `module:qualname`; Ruff keys are
`module:RULE`, with a separate pair for each rule being transferred. Supply any
`--src` / `--baseline` options after the pairs. The operation:

- Requires the old key's allowance to exist. Function sources must be absent from
  the tree, including overload stubs. A moved function can leave its old module in
  place. File sources may be deleted or retained with a smaller measured value.
- Requires every destination to be a current offender of the same kind (file or
  function for size; the same ratcheted rule for Ruff). Destinations already in the
  baseline, duplicate destinations and source/destination cycles are refused.
- Allows functions to move one-to-one. A file's allowance may split across several
  destination files by repeating its old key. Every destination gets its measured
  value. Their sum plus the retained old file's current size must not exceed the
  recorded allowance. A retained old file records its measured size, even if now
  below the limit. Ruff applies the same budget to finding counts separately for
  each rule (including a zero retained count), without pooling rules or source files.
  An unchanged/grown source or insufficient remaining allowance refuses the batch.
- Validates the entire batch before writing. Any invalid pair refuses the whole
  batch without changing the baseline. Missing baselines and malformed command
  arguments exit 2; invalid transfers exit 1; successful transfers exit 0 and print
  each transferred key/value.

On success deleted sources' entries are removed; retained sources and destinations
record current measurements. Unrelated entries stay unchanged. `move` does not stamp other improvements
or waive unrelated regressions. Finish all reviewed transfers, then run both checks;
unlisted offenders still fail. Final baseline diffs and the move list belong to the
coordinator's PR. Builders do not run transfers against the rollout baselines.

### Semgrep rules that stop a shared definition being copied

Each rule has one definition site; the shapes below, written anywhere else, are a match.
Path scope and `allow_in` (module:function) live in the rule's `metadata`; `allow_in` lists
only places that need the copy today. Rules read the spelling in the source: they do not
follow import aliases (`RunStatus as Status`) or aliased callables, except a writer bound
to a local name first (`emit = _emit_event_safe if ...`), which is matched for run.started
and run.error.

| Rule | Shapes matched (definition site) and kept exceptions |
|---|---|
| `stored-event-name-literal` | a stored event name as a string (`StoredEventType`, `_lifecycle.py`; migrations excluded) |
| `lifecycle-reason-literal` | a reason string as a keyword or dict value, comparison, tuple/list/set membership, assignment or conditional-expression value (`_lifecycle.py`, `_codes.py`); the words `requested` and `budget` only where the name or key contains "reason" |
| `storage-format-literal` | the number 2 compared (either side), tested with `in`, or assigned for `storage_format` (`STORAGE_FORMAT_CHECKPOINTED`); format 1 is not matched |
| `meta-key-literal` | a `dendrux.*` key as a dict key, subscript, get/pop/setdefault argument, `in` operand or constant assignment (`_meta_keys.py`; OTel excluded); docstrings, module names and logger names are not keys |
| `default-sqlite-url-literal` | `./dendrux.db` inside a string outside `db/session.py` (`_codes.py` messages excluded); prose in a docstring also matches |
| `connection-loss-code-copy` | `startswith("08")` (also in a tuple), `[:2]`/`[0:2]` tests against `"08"`, and `57P01` to `57P03`, outside `_is_connection_loss_code`; `53300` is not matched |
| `new-run-fingerprint-outside-helper` | `compute_idempotency_fingerprint(` and the output-type name f-string under `runtime/`, `agent.py`, `loops/`, outside `_new_run_metadata`; kept: `_idempotent_hit_after_image_error` |
| `run-started-outside-startup` | `StoredEventType.RUN_STARTED` passed to an event write (also through a local writer alias) outside `_start_new_run` |
| `error-finalize-outside-helper` | finalizing with the error status, or writing `run.error`, outside `_persist_run_failure`; kept: `_reclaim._handle_rebuild_failure` |
| `cancel-finalize-outside-helper` | finalizing with the cancelled status outside `_persist_run_cancellation`; kept: `_finalize_loop_cancelled`, `_pause_with_event`, `Agent._cancel_one` |

`hardcoded-terminal-status-set` also matches `RunStatus.X.value` operands.

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
  It follows clock samples through collection appends, iteration and comprehensions too,
  including derived bounds such as `max(stalls) < 0.025`.
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

## Split tooling (moving code between modules)

Three tools prove a split changed nothing but where code lives. All are pure AST work and
need no database.

| Tool | Use |
|---|---|
| `tools/move_check.py` | Fails closed. Compares the old module at a git ref with the old and new modules now. Every top-level statement must be present exactly once, in one module, with an identical `ast.dump`, in the same relative order, with the same count of repeated names. Only function, class and constant definitions may move; any other statement (calls, `if`/`try`/`for`/`with`, `__all__`) must stay unchanged in the old module (an assignment whose value calls something needs `--allow-effects NAME`). `from __future__` imports must match in every module. Ignored: import statements, `if TYPE_CHECKING:` blocks that only import, the module docstring, a literal `__all__` in a *new* module. A statement may not change kind (a class may not become a function or an assignment). A movable constant without `--allow-effects` is a literal, a name, a dotted name, or a tuple, list, set or dict of those; anything else (subscripts, operators, calls, comprehensions) needs the allowance. A method may move to a new mixin class that has no decorator, keyword or base (other than an `--allow-new` name) and is listed before the original bases; it fails (manual review) if it uses zero-argument `super()`, `__class__`, a name Python mangles (`__x`), or a decorator, default or annotation that reads a name bound in the class body. Only `TYPE_CHECKING` imported from `typing` (and not rebound) is a type-only guard; any other `if` is executable and compared. Anything missing, duplicated, extra, reordered or changed fails with a diff of the first mismatch. |
| `tools/patch_codemod.py` | Rewrites `unittest.mock.patch`, `patch.object`, `mocker.patch` and `monkeypatch.setattr/delattr` targets in `tests/` from the old module to the module that now defines the name (string targets, positional or keyword, and module aliases). Calls are recognised by what they resolve to through the file's imports and scopes, never by spelling; MonkeyPatch objects are recognised only when their factory resolves to `pytest.MonkeyPatch`. Relative imports are resolved against the file's package. A shadowed alias is left alone with a note; an alias bound more than one way, or a patch helper imported relatively, is left alone and reported as unresolved. An existing import is reused only if it is an unconditional module-level import before the use; otherwise a fresh import goes into the leading import block. Dry run prints a diff; `--write` applies it. |
| `tools/stale_patch_lint.py` | `make stale-patches`, part of `make lint` (so `make ci`) and the CI quality job: fails when a test patches a name its module only re-exports. A module reads an import only when code inside a function resolves to that module-level binding; a parameter of the same name, an import-time copy (`Y = X`), a decorator or a default argument is not a read. Patching below a re-export (`mod.X.attr`) mutates the shared object and is allowed. Relative imports count as imports. A patch site it cannot resolve (ambiguous binding, relative patch helper) is printed as `unresolved` and fails the lint: a human must read it. |

To prove a move PR (run from `packages/python`; `BASE` is the ref before the move):

```sh
make move-check BASE=<ref> OLD=src/dendrux/runtime/runner.py \
  NEW="src/dendrux/runtime/a.py src/dendrux/runtime/b.py" \
  MOVE_ARGS="--require-reexports --emit-map /abs/path/map.json"
python tools/patch_codemod.py --map /abs/path/map.json tests          # read the diff
python tools/patch_codemod.py --map /abs/path/map.json --write tests
ruff check --select F401 --fix tests && ruff format tests
make stale-patches
```

`--allow-new NAME...` admits a named new definition (for example a shared attribute base) and
the output lists it, so a reviewer reads it by hand. `--require-reexports` also fails unless each
moved public name has an unconditional, absolute, module-level `from <new module> import name` in
the old module (a deleted old module fails too) (a type-checking-only, function-local, conditional, relative or wrong-source import
does not count); names listed in the old `__all__` must always stay bound. Not proved: import
order and cycles (`make arch`, `make cycles`), registration decorators, and the body of any
`--allow-new` definition.

## Test timeout

`pytest-timeout` (dev extra, exact pin) stops any test that runs longer than `timeout = 300`
seconds and fails it with the stack. It sits above `faulthandler_timeout = 240`, which dumps
every thread's stack at 240 s, so a hang shows where it is stuck before the test is failed.
Both are in `pyproject.toml`. The benchmarks (`make bench` and the CI `benchmarks` job) run with
the same explicit `--timeout=600`, because the migration benchmark legitimately runs longer than a
unit test; a wiring test keeps the two flags equal and inside the job's time limit. To give one test
more time, mark it `@pytest.mark.timeout(<seconds>)` and say why.

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
