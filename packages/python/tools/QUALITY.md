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
| `make crap` | `tools/crap_ratchet.py` | A new function scores CRAP above 30, a changed function scores above 30, or an existing function's score rises. Needs `coverage.json` from `make test`. Baseline: `crap_baseline.json`. |

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
- CRAP: add tests for the function, or make it simpler.

## Stamping after an improvement

When a change fixes problems, the baselines are stale (the check prints "improved; run
`make quality-stamp`"). Run `make quality-stamp` (the CRAP part needs a fresh `coverage.json`
from `make test`) and commit the baseline diff with the fix. The diff must only remove or
lower entries.

**Baselines may only shrink unless the coordinator approves.** Never stamp over a new or
worse entry. If one is truly unavoidable, the coordinator approves it, and the commit uses
`stamp --allow-regress --reason "..."` (the CRAP ratchet records it as a waiver in its
baseline) and says why in the commit message.

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
