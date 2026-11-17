# Split move evidence

The 0.7 split (runner and SQL store) is complete. Its manifests, emitted/expanded
maps and the SQL base receipt were retired, and so were the two temporary CI jobs
(`move-manifests`, `move-proof`). This directory holds no active manifest, so
`make move-proof` / `python tools/move_ci.py` reports "no active manifests" and
exits 0.

The reusable tools stay for future splits: `tools/move_check.py`,
`tools/move_ci.py`, `tools/_move_*.py`, `tools/move_keys.py`,
`tools/patch_codemod.py`, the stale-patch lint and `make move-check`. The
runnable representative input is `tests/tooling/fixtures/moves/manifest.json`.

## Starting a future split

1. Pick the immutable base SHA and write a v1 manifest here, one active
   cumulative manifest per original module (for example `pr1.json`). Include
   every created owner and the residual facade, and no future owners.
2. Generate the evidence with `make move-check BASE=<sha> OLD=<path>
   NEW="<paths>" MOVE_ARGS="--move-manifest tools/moves/pr1.json --emit-map
   tools/moves/pr1-emitted.json"`. Track `pr1-emitted.json` and
   `pr1-expanded.json` (the emitted map plus every imported seam's owner module)
   beside it. Neither is a manifest.
3. `make move-proof` verifies every tracked manifest from its base. It needs
   git-tracked inputs and full history.
4. To run it in CI, restore a `move-proof` job (full-history checkout, runs
   `python tools/move_ci.py`) for the lifetime of the split, then retire the
   manifests and the job again when the split is done.
5. A SQL-style split whose base is not the runner pin also needs a tracked
   `sql-base.json` receipt of `{base, source_sha256}` (see `_check_base`).

## Manifest schema (v1; unknown fields fail)

| Field | Meaning |
| --- | --- |
| `version` | Integer `1`. |
| `base` | Full immutable git SHA. |
| `old` | Original path relative to `packages/python`. |
| `source_sha256` | SHA256 of the exact original source at `base`. |
| `sources` | Every cumulative destination path, including `old`, mapped to its exact current UTF-8 SHA256. |
| `owners` | Every original top-level statement key and class method key mapped to its current destination path; retained roots and methods count too. Repeated overload roots have one inventory entry. |
| `lexical` | Entries `{root, module, name, target, count}`; `target` is the absolute owner attribute, `count` the exact number of replaced global reads. Locals cannot be normalized. |
| `hoists` | Entries `{root, module, statement, comment}` for a leading unconditional import and its exact comment tokens. Optional `eager` lists `{module, path, source, sha256}` intermediate dependency snapshots, verified against base and current sources. |
| `annotations` | Entries `{root, module, host, count}` for reviewed SQL annotation edits. |
| `metadata` | Entries `{root, module, attribute, value}` for original classes and public runner functions; moved originals require their original `__module__` pin. |
| `scaffolding` | `_StoreHost` / `_SweepStore` mapped to `{module, source}` with the exact reviewed class declaration. |
| `effects` | Named moved singleton assignments only (`logger`, `_STOP_REASONS`, `_UNSET_DEPTH`, `_WAIT_MARK_LABELS`); `logging.getLogger(__name__)` is pinned to the original module-name literal. |
| `seams` | Imported patch-seam local name to absolute owning attribute; include every activated seam and all residual consumers. |
| `providers` | Original imported base identity to `{path, sha256, class}`, verified against both `base` and current tracked source. |

Use only the move-check CLI or `make move-proof` as proof gates. The in-memory
`check_move` API trusts its snapshots and is for comparison and fixtures.
Both checkers reject star imports and unexplained unconditional top-level
imports, and inspect guarded (`TYPE_CHECKING`) imports too.
