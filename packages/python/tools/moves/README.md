# Temporary split evidence

This directory intentionally contains no real move manifest in T2b. The runnable
representative input is `tests/tooling/fixtures/moves/manifest.json` (its synthetic
base is for the in-memory checker, not the git CI runner).

Keep **one active cumulative manifest per original module** here, named `pr1.json`
for the first runner move, for example. Archive predecessor evidence outside code
and replace it when the next reviewed cumulative manifest lands. Every created
owner and the residual facade must be included. Do not include future owners.

A v1 manifest has these exact fields (unknown fields fail):

| Field | Meaning |
| --- | --- |
| `version` | Integer `1`. |
| `base` | Full immutable git SHA; runner uses `c2e3169700c14341f85599b23066a43886234123`. |
| `old` | Original path relative to `packages/python`. |
| `source_sha256` | SHA256 of the exact original source at `base`. |
| `sources` | Every cumulative destination path, including `old`, mapped to its exact current UTF-8 SHA256. |
| `owners` | Every original top-level statement key and class method key mapped to its current destination path; retained roots and methods count too. Repeated overload roots have one inventory entry. |
| `lexical` | Entries `{root, module, name, target, count}`; `module` is a destination path, `target` is the absolute owner attribute, and `count` is the exact number of replaced global reads. Locals cannot be normalized. |
| `hoists` | Entries `{root, module, statement, comment}` for a leading unconditional import and its exact comment tokens. A small inert stdlib set is accepted directly; other modules must already load through unconditional original imports. Optional `eager` is an ordered list of `{module, path, source, sha256}` intermediate dependency snapshots proving that path; CLI/CI verifies each against base and current sources. Optional imports and imports following executable statements are refused. |
| `annotations` | Entries `{root, module, host, count}` for the four reviewed SQL annotation edits only. |
| `metadata` | Entries `{root, module, attribute, value}` for original classes and public runner functions only; moved originals require their original `__module__` pin. SQL methods cannot have pins. |
| `scaffolding` | `_StoreHost` / `_SweepStore` mapped to `{module, source}` containing the exact reviewed class declaration. Host runtime methods, method annotation slots, changed signatures and copied implementations fail. |
| `effects` | Named moved singleton assignments only: `logger`, `_STOP_REASONS`, `_UNSET_DEPTH`, `_WAIT_MARK_LABELS`. Their RHS and count still undergo ordinary AST comparison. The sole RHS normalization pins `logging.getLogger(__name__)` to the original module-name literal; leaving `__name__` in a new owner or changing its logger constructor fails. |
| `seams` | Imported patch-seam local name → absolute owning attribute. Include every activated seam and all residual consumers. `current_skill_session` may retain the reviewed `_run_entry` alias. This inventory requires reviewer verification against the frozen imported-seam evidence. |
| `providers` | Original imported base identity → `{path, sha256, class}`. `class` is the exact AST-equivalent original class source; the file hash and class are verified against both `base` and current tracked source. Base-free original provider classes are supported. |

All root/module references use the same path keys as `sources`. For a moved method,
`root` remains the original `SQLStateStore.method`; the checker discovers its new
operation host. Operation classes must never be put in `--allow-new`.

Track `pr1-emitted.json` beside the manifest (the result of `--emit-map`) and
`pr1-expanded.json` (that map plus every imported seam's actual owner module).
The CI runner compares both maps with the proven inventory. Neither is a manifest.
SQL also needs tracked `sql-base.json` containing exactly `{base, source_sha256}`
for the actual post-T2 prerequisite commit. Do not substitute the runner pin for it.

Run from `packages/python`:

```sh
make move-check BASE=c2e3169700c14341f85599b23066a43886234123 OLD=src/dendrux/runtime/runner.py \
  NEW="src/dendrux/runtime/_run_errors.py src/dendrux/runtime/_run_pending.py src/dendrux/runtime/_run_contracts.py src/dendrux/runtime/_run_observers.py" \
  MOVE_ARGS="--require-reexports --allow-effects logger _STOP_REASONS _UNSET_DEPTH --move-manifest tools/moves/pr1.json --emit-map tools/moves/pr1-emitted.json"
make move-proof
```

`move-proof` requires git-tracked inputs and full history. With no manifest it fails;
CI discovery skips the proof job only while no split sources exist. Deleting a
manifest while retaining split sources fails discovery. Changing any source hash,
base, provider snapshot or cumulative map fails the proof.

After PR10, archive evidence outside code and remove the manifests/maps/receipt and
the two temporary workflow jobs. Keep the reusable checkers, fixtures, owner guards,
architecture contract and key-transfer tool. Remove discovery wiring with the jobs.

Review hardening: `seams` entries must have an original imported binding and at least
one matching lexical edit; unused entries fail. Metadata pins must follow their
class/function definition. Both checker modes reject all star imports and unexplained
unconditional top-level imports. Imports must preserve original runtime bindings or
be explained by exact hoists, lexical owners/seams, re-exports, provider bases or
scaffold typing; arbitrary additions beside an allowed import still fail.

Use only the move-check CLI or `make move-proof` as proof gates. The in-memory
`check_move` API deliberately trusts its snapshots for comparison/fixtures; the two
supported gates verify immutable/current provider sources before invoking it.
