# Formal model: run ownership, cancel and recovery

Dendrux promises that a run is never run twice and never lost. That promise rests on a few rules
about the run table: who may write a run, how cancellation stops it, and how `sweep()` and
`reclaim()` recover it after a crash. `DendruxRuns.tla` writes those rules down as a TLA+ state
machine. The TLC model checker then explores every interleaving of a few workers, one root run
with two subagents, crashes, sweeps and cancels, and checks the properties below in every
reachable state.

This checks the **design**, not the Python code. The ordinary test suite checks the code. The
table in [The bridge to the code](#the-bridge-to-the-code) maps each model action to the
function that implements it, so a change to one of those functions is a prompt to update the
model.

## What is modelled

| Part | In the model |
|---|---|
| Runs | One managed root and two children. One child runs the ReAct loop (reclaimable); the other runs SingleCall (never reclaimed, sweep fails it) |
| Statuses | `absent` (row not created), `running`, `paused` (any external pause: client tool, human input, approval, requested pause), `waiting` (stored as `paused` with a managed wait on children), and the terminal `success`, `error`, `cancelled` |
| Owner epoch | `claim_epoch` per run. Claims of a paused run, adoptions, reclaims, the child bind (for the parent) and sweep's CASes bump it; the owner's own pause or finalize and the canceller's CAS do not. Every owner write is fenced on it |
| Owners | An owner task holds one epoch of one run. A worker process can crash at any step, which removes all its tasks |
| Sweep | Marks a stale ReAct row reclaimable (epoch + 1), finalizes a stale row that cannot be reclaimed (or has a cancel request), fails an old external pause. It may treat **any** running row as stale: an owner that looks dead may be alive but slow |
| Reclaim | Takes a marked row at epoch + 1, or finalizes it `cancelled` if a cancel request is set |
| Cancel | `cancel_run` on the root, which always cascades for a managed root: mark the root, scan its children, mark them, then per run `request_cancel` and a CAS over the pause statuses only, children before the root. A canceller can run in another process (no task registry), can crash at any step, and can be called again |
| Children | The parent binds one or two children in one transaction (refused if the parent is cancel-requested or fenced), parks as a managed wait, and later adopts: a read (all children terminal, parent not cancelled) and then a claim |

## Properties

Safety, checked in every reachable state:

| Property | Meaning |
|---|---|
| `SingleWriter` | At most one owner task can pass a run's write fence at any time |
| `StaleOwnerFenced` | A task that can pass the fence is the task the run was last handed to. Once a run is swept, marked or finalized by someone else, its old owner's writes fail |
| `StaleWriteFails` | No owner write (a fenced write or a finalize) ever succeeded for an owner the run had been taken from |
| `NoCancellerFinalizesRunning` | A canceller never moves a `running` row to a terminal status. The owner stops itself at its next checkpoint |
| `OrphanWindow` | A child may outlive its terminal parent only while `running` with its own cancel request set; it then stops at its next checkpoint or sweep settles it. It is never paused or unflagged under a terminal parent |
| `TerminalFinal` | A terminal run never changes status again. A cancelled run is never revived |
| `RequestKept` | A cancel request is cleared only when the run becomes terminal |

Liveness, under the fairness assumptions below:

| Property | Meaning |
|---|---|
| `CancelLeadsToTerminal` | A cancel request eventually leads to a terminal run |
| `CrashedOwnerRecovered` | A running run whose owner is gone is eventually owned again (reclaimed) or settled (terminal) |
| `OrphansSettle` | The children of a terminal parent eventually become terminal |

Fairness assumptions (`Fairness` in the spec):

- A live owner task keeps reaching checkpoints, and after it sees a cancel it finalizes.
- The developer calls `sweep(on_stale="reclaim")` and `reclaim()` for what sweep lists, for rows
  whose owner is gone. A live owner renews its lease, so a sane threshold does not keep sweeping
  it. Safety still lets sweep hit live owners at any time.
- A `cancel_run` call that has started runs to its end. If it crashes, the caller calls it
  again (`cancel_run` is repeatable).
- Nothing forces a crash, a resume of a paused run, an adoption, an abandoned-waiting sweep or
  `on_stale="error"`.

Each assumption is one weak-fairness condition over a group of steps. In this bounded model that
is as strong as one condition per task or per run: every step in a group changes the state, and
a behavior has only finitely many of them, so no enabled step can be starved forever.

The liveness of cancellation depends on the retry assumption. If a `cancel_run` call dies after
marking a root that is waiting on its children, and nobody calls it again, the root stays
`paused` with the request set: adoption refuses a cancel-requested parent, and the waiting sweep
leaves managed waits alone. `bugs/NoCancelRetry.cfg` shows this trace. The fix is the
documented one: call `cancel_run` again.

## Bug variants

Each bug switch (a constant in the spec) breaks one rule. With the switch on, TLC must find a
counterexample. This shows the properties are strong enough to notice the bug, in the same way a
revert check shows a test is.

| Config | Bug | Caught by |
|---|---|---|
| `bugs/BugCancelRunning.cfg` | The cancel CAS also accepts `running` | `NoCancellerFinalizesRunning` |
| `bugs/BugSweepNoBump.cfg` | Sweep's CASes do not bump the claim epoch | `StaleWriteFails`: the owner sweep declared dead keeps writing |
| `bugs/BugFinalizeNoEpoch.cfg` | The owner's finalize checks the status but not the epoch | `StaleWriteFails`: a fenced owner still finalizes the run |
| `bugs/BugPauseIgnoresCancel.cfg` | The owner's pause CAS ignores `cancel_requested` | `OrphanWindow`: a child pauses after the cascade settled it, under a cancelled parent |
| `bugs/BugBindIgnoresCancel.cfg` | The child bind ignores the parent's cancel mark | `OrphanWindow`: a child is created after the cascade scanned, and is never flagged |
| `bugs/NoCancelRetry.cfg` | No bug; the retry assumption is dropped | `CancelLeadsToTerminal` (see above) |

## Configs, bounds and why they are enough

| Config | Checks | Workers | Cancellers | Children | Crashes (worker, per canceller) | Pauses per run |
|---|---|---|---|---|---|---|
| `DendruxRuns.cfg` | safety | 2 (symmetric) | 1 | 2 (one ReAct, one SingleCall) | 1, 1 | 1 |
| `DendruxRunsTwoCancels.cfg` | safety | 2 (symmetric) | 2 | 2 | 0, 0 | 1 |
| `DendruxRunsLive.cfg` | liveness | 1 | 1 | 2 | 1, 1 | 0 |
| `DendruxRunsLivePauses.cfg` | liveness | 1 | 1 | 1 (ReAct) | 1, 1 | 1 |

All configs use reclaim limit 1 and a parallel limit of 2. TLC cannot use symmetry when it checks
liveness, and a liveness check costs far more per state, so the liveness configs are smaller and
split in two.

| Bound | Why it is enough |
|---|---|
| 2 workers (safety) | Enough for a slow owner and a second worker that reclaims or resumes the same run. A third adds no new kind of conflict: every write is a CAS on one row, so each conflict is between the current holder and one other writer. A single worker can also hold a stale task and a fresh one for the same run |
| 1 worker (liveness) | Liveness is about recovery after a crash; the worker can reclaim its own runs, so a second worker adds states but no new recovery path |
| 1 root, 2 children | Two children give a real fan-out (N = 2), the all-terminal fan-in, and siblings that settle in either order. One ReAct and one SingleCall child cover both recovery paths |
| Depth 1 | Each cascade level repeats the same steps: mark a frontier, scan its children, mark them, settle deepest first. Binds under a parent serialize with that parent's mark on the parent row lock, and the root-stopped read covers every depth |
| 2 cancellers | Concurrent cancels, and a mixed Agent and module caller. Run without crashes to keep the run time down; crashes are covered with one canceller |
| 1 crash | One crash at any step already produces every recovery state: an unowned running row, a marked row, an unadopted wait, a half-done cancel. More crashes repeat them |
| 1 pause per run | One pause and one claim per run exercise pause, resume, cancel of a paused run and the abandoned sweep, and keep the state space finite |
| Reclaim limit 1 | One reclaim, then the limit path (sweep finalizes with `reclaim_limit`) |

These bounds are an engineering judgement, not a proof that larger instances behave the same.

## Simplifications and what they leave out

- **Transactions are atomic and never fail.** Each action is one committed transaction or a lost
  CAS. Database errors, retries, lost acknowledgements, read-back and timeouts are not modelled.
  The model says nothing about errors such as `PersistenceError`; it only says that no
  interleaving of successful transactions breaks a property.
- **Time is not modelled.** "Stale" is a free choice of sweep, so sweep may act on a live owner at
  any time. Lease renewal, thresholds and clocks are left out.
- **Fenced writes are one action.** Checkpoints, tool-call completion, submission consumption and
  trace writes all use the same fence (epoch only, not status). They are one `OwnerWrite`.
- **Terminal statuses are folded.** `max_iterations` behaves like `success` or `error` here.
  `pending` is never stored by these code paths, so it is left out.
- **One fan-out round.** The parent binds children once. A second delegation round repeats the same
  steps.
- **Siblings settle in any order.** The code settles them in id order; the model allows every
  order, which includes that one.
- **Same-process signals are subsumed.** An Agent canceller in the owner's process also cancels
  the owner's task. In the model that is the owner seeing the request at once, which is one of
  the interleavings already allowed. The module `dendrux.cancel_run` has no task registry; the
  owner stops at a later checkpoint, which is also allowed.
- **`release_run_claim` is the pause action.** Its CAS (running, epoch, no cancel request) and its
  cancel fallback are the same as the owner's pause.
- **Pre-checks are reads, not guards.** The adoption read (all children terminal, parent not
  cancelled) is a separate step from the claim, so a cancel can land between them, as in the code.
  The routing pre-check before a resume of a paused child is not modelled: the claim may happen
  under a stopped root, and the child then stops at its first checkpoint.
- **Not modelled:** tenants, governance and skill checks, idempotency keys, events and their
  ordering, pause requests (`request_pause`), notes, budgets, guardrails, the provider, and the
  legacy (non-checkpointed) storage format.

## The bridge to the code

Paths are under `packages/python/src/dendrux/`.

| TLA+ action | Code |
|---|---|
| `Start` | `runtime/stores/sql/_sql_runs.py:_SqlRuns.create_run` (row created `running`, epoch 0) |
| `OwnerWrite` | `runtime/stores/sql/_sql_writes.py:_SqlWrites._fence`, reached through `runtime/state.py:_ClaimedStateStore.__getattr__` (injects the claim epoch) from `checkpoint_step`, `complete_tool_call`, `consume_submission` and the other claimed writes |
| `OwnerCheckpoint` | `runtime/state.py:_ClaimedStateStore.get_run_control`; for managed runs `runtime/_managed_cancel.py:checkpoint` and `_managed_control`, reading `runtime/stores/sql/_managed_reads.py:get_managed_run_control` (own request or root stopped, `_managed_stop_predicate`) |
| `OwnerStop` | `runtime/_run_outcome.py:_finalize_loop_cancelled` and `runtime/_managed_cancel.py:cancelled_result`, through `runtime/_run_events.py:_transition_with_event` to `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.finalize_with_event` / `_cas_finalize` (status `running`, epoch) |
| `OwnerFinish` | `runtime/_run_outcome.py:_persist_loop_outcome` (terminal CAS on `running` and the epoch); in-process cancellation `runtime/_run_events.py:_persist_task_cancellation`; the fan-out driver's cleanup of its own claims (`runtime/_fanout_ledger.py`) |
| `OwnerPause` | `runtime/_run_outcome.py:_persist_pause` → `runtime/_run_events.py:_pause_with_event` → `runtime/stores/sql/_sql_pause.py:_SqlPause.pause_with_event` / `try_pause_run`; also `runtime/stores/sql/_sql_claims.py:_SqlClaims.release_run_claim` |
| `OwnerBind` | `runtime/stores/sql/_sql_managed.py:_SqlManaged.bind_managed_child` (`_admit_parent`, `_bind_park_parent`, `_bind_child`) and `runtime/stores/sql/_sql_fanout.py:bind_managed_children` |
| `ClaimPaused` | `runtime/stores/sql/_sql_claims.py:_SqlClaims.claim_paused_run` and `submit_and_claim` |
| `AdoptRead` | `runtime/_managed_fanout.py:settle_parent` (children terminal) and `runtime/_managed_claim.py:claim_fanout_adoption` (`cancellation_requested`, pause read) |
| `AdoptClaim` | `runtime/stores/sql/_sql_claims.py:_SqlClaims.submit_and_claim` with `runtime/_managed_claim.py:fanout_adoption_validator` |
| `Crash` | Process death; nothing in the code runs |
| `SweepMark` | `runtime/stores/sql/_sql_leases.py:_SqlLeases._scan_expired_runs`, mark branch (reached from `runtime/sweep.py:sweep`) |
| `SweepStale` | `runtime/stores/sql/_sql_leases.py:_SqlLeases._scan_expired_runs`, finalize branch; `runtime/stores/sql/_sql_sweep_scan.py:_SweepScan.sweep_row` for running rows |
| `SweepAbandoned` | `runtime/stores/sql/_sql_sweep_scan.py:_SweepScan.sweep_row` for waiting rows; managed waits skipped by `_SweepScan.skipped` |
| `Reclaim` | `runtime/stores/sql/_sql_leases.py:_SqlLeases.reclaim_run` |
| `CStart`, `CRetry` | `agent.py:Agent.cancel_run` (and the module `dendrux.cancel_run`, which shares the same service) with `runtime/_managed_cancel.py:cancel_scope` |
| `CMarkRoot`, `CMarkKids` | `runtime/_managed_cancel.py:mark_subtree` → `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.mark_cancel_requested` |
| `CScan` | `runtime/_managed_cancel.py:_scan_frontier` → `get_managed_cancel_children_page` |
| `CPick` | `agent.py:Agent._cancel_one` → `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.request_cancel` (order from `runtime/_managed_cancel.py:cancel_descendants`) |
| `CCas` | `agent.py:Agent._cancel_one` → `runtime/_run_events.py:_transition_with_event` over `PAUSE_STATUS_VALUES` → `finalize_with_event` |
| `CDone`, `CCrash` | End of `Agent.cancel_run`; the call dying at any await |

## Running the checks

You need Java 11 or later and `tla2tools.jar` v1.7.4
(`https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar`, sha256
`936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`).

```bash
formal/check.sh            # every config, as CI runs it
formal/check.sh quick      # skip the two long safety runs
cd packages/python && make tla    # the same, from the package directory
```

`TLA2TOOLS_JAR` points at the jar (default `~/.local/share/tlaplus/tla2tools.jar`), `JAVA` at the
Java binary (default `$JAVA_HOME/bin/java`, else `java`), and `TLC_WORKERS` sets the TLC threads.
The script passes only when the four real configs pass and every bug config is caught with its
expected property. CI runs it in the `tla-model-check` job.

To run one config by hand:

```bash
cd formal
java -cp tla2tools.jar tlc2.TLC -workers auto -config bugs/BugSweepNoBump.cfg DendruxRuns.tla
```
