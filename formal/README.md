# Formal model: run ownership, cancel and recovery

`DendruxRuns.tla` is a TLA+ model that **checks the ownership, cancellation and recovery protocol
within the stated bounds and assumptions**. It covers the run table's rules:
- who may write a run;
- how cancellation stops a run and its subagents;
- how `sweep()` and `reclaim()` recover a run after a crash.

The TLC model checker explores every interleaving of a few workers, one root run with up to two
subagents, sweeps and cancels. It checks the properties below in every reachable state.

What this is, and what it is not:
- It checks the **design**, not the Python code. The test suite checks the code. The table in
  [The bridge to the code](#the-bridge-to-the-code) maps each model action to the function that
  implements it, so a change to one of those functions is a prompt to update the model.
- It checks **durable writes that pass the epoch fence**, not execution. An owner that has lost
  its claim may still run a tool or a provider call before its next write fails. A recovered
  run may repeat work that the old owner had started. The model says that the old owner's writes
  do not commit. It does not say that work or external side effects happen once.
- It is **bounded**. The bounds and the reasons for them are below. They are an engineering
  judgement, not a proof that larger instances behave the same.

## What is modelled

| Part | In the model |
|---|---|
| Runs | One managed root and up to two children. One child runs the ReAct loop (reclaimable). The other runs SingleCall (never reclaimed; sweep fails it) |
| Statuses | `absent` (row not created), `running`, `waiting` (an external pause: client tool, human input, approval), `paused` (after `request_pause`), `managed` (stored as `paused` with a managed wait on children), and the terminal `success`, `error`, `cancelled` |
| Owner epoch | `claim_epoch` per run. These bump it: claiming a paused run, an adoption, a reclaim, the child bind (for the parent), and every sweep CAS. These do not: the owner's own pause or finalize, and the canceller's CAS. Every owner write is fenced on it |
| Owners | An owner task holds one epoch of one run. After pausing at an external pause, a task may stay "parked" until it returns; it can still run its loop-cancel CAS |
| Recovery | Sweep reads a stale ReAct row and later marks it reclaimable with a CAS (epoch + 1). Sweep finalizes a stale row that cannot be reclaimed or has a cancel request. Sweep fails an old external pause. A progress write resets `reclaim_attempts`, which opens a new recovery episode. Sweep may treat **any** running row as stale: an owner that looks dead may be alive but slow |
| Reclaim | Takes a marked row at epoch + 1. If a cancel request is set, it finalizes the row `cancelled` instead |
| Cancel | `cancel_run` on the root always cascades for a managed root. It marks the root, scans the children and marks them. Then, for each run, it calls `request_cancel` and a CAS over the pause statuses only, children before the root. A scan may be incomplete: the `max_nodes` bound truncates it, or a child with a corrupt header is skipped. Left-out children are reported and not settled by that call. A canceller can run in another process (no task registry). It can crash at any step and be called again |
| Children | The parent binds one or two children in one transaction. The bind is refused if the parent is cancel-requested or fenced. The parent then parks as a managed wait. Adoption is two steps: a read (all children terminal, parent not cancelled), then a claim |

## Properties

### Safety, checked in every reachable state

| Property | Meaning |
|---|---|
| `SingleWriter` | At most one owner task can pass a run's write fence at any time |
| `StaleOwnerFenced` | A working task that can pass the fence is the task the run was last handed to. Once a run is swept, marked, claimed or finalized by someone else, its old owner's fenced writes fail |
| `StaleWriteFails` | No owner write (a fenced write or a finalize) ever committed for an owner the run had been taken from |
| `NoCancellerFinalizesRunning` | A canceller never moves a `running` row to a terminal status. The owner stops itself at its next checkpoint |
| `OrphanWindow` | Consider a child that the settling cascade covered. It may outlive its terminal parent only while `running` with its own cancel request set. It then stops at its next checkpoint, or sweep settles it. It is never paused, and never unflagged, under a terminal parent. Children an incomplete cascade left out are reported by that call and are not covered by this property: a left-out paused child can stay paused under a cancelled parent |
| `TerminalFinal` | A terminal run never changes status again. A cancelled run is never revived |
| `RequestKept` | A cancel request is cleared only when the run becomes terminal |

### Liveness, under the fairness assumptions below and with a complete cancel traversal

| Property | Meaning |
|---|---|
| `CancelLeadsToTerminal` | A cancel request eventually leads to a terminal run |
| `CrashedOwnerRecovered` | A **running** run whose owner is gone is eventually owned again (reclaimed) or settled (terminal). It says nothing about paused runs or managed waits |
| `OrphansSettle` | The children of a terminal parent eventually become terminal |

These liveness results do not show that a workflow completes. Nothing obliges anyone to resume
a paused run or adopt finished children.

### Fairness assumptions

These are obligations on the developer's code and on owner progress. They are not services
Dendrux runs.
- A live owner task keeps reaching checkpoints, and after it sees a cancel it finalizes.
- The developer calls `sweep(on_stale="reclaim")`, and calls `reclaim()` for what sweep lists, for
  rows whose owner is gone. A live owner renews its lease, so a sane threshold does not keep
  sweeping it. Safety still lets sweep hit live owners at any time.
- A `cancel_run` call that has started runs to its end. If it crashes, the caller calls it again.
- Nothing forces a crash, a resume of a paused run, an adoption, an abandoned-waiting sweep or
  `on_stale="error"`.

Each assumption is one weak-fairness condition over a group of steps. In this bounded model, that
is as strong as one condition per task or per run: every step in a group changes the state, and a
behavior has only finitely many such steps. That argument does not carry over to an unbounded
workload. There, the obligations must hold per eligible run, including every page of a sweep.

### A cancel that dies must be called again

The liveness of cancellation depends on the retry assumption. Suppose a `cancel_run` call dies
after marking a root that is waiting on its children, and nobody calls it again. The root then
stays `paused` with the request set:
- adoption refuses a cancel-requested parent;
- `reclaim()` of the root refuses a stopped root;
- the abandoned-waiting sweep leaves managed waits alone.

`bugs/NoCancelRetry.cfg` shows this trace. Only another `cancel_run` settles the root.

## Bug variants and reachability probes

Each bug switch (a constant in the spec) breaks one rule. With the switch on, TLC must find a
counterexample. This shows the properties are strong enough to notice the bug, in the same way a
revert check shows a test is. The bug configs leave out `StaleOwnerFenced`, so each trace ends at
the bad write itself.

| Config | Bug | Caught by |
|---|---|---|
| `bugs/BugCancelRunning.cfg` | The cancel CAS also accepts `running` | `NoCancellerFinalizesRunning` |
| `bugs/BugSweepNoBump.cfg` | Sweep's CASes do not bump the claim epoch | `StaleWriteFails`: the owner sweep declared dead keeps writing |
| `bugs/BugFinalizeNoEpoch.cfg` | The owner's finalize checks the status but not the epoch | `StaleWriteFails`: a fenced owner still finalizes the run |
| `bugs/BugPauseIgnoresCancel.cfg` | The owner's pause CAS ignores `cancel_requested` | `OrphanWindow`: a child pauses after the cascade settled it, under a cancelled parent |
| `bugs/BugBindIgnoresCancel.cfg` | The child bind ignores the parent's cancel mark | `OrphanWindow`: a child is created after the cascade scanned, and is never flagged |
| `bugs/NoCancelRetry.cfg` | No bug; the retry assumption is dropped | `CancelLeadsToTerminal` |

The reachability probes in `probes/` show the properties are not vacuous. Each probe is a
predicate in `DendruxRuns.tla` that TLC must report as violated. The violation is the witness
that the named behavior can happen:

| Probe | Behavior shown |
|---|---|
| `ProbeOrphanWindowOpen` | A cancelled root with a covered child still running and flagged |
| `ProbeAdopted` | A parent adopted after its children finished |
| `ProbeReclaimed` | A run reclaimed and owned by the reclaimer |
| `ProbeSlowOwnerFenced` | A slow owner fenced while another task owns the same running run |
| `ProbeSecondRecoveryEpisode` | A progress write resets `reclaim_attempts` after a reclaim, and the run is reclaimed again |
| `ProbeCancelledPausedKid` | The canceller's CAS cancels a child that was paused just before |
| `ProbeKidSawRootStop` | A child's owner, at a checkpoint at its own epoch, stops because the root stopped, without a request of its own |
| `ProbeOwnerSawOwnCancel` | A child's owner, at a checkpoint at its own epoch, stops on its own cancel request (a child that cannot pause, so the step is the checkpoint) |
| `ProbeMarkAfterCancel` | Sweep marks a row reclaimable after a cancel committed between its read and its CAS |
| `ProbeLateCancelWins` | An owner's loop-cancel CAS wins on the external pause it left |
| `ProbeTruncatedLeavesPausedKid` | A truncated cascade cancels the root and leaves a paused child behind |

## Configs, bounds and why they are enough

| Config | Checks | Workers | Cancellers | Children | Runs that may pause (kinds) | Progress resets | Crashes (worker, per canceller) | Cancel traversal |
|---|---|---|---|---|---|---|---|---|
| `DendruxRuns.cfg` | safety | 2 (symmetric) | 1 | 2 | child c1 (WAITING_*) | 0 | none | complete |
| `DendruxRunsRecovery.cfg` | safety | 2 (symmetric) | 1 | 2 | none | 1 | none | complete |
| `DendruxRunsTruncation.cfg` | safety | 1 | 1 | 2 | child c1 (WAITING_*) | 0 | none | may be incomplete |
| `DendruxRunsTwoCancels.cfg` | safety | 1 | 2 | 2 | child c1 (WAITING_*) | 0 | none | complete |
| `DendruxRunsLive.cfg` | liveness and safety | 1 | 1 | 2 | none | 0 | 1, 1 | complete |
| `DendruxRunsLivePauses.cfg` | liveness and safety | 1 | 1 | 1 (ReAct) | root and child (WAITING_* and PAUSED) | 0 | 1, 1 | complete |
| `DendruxRunsLiveRecovery.cfg` | liveness and safety | 1 | 1 | 1 (ReAct) | none | 1 | 1, 1 | complete |

All configs use reclaim limit 1 and a parallel limit of 2. With two children, one runs ReAct and
one runs SingleCall.

The features multiply the state space, so each config turns on only some of them. Within a
config, every interleaving of the features that are on is checked. Interleavings of features in
different configs are not. A pause on every run multiplies the state space about a hundredfold, so
the safety configs let one child pause, and the root's own pauses (both kinds) are checked in
`DendruxRunsLivePauses.cfg`, which also checks every safety property. Combinations not checked
together:
- truncation with two workers, with progress resets, or with root pauses;
- progress resets with pauses;
- two cancellers with truncation, progress resets or root pauses;
- two workers with root pauses.

| Bound | Why it is judged enough |
|---|---|
| No crash bound for safety | For safety, a crashed task is the same as a task that never acts again, and safety checks every prefix. The task-based invariants are only stronger while a dead task is still counted. A crashed canceller that is called again is the same as a second canceller, which `DendruxRunsTwoCancels.cfg` checks. Liveness needs crashes, and the liveness configs have them |
| 2 workers | Enough for a slow owner and a second worker that reclaims, resumes or adopts the same run, and for two adopters racing. A third adds no new kind of conflict: every write is a CAS on one row, so each conflict is between the current holder and one other writer. One worker can also hold a stale task and a fresh one for the same run, which is why the one-worker configs still cover the slow-owner case |
| 1 root, up to 2 children | Two children give a real fan-out (N = 2), the all-terminal fan-in, siblings that settle in either order, and a cascade that covers one child and leaves the other out. One ReAct and one SingleCall child cover both recovery paths |
| Depth 1 | Each cascade level repeats the same steps: mark a frontier, scan its children, mark them, settle deepest first. Binds under a parent serialize with that parent's mark on the parent row lock, and the root-stopped read covers every depth. Nested waiting parents and paged frontiers are not checked |
| 1 pause, on chosen runs | One pause and one claim exercise pause, resume, cancel of a paused run, the late owner cancel and the abandoned sweep, and keep the state space finite. Repeated pauses are not checked |
| Reclaim limit 1, one progress reset | One reclaim, then either the limit path (sweep finalizes with `reclaim_limit`) or, after a progress reset, a second recovery episode. Longer chains of episodes are not checked |

## Simplifications and what they leave out

- **Transactions are atomic and never fail.** Each durable action is one committed transaction
  or a lost CAS. Database errors, retries, lost acknowledgements, commit read-back, lock behavior
  and timeouts are not modelled. The model says nothing about errors such as `PersistenceError`.
- **Some steps are reads, not transactions.** These steps are separate reads whose result is used
  later, as in the code:
  - sweep's candidate read before its mark CAS, so a cancel can commit in between;
  - the adoption read;
  - the cancel scan.

  Call entry and exit, and process death, are steps too.
- **Sweep's finalize is one step.** Its CAS reads the cancel request when it commits. The
  `on_stale="error"` mode lets sweep finalize any running row, which covers every snapshot the
  read could have taken.
- **Time is not modelled.** "Stale" is a free choice of sweep, so sweep may act on a live owner at
  any time. Lease renewal, thresholds and clocks are left out.
- **Fenced writes are two actions.** Checkpoints, tool-call completion, submission consumption and
  trace writes all use the same fence: epoch only, not status. An advancing write (a fresh turn,
  or an advancing submission) also resets `reclaim_attempts`. A parked owner makes no fenced write
  after its pause commit; the built-in recorder writes nothing when a run finishes.
- **The owner's cancel CASes.** The loop-cancel CAS accepts `running` and the WAITING_* statuses at
  the owner's epoch, but not `paused`. The model applies it after a checkpoint saw a cancel, and,
  for a parked owner, after its pause. The runner may never reach the parked case; the model
  allows it anyway. Task cancellation, the managed-stop path and normal completion accept
  `running` only.
- **Terminal statuses are folded.** `max_iterations` behaves like `success` or `error` here.
  `pending` is never stored by these code paths, so it is left out. A terminal row's lease mark is
  dropped, because it no longer matters; the stored column may keep it.
- **One fan-out round.** The parent binds children once. A second delegation round, with its own
  wait generation and adoption, is not checked.
- **Valid managed headers, except as a skip.** A corrupt header appears only as a child the scan
  leaves out. Missing, foreign-tenant or corrupt roots are not modelled.
- **Siblings settle in any order.** The code settles them in id order; the model allows every
  order, which includes that one.
- **Same-process signals are subsumed.** An Agent canceller in the owner's process also cancels the
  owner's task. In the model, that is the owner seeing the request at once, which is one of the
  interleavings already allowed. The module `dendrux.cancel_run` has no task registry; the owner
  stops at a later checkpoint, which is also allowed.
- **`release_run_claim` is the pause action.** Its CAS (running, epoch, no cancel request) and its
  cancel fallback match the owner's pause. In the code the fallback commits in the same
  transaction; the model allows a gap, which is more permissive for safety.
- **Routing pre-checks are left out.** The routing pre-check before a resume of a paused child is
  not modelled. The claim may happen under a stopped root, and the child then stops at its first
  checkpoint.
- **Not modelled:**
  - tenants, governance and skill checks, idempotency keys;
  - events and their ordering, notes, budgets, guardrails, the provider;
  - the legacy (non-checkpointed) storage format;
  - duplicate custody of one claim inside one process.

## The bridge to the code

Paths are under `packages/python/src/dendrux/`.

| TLA+ action | Code |
|---|---|
| `Start` | `runtime/stores/sql/_sql_runs.py:_SqlRuns.create_run` (row created `running`, epoch 0) |
| `OwnerWrite` | `runtime/stores/sql/_sql_writes.py:_SqlWrites._fence`, reached through `runtime/state.py:_ClaimedStateStore.__getattr__` (injects the claim epoch) from `checkpoint_step`, `complete_tool_call`, `consume_submission` and the other claimed writes |
| `OwnerAdvance` | The same fenced writes with an advancing turn or submission: `runtime/stores/sql/_sql_support.py:_checkpoint_progress` and `_submission_progress` reset `reclaim_attempts` |
| `OwnerCheckpoint` | `runtime/state.py:_ClaimedStateStore.get_run_control`; for managed runs, `runtime/_managed_cancel.py:checkpoint` and `_managed_control`, which read `runtime/stores/sql/_managed_reads.py:get_managed_run_control` (own request or root stopped, `_managed_stop_predicate`) |
| `OwnerStop`, `OwnerLateCancel` | `runtime/_run_outcome.py:_finalize_loop_cancelled` with `CANCEL_FINALIZE_STATUS_VALUES`; the managed-stop path `runtime/_managed_cancel.py:cancelled_result` uses `running` only. Both go through `runtime/_run_events.py:_transition_with_event` to `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.finalize_with_event` / `_cas_finalize` |
| `OwnerFinish` | `runtime/_run_outcome.py:_persist_loop_outcome` (terminal CAS on `running` and the epoch); in-process cancellation `runtime/_run_events.py:_persist_task_cancellation`; the fan-out driver's cleanup of its own claims (`runtime/_fanout_ledger.py`) |
| `OwnerPause`, `OwnerLeave` | `runtime/_run_outcome.py:_persist_pause` → `runtime/_run_events.py:_pause_with_event` → `runtime/stores/sql/_sql_pause.py:_SqlPause.pause_with_event` / `try_pause_run`; also `runtime/stores/sql/_sql_claims.py:_SqlClaims.release_run_claim` |
| `OwnerBind` | `runtime/stores/sql/_sql_managed.py:_SqlManaged.bind_managed_child` (`_admit_parent`, `_bind_park_parent`, `_bind_child`) and `runtime/stores/sql/_sql_fanout.py:bind_managed_children` |
| `ClaimPaused` | `runtime/stores/sql/_sql_claims.py:_SqlClaims.claim_paused_run` and `submit_and_claim` |
| `AdoptRead` | `runtime/_managed_fanout.py:settle_parent` (children terminal) and `runtime/_managed_claim.py:claim_fanout_adoption` (`cancellation_requested`, pause read) |
| `AdoptClaim` | `runtime/stores/sql/_sql_claims.py:_SqlClaims.submit_and_claim` with `runtime/_managed_claim.py:fanout_adoption_validator` |
| `Crash` | Process death; nothing in the code runs |
| `SweepRead`, `SweepMark` | `runtime/stores/sql/_sql_leases.py:_SqlLeases._scan_expired_runs`: the page read, then the mark CAS (status, epoch, lease mark unset; no cancel recheck). Reached from `runtime/sweep.py:sweep` |
| `SweepStale` | `runtime/stores/sql/_sql_leases.py:_SqlLeases._scan_expired_runs`, finalize branch; `runtime/stores/sql/_sql_sweep_scan.py:_SweepScan.sweep_row` for running rows |
| `SweepAbandoned` | `runtime/stores/sql/_sql_sweep_scan.py:_SweepScan.sweep_row` for waiting rows; managed waits are skipped by `_SweepScan.skipped` |
| `Reclaim` | `runtime/stores/sql/_sql_leases.py:_SqlLeases.reclaim_run` |
| `CStart`, `CRetry` | `agent.py:Agent.cancel_run`, and the module `dendrux.cancel_run`, which shares the same service; both use `runtime/_managed_cancel.py:cancel_scope` |
| `CMarkRoot`, `CMarkKids` | `runtime/_managed_cancel.py:mark_subtree` → `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.mark_cancel_requested` |
| `CScan` | `runtime/_managed_cancel.py:_scan_frontier` → `get_managed_cancel_children_page`; truncation at the `max_nodes` bound; skipped corrupt headers (`ManagedCancelPages`, `cascade_truncated`) |
| `CPick` | `agent.py:Agent._cancel_one` → `runtime/stores/sql/_sql_lifecycle.py:_SqlLifecycle.request_cancel` (order from `runtime/_managed_cancel.py:cancel_descendants`) |
| `CCas` | `agent.py:Agent._cancel_one` → `runtime/_run_events.py:_transition_with_event` over `PAUSE_STATUS_VALUES` → `finalize_with_event` |
| `CDone`, `CCrash` | End of `Agent.cancel_run`; the call dying at any await |

## Running the checks

You need Java 11 or later and `tla2tools.jar` v1.7.4
(`https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar`, sha256
`936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`).

```bash
formal/check.sh            # every config, bug variant and probe, as CI runs it
formal/check.sh quick      # skip the long safety runs
cd packages/python && make tla    # the same, from the package directory
```

Environment variables:
- `TLA2TOOLS_JAR` points at the jar (default `~/.local/share/tlaplus/tla2tools.jar`).
- `JAVA` points at the Java binary (default `$JAVA_HOME/bin/java`, else `java`).
- `TLC_WORKERS` sets the number of TLC threads.
- `TLC_LOG_DIR` keeps each config's full output.

The script passes only when all three of these hold:
- every real config passes;
- every bug config is caught with its expected property;
- every probe is reached.

CI runs it in the `tla-model-check` job.

To run one config by hand:

```bash
cd formal
java -cp tla2tools.jar tlc2.TLC -workers auto -config bugs/BugSweepNoBump.cfg DendruxRuns.tla
```
