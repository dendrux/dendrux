----------------------------- MODULE DendruxRuns -----------------------------
(***************************************************************************)
(* Run ownership, cancellation and recovery in Dendrux, as a state machine.*)
(*                                                                         *)
(* This models the DESIGN of the run table protocol, not the Python code.  *)
(* Each action is one database transaction: a compare-and-set (CAS) on one *)
(* row, or one statement over a set of rows. Transactions are atomic and   *)
(* either commit or lose their CAS; database errors are not modelled.      *)
(*                                                                         *)
(* Actors:                                                                 *)
(*   - workers: processes that own runs through "tasks" (an owner task     *)
(*     holds one claim epoch of one run). A worker can crash at any step,  *)
(*     which removes all of its tasks.                                     *)
(*   - sweep: the developer-called recovery scan. It may treat ANY running *)
(*     row as stale, because an owner that looks dead may be alive but     *)
(*     slow. That is the dangerous case.                                   *)
(*   - cancellers: Agent.cancel_run or dendrux.cancel_run, possibly in     *)
(*     another process (no task registry). A canceller can crash and is    *)
(*     then retried.                                                       *)
(*                                                                         *)
(* formal/README.md maps every action to the code that implements it.     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Workers,          \* owner processes
    Cancellers,       \* independent cancel_run callers
    Root,             \* the managed root run
    Kids,             \* its managed children (bound by one fan-out)
    SingleCallKids,   \* children on the SingleCall loop: never reclaimed
    NoWorker,         \* model value: "no holder"
    NoRun,            \* model value: "no current target"
    MaxParallel,      \* max_parallel_subagents
    ReclaimLimit,     \* reclaim_limit
    MaxPauses,        \* bound on owner pauses per run (keeps the model finite)
    MaxCrashes,       \* bound on worker crashes
    MaxCancelCrashes, \* bound on crashes per canceller
    CancelRetried,    \* liveness assumption: a crashed cancel_run is called again
    \* Bug switches. All FALSE in the real design.
    BugCancelRunning,     \* the cancel CAS also accepts RUNNING
    BugSweepNoBump,       \* sweep's CASes do not bump the claim epoch
    BugFinalizeNoEpoch,   \* the owner's finalize skips the epoch check
    BugPauseIgnoresCancel,\* the owner's pause CAS ignores cancel_requested
    BugBindIgnoresCancel  \* the child bind ignores the parent's cancel mark

ASSUME /\ Kids # {} /\ Root \notin Kids /\ SingleCallKids \subseteq Kids
       /\ MaxParallel \in Nat \ {0} /\ ReclaimLimit \in Nat \ {0}
       /\ MaxPauses \in Nat /\ MaxCrashes \in Nat /\ MaxCancelCrashes \in Nat
       /\ NoWorker \notin Workers /\ NoRun \notin {Root} \cup Kids

Runs      == {Root} \cup Kids
Terminal  == {"success", "error", "cancelled"}
\* "paused" is any external pause (WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT,
\* WAITING_APPROVAL, or a requested PAUSED). "waiting" is the stored status
\* PAUSED with pause_reason "managed_wait": the parent waits on its children.
Paused    == {"paused", "waiting"}
Statuses  == {"absent", "running"} \cup Paused \cup Terminal
NoHolder  == <<NoWorker, 0>>
CPhases   == {"idle", "markRoot", "scan", "markKids", "settle", "done"}

VARIABLES
    status,       \* run -> Statuses ("absent" = row not created yet)
    epoch,        \* run -> claim_epoch
    cancelReq,    \* run -> cancel_requested
    leaseExp,     \* run -> lease_expired_at is set
    attempts,     \* run -> reclaim_attempts
    pauses,       \* run -> owner pauses so far (model bound only)
    tasks,        \* set of owner tasks [w, r, e, pc]
    adoptPending, \* worker -> read "all children terminal, not cancelled"
    crashes,      \* worker crashes so far
    cpc,          \* canceller -> phase
    ckids,        \* canceller -> children found by its scan
    ctodo,        \* canceller -> runs still to settle
    ccur,         \* canceller -> run between request_cancel and its CAS
    ccrashes,     \* canceller -> crashes so far
    \* Ghost (history) variables, used only by the properties.
    holder,       \* run -> <<worker, epoch>> of the task the protocol says owns it
    staleWrite,   \* an owner write succeeded although the run was taken from it
    cancelRunning \* a canceller moved a RUNNING row to a terminal status

runVars   == <<status, epoch, cancelReq, leaseExp, attempts, pauses>>
procVars  == <<tasks, adoptPending, crashes>>
cVars     == <<cpc, ckids, ctodo, ccur, ccrashes>>
ghostVars == <<holder, staleWrite, cancelRunning>>
vars      == <<runVars, procVars, cVars, ghostVars>>

Task(w, r, e) == [w |-> w, r |-> r, e |-> e, pc |-> "work"]
Stopping(t)   == [t EXCEPT !.pc = "stop"]

\* The fence of every owner write: claim_epoch must equal the task's epoch.
Gate(t) == epoch[t.r] = t.e
\* Ghost: is t the owner the protocol last handed this run to?
Mine(t) == holder[t.r] = <<t.w, t.e>>
\* An owner task holding the run's current epoch exists.
Owned(r) == \E t \in tasks : t.r = r /\ t.e = epoch[r]
Unowned(r) == status[r] = "running" /\ ~Owned(r)

\* is_managed_root_stopped: the root is marked or terminal.
RootStopped == cancelReq[Root] \/ status[Root] \in Terminal
\* get_managed_run_control: own cancel flag, or (for a child) root stopped.
CancelSeen(r) == cancelReq[r] \/ (r \in Kids /\ RootStopped)

\* A sweep CAS bumps the epoch, unless the bug is switched on.
SweepBump(r) == IF BugSweepNoBump THEN epoch ELSE [epoch EXCEPT ![r] = @ + 1]

\* The shared effect of every finalize: terminal status, flags cleared.
SetTerminal(r, s) ==
    /\ status'    = [status    EXCEPT ![r] = s]
    /\ cancelReq' = [cancelReq EXCEPT ![r] = FALSE]
    /\ leaseExp'  = [leaseExp  EXCEPT ![r] = FALSE]
    /\ holder'    = [holder    EXCEPT ![r] = NoHolder]

TypeOK ==
    /\ status \in [Runs -> Statuses]
    /\ epoch \in [Runs -> Nat]
    /\ cancelReq \in [Runs -> BOOLEAN]
    /\ leaseExp \in [Runs -> BOOLEAN]
    /\ attempts \in [Runs -> 0..ReclaimLimit]
    /\ pauses \in [Runs -> 0..MaxPauses]
    /\ \A t \in tasks : t.w \in Workers /\ t.r \in Runs /\ t.e \in Nat
                        /\ t.pc \in {"work", "stop"}
    /\ adoptPending \in [Workers -> BOOLEAN]
    /\ crashes \in 0..MaxCrashes
    /\ cpc \in [Cancellers -> CPhases]
    /\ ckids \in [Cancellers -> SUBSET Kids]
    /\ ctodo \in [Cancellers -> SUBSET Runs]
    /\ ccur \in [Cancellers -> Runs \cup {NoRun}]
    /\ ccrashes \in [Cancellers -> 0..MaxCancelCrashes]
    /\ holder \in [Runs -> (Workers \cup {NoWorker}) \X Nat]
    /\ staleWrite \in BOOLEAN /\ cancelRunning \in BOOLEAN

Init ==
    /\ status = [r \in Runs |-> "absent"]
    /\ epoch = [r \in Runs |-> 0]
    /\ cancelReq = [r \in Runs |-> FALSE]
    /\ leaseExp = [r \in Runs |-> FALSE]
    /\ attempts = [r \in Runs |-> 0]
    /\ pauses = [r \in Runs |-> 0]
    /\ tasks = {}
    /\ adoptPending = [w \in Workers |-> FALSE]
    /\ crashes = 0
    /\ cpc = [k \in Cancellers |-> "idle"]
    /\ ckids = [k \in Cancellers |-> {}]
    /\ ctodo = [k \in Cancellers |-> {}]
    /\ ccur = [k \in Cancellers |-> NoRun]
    /\ ccrashes = [k \in Cancellers |-> 0]
    /\ holder = [r \in Runs |-> NoHolder]
    /\ staleWrite = FALSE
    /\ cancelRunning = FALSE

-----------------------------------------------------------------------------
(* Workers and their owner tasks                                           *)
-----------------------------------------------------------------------------

\* create_run: the root row is created RUNNING at epoch 0, owned by its caller.
Start(w) ==
    /\ status[Root] = "absent"
    /\ status' = [status EXCEPT ![Root] = "running"]
    /\ tasks' = tasks \cup {Task(w, Root, 0)}
    /\ holder' = [holder EXCEPT ![Root] = <<w, 0>>]
    /\ UNCHANGED <<epoch, cancelReq, leaseExp, attempts, pauses,
                   adoptPending, crashes, cVars, staleWrite, cancelRunning>>

\* A fenced write (_fence: checkpoint_step, complete_tool_call,
\* consume_submission, ...). It checks the epoch only, not the status.
\* A lost fence raises RunFencedError and the task stops.
OwnerWrite(t) ==
    /\ t.pc = "work"
    /\ IF Gate(t)
         THEN /\ staleWrite' = (staleWrite \/ ~Mine(t))
              /\ UNCHANGED tasks
         ELSE /\ tasks' = tasks \ {t}
              /\ UNCHANGED staleWrite
    /\ UNCHANGED <<runVars, adoptPending, crashes, cVars, holder, cancelRunning>>

\* The control read at each checkpoint (get_run_control on the claimed store,
\* or the managed admission read). Wrong epoch or not RUNNING: fenced, stop.
\* Cancel seen: go finalize as cancelled. Otherwise nothing changes.
OwnerCheckpoint(t) ==
    /\ t.pc = "work"
    /\ \/ /\ epoch[t.r] # t.e \/ status[t.r] # "running"
          /\ tasks' = tasks \ {t}
       \/ /\ epoch[t.r] = t.e /\ status[t.r] = "running" /\ CancelSeen(t.r)
          /\ tasks' = (tasks \ {t}) \cup {Stopping(t)}
    /\ UNCHANGED <<runVars, adoptPending, crashes, cVars, ghostVars>>

\* The owner's terminal CAS: status IN (running) AND claim_epoch = mine.
OwnerFinalize(t, s) ==
    /\ tasks' = tasks \ {t}
    /\ IF status[t.r] = "running" /\ (Gate(t) \/ BugFinalizeNoEpoch)
         THEN /\ SetTerminal(t.r, s)
              /\ staleWrite' = (staleWrite \/ ~Mine(t))
         ELSE UNCHANGED <<status, cancelReq, leaseExp, holder, staleWrite>>
    /\ UNCHANGED <<epoch, attempts, pauses, adoptPending, crashes, cVars,
                   cancelRunning>>

\* After a checkpoint saw the cancel: finalize as cancelled.
OwnerStop(t) == t.pc = "stop" /\ OwnerFinalize(t, "cancelled")

\* The loop ends (success / error / max_iterations, folded into success and
\* error), or the owner's task is cancelled in-process (caller cancellation,
\* a same-process cancel_run signal, the fan-out driver's cleanup), which
\* finalizes cancelled under the same claim. Completion may win over a
\* pending cancel request: cancellation is cooperative.
OwnerFinish(t, s) == t.pc = "work" /\ OwnerFinalize(t, s)

\* try_pause_run / pause_with_event: RUNNING AND NOT cancel_requested AND
\* epoch = mine. If the CAS loses on a row that is still RUNNING, the owner
\* finalizes cancelled (_persist_pause). Also covers release_run_claim.
OwnerPause(t) ==
    /\ t.pc = "work"
    /\ pauses[t.r] < MaxPauses
    /\ IF status[t.r] = "running" /\ Gate(t)
          /\ (~cancelReq[t.r] \/ BugPauseIgnoresCancel)
         THEN /\ status' = [status EXCEPT ![t.r] = "paused"]
              /\ pauses' = [pauses EXCEPT ![t.r] = @ + 1]
              /\ holder' = [holder EXCEPT ![t.r] = NoHolder]
              /\ tasks' = tasks \ {t}
         ELSE /\ UNCHANGED <<status, pauses, holder>>
              /\ tasks' = IF status[t.r] = "running"
                            THEN (tasks \ {t}) \cup {Stopping(t)}
                            ELSE tasks \ {t}
    /\ UNCHANGED <<epoch, cancelReq, leaseExp, attempts, adoptPending, crashes,
                   cVars, staleWrite, cancelRunning>>

\* bind_managed_children (N >= 2) or bind_managed_child (N = 1): one
\* transaction under the parent row lock. Admission: parent RUNNING, epoch =
\* mine, not cancelled, root not stopped. Then the parent parks as a managed
\* wait with its epoch retired (+1), and each child is created RUNNING at
\* epoch 0, its claim handed to an owner task in the same worker.
OwnerBind(t, S) ==
    /\ t.pc = "work" /\ t.r = Root
    /\ S # {} /\ Cardinality(S) <= MaxParallel
    /\ \A c \in Kids : status[c] = "absent"
    /\ IF status[Root] = "running" /\ Gate(t)
          /\ (~cancelReq[Root] \/ BugBindIgnoresCancel)
         THEN /\ status' = [r \in Runs |->
                              IF r = Root THEN "waiting"
                              ELSE IF r \in S THEN "running" ELSE status[r]]
              /\ epoch' = [epoch EXCEPT ![Root] = @ + 1]
              /\ tasks' = (tasks \ {t}) \cup {Task(t.w, c, 0) : c \in S}
              /\ holder' = [r \in Runs |->
                              IF r = Root THEN NoHolder
                              ELSE IF r \in S THEN <<t.w, 0>> ELSE holder[r]]
         ELSE /\ UNCHANGED <<status, epoch, holder>>
              \* Refused as cancelled: the parent goes to its cancel path.
              \* Refused as fenced: the task stops.
              /\ tasks' = IF status[Root] = "running" /\ cancelReq[Root]
                            THEN (tasks \ {t}) \cup {Stopping(t)}
                            ELSE tasks \ {t}
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, adoptPending, crashes,
                   cVars, staleWrite, cancelRunning>>

\* claim_paused_run / submit_and_claim: CAS status = paused -> RUNNING,
\* epoch + 1. It does not read cancel_requested: a request set on a paused
\* run is kept for the new owner, which stops at its first checkpoint.
ClaimPaused(w, r) ==
    /\ status[r] = "paused"
    /\ status' = [status EXCEPT ![r] = "running"]
    /\ epoch' = [epoch EXCEPT ![r] = @ + 1]
    /\ tasks' = tasks \cup {Task(w, r, epoch[r] + 1)}
    /\ holder' = [holder EXCEPT ![r] = <<w, epoch[r] + 1>>]
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, adoptPending, crashes,
                   cVars, staleWrite, cancelRunning>>

\* Fan-in, step 1 (settle_parent, claim_fanout_adoption): read every bound
\* child terminal and the parent not cancelled. A separate read, so a
\* cancel mark can land between it and the claim.
AdoptRead(w) ==
    /\ ~adoptPending[w]
    /\ status[Root] = "waiting"
    /\ \A c \in Kids : status[c] \in Terminal \cup {"absent"}
    /\ ~cancelReq[Root]
    /\ adoptPending' = [adoptPending EXCEPT ![w] = TRUE]
    /\ UNCHANGED <<runVars, tasks, crashes, cVars, ghostVars>>

\* Fan-in, step 2: submit_and_claim on the parent (expected PAUSED, same
\* wait generation). One winner; epoch + 1 fences every earlier writer.
AdoptClaim(w) ==
    /\ adoptPending[w]
    /\ adoptPending' = [adoptPending EXCEPT ![w] = FALSE]
    /\ IF status[Root] = "waiting"
         THEN /\ status' = [status EXCEPT ![Root] = "running"]
              /\ epoch' = [epoch EXCEPT ![Root] = @ + 1]
              /\ tasks' = tasks \cup {Task(w, Root, epoch[Root] + 1)}
              /\ holder' = [holder EXCEPT ![Root] = <<w, epoch[Root] + 1>>]
         ELSE UNCHANGED <<status, epoch, tasks, holder>>
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, crashes, cVars,
                   staleWrite, cancelRunning>>

\* A worker process dies: all its tasks and reads vanish.
Crash(w) ==
    /\ crashes < MaxCrashes
    /\ \E t \in tasks : t.w = w
    /\ tasks' = {t \in tasks : t.w # w}
    /\ adoptPending' = [adoptPending EXCEPT ![w] = FALSE]
    /\ crashes' = crashes + 1
    /\ UNCHANGED <<runVars, cVars, ghostVars>>

-----------------------------------------------------------------------------
(* sweep() and reclaim()                                                   *)
-----------------------------------------------------------------------------

\* mark_lease_expired, the "reclaim" branch: a stale ReAct row, under its
\* reclaim limit, not cancel-requested, is marked reclaimable and its epoch
\* bumped, which fences the old owner if it is still alive.
SweepMark(r) ==
    /\ status[r] = "running"
    /\ ~leaseExp[r] /\ ~cancelReq[r]
    /\ r \notin SingleCallKids
    /\ attempts[r] < ReclaimLimit
    /\ leaseExp' = [leaseExp EXCEPT ![r] = TRUE]
    /\ epoch' = SweepBump(r)
    /\ holder' = [holder EXCEPT ![r] = NoHolder]
    /\ UNCHANGED <<status, cancelReq, attempts, pauses, procVars, cVars,
                   staleWrite, cancelRunning>>

\* mark_lease_expired's finalize branch (and sweep_stale_runs): a stale
\* RUNNING row with a reason (SingleCall loop, reclaim limit reached, or
\* on_stale="error") or a pending cancel request is finalized: CANCELLED if
\* the request is set, else ERROR. CAS on status and epoch; epoch + 1.
SweepStale(r, mode) ==
    /\ status[r] = "running"
    /\ \/ mode = "error"
       \/ cancelReq[r] \/ r \in SingleCallKids \/ attempts[r] >= ReclaimLimit
    /\ SetTerminal(r, IF cancelReq[r] THEN "cancelled" ELSE "error")
    /\ epoch' = SweepBump(r)
    /\ UNCHANGED <<attempts, pauses, procVars, cVars, staleWrite, cancelRunning>>

\* sweep_abandoned_runs: an old external pause becomes ERROR. Managed
\* waits ("waiting") are exempt, by design.
SweepAbandoned(r) ==
    /\ status[r] = "paused"
    /\ SetTerminal(r, "error")
    /\ epoch' = SweepBump(r)
    /\ UNCHANGED <<attempts, pauses, procVars, cVars, staleWrite, cancelRunning>>

\* reclaim_run: a marked ReAct row under its limit. With a pending cancel it
\* is finalized CANCELLED (epoch + 1) and the reclaim is refused; otherwise
\* the caller becomes the owner at epoch + 1.
Reclaim(w, r) ==
    /\ status[r] = "running" /\ leaseExp[r]
    /\ r \notin SingleCallKids
    /\ attempts[r] < ReclaimLimit
    /\ epoch' = [epoch EXCEPT ![r] = @ + 1]
    /\ IF cancelReq[r]
         THEN /\ SetTerminal(r, "cancelled")
              /\ UNCHANGED <<attempts, tasks>>
         ELSE /\ leaseExp' = [leaseExp EXCEPT ![r] = FALSE]
              /\ attempts' = [attempts EXCEPT ![r] = @ + 1]
              /\ tasks' = tasks \cup {Task(w, r, epoch[r] + 1)}
              /\ holder' = [holder EXCEPT ![r] = <<w, epoch[r] + 1>>]
              /\ UNCHANGED <<status, cancelReq>>
    /\ UNCHANGED <<pauses, adoptPending, crashes, cVars, staleWrite,
                   cancelRunning>>

-----------------------------------------------------------------------------
(* cancel_run on the managed root (always a cascade)                       *)
-----------------------------------------------------------------------------

CStart(k) ==
    /\ cpc[k] = "idle"
    /\ status[Root] # "absent"
    /\ cpc' = [cpc EXCEPT ![k] = "markRoot"]
    /\ UNCHANGED <<runVars, procVars, ckids, ctodo, ccur, ccrashes, ghostVars>>

\* A retry of a cancel_run call that crashed.
CRetry(k) == ccrashes[k] > 0 /\ CStart(k)

\* mark_cancel_requested([root]): one statement, non-terminal rows only.
\* It serializes with a bind on the parent row lock.
CMarkRoot(k) ==
    /\ cpc[k] = "markRoot"
    /\ cancelReq' = [cancelReq EXCEPT ![Root] = @ \/ status[Root] \notin Terminal]
    /\ cpc' = [cpc EXCEPT ![k] = "scan"]
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, procVars,
                   ckids, ctodo, ccur, ccrashes, ghostVars>>

\* get_managed_cancel_children_page: the committed children of the frontier.
CScan(k) ==
    /\ cpc[k] = "scan"
    /\ ckids' = [ckids EXCEPT ![k] = {c \in Kids : status[c] # "absent"}]
    /\ cpc' = [cpc EXCEPT ![k] = "markKids"]
    /\ UNCHANGED <<runVars, procVars, ctodo, ccur, ccrashes, ghostVars>>

\* mark_cancel_requested(children): the next frontier, one statement.
CMarkKids(k) ==
    /\ cpc[k] = "markKids"
    /\ cancelReq' = [r \in Runs |->
                       IF r \in ckids[k] /\ status[r] \notin Terminal
                         THEN TRUE ELSE cancelReq[r]]
    /\ ctodo' = [ctodo EXCEPT ![k] = ckids[k] \cup {Root}]
    /\ cpc' = [cpc EXCEPT ![k] = "settle"]
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, procVars,
                   ckids, ccur, ccrashes, ghostVars>>

\* _cancel_one, part 1: request_cancel (non-terminal only). Children are
\* settled before the root (deepest first); siblings in any order.
CPick(k, r) ==
    /\ cpc[k] = "settle" /\ ccur[k] = NoRun
    /\ r \in ctodo[k]
    /\ r \in Kids \/ ctodo[k] \cap Kids = {}
    /\ ctodo' = [ctodo EXCEPT ![k] = @ \ {r}]
    /\ IF status[r] \notin Terminal
         THEN /\ cancelReq' = [cancelReq EXCEPT ![r] = TRUE]
              /\ ccur' = [ccur EXCEPT ![k] = r]
         ELSE UNCHANGED <<cancelReq, ccur>>
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, procVars,
                   cpc, ckids, ccrashes, ghostVars>>

\* _cancel_one, part 2: finalize_with_event CAS over the pause statuses only.
\* No epoch is read or bumped. A RUNNING row is left to its owner.
CCas(k) ==
    /\ ccur[k] # NoRun
    /\ LET r == ccur[k] IN
         IF status[r] \in Paused \/ (BugCancelRunning /\ status[r] = "running")
           THEN /\ SetTerminal(r, "cancelled")
                /\ cancelRunning' = (cancelRunning \/ status[r] = "running")
           ELSE UNCHANGED <<status, cancelReq, leaseExp, holder, cancelRunning>>
    /\ ccur' = [ccur EXCEPT ![k] = NoRun]
    /\ UNCHANGED <<epoch, attempts, pauses, procVars, cpc, ckids, ctodo,
                   ccrashes, staleWrite>>

CDone(k) ==
    /\ cpc[k] = "settle" /\ ctodo[k] = {} /\ ccur[k] = NoRun
    /\ cpc' = [cpc EXCEPT ![k] = "done"]
    /\ UNCHANGED <<runVars, procVars, ckids, ctodo, ccur, ccrashes, ghostVars>>

CCrash(k) ==
    /\ cpc[k] \in {"markRoot", "scan", "markKids", "settle"}
    /\ ccrashes[k] < MaxCancelCrashes
    /\ cpc' = [cpc EXCEPT ![k] = "idle"]
    /\ ckids' = [ckids EXCEPT ![k] = {}]
    /\ ctodo' = [ctodo EXCEPT ![k] = {}]
    /\ ccur' = [ccur EXCEPT ![k] = NoRun]
    /\ ccrashes' = [ccrashes EXCEPT ![k] = @ + 1]
    /\ UNCHANGED <<runVars, procVars, ghostVars>>

-----------------------------------------------------------------------------

OwnerStep(t) ==
    \/ OwnerWrite(t)
    \/ OwnerCheckpoint(t)
    \/ OwnerStop(t)
    \/ OwnerPause(t)
    \/ \E s \in {"success", "error", "cancelled"} : OwnerFinish(t, s)
    \/ \E S \in SUBSET Kids : OwnerBind(t, S)

Next ==
    \/ \E w \in Workers :
          \/ Start(w) \/ Crash(w) \/ AdoptRead(w) \/ AdoptClaim(w)
          \/ \E r \in Runs : ClaimPaused(w, r) \/ Reclaim(w, r)
    \/ \E t \in tasks : OwnerStep(t)
    \/ \E r \in Runs :
          \/ SweepMark(r) \/ SweepAbandoned(r)
          \/ \E m \in {"reclaim", "error"} : SweepStale(r, m)
    \/ \E k \in Cancellers :
          \/ CStart(k) \/ CMarkRoot(k) \/ CScan(k) \/ CMarkKids(k)
          \/ \E r \in Runs : CPick(k, r)
          \/ CCas(k) \/ CDone(k) \/ CCrash(k)

(***************************************************************************)
(* Fairness (used only for the liveness properties):                       *)
(*  - a live owner task keeps reaching checkpoints, and finalizes after    *)
(*    seeing a cancel;                                                     *)
(*  - the developer runs sweep(on_stale="reclaim") and reclaim() for what  *)
(*    sweep lists, for rows whose owner is gone (a live owner renews its   *)
(*    lease, so a sane threshold does not keep sweeping it);               *)
(*  - a started cancel_run runs to its end; with CancelRetried, a crashed  *)
(*    cancel_run is called again.                                          *)
(* Nothing forces a crash, a resume of a paused run, an adoption, an       *)
(* abandoned-waiting sweep or on_stale="error".                            *)
(***************************************************************************)
Fairness ==
    /\ \A w \in Workers, r \in Runs :
          /\ WF_vars(\E t \in tasks : t.w = w /\ t.r = r /\ OwnerCheckpoint(t))
          /\ WF_vars(\E t \in tasks : t.w = w /\ t.r = r /\ OwnerStop(t))
    /\ \A r \in Runs :
          /\ WF_vars(Unowned(r) /\ SweepMark(r))
          /\ WF_vars(Unowned(r) /\ SweepStale(r, "reclaim"))
          /\ WF_vars(\E w \in Workers : Reclaim(w, r))
    /\ \A k \in Cancellers :
          /\ WF_vars(CMarkRoot(k)) /\ WF_vars(CScan(k)) /\ WF_vars(CMarkKids(k))
          /\ WF_vars(\E r \in Runs : CPick(k, r))
          /\ WF_vars(CCas(k)) /\ WF_vars(CDone(k))
          /\ (CancelRetried => WF_vars(CRetry(k)))

SafetySpec == Init /\ [][Next]_vars

\* Workers are interchangeable (safety runs only; TLC does not support
\* symmetry with liveness checking).
WorkerSymmetry == Permutations(Workers)
Spec       == SafetySpec /\ Fairness

-----------------------------------------------------------------------------
(* Safety                                                                  *)
-----------------------------------------------------------------------------

\* Single writer: at most one task can pass a run's write fence at a time.
SingleWriter ==
    \A r \in Runs : Cardinality({t \in tasks : t.r = r /\ t.e = epoch[r]}) <= 1

\* Fencing: a task that can pass the fence is the one the protocol handed
\* the run to. Once a run is taken away (swept, marked, finalized by someone
\* else), its old owner's writes must fail.
StaleOwnerFenced ==
    \A t \in tasks : Gate(t) => Mine(t)

\* No owner write (fenced write or finalize) ever succeeded for an owner
\* the run had been taken from.
StaleWriteFails == ~staleWrite

\* A canceller never moves a RUNNING row to a terminal status.
NoCancellerFinalizesRunning == ~cancelRunning

\* The orphan window, as the code has it: a child may outlive its terminal
\* parent only while RUNNING with its own cancel request set (it stops at
\* its next checkpoint, or sweep settles it). It is never paused or
\* unflagged under a terminal parent.
OrphanWindow ==
    \A c \in Kids :
        (status[Root] \in Terminal /\ status[c] \notin Terminal \cup {"absent"})
            => (status[c] = "running" /\ cancelReq[c])

\* Terminal is final: a terminal run never changes status (a cancelled run
\* is never revived).
TerminalFinal ==
    [][\A r \in Runs : status[r] \in Terminal => status'[r] = status[r]]_vars

\* Request kept: a cancel request is cleared only when the run goes terminal.
RequestKept ==
    [][\A r \in Runs : (cancelReq[r] /\ ~cancelReq'[r]) => status'[r] \in Terminal]_vars

-----------------------------------------------------------------------------
(* Liveness                                                                *)
-----------------------------------------------------------------------------

\* A cancel request eventually leads to a terminal run.
CancelLeadsToTerminal ==
    \A r \in Runs : cancelReq[r] ~> status[r] \in Terminal

\* A running run whose owner is gone (crashed, or fenced out) is eventually
\* owned again (reclaimed) or settled (terminal).
CrashedOwnerRecovered ==
    \A r \in Runs : Unowned(r) ~> (status[r] \in Terminal \/ Owned(r))

\* Children of a terminal parent eventually settle.
OrphansSettle ==
    \A c \in Kids :
        (status[Root] \in Terminal /\ status[c] # "absent") ~> status[c] \in Terminal
=============================================================================
