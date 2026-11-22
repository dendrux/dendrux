----------------------------- MODULE DendruxRuns -----------------------------
(***************************************************************************)
(* Run ownership, cancellation and recovery in Dendrux, as a state machine.*)
(*                                                                         *)
(* This models the DESIGN of the run-table protocol, not the Python code,  *)
(* within small bounds and stated assumptions (formal/README.md).          *)
(*                                                                         *)
(* Durable actions are database transactions: a compare-and-set (CAS) on   *)
(* one row, or one statement over a set of rows. They are atomic and       *)
(* either commit or lose their CAS; database errors, lost                  *)
(* acknowledgements and read-backs are not modelled. Other actions are     *)
(* reads whose result is used later (sweep's candidate read, the adoption  *)
(* read, the cancel scan), call entry and exit, and process death.         *)
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
(* The model checks durable writes that pass the epoch fence. It does not  *)
(* check execution or external side effects: a fenced-out owner may still  *)
(* run a tool or a provider call before its next write fails.              *)
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
    NoCand,           \* model value: sweep holds no candidate
    MaxParallel,      \* max_parallel_subagents
    ReclaimLimit,     \* reclaim_limit
    MaxPauses,        \* bound on owner pauses per run (keeps the model finite)
    PauseKinds,       \* pause statuses an owner may take: subset of {"waiting", "paused"}
    PauseRuns,        \* runs whose owner may pause (keeps the model finite)
    MaxResets,        \* bound on progress writes that reset reclaim_attempts (all runs)
    MaxCrashes,       \* bound on worker crashes
    MaxCancelCrashes, \* bound on crashes per canceller
    IncompleteScan,   \* a cancel traversal may be truncated (max_nodes) or skip children
    CancelRetried,    \* liveness assumption: a crashed cancel_run is called again
    \* Bug switches. All FALSE in the real design.
    BugCancelRunning,     \* the cancel CAS also accepts RUNNING
    BugSweepNoBump,       \* sweep's CASes do not bump the claim epoch
    BugFinalizeNoEpoch,   \* the owner's finalize skips the epoch check
    BugPauseIgnoresCancel,\* the owner's pause CAS ignores cancel_requested
    BugBindIgnoresCancel  \* the child bind ignores the parent's cancel mark

ASSUME /\ Kids # {} /\ Root \notin Kids /\ SingleCallKids \subseteq Kids
       /\ MaxParallel \in Nat \ {0} /\ ReclaimLimit \in Nat \ {0}
       /\ MaxPauses \in Nat /\ MaxResets \in Nat
       /\ PauseKinds \subseteq {"waiting", "paused"} /\ PauseRuns \subseteq {Root} \cup Kids
       /\ MaxCrashes \in Nat /\ MaxCancelCrashes \in Nat
       /\ NoWorker \notin Workers /\ NoRun \notin {Root} \cup Kids

Runs      == {Root} \cup Kids
Terminal  == {"success", "error", "cancelled"}
\* Stored statuses:
\*   "waiting" = WAITING_CLIENT_TOOL, WAITING_HUMAN_INPUT or WAITING_APPROVAL
\*               (an external pause);
\*   "paused"  = PAUSED after request_pause;
\*   "managed" = PAUSED with pause_reason "managed_wait": the parent waits
\*               on its children.
ExtPaused   == {"waiting", "paused"}
Paused      == ExtPaused \cup {"managed"}    \* PAUSE_STATUS_VALUES
\* The owner's loop-cancel CAS (CANCEL_FINALIZE_STATUS_VALUES): WAITING_*
\* and RUNNING, not PAUSED.
LoopCancelOK == {"running", "waiting"}
Statuses  == {"absent", "running"} \cup Paused \cup Terminal
NoHolder  == <<NoWorker, 0>>
CPhases   == {"idle", "markRoot", "scan", "markKids", "settle", "done"}

VARIABLES
    status,       \* run -> Statuses ("absent" = row not created yet)
    epoch,        \* run -> claim_epoch
    cancelReq,    \* run -> cancel_requested
    leaseExp,     \* run -> lease_expired_at is set (projected away at terminal)
    attempts,     \* run -> reclaim_attempts
    pauses,       \* run -> owner pauses so far (model bound only)
    resets,       \* runs whose reclaim_attempts a progress write has reset (model bound only)
    tasks,        \* set of owner tasks [w, r, e, pc]
    adoptPending, \* worker -> read "all children terminal, not cancelled"
    crashes,      \* worker crashes so far
    scand,        \* sweep's mark candidate read: NoCand or [r, e]
    cpc,          \* canceller -> phase
    ckids,        \* canceller -> children its scan covers
    cleft,        \* canceller -> existing children its scan left out (truncated or skipped)
    ctodo,        \* canceller -> runs still to settle
    ccur,         \* canceller -> run between request_cancel and its CAS
    ccrashes,     \* canceller -> crashes so far
    \* Ghost (history) variables, used only by the properties.
    holder,       \* run -> <<worker, epoch>> of the owner the protocol last handed it to
    staleWrite,   \* an owner write succeeded although the run was taken from it
    cancelRunning,\* a canceller moved a RUNNING row to a terminal status
    exempt        \* children a truncated cascade left out when it settled the root

runVars   == <<status, epoch, cancelReq, leaseExp, attempts, pauses, resets>>
procVars  == <<tasks, adoptPending, crashes, scand>>
cVars     == <<cpc, ckids, cleft, ctodo, ccur, ccrashes>>
ghostVars == <<holder, staleWrite, cancelRunning, exempt>>
vars      == <<runVars, procVars, cVars, ghostVars>>

\* pc: "work" (running its loop), "stop" (saw a cancel, will finalize),
\* "parked" (paused its run at an external pause and has not returned yet).
Task(w, r, e) == [w |-> w, r |-> r, e |-> e, pc |-> "work"]
WithPc(t, p)  == [t EXCEPT !.pc = p]

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

\* The shared effect of every finalize: terminal status, flags cleared. The
\* lease mark is projected away at terminal (it no longer matters).
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
    /\ resets \subseteq Runs /\ Cardinality(resets) <= MaxResets
    /\ \A t \in tasks : t.w \in Workers /\ t.r \in Runs /\ t.e \in Nat
                        /\ t.pc \in {"work", "stop", "parked"}
    /\ adoptPending \in [Workers -> BOOLEAN]
    /\ crashes \in 0..MaxCrashes
    /\ scand = NoCand \/ (scand.r \in Runs /\ scand.e \in Nat)
    /\ cpc \in [Cancellers -> CPhases]
    /\ ckids \in [Cancellers -> SUBSET Kids]
    /\ cleft \in [Cancellers -> SUBSET Kids]
    /\ ctodo \in [Cancellers -> SUBSET Runs]
    /\ ccur \in [Cancellers -> Runs \cup {NoRun}]
    /\ ccrashes \in [Cancellers -> 0..MaxCancelCrashes]
    /\ holder \in [Runs -> (Workers \cup {NoWorker}) \X Nat]
    /\ staleWrite \in BOOLEAN /\ cancelRunning \in BOOLEAN
    /\ exempt \subseteq Kids

Init ==
    /\ status = [r \in Runs |-> "absent"]
    /\ epoch = [r \in Runs |-> 0]
    /\ cancelReq = [r \in Runs |-> FALSE]
    /\ leaseExp = [r \in Runs |-> FALSE]
    /\ attempts = [r \in Runs |-> 0]
    /\ pauses = [r \in Runs |-> 0]
    /\ resets = {}
    /\ tasks = {}
    /\ adoptPending = [w \in Workers |-> FALSE]
    /\ crashes = 0
    /\ scand = NoCand
    /\ cpc = [k \in Cancellers |-> "idle"]
    /\ ckids = [k \in Cancellers |-> {}]
    /\ cleft = [k \in Cancellers |-> {}]
    /\ ctodo = [k \in Cancellers |-> {}]
    /\ ccur = [k \in Cancellers |-> NoRun]
    /\ ccrashes = [k \in Cancellers |-> 0]
    /\ holder = [r \in Runs |-> NoHolder]
    /\ staleWrite = FALSE
    /\ cancelRunning = FALSE
    /\ exempt = {}

-----------------------------------------------------------------------------
(* Workers and their owner tasks                                           *)
-----------------------------------------------------------------------------

\* create_run: the root row is created RUNNING at epoch 0, owned by its caller.
Start(w) ==
    /\ status[Root] = "absent"
    /\ status' = [status EXCEPT ![Root] = "running"]
    /\ tasks' = tasks \cup {Task(w, Root, 0)}
    /\ holder' = [holder EXCEPT ![Root] = <<w, 0>>]
    /\ UNCHANGED <<epoch, cancelReq, leaseExp, attempts, pauses, resets,
                   adoptPending, crashes, scand, cVars,
                   staleWrite, cancelRunning, exempt>>

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
    /\ UNCHANGED <<runVars, adoptPending, crashes, scand, cVars,
                   holder, cancelRunning, exempt>>

\* An advancing fenced write (a fresh assistant or tool turn, or an
\* advancing submission): the same fence, and reclaim_attempts resets to 0,
\* which opens a new recovery episode. MaxResets bounds the episodes. A
\* reset of a count already at 0 changes nothing and is OwnerWrite.
OwnerAdvance(t) ==
    /\ t.pc = "work"
    /\ Gate(t)
    /\ attempts[t.r] > 0
    /\ t.r \notin resets /\ Cardinality(resets) < MaxResets
    /\ attempts' = [attempts EXCEPT ![t.r] = 0]
    /\ resets' = resets \cup {t.r}
    /\ staleWrite' = (staleWrite \/ ~Mine(t))
    /\ UNCHANGED <<status, epoch, cancelReq, leaseExp, pauses, procVars, cVars,
                   holder, cancelRunning, exempt>>

\* The control read at each checkpoint (get_run_control on the claimed store,
\* or the managed admission read). Wrong epoch or not RUNNING: fenced, stop.
\* Cancel seen: go finalize as cancelled. Otherwise nothing changes.
OwnerCheckpoint(t) ==
    /\ t.pc = "work"
    /\ \/ /\ epoch[t.r] # t.e \/ status[t.r] # "running"
          /\ tasks' = tasks \ {t}
       \/ /\ epoch[t.r] = t.e /\ status[t.r] = "running" /\ CancelSeen(t.r)
          /\ tasks' = (tasks \ {t}) \cup {WithPc(t, "stop")}
    /\ UNCHANGED <<runVars, adoptPending, crashes, scand, cVars, ghostVars>>

\* An owner terminal CAS: status IN allowed AND claim_epoch = mine.
OwnerFinalize(t, s, allowed) ==
    /\ tasks' = tasks \ {t}
    /\ IF status[t.r] \in allowed /\ (Gate(t) \/ BugFinalizeNoEpoch)
         THEN /\ SetTerminal(t.r, s)
              /\ staleWrite' = (staleWrite \/ ~Mine(t))
         ELSE UNCHANGED <<status, cancelReq, leaseExp, holder, staleWrite>>
    /\ UNCHANGED <<epoch, attempts, pauses, resets, adoptPending, crashes, scand,
                   cVars, cancelRunning, exempt>>

\* After a checkpoint saw the cancel: the loop-cancel CAS over WAITING_* and
\* RUNNING (the managed-stop path uses RUNNING only, a subset).
OwnerStop(t) == t.pc = "stop" /\ OwnerFinalize(t, "cancelled", LoopCancelOK)

\* The loop ends (success / error / max_iterations, folded into success and
\* error), or the owner's task is cancelled in-process (caller cancellation,
\* a same-process cancel_run signal, the fan-out driver's cleanup), which
\* finalizes cancelled under the same claim. These CASes accept RUNNING
\* only. Completion may win over a pending cancel request: cancellation is
\* cooperative.
OwnerFinish(t, s) == t.pc = "work" /\ OwnerFinalize(t, s, {"running"})

\* The owner's loop-cancel CAS after it paused at an external pause and
\* before it returned (the WAITING_* part of CANCEL_FINALIZE_STATUS_VALUES).
\* The runner may never reach this schedule; the model allows it anyway.
OwnerLateCancel(t) == t.pc = "parked" /\ OwnerFinalize(t, "cancelled", LoopCancelOK)

\* A parked owner returns.
OwnerLeave(t) ==
    /\ t.pc = "parked"
    /\ tasks' = tasks \ {t}
    /\ UNCHANGED <<runVars, adoptPending, crashes, scand, cVars, ghostVars>>

\* try_pause_run / pause_with_event: RUNNING AND NOT cancel_requested AND
\* epoch = mine, to an external pause (WAITING_*) or a requested PAUSED.
\* No epoch bump: the pausing owner stays the holder until someone claims.
\* If the CAS loses on a row that is still RUNNING, the owner finalizes
\* cancelled (_persist_pause). Also covers release_run_claim.
OwnerPause(t, kind) ==
    /\ t.pc = "work"
    /\ t.r \in PauseRuns /\ pauses[t.r] < MaxPauses
    /\ IF status[t.r] = "running" /\ Gate(t)
          /\ (~cancelReq[t.r] \/ BugPauseIgnoresCancel)
         THEN /\ status' = [status EXCEPT ![t.r] = kind]
              /\ pauses' = [pauses EXCEPT ![t.r] = @ + 1]
              /\ tasks' = IF kind = "waiting"
                            THEN (tasks \ {t}) \cup {WithPc(t, "parked")}
                            ELSE tasks \ {t}
         ELSE /\ UNCHANGED <<status, pauses>>
              /\ tasks' = IF status[t.r] = "running"
                            THEN (tasks \ {t}) \cup {WithPc(t, "stop")}
                            ELSE tasks \ {t}
    /\ UNCHANGED <<epoch, cancelReq, leaseExp, attempts, resets, adoptPending,
                   crashes, scand, cVars, ghostVars>>

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
                              IF r = Root THEN "managed"
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
                            THEN (tasks \ {t}) \cup {WithPc(t, "stop")}
                            ELSE tasks \ {t}
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, resets, adoptPending,
                   crashes, scand, cVars, staleWrite, cancelRunning, exempt>>

\* claim_paused_run / submit_and_claim: CAS status = (external pause) ->
\* RUNNING, epoch + 1. It does not read cancel_requested: a request set on
\* a paused run is kept for the new owner, which stops at its first
\* checkpoint.
ClaimPaused(w, r) ==
    /\ status[r] \in ExtPaused
    /\ status' = [status EXCEPT ![r] = "running"]
    /\ epoch' = [epoch EXCEPT ![r] = @ + 1]
    /\ tasks' = tasks \cup {Task(w, r, epoch[r] + 1)}
    /\ holder' = [holder EXCEPT ![r] = <<w, epoch[r] + 1>>]
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, resets, adoptPending,
                   crashes, scand, cVars, staleWrite, cancelRunning, exempt>>

\* Fan-in, step 1 (settle_parent, claim_fanout_adoption): read every bound
\* child terminal and the parent not cancelled. A separate read, so a
\* cancel mark can land between it and the claim.
AdoptRead(w) ==
    /\ ~adoptPending[w]
    /\ status[Root] = "managed"
    /\ \A c \in Kids : status[c] \in Terminal \cup {"absent"}
    /\ ~cancelReq[Root]
    /\ adoptPending' = [adoptPending EXCEPT ![w] = TRUE]
    /\ UNCHANGED <<runVars, tasks, crashes, scand, cVars, ghostVars>>

\* Fan-in, step 2: submit_and_claim on the parent (expected PAUSED, same
\* wait generation). One winner; epoch + 1 fences every earlier writer.
AdoptClaim(w) ==
    /\ adoptPending[w]
    /\ adoptPending' = [adoptPending EXCEPT ![w] = FALSE]
    /\ IF status[Root] = "managed"
         THEN /\ status' = [status EXCEPT ![Root] = "running"]
              /\ epoch' = [epoch EXCEPT ![Root] = @ + 1]
              /\ tasks' = tasks \cup {Task(w, Root, epoch[Root] + 1)}
              /\ holder' = [holder EXCEPT ![Root] = <<w, epoch[Root] + 1>>]
         ELSE UNCHANGED <<status, epoch, tasks, holder>>
    /\ UNCHANGED <<cancelReq, leaseExp, attempts, pauses, resets, crashes, scand,
                   cVars, staleWrite, cancelRunning, exempt>>

\* A worker process dies: all its tasks and reads vanish.
Crash(w) ==
    /\ crashes < MaxCrashes
    /\ (\E t \in tasks : t.w = w) \/ adoptPending[w]
    /\ tasks' = {t \in tasks : t.w # w}
    /\ adoptPending' = [adoptPending EXCEPT ![w] = FALSE]
    /\ crashes' = crashes + 1
    /\ UNCHANGED <<runVars, scand, cVars, ghostVars>>

-----------------------------------------------------------------------------
(* sweep() and reclaim()                                                   *)
-----------------------------------------------------------------------------

\* mark_lease_expired, "reclaim" branch, in two steps. It reads a page of
\* RUNNING rows and picks, from that snapshot, a ReAct row with no lease
\* mark, no cancel request and attempts under the limit. Later it runs the
\* row's CAS: status RUNNING, claim_epoch = the epoch read, lease mark still
\* unset; epoch + 1. The CAS does not recheck the cancel request, so a
\* cancel can commit between the read and the mark.
SweepRead(r) ==
    /\ scand = NoCand
    /\ status[r] = "running"
    /\ ~leaseExp[r] /\ ~cancelReq[r]
    /\ r \notin SingleCallKids /\ attempts[r] < ReclaimLimit
    /\ scand' = [r |-> r, e |-> epoch[r]]
    /\ UNCHANGED <<runVars, tasks, adoptPending, crashes, cVars, ghostVars>>

SweepMark ==
    /\ scand # NoCand
    /\ scand' = NoCand
    /\ LET r == scand.r IN
         IF status[r] = "running" /\ epoch[r] = scand.e /\ ~leaseExp[r]
           THEN /\ epoch' = SweepBump(r)
                /\ leaseExp' = [leaseExp EXCEPT ![r] = TRUE]
                /\ holder' = [holder EXCEPT ![r] = NoHolder]
           ELSE UNCHANGED <<epoch, leaseExp, holder>>
    /\ UNCHANGED <<status, cancelReq, attempts, pauses, resets, tasks,
                   adoptPending, crashes, cVars, staleWrite, cancelRunning, exempt>>

\* mark_lease_expired's finalize branch (and sweep_stale_runs): a stale
\* RUNNING row with a reason (SingleCall loop, reclaim limit reached, or
\* on_stale="error") or a pending cancel request is finalized: CANCELLED if
\* the request is set, else ERROR. CAS on status and epoch; epoch + 1. One
\* atomic step: the written status reads the flag at commit time, and mode
\* "error" already lets sweep finalize any running row, which covers every
\* snapshot the read could have taken.
SweepStale(r, mode) ==
    /\ status[r] = "running"
    /\ \/ mode = "error"
       \/ cancelReq[r] \/ r \in SingleCallKids \/ attempts[r] >= ReclaimLimit
    /\ SetTerminal(r, IF cancelReq[r] THEN "cancelled" ELSE "error")
    /\ epoch' = SweepBump(r)
    /\ UNCHANGED <<attempts, pauses, resets, procVars, cVars, staleWrite,
                   cancelRunning, exempt>>

\* sweep_abandoned_runs: an old external pause becomes ERROR. Managed
\* waits are exempt, by design. Its CAS rechecks every field it read, so
\* it is one atomic step here.
SweepAbandoned(r) ==
    /\ status[r] \in ExtPaused
    /\ SetTerminal(r, "error")
    /\ epoch' = SweepBump(r)
    /\ UNCHANGED <<attempts, pauses, resets, procVars, cVars, staleWrite,
                   cancelRunning, exempt>>

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
    /\ UNCHANGED <<pauses, resets, adoptPending, crashes, scand, cVars,
                   staleWrite, cancelRunning, exempt>>

-----------------------------------------------------------------------------
(* cancel_run on the managed root (always a cascade)                       *)
-----------------------------------------------------------------------------

CStart(k) ==
    /\ cpc[k] = "idle"
    /\ status[Root] # "absent"
    /\ cpc' = [cpc EXCEPT ![k] = "markRoot"]
    /\ UNCHANGED <<runVars, procVars, ckids, cleft, ctodo, ccur, ccrashes,
                   ghostVars>>

\* A retry of a cancel_run call that crashed.
CRetry(k) == ccrashes[k] > 0 /\ CStart(k)

\* mark_cancel_requested([root]): one statement, non-terminal rows only.
\* It serializes with a bind on the parent row lock.
CMarkRoot(k) ==
    /\ cpc[k] = "markRoot"
    /\ cancelReq' = [cancelReq EXCEPT ![Root] = @ \/ status[Root] \notin Terminal]
    /\ cpc' = [cpc EXCEPT ![k] = "scan"]
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, resets, procVars,
                   ckids, cleft, ctodo, ccur, ccrashes, ghostVars>>

\* get_managed_cancel_children_page: the committed children of the frontier.
\* With IncompleteScan the traversal may cover only some of them: the
\* max_nodes bound truncates it, or a child with a corrupt header is
\* skipped. The left-out children are reported (cascade_truncated, skipped
\* outcomes) and never marked or settled by this call.
CScan(k) ==
    /\ cpc[k] = "scan"
    /\ LET existing == {c \in Kids : status[c] # "absent"} IN
         \E S \in SUBSET existing :
            /\ IncompleteScan \/ S = existing
            /\ ckids' = [ckids EXCEPT ![k] = S]
            /\ cleft' = [cleft EXCEPT ![k] = existing \ S]
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
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, resets, procVars,
                   ckids, cleft, ccur, ccrashes, ghostVars>>

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
    /\ UNCHANGED <<status, epoch, leaseExp, attempts, pauses, resets, procVars,
                   cpc, ckids, cleft, ccrashes, ghostVars>>

\* _cancel_one, part 2: finalize_with_event CAS over the pause statuses only.
\* No epoch is read or bumped. A RUNNING row is left to its owner.
CCas(k) ==
    /\ ccur[k] # NoRun
    /\ LET r == ccur[k] IN
         IF status[r] \in Paused \/ (BugCancelRunning /\ status[r] = "running")
           THEN /\ SetTerminal(r, "cancelled")
                /\ cancelRunning' = (cancelRunning \/ status[r] = "running")
                /\ exempt' = IF r = Root THEN exempt \cup cleft[k] ELSE exempt
           ELSE UNCHANGED <<status, cancelReq, leaseExp, holder, cancelRunning,
                            exempt>>
    /\ ccur' = [ccur EXCEPT ![k] = NoRun]
    /\ UNCHANGED <<epoch, attempts, pauses, resets, procVars, cpc, ckids, cleft,
                   ctodo, ccrashes, staleWrite>>

CDone(k) ==
    /\ cpc[k] = "settle" /\ ctodo[k] = {} /\ ccur[k] = NoRun
    /\ cpc' = [cpc EXCEPT ![k] = "done"]
    /\ UNCHANGED <<runVars, procVars, ckids, cleft, ctodo, ccur, ccrashes,
                   ghostVars>>

CCrash(k) ==
    /\ cpc[k] \in {"markRoot", "scan", "markKids", "settle"}
    /\ ccrashes[k] < MaxCancelCrashes
    /\ cpc' = [cpc EXCEPT ![k] = "idle"]
    /\ ckids' = [ckids EXCEPT ![k] = {}]
    /\ cleft' = [cleft EXCEPT ![k] = {}]
    /\ ctodo' = [ctodo EXCEPT ![k] = {}]
    /\ ccur' = [ccur EXCEPT ![k] = NoRun]
    /\ ccrashes' = [ccrashes EXCEPT ![k] = @ + 1]
    /\ UNCHANGED <<runVars, procVars, ghostVars>>

-----------------------------------------------------------------------------

OwnerStep(t) ==
    \/ OwnerWrite(t)
    \/ OwnerAdvance(t)
    \/ OwnerCheckpoint(t)
    \/ OwnerStop(t)
    \/ OwnerLateCancel(t)
    \/ OwnerLeave(t)
    \/ \E kind \in PauseKinds : OwnerPause(t, kind)
    \/ \E s \in {"success", "error", "cancelled"} : OwnerFinish(t, s)
    \/ \E S \in SUBSET Kids : OwnerBind(t, S)

Next ==
    \/ \E w \in Workers :
          \/ Start(w) \/ Crash(w) \/ AdoptRead(w) \/ AdoptClaim(w)
          \/ \E r \in Runs : ClaimPaused(w, r) \/ Reclaim(w, r)
    \/ \E t \in tasks : OwnerStep(t)
    \/ \E r \in Runs :
          \/ SweepRead(r) \/ SweepAbandoned(r)
          \/ \E m \in {"reclaim", "error"} : SweepStale(r, m)
    \/ SweepMark
    \/ \E k \in Cancellers :
          \/ CStart(k) \/ CMarkRoot(k) \/ CScan(k) \/ CMarkKids(k)
          \/ \E r \in Runs : CPick(k, r)
          \/ CCas(k) \/ CDone(k) \/ CCrash(k)

(***************************************************************************)
(* Fairness (used only for the liveness properties). These are conditional *)
(* obligations on the developer's code and the owner's progress, not        *)
(* services the library runs:                                               *)
(*  - a live owner task keeps reaching checkpoints, and finalizes after    *)
(*    seeing a cancel;                                                     *)
(*  - the developer runs sweep(on_stale="reclaim") and reclaim() for what  *)
(*    sweep lists, for rows whose owner is gone (a live owner renews its   *)
(*    lease, so a sane threshold does not keep sweeping it);               *)
(*  - a started cancel_run runs to its end; with CancelRetried, a crashed  *)
(*    cancel_run is called again.                                          *)
(* Nothing forces a crash, a resume of a paused run, an adoption, an       *)
(* abandoned-waiting sweep or on_stale="error".                            *)
(*                                                                         *)
(* Each group is one weak-fairness condition over a disjunction, which    *)
(* keeps TLC's liveness check small. In this bounded model that is as     *)
(* strong as one condition per task, run or step: every step in a group   *)
(* changes the state, and a behavior has finitely many of them (epochs,   *)
(* tasks, recovery episodes and canceller phases are bounded), so no      *)
(* enabled step can be starved forever. That argument does not carry over *)
(* to an unbounded workload, where fairness must hold per eligible run.   *)
(***************************************************************************)
OwnerProgress ==
    \E t \in tasks : OwnerCheckpoint(t) \/ OwnerStop(t)

Recovery ==
    \/ \E r \in Runs : Unowned(r) /\ (SweepRead(r) \/ SweepStale(r, "reclaim"))
    \/ SweepMark
    \/ \E r \in Runs, w \in Workers : Reclaim(w, r)

CancelProgress ==
    \E k \in Cancellers :
        \/ CMarkRoot(k) \/ CScan(k) \/ CMarkKids(k) \/ CCas(k) \/ CDone(k)
        \/ \E r \in Runs : CPick(k, r)

Fairness ==
    /\ WF_vars(OwnerProgress)
    /\ WF_vars(Recovery)
    /\ WF_vars(CancelProgress)
    /\ CancelRetried => WF_vars(\E k \in Cancellers : CRetry(k))

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

\* Fencing: a working task that can pass the fence is the one the protocol
\* handed the run to. Once a run is taken away (swept, marked, claimed,
\* finalized by someone else), its old owner's writes must fail. A parked
\* task (after its pause commit) makes no fenced write; its only write is
\* the loop-cancel CAS, whose status predicate StaleWriteFails covers. The
\* canceller's CAS does not bump the epoch, so a parked task may still match
\* the epoch of a run cancelled while paused.
StaleOwnerFenced ==
    \A t \in tasks : (t.pc # "parked" /\ Gate(t)) => Mine(t)

\* No owner write (fenced write or finalize) ever succeeded for an owner
\* the run had been taken from.
StaleWriteFails == ~staleWrite

\* A canceller never moves a RUNNING row to a terminal status.
NoCancellerFinalizesRunning == ~cancelRunning

\* The orphan window, as the code has it: a child the settling cascade
\* covered may outlive its terminal parent only while RUNNING with its own
\* cancel request set (it stops at its next checkpoint, or sweep settles
\* it). Children a truncated or skipping cascade left out (exempt) are
\* reported by that call, not settled, and are not covered here.
OrphanWindow ==
    \A c \in Kids \ exempt :
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
(* Liveness (checked with IncompleteScan = FALSE: full traversal)          *)
-----------------------------------------------------------------------------

\* A cancel request eventually leads to a terminal run.
CancelLeadsToTerminal ==
    \A r \in Runs : cancelReq[r] ~> status[r] \in Terminal

\* A running run whose owner is gone (crashed, or fenced out) is eventually
\* owned again (reclaimed) or settled (terminal). This covers RUNNING rows
\* only; it says nothing about paused runs or managed waits.
CrashedOwnerRecovered ==
    \A r \in Runs : Unowned(r) ~> (status[r] \in Terminal \/ Owned(r))

\* Children of a terminal parent eventually settle.
OrphansSettle ==
    \A c \in Kids :
        (c \notin exempt /\ status[Root] \in Terminal /\ status[c] # "absent")
            ~> (status[c] \in Terminal \/ c \in exempt)

-----------------------------------------------------------------------------
(* Reachability probes (formal/probes/). Each must be VIOLATED: the        *)
(* violation is the witness that the named behavior is reachable, so the   *)
(* properties above are not vacuous.                                       *)
-----------------------------------------------------------------------------

\* A cancelled root with a covered child still running: the orphan window.
ProbeOrphanWindowOpen ==
    ~(status[Root] = "cancelled"
      /\ \E c \in Kids \ exempt : status[c] = "running" /\ cancelReq[c])

\* The parent was adopted: it runs again after its children finished.
ProbeAdopted ==
    ~(status[Root] = "running" /\ \E c \in Kids : status[c] \in Terminal)

\* A run was reclaimed and is owned by the reclaimer.
ProbeReclaimed ==
    ~(\E r \in Runs : attempts[r] = 1 /\ status[r] = "running" /\ Owned(r))

\* A slow owner is fenced while another task owns the same running run.
ProbeSlowOwnerFenced ==
    ~(\E t \in tasks : ~Gate(t) /\ status[t.r] = "running" /\ Owned(t.r))

\* A progress write reset reclaim_attempts after a reclaim, and the run
\* was reclaimed again.
ProbeSecondRecoveryEpisode ==
    ~(\E r \in resets : attempts[r] = 1 /\ Owned(r)
                     /\ status[r] = "running")

\* The canceller's CAS cancels a child that was paused (external or
\* requested) just before.
ProbeCancelledPausedKid ==
    [][~(\E k \in Cancellers :
           /\ ccur[k] \in Kids
           /\ status[ccur[k]] \in ExtPaused
           /\ status'[ccur[k]] = "cancelled"
           \* the canceller's CCas step, not the owner's own CAS
           /\ ccur'[k] = NoRun /\ tasks' = tasks)]_vars

\* A child's owner, at a checkpoint with its own epoch on a RUNNING row,
\* stops because the root stopped, with no request of its own (a lost pause
\* CAS needs the child's own request, so this step is the checkpoint).
ProbeKidSawRootStop ==
    [][~(\E t \in tasks :
           /\ t.r \in Kids /\ t.pc = "work"
           /\ Gate(t) /\ status[t.r] = "running"
           /\ ~cancelReq[t.r] /\ RootStopped
           /\ tasks' = (tasks \ {t}) \cup {WithPc(t, "stop")})]_vars

\* An owner, at a checkpoint with its own epoch on a RUNNING row, stops on
\* its own cancel request. Restricted to a child that cannot pause, so the
\* step is the checkpoint, not a lost pause CAS or a refused bind (both
\* have the same effect).
ProbeOwnerSawOwnCancel ==
    [][~(\E t \in tasks :
           /\ t.pc = "work" /\ t.r \in Kids \ PauseRuns
           /\ Gate(t) /\ status[t.r] = "running" /\ cancelReq[t.r]
           /\ tasks' = (tasks \ {t}) \cup {WithPc(t, "stop")})]_vars

\* Sweep marks a row reclaimable after a cancel committed between its
\* candidate read and its CAS.
ProbeMarkAfterCancel ==
    [][~(/\ scand # NoCand
         /\ cancelReq[scand.r]
         /\ leaseExp'[scand.r] /\ ~leaseExp[scand.r])]_vars

\* The owner's late loop-cancel CAS wins on an external pause.
ProbeLateCancelWins ==
    [][~(\E t \in tasks :
           /\ t.pc = "parked" /\ status[t.r] = "waiting"
           /\ status'[t.r] = "cancelled" /\ tasks' = tasks \ {t})]_vars

\* A truncated cascade settles the root and leaves a paused child behind.
ProbeTruncatedLeavesPausedKid ==
    ~(status[Root] = "cancelled" /\ \E c \in exempt : status[c] \in ExtPaused)
=============================================================================
