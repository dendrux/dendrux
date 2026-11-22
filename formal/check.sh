#!/usr/bin/env bash
# Run every TLC check of formal/DendruxRuns.tla and compare with the expected result.
#
#   formal/check.sh            all checks
#   formal/check.sh quick      skip the two long safety runs
#
# Environment:
#   TLA2TOOLS_JAR  path to tla2tools.jar v1.7.4 (default ~/.local/share/tlaplus/tla2tools.jar)
#   JAVA           java 11+ binary (default: $JAVA_HOME/bin/java, else java on PATH)
#   TLC_WORKERS    TLC worker threads (default: auto)
#   TLC_LOG_DIR    keep each config's full TLC output here (default: discarded)
#
# Exit status is 0 only when every passing config passes and every bug config
# is caught with the expected property.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
jar="${TLA2TOOLS_JAR:-$HOME/.local/share/tlaplus/tla2tools.jar}"
if [[ -n "${JAVA_HOME:-}" ]]; then default_java="$JAVA_HOME/bin/java"; else default_java=java; fi
java="${JAVA:-$default_java}"
workers="${TLC_WORKERS:-auto}"
mode="${1:-all}"

if [[ ! -f "$jar" ]]; then
  echo "tla2tools.jar not found at $jar (set TLA2TOOLS_JAR)" >&2
  exit 2
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# config | expected: PASS, or the name of the property TLC must report as violated
checks=(
  "DendruxRuns.cfg|PASS"
  "DendruxRunsTwoCancels.cfg|PASS"
  "DendruxRunsLive.cfg|PASS"
  "DendruxRunsLivePauses.cfg|PASS"
  "bugs/BugCancelRunning.cfg|NoCancellerFinalizesRunning"
  "bugs/BugSweepNoBump.cfg|StaleWriteFails"
  "bugs/BugFinalizeNoEpoch.cfg|StaleWriteFails"
  "bugs/BugPauseIgnoresCancel.cfg|OrphanWindow"
  "bugs/BugBindIgnoresCancel.cfg|OrphanWindow"
  "bugs/NoCancelRetry.cfg|CancelLeadsToTerminal"
)

failed=0
for entry in "${checks[@]}"; do
  cfg="${entry%%|*}"
  expected="${entry##*|}"
  if [[ "$mode" == "quick" && ( "$cfg" == "DendruxRuns.cfg" || "$cfg" == "DendruxRunsTwoCancels.cfg" ) ]]; then
    echo "SKIP  $cfg (quick mode)"
    continue
  fi
  log="$scratch/$(basename "$cfg" .cfg).log"
  start=$(date +%s)
  set +e
  (cd "$here" && "$java" -XX:+UseParallelGC -cp "$jar" tlc2.TLC \
      -workers "$workers" -deadlock -cleanup \
      -metadir "$scratch/meta-$(basename "$cfg" .cfg)" \
      -config "$cfg" DendruxRuns.tla) >"$log" 2>&1
  code=$?
  set -e
  secs=$(( $(date +%s) - start ))
  if [[ -n "${TLC_LOG_DIR:-}" ]]; then
    mkdir -p "$TLC_LOG_DIR"
    cp "$log" "$TLC_LOG_DIR/"
  fi
  states="$(grep -E 'states generated' "$log" | tail -1 || true)"
  if [[ "$expected" == "PASS" ]]; then
    if [[ $code -eq 0 ]] && grep -q "Model checking completed. No error has been found." "$log"; then
      echo "PASS  $cfg (${secs}s) $states"
    else
      echo "FAIL  $cfg: expected no violation, TLC exit $code"
      cat "$log"
      failed=1
    fi
  else
    if [[ "$expected" == "CancelLeadsToTerminal" ]]; then
      # TLC does not name a violated liveness property; this config checks only that one.
      pattern="Temporal properties were violated"
    else
      pattern="Invariant ${expected} is violated"
    fi
    if [[ $code -ne 0 ]] && grep -q "$pattern" "$log"; then
      echo "CAUGHT $cfg: $expected violated (${secs}s) $states"
    else
      echo "MISSED $cfg: expected a violation of $expected, TLC exit $code"
      cat "$log"
      failed=1
    fi
  fi
done
exit $failed
