#!/bin/bash
# Real-audio A/B of the Keystone director (KDIR=1) against the VOI gate (KDIR=0): headstart mode, Wikipedia web lane,
# the configured ear, COST_MODE parity, the WARM store (runs/index/warm_store copied fresh for every run), 58-question
# set + showcase, conditions interleaved per round (paired by question). The shared run_ab lock is held only for each
# timed run, with >= 3 minutes between holds.
cd "$(dirname "$0")/../.."
SETS=${SETS:-"q58 showcase"}; ROUNDS=${ROUNDS:-1}; GAP=${GAP:-180}
PY=.venv/bin/python
export WEB_BACKEND=wikipedia COST_MODE=parity
LOCK=${RUN_AB_LOCK:-../headstart/runs/.run_ab.lock}
PRIV=$(mktemp -u "${TMPDIR:-/tmp}/kdir_ab_lock.XXXX")
acq() {
  while true; do
    if ( set -o noclobber; echo "kdir pid=$$ since=$(date +%H:%M:%S) args=KDIR A/B $1" > "$LOCK" ) 2>/dev/null; then
      return; fi
    pid=$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -f "$LOCK"; continue; fi
    sleep 1
  done
}
rel() { rm -f "$LOCK"; }
trap rel EXIT
WARM=$(mktemp -d "${TMPDIR:-/tmp}/kdir_warm.XXXX")
run() {   # run <name> "<VAR=x ...>" [run_ab args...]
  local name=$1 envs=$2; shift 2
  rm -rf "$WARM"; cp -R runs/index/warm_store "$WARM"
  acq "$name"
  env $envs INDEX=1 INDEX_PERSIST=1 INDEX_RESET=never KEYSTONE_DIR=$WARM RUN_AB_LOCK="$PRIV" $PY eval/run_ab.py \
    --modes headstart --no-latest --owner kdir --out runs/index/ab_kdir_${name}.json "$@" > runs/index/ab_kdir_${name}.log 2>&1
  rel
  sleep "$GAP"
}
for r in $(seq 1 "$ROUNDS"); do
  for set in $SETS; do
    Q=(); [ "$set" = showcase ] && Q=(--questions eval/showcase/questions.jsonl)
    if [ $((r % 2)) = 1 ]; then A="voi KDIR=0"; B="kdir KDIR=1"; else A="kdir KDIR=1"; B="voi KDIR=0"; fi
    run ${set}_${A% *}_r$r "${A#* }" "${Q[@]}"
    run ${set}_${B% *}_r$r "${B#* }" "${Q[@]}"
  done
done
echo done
