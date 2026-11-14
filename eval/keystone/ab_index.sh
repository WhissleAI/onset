#!/bin/bash
# Real-audio A/B of the Keystone index (headstart mode, Wikipedia web lane, the configured ear): INDEX off vs on with a
# cold store per (question, mode) run (fresh sessions) vs on with a WARM store (the replayed workload store,
# runs/index/warm_store: other users asked related questions before). 58-question set + showcase.
# Takes the shared run_ab lock (same file + format as run_ab.py) for each timed run only, polling every second.
cd "$(dirname "$0")/../.."
SETS=${SETS:-"q58 showcase"}; VARIANTS=${VARIANTS:-"off cold warm"}; SUF=${SUF:-}
PY=.venv/bin/python
export WEB_BACKEND=wikipedia
LOCK=${RUN_AB_LOCK:-../headstart/runs/.run_ab.lock}
PRIV=$(mktemp -u "${TMPDIR:-/tmp}/ks_ab_lock.XXXX")
acq() {
  while true; do
    if ( set -o noclobber; echo "index pid=$$ since=$(date +%H:%M:%S) args=keystone A/B $1" > "$LOCK" ) 2>/dev/null; then
      return; fi
    pid=$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -f "$LOCK"; continue; fi     # stale
    sleep 1
  done
}
rel() { rm -f "$LOCK"; }
trap rel EXIT
run() {   # run <name> "<VAR=x ...>" [run_ab args...]
  local name=$1 envs=$2; shift 2
  acq "$name"
  env $envs RUN_AB_LOCK="$PRIV" $PY eval/run_ab.py --modes headstart --no-latest --owner index \
    --out runs/index/ab_${name}.json "$@" > runs/index/ab_${name}.log 2>&1
  rel
}
WARM=$(mktemp -d "${TMPDIR:-/tmp}/ks_warm.XXXX")
for set in $SETS; do
  Q=(); [ "$set" = showcase ] && Q=(--questions eval/showcase/questions.jsonl)
  [[ " $VARIANTS " == *" off "* ]] && run ${set}_off$SUF "INDEX=0" "${Q[@]}"
  [[ " $VARIANTS " == *" cold "* ]] && run ${set}_cold$SUF "INDEX=1 INDEX_PERSIST=0" "${Q[@]}"
  rm -rf "$WARM"; cp -R runs/index/warm_store "$WARM"
  [[ " $VARIANTS " == *" warm "* ]] && run ${set}_warm$SUF "INDEX=1 INDEX_PERSIST=1 INDEX_RESET=never KEYSTONE_DIR=$WARM" "${Q[@]}"
done
echo done
