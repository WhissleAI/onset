#!/bin/bash
# Real-audio A/B of the Keystone index (headstart mode, Wikipedia web lane, Arc ear): INDEX off vs on with a cold store
# per (question, mode) run (fresh sessions) vs on with a WARM store (the replayed workload store, runs/index/warm_store:
# other users asked related questions before). 58-question set + showcase. Takes the shared run_ab lock per run.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export WEB_BACKEND=wikipedia RUN_AB_LOCK=${RUN_AB_LOCK:-../headstart/runs/.run_ab.lock}
COMMON="--modes headstart --no-latest --wait-lock 14400 --owner index"
WARM=$(mktemp -d "${TMPDIR:-/tmp}/ks_warm.XXXX")
w() { while [ -f "$RUN_AB_LOCK" ]; do sleep 1; done; }       # poll the shared lock every second (run_ab polls 30 s)
for set in q58 showcase; do
  Q=""; [ "$set" = showcase ] && Q="--questions eval/showcase/questions.jsonl"
  w; INDEX=0 $PY eval/run_ab.py $Q $COMMON --out runs/index/ab_${set}_off.json > runs/index/ab_${set}_off.log 2>&1
  w; INDEX=1 INDEX_PERSIST=0 $PY eval/run_ab.py $Q $COMMON --out runs/index/ab_${set}_cold.json > runs/index/ab_${set}_cold.log 2>&1
  rm -rf "$WARM"; cp -R runs/index/warm_store "$WARM"
  w; INDEX=1 INDEX_PERSIST=1 INDEX_RESET=never KEYSTONE_DIR="$WARM" $PY eval/run_ab.py $Q $COMMON \
    --out runs/index/ab_${set}_warm.json > runs/index/ab_${set}_warm.log 2>&1
done
echo done
