#!/bin/bash
# Director v2 end-to-end A/B: FLUX_SOFT=director (the v2 model in server/director/) vs FLUX_SOFT=fused, real Flux ear,
# Wikipedia web lane (never Tavily: TAVILY_API_KEY is emptied), showcase (29) + the default 58 set. Each run_ab call
# takes runs/.run_ab.lock for its replay only (--wait-lock); pairs alternate which policy goes first.
#   eval/ab_director_v2.sh <tag> [rep]
cd "$(dirname "$0")/.."
TAG=${1:-v2}; REP=${2:-1}
export TAVILY_API_KEY= WEB_BACKEND=wikipedia EAR=flux DIRECTOR=on DIRECTOR_STREAMS=1 DIRECTOR_ARC_EAR=0 RUN_AB_OWNER=director-v2
run() {   # policy set out
  FLUX_SOFT=$1 .venv/bin/python eval/run_ab.py $2 --no-latest --wait-lock 7200 --owner director-v2 \
    --out runs/ab_${TAG}_$3_$1_r$REP.json > runs/ab_${TAG}_$3_$1_r$REP.log 2>&1
}
for set in showcase q58; do
  args=$([ $set = showcase ] && echo "--questions eval/showcase/questions.jsonl" || echo "")
  if [ $((REP % 2)) -eq 1 ]; then run director "$args" $set; run fused "$args" $set
  else run fused "$args" $set; run director "$args" $set; fi
done
echo done
