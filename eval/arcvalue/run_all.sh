#!/bin/bash
# Arc-value replays (docs/ARC_VALUE.md): three arms per category, Wikipedia lane, no Tavily, the shared CPU lock taken
# per replay (run_ab.py --wait-lock; run_conv.py via sim/with_lock.sh). Each step skips if its output exists.
#   eval/arcvalue/run_all.sh [step ...]      steps: corr fact show multi live noisy conv
set -u
cd "$(dirname "$0")/../.."
export WEB_BACKEND=wikipedia RUN_AB_LOCK=/Users/karan/Desktop/work/whissle/headstart/runs/.run_ab.lock RUN_AB_OWNER=arcvalue
PY=.venv/bin/python
M=baseline,headstart,plain
R=runs/arcvalue
mkdir -p $R
ab() {  # out args...
  local out=$1; shift
  grep -qs "^wrote" "${out%.json}.log" && { echo "skip $out"; return; }
  echo "$(date +%T) run_ab -> $out"
  $PY eval/run_ab.py --modes $M --no-latest --wait-lock 7200 --out "$out" "$@" > "${out%.json}.log" 2>&1
  echo "$(date +%T) done $out rc=$?"
}
steps="${*:-corr fact show multi live noisy conv}"
for s in $steps; do case $s in
  corr)  for r in 1 2; do ab $R/corr_r$r.json --questions eval/arcvalue/corrections.jsonl --audio-dir eval/audio_arcvalue --kinds all; done ;;
  fact)  for r in 1 2; do ab $R/fact58_r$r.json; done ;;
  show)  for r in 1 2; do ab $R/showcase_r$r.json --questions eval/showcase/questions.jsonl --kinds all; done ;;
  multi) ab $R/multi_r1.json --questions eval/multipart/questions.jsonl --kinds all ;;
  live)  ab $R/live.json --questions eval/arcvalue/livedata.jsonl --audio-dir eval/audio_arcvalue --kinds all
         $PY eval/arcvalue/score_live.py $R/live.json > $R/live_scored.log 2>&1 ;;
  noisy) for snr in 20 10; do ab $R/noisy_snr$snr.json --questions eval/arcvalue/showcase_scored.jsonl --audio-dir eval/audio_arcvalue_noisy/snr$snr --kinds all; done ;;
  conv)  [ -s $R/conv.json ] && echo "skip conv" || { echo "$(date +%T) run_conv"; MAX_MIN=120 sim/with_lock.sh arcvalue $PY eval/run_conv.py --modes $M --tag arcvalue_tmp > $R/conv.log 2>&1; mv runs/conv_arcvalue_tmp.json $R/conv.json; echo "$(date +%T) done conv"; } ;;
esac; done
