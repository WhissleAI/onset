#!/bin/bash
# Continuous training loop for Onset's learned parts (calibrator, Keystone director, TTLs, Mamba search director).
#
#   scripts/train_cycle.sh start      background loop (nohup; survives the shell), a cycle every TRAIN_EVERY_MIN (75)
#   scripts/train_cycle.sh stop       stop after the current step (STOP flag), then kill the loop if it lingers
#   scripts/train_cycle.sh status     loop pid, last cycle, model versions, last lines of runs/train/cycle.log
#   scripts/train_cycle.sh once       one cycle in the foreground
#   scripts/train_cycle.sh dry-run    one small cycle (cycle 0): gates computed, nothing installed or written to docs
#
# The user's machine comes first: every step runs `nice -n 10`, one model-loading process at a time, and no new heavy
# step starts while runs/.quiet exists in the main checkout (the lead creates it around the user's test windows) or
# while swap used > TRAIN_SWAP_MAX_GB (16). Logs: runs/train/cycle.log; per cycle: runs/train/cycles/c<N>/.
# Rollback a model: .venv/bin/python eval/train/registry.py rollback <calib|kdir|ttl|director> [vNNN]
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
PY="$ROOT/.venv/bin/python"
T="$ROOT/runs/train"
PID="$T/loop.pid"
EVERY_MIN=${TRAIN_EVERY_MIN:-75}
mkdir -p "$T"

running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }

loop() {
  echo $$ > "$PID"
  rm -f "$T/STOP"
  echo "[$(date '+%F %T')] loop start (pid $$, every ${EVERY_MIN} min)" >> "$T/cycle.log"
  while [ ! -f "$T/STOP" ]; do
    t0=$(date +%s)
    nice -n 10 "$PY" eval/train/cycle.py >> "$T/cycle.out" 2>&1
    echo "[$(date '+%F %T')] cycle exit $?" >> "$T/cycle.log"
    # sleep until the next slot, waking every 30 s to honour stop
    while [ ! -f "$T/STOP" ] && [ $(( $(date +%s) - t0 )) -lt $(( EVERY_MIN * 60 )) ]; do sleep 30; done
  done
  echo "[$(date '+%F %T')] loop stopped" >> "$T/cycle.log"
  rm -f "$PID" "$T/STOP"
}

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PID"))"; exit 0; fi
    nohup "$0" __loop > /dev/null 2>&1 &
    sleep 1; echo "started (pid $(cat "$PID" 2>/dev/null)); log: runs/train/cycle.log; stop: scripts/train_cycle.sh stop" ;;
  __loop) loop ;;
  stop)
    touch "$T/STOP"
    if running; then
      p=$(cat "$PID")
      echo "stop requested; waiting up to 60 s for pid $p"
      for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
      if kill -0 "$p" 2>/dev/null; then
        pkill -TERM -P "$p" 2>/dev/null; pkill -TERM -f "eval/train/(cycle|calib_step|kdir_step|director_step|synth|collect).py" 2>/dev/null
        kill -TERM "$p" 2>/dev/null; echo "killed loop $p and its current step"
      fi
    fi
    rm -f "$PID"; echo "stopped" ;;
  status)
    if running; then echo "loop running (pid $(cat "$PID"))"; else echo "loop not running"; fi
    [ -f "$ROOT/../headstart/runs/.quiet" ] && echo "quiet flag present: no new heavy steps"
    sysctl vm.swapusage 2>/dev/null
    "$PY" eval/train/registry.py status 2>/dev/null
    tail -n 15 "$T/cycle.log" 2>/dev/null ;;
  once) nice -n 10 "$PY" eval/train/cycle.py ;;
  dry-run) nice -n 10 "$PY" eval/train/cycle.py --dry-run ;;
  *) echo "usage: $0 start|stop|status|once|dry-run"; exit 2 ;;
esac
