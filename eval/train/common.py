"""Shared plumbing for the continuous training loop (scripts/train_cycle.sh, eval/train/cycle.py).

Paths (all git-ignored, local only):
  runs/train/            real/ (harvested rows), synth/cN/ (generated sessions), cycles/cN/ (per-cycle reports),
                         cycle.log, latest.json, loop.pid, state.json
  runs/models/<model>/   versioned model files + history.jsonl + CURRENT (eval/train/registry.py)
Scheduling rules (the user's machine comes first): one heavy step at a time, `nice -n 10`, no new heavy step while
the quiet flag exists (QUIET_FLAG, the lead creates it around the user's test windows) or swap used > SWAP_MAX_GB.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")
TRAIN = Path(os.environ.get("TRAIN_DIR", ROOT / "runs/train"))
MODELS = Path(os.environ.get("TRAIN_MODELS_DIR", ROOT / "runs/models"))
MAIN = Path(os.environ.get("HEADSTART_MAIN", ROOT.parent / "headstart"))      # the main checkout (shared runs/)
QUIET_FLAG = Path(os.environ.get("TRAIN_QUIET_FLAG", MAIN / "runs/.quiet"))
STOP_FLAG = TRAIN / "STOP"
SWAP_MAX_GB = float(os.environ.get("TRAIN_SWAP_MAX_GB", "16"))
APP_DATA = Path(os.environ.get("TRAIN_APP_DATA", Path.home() / "Library/Application Support/Onset/data"))


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str, path: Path | None = None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    p = path or TRAIN / "cycle.log"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def read_jsonl(p: Path) -> list[dict]:
    if not Path(p).exists():
        return []
    out = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def append_jsonl(p: Path, rows) -> int:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
            n += 1
    return n


def write_json(p: Path, d) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(p)


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------------------------------------ the user first
def swap_used_gb(text: str | None = None) -> float:
    """`sysctl vm.swapusage` -> used GB (0 if unreadable)."""
    if text is None:
        try:
            text = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return 0.0
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", text or "")
    if not m:
        return 0.0
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


def quiet() -> bool:
    return QUIET_FLAG.exists()


def stopped() -> bool:
    return STOP_FLAG.exists()


def may_start_heavy(swap_gb: float | None = None, quiet_flag: bool | None = None) -> tuple[bool, str]:
    """(ok, why not). A heavy step (a model-loading process) starts only when the user isn't testing and swap is
    below the cap."""
    if stopped():
        return False, "stop requested"
    q = quiet() if quiet_flag is None else quiet_flag
    if q:
        return False, f"quiet flag {QUIET_FLAG.name} present"
    s = swap_used_gb() if swap_gb is None else swap_gb
    if s > SWAP_MAX_GB:
        return False, f"swap used {s:.1f} GB > {SWAP_MAX_GB:.0f} GB"
    return True, ""


def wait_for_slot(max_wait_s: float = 3600, poll_s: float = 60, sleep=time.sleep) -> bool:
    """Block until a heavy step may start (polling); False when stop was requested or max_wait_s passed."""
    t0 = time.time()
    said = ""
    while True:
        ok, why = may_start_heavy()
        if ok:
            return True
        if stopped() or time.time() - t0 > max_wait_s:
            return False
        if why != said:
            log(f"waiting: {why}")
            said = why
        sleep(poll_s)


def run_heavy(cmd: list[str], logname: str, cwd: Path | None = None, env: dict | None = None,
              timeout: float = 3600) -> int:
    """One model-loading process, niced, after the quiet / swap check. Output to the cycle's log file. Returns the
    exit code (-1: not started)."""
    if not wait_for_slot():
        log(f"skip {logname}: no slot (stop / quiet / swap)")
        return -1
    lp = Path(logname)
    lp.parent.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, WEB_BACKEND="wikipedia", TAVILY_API_KEY="", PYTHONUNBUFFERED="1", **(env or {}))
    with lp.open("a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        try:
            p = subprocess.run(["nice", "-n", "10", *cmd], cwd=str(cwd or ROOT), env=e, stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write(f"timeout after {timeout} s\n")
            return 124


def sys_path():
    for p in (ROOT, ROOT / "eval", ROOT / "eval/keystone", ROOT / "eval/train"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
