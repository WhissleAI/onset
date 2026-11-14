"""Planner on vs off, PAIRED: every (question, mode) is run twice back to back in one process, once with PLAN=1 and
once with PLAN=0 (order alternating), so machine load hits both arms alike (the laptop runs other jobs; separate
runs minutes apart differed by seconds in BOTH modes, planner or not).

  .venv/bin/python eval/multipart/ab_toggle.py --questions eval/multipart/questions.jsonl \
      --audio-dir eval/audio_multipart --out runs/plan/tg_mp.json
  .venv/bin/python eval/multipart/ab_toggle.py --questions eval/showcase/questions.jsonl --modes headstart \
      --out runs/plan/tg_sc.json
  .venv/bin/python eval/multipart/ab_toggle.py --modes headstart --out runs/plan/tg_58.json      # the 58 set

Takes the shared CPU lock (runs/.run_ab.lock, RUN_AB_LOCK) for the whole replay, like run_ab.py. Rows are run_ab rows
plus "plan_on"; the summary compares the arms per mode (paired ready deltas, accuracy, paid / free calls, and the time
the planner itself took per frame: decompose() is timed on every frame of the PLAN=1 arm).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import run_ab  # noqa: E402

DECOMP: list[float] = []


def _timed_decompose():
    from server import planner
    orig = planner.decompose

    def timed(f):
        t = time.perf_counter()
        try:
            return orig(f)
        finally:
            DECOMP.append((time.perf_counter() - t) * 1000)
    planner.decompose = timed


def summary(rows: list[dict]) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        by: dict = {}
        for r in rows:
            if r["mode"] == mode:
                by.setdefault(r["id"], {})[r["plan_on"]] = r
        pairs = [(v[True], v[False]) for v in by.values() if True in v and False in v
                 and v[True]["ready_ms"] is not None and v[False]["ready_ms"] is not None]
        if not pairs:
            continue
        d = [off["ready_ms"] - on["ready_ms"] for on, off in pairs]
        arm = {}
        for k, idx in (("on", 0), ("off", 1)):
            rs = [p[idx] for p in pairs]
            sc = [r for r in rs if r["correct"] is not None]
            arm[k] = {"p50_ready_ms": run_ab.pct([r["ready_ms"] for r in rs], 50),
                      "p90_ready_ms": run_ab.pct([r["ready_ms"] for r in rs], 90),
                      "mean_ready_ms": round(statistics.mean([r["ready_ms"] for r in rs])),
                      "p50_sources_ms": run_ab.pct([r["sources_ms"] for r in rs], 50),
                      "accuracy": round(sum(1 for r in sc if r["correct"]) / len(sc), 3) if sc else None,
                      "paid_per_q": round(statistics.mean([r.get("web_paid") if r.get("web_paid") is not None
                                                           else (r.get("paid_calls") or 0) for r in rs]), 2),
                      "free_per_q": round(statistics.mean([r.get("free_calls") or 0 for r in rs]), 2),
                      "plan_path": sum(1 for r in rs if r.get("path") == "plan")}
        out[mode] = {"n_pairs": len(pairs), **{f"{k}_{kk}": vv for k, v in arm.items() for kk, vv in v.items()},
                     "p50_on_faster_ms": run_ab.pct(d, 50), "mean_on_faster_ms": round(statistics.mean(d)),
                     "on_faster": sum(1 for x in d if x > 0),
                     "same_answer_path": sum(1 for on, off in pairs if on.get("path") == off.get("path"))}
    if DECOMP:
        out["decompose_ms"] = {"n_frames": len(DECOMP), "p50": round(statistics.median(DECOMP), 3),
                               "p99": round(sorted(DECOMP)[int(len(DECOMP) * .99)], 3), "max": round(max(DECOMP), 3)}
    return out


async def main_async(a) -> int:
    from server import answer, config, ear, embed, lanes
    qs = run_ab.load_questions(path=Path(a.questions).resolve() if a.questions else None)
    if not a.questions and config.WEB_BACKEND == "wikipedia":
        qs = [q for q in qs if q.get("kind") != "current"]
    if a.ids:
        qs = [q for q in qs if q["id"] in a.ids.split(",")]
    qs = [q for q in qs if (run_ab.AUDIO_DIR / q["file"]).exists()]
    modes = a.modes.split(",")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ear.warmup)
    await embed.arun(embed.warmup)
    await lanes.warm_web()
    await answer.warm()
    _timed_decompose()
    wq = next((q for q in run_ab.load_questions() if q.get("kind") == "current"), None)
    if wq:
        for mode in modes:
            await run_ab.run_one(wq, mode, audio_dir=run_ab.AUDIO)
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": run_ab.run_config(len(lanes.index())),
           "questions": a.questions or "eval/questions.jsonl (+ local)", "summary": {}, "rows": []}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(qs)} questions x {modes} x {a.flag} on/off", flush=True)
    for qi, q in enumerate(qs):
        for mi, mode in enumerate(modes):
            arms = [True, False] if (qi + mi) % 2 == 0 else [False, True]
            for on in arms:
                setattr(config, a.flag, on)
                try:
                    r = await run_ab.run_one(q, mode)
                except Exception as e:
                    r = {"id": q["id"], "mode": mode, "ready_ms": None, "correct": None, "error": repr(e)[:200]}
                r["plan_on"] = on
                res["rows"].append(r)
                print(f"  {q['id']} {mode:9s} plan={'on ' if on else 'off'} ready={r.get('ready_ms')} "
                      f"src={r.get('sources_ms')} {r.get('path')} paid={r.get('web_paid')} ok={r.get('correct')}",
                      flush=True)
                await asyncio.sleep(run_ab.PAUSE_BETWEEN_S)
        res["summary"] = summary(res["rows"])
        out.write_text(json.dumps(res, indent=1, default=float))
    if a.flag == "PLAN":
        config.PLAN = True
    res["summary"] = summary(res["rows"])
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["summary"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="")
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--modes", default="headstart,baseline")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flag", default="PLAN", help="config flag toggled per arm (PLAN, LOOP_ASR_FIX, VERIFY_RESEARCH)")
    ap.add_argument("--wait-lock", type=float, default=3600)
    a = ap.parse_args()
    if a.audio_dir:
        run_ab.AUDIO_DIR = Path(a.audio_dir).resolve()
    lock = Path(os.environ.get("RUN_AB_LOCK", ROOT / "runs" / ".run_ab.lock"))
    deadline = time.time() + a.wait_lock
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            held = lock.read_text().strip() if lock.exists() else ""
            import re
            m = re.search(r"pid=(\d+)", held)
            alive = False
            if m:
                try:
                    os.kill(int(m.group(1)), 0)
                    alive = True
                except (ProcessLookupError, PermissionError) as e:
                    alive = isinstance(e, PermissionError)
            if not alive:
                lock.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise SystemExit(f"lock held: {held}")
            time.sleep(30)
    os.write(fd, f"plan-toggle pid={os.getpid()} since={time.strftime('%H:%M:%S')} args={' '.join(sys.argv[1:])}\n".encode())
    os.close(fd)
    os.chdir(ROOT)
    try:
        rc = asyncio.run(main_async(a))
    finally:
        lock.unlink(missing_ok=True)
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
