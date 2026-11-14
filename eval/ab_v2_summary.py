"""Summarize the Director v2 A/Bs (eval/ab_director_v2.sh): per question set and policy, Head Start ready time
(p50 / p90 / mean over reps), endpoint lag, accuracy, web calls; the baseline's p50 per run (run-to-run noise); the
paired director - fused difference per (question, rep) with a bootstrap CI.

  .venv/bin/python eval/ab_v2_summary.py [--tag v2] [--out runs/ab_v2_summary.json]
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / f"ab_{a.tag}_*_r*.json"))):
        m = re.search(rf"ab_{a.tag}_(\w+?)_(director|fused)_r(\d+)\.json$", p)
        if not m:
            continue
        runs[(m.group(1), m.group(2), int(m.group(3)))] = json.load(open(p))
    res = {}
    for qs in sorted({k[0] for k in runs}):
        res[qs] = {}
        hs = {}
        for pol in ("director", "fused"):
            rows, bl = [], []
            for (s, po, rep), d in runs.items():
                if s != qs or po != pol:
                    continue
                for r in d["rows"]:
                    if r.get("mode") == "headstart" and r.get("ready_ms") is not None:
                        rows.append(r)
                        hs[(pol, r["id"], rep)] = r["ready_ms"]
                b = [r["ready_ms"] for r in d["rows"] if r.get("mode") == "baseline" and r.get("ready_ms") is not None]
                if b:
                    bl.append(round(float(np.median(b))))
            if not rows:
                continue
            x = np.array([r["ready_ms"] for r in rows], float)
            lag = [r["endpoint_lag_ms"] for r in rows if r.get("endpoint_lag_ms") is not None]
            sc = [r["correct"] for r in rows if r.get("correct") is not None]
            res[qs][pol] = {"n": len(rows), "p50": round(float(np.median(x))), "p90": round(float(np.percentile(x, 90))),
                            "mean": round(float(x.mean())), "endpoint_lag_p50": round(float(np.median(lag))) if lag else None,
                            "accuracy": f"{sum(sc)}/{len(sc)}" if sc else None,
                            "web_calls_per_q": round(float(np.mean([r.get("web_calls") or 0 for r in rows])), 2),
                            "paid_per_q": round(float(np.mean([r.get("paid_calls") or 0 for r in rows])), 2),
                            "baseline_p50_per_run": bl}
        keys = [(q, rep) for (pol, q, rep) in hs if pol == "director" and ("fused", q, rep) in hs]
        if keys:
            d = np.array([hs[("director", q, rep)] - hs[("fused", q, rep)] for q, rep in keys], float)
            rng = np.random.default_rng(0)
            bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
            bm = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
            res[qs]["paired director - fused"] = {
                "n_pairs": len(d), "median_ms": round(float(np.median(d))),
                "median_ci": [round(float(np.percentile(bs, 2.5))), round(float(np.percentile(bs, 97.5)))],
                "mean_ms": round(float(d.mean())), "mean_ci": [round(float(np.percentile(bm, 2.5))), round(float(np.percentile(bm, 97.5)))],
                "director_faster": f"{int((d < 0).sum())}/{len(d)}"}
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
