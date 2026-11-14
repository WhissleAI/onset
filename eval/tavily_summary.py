"""Summarize the real-Tavily A/B runs into a tracked, numbers-only results file (no paths, no answers, no keys).

  .venv/bin/python eval/tavily_summary.py --set showcase=runs/tavily/ab_showcase.json --set current=... \
      --out eval/results/tavily_ab_2026-11-14.json

Per set and mode: questions, accuracy (keyword-scored rows with gold), credits / question (Tavily usage.credits as
metered per call), paid calls / question, p50 / p90 time to answer (first answer token after the last spoken word),
and the paired gain (baseline - Onset per question, median) with how many questions Onset was faster on.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(len(xs) - 1, int(k) + 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def arm(rs: list[dict]) -> dict:
    scored = [r for r in rs if r.get("correct") is not None]
    ready = [r.get("ready_ms") for r in rs]
    return {"questions": len(rs), "correct": sum(1 for r in scored if r["correct"]), "scored": len(scored),
            "accuracy": round(sum(1 for r in scored if r["correct"]) / len(scored), 3) if scored else None,
            "credits_per_q": round(statistics.mean([r.get("credits") or 0 for r in rs]), 3) if rs else None,
            "credits_total": round(sum(r.get("credits") or 0 for r in rs), 1),
            "paid_calls_per_q": round(statistics.mean([r.get("paid_calls") or 0 for r in rs]), 3) if rs else None,
            "ready_p50_ms": pct(ready, .5), "ready_p90_ms": pct(ready, .9),
            "answered": sum(1 for x in ready if x is not None),
            "errors": sum(1 for r in rs if r.get("error")),
            "web_fallback_rows": sum(1 for r in rs if (r.get("web_backend") or "tavily") not in ("tavily", None))}


def summarize(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["mode"], []).append(r)
    out = {m: arm(rs) for m, rs in sorted(by.items())}
    hs = {r["id"]: r for r in by.get("headstart", [])}
    bl = {r["id"]: r for r in by.get("baseline", [])}
    d = [bl[i]["ready_ms"] - hs[i]["ready_ms"] for i in hs if i in bl
         and hs[i].get("ready_ms") is not None and bl[i].get("ready_ms") is not None]
    out["paired"] = {"n": len(d), "onset_faster": sum(1 for x in d if x > 0),
                     "median_gain_ms": round(statistics.median(d)) if d else None,
                     "mean_gain_ms": round(statistics.mean(d)) if d else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], help="name=path to a run_ab.py output")
    ap.add_argument("--extra", default="", help="json file merged in as-is (extract / site-brain / credit ledger)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"what": "Real Tavily A/B, Onset (mode headstart) vs plain search after the end of turn (mode baseline). "
                   "Same ear (Deepgram Flux), director, router, Keystone, answer model; COST_MODE parity; "
                   "WEB_BACKEND=tavily (search_depth fast, 1 credit); demo .env settings.",
           "date": "2026-11-14", "sets": {}}
    allrows = []
    for s in a.set:
        name, path = s.split("=", 1)
        d = json.loads(Path(path).read_text())
        rows = d["rows"]
        allrows += rows
        res["sets"][name] = summarize(rows)
        cfg = d.get("config") or {}
        res["sets"][name]["git"] = cfg.get("git")
        res["sets"][name]["created"] = d.get("created")
    res["all"] = summarize(allrows)
    if a.extra:
        res.update(json.loads(Path(a.extra).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("sets", "all")}, indent=1))


if __name__ == "__main__":
    main()
