"""Summarise the Keystone real-audio A/B (eval/keystone/ab_index.sh): index off / cold / warm, per question set.

    .venv/bin/python eval/keystone/ab_summary.py   -> prints a markdown table, writes runs/index/ab_summary.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs/index"


def summ(rows):
    rs = [r for r in rows if r.get("mode") == "headstart" and r.get("ready_ms") is not None]
    if not rs:
        return None
    sc = [r for r in rs if r.get("correct") is not None and r.get("kind") not in ("current",)]
    tiers = {}
    for r in rs:
        for k, v in (r.get("tiers") or {}).items():
            tiers[k] = tiers.get(k, 0) + v
    ready = sorted(r["ready_ms"] for r in rs)
    return {"n": len(rs), "ready_p50": statistics.median(ready), "ready_p90": ready[int(0.9 * (len(ready) - 1))],
            "correct": round(sum(1 for r in sc if r["correct"]) / len(sc), 3) if sc else None, "n_scored": len(sc),
            "paid_per_q": round(statistics.mean([(r.get("paid_calls") if r.get("paid_calls") is not None
                                                  else r.get("web_calls") or 0) for r in rs]), 3),
            "index_answer": sum(1 for r in rs if r.get("answer_tier") == "index"),
            "index_fires": tiers.get("index", 0), "tiers": tiers}


def main():
    out = {}
    print("| set | index | n | ready p50 ms | ready p90 ms | correct | paid / q | answers from index | index-served fires |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in ("q58", "showcase"):
        for v in ("off", "cold", "warm", "off2", "warm2", "off3", "warm3"):
            p = RUNS / f"ab_{s}_{v}.json"
            if not p.exists():
                continue
            x = summ(json.loads(p.read_text()).get("rows") or [])
            out[f"{s}_{v}"] = x
            if x:
                print(f"| {s} | {v} | {x['n']} | {x['ready_p50']:.0f} | {x['ready_p90']:.0f} | {x['correct']} | "
                      f"{x['paid_per_q']} | {x['index_answer']} | {x['index_fires']} |")
    (RUNS / "ab_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
