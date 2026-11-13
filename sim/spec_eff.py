"""Speculation-efficiency A/B in the simulator: old fire policy (SPEC_EFF=0) vs new (SPEC_EFF=1) on the 80 real Flux
traces, both modes, server defaults for the chosen backend (no CURRENT overrides), N seeds.

  WEB_BACKEND=wikipedia .venv/bin/python -m sim.spec_eff --profile wikipedia --seeds 3 --out runs/s/eff_wiki.json
  WEB_BACKEND=tavily    .venv/bin/python -m sim.spec_eff --profile tavily --seeds 3 --set SPEC_PAID_MAX=2

Variants: --variant NAME:K=V,K=V (repeatable) adds Head Start runs with those settings (on top of SPEC_EFF=1).
CACHE_SHARED=1 in a variant keeps the search cache across the question set (the cross-question cache on; otherwise
it is cleared before every run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import fakes, flux_traces  # noqa: E402
from sim.simulate import Simulator, summarize  # noqa: E402

KEYS = ["n", "p50_ready_ms", "p90_ready_ms", "mean_ready_ms", "src_ok_rate", "grounded_rate", "paid_calls_per_q",
        "web_calls_per_q", "search_cached_per_q", "spec_fires_per_q", "wasted_fires_per_q", "draft_reuse_rate",
        "p50_used_fire_lead_ms", "used_fire_rate", "p50_first_fire_lead_ms", "paths", "llm_requests_per_q", "errors",
        "web_missing"]


def parse(v):
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def agree(rows, bl):
    """Paired accuracy proxy vs the baseline of the same (question, seed): src_ok both / only BL / only HS."""
    both = only_bl = only_hs = 0
    for r in rows:
        b = bl.get((r["id"], r["seed"]))
        if not b or r.get("src_ok") is None:
            continue
        both += bool(r["src_ok"] and b["src_ok"])
        only_bl += bool(b["src_ok"] and not r["src_ok"])
        only_hs += bool(r["src_ok"] and not b["src_ok"])
    return {"both_ok": both, "only_bl_ok": only_bl, "only_hs_ok": only_hs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="wikipedia")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--set", action="append", default=[], help="K=V applied to every run")
    ap.add_argument("--variant", action="append", default=[], help="NAME:K=V,K=V (Head Start, SPEC_EFF=1 + these)")
    ap.add_argument("--no-old", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fetch", type=int, default=2, help="rounds: fetch Wikipedia queries the runs missed, then rerun")
    a = ap.parse_args()
    base = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        base[k] = parse(v)
    sim = Simulator()
    for r in range(a.fetch + 1):
        out = run(sim, a, base)
        if not sim.caches.missing_web or r == a.fetch:
            return out
        import asyncio
        print(f"fetching {len(sim.caches.missing_web)} missing queries, rerunning", flush=True)
        asyncio.run(sim.caches.fetch_missing())
        sim.caches.save()


def run(sim, a, base):
    traces = flux_traces.load(ids=set(a.ids.split(",")) if a.ids else None)
    prof = fakes.PROFILES[a.profile]
    seeds = range(a.seeds)
    t = time.perf_counter()
    res = {"profile": a.profile, "seeds": a.seeds, "base": base, "runs": {}}
    bl_rows = sim.run_set(traces, dict(base), prof, seeds=seeds, modes=("baseline",))
    bl = {(r["id"], r["seed"]): r for r in bl_rows}
    res["runs"]["baseline"] = {"summary": summarize(bl_rows, "baseline"), "rows": bl_rows}
    runs = [] if a.no_old else [("old", {"SPEC_EFF": False})]
    runs.append(("new", {"SPEC_EFF": True}))
    for spec in a.variant:
        name, kvs = spec.split(":", 1)
        cfg = {"SPEC_EFF": True}
        for kv in kvs.split(","):
            k, v = kv.split("=", 1)
            cfg[k] = parse(v)
        runs.append((name, cfg))
    for name, cfg in runs:
        rows = sim.run_set(traces, {**base, **cfg}, prof, seeds=seeds, modes=("headstart",))
        for r in rows:
            b = bl.get((r["id"], r["seed"]))
            if b is not None:
                r["bl_top1"] = bool(r["src_ids"]) and r["src_ids"][0] in b["src_ids"][:3]
        s = summarize(rows, "headstart")
        s["vs_baseline"] = agree(rows, bl)
        res["runs"][name] = {"cfg": cfg, "summary": s, "rows": rows}
        print(name, json.dumps({k: s.get(k) for k in KEYS + ["vs_baseline", "bl_top1_rate"]}), flush=True)
    sb = res["runs"]["baseline"]["summary"]
    print("baseline", json.dumps({k: sb.get(k) for k in KEYS}))
    sim.caches.save()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"{time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
