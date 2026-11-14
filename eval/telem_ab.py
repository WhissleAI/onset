"""Retrieval A/B on Telem: Tavily (through Telem) vs Telem auto-routing vs Telem per-family engines.

Retrieval level only (no answer model, no Onset server, no models loaded): each arm searches each question once and
we score whether the gold answer appears in the top-K sources Onset would hand the answer model. Arms:
  tavily   Telem with providers.include=["tavily"]: Tavily's engine through Telem's router, billed to Telem credit
           (the brief forbids Tavily-key spend). Latency includes Telem's routing hop.
  auto     Telem auto_routing="accuracy" (Telem picks the engines per query).
  family   server/telem.py FAMILY_PROVIDERS, family = the question's labelled family (eval/telem/questions.jsonl;
           in the app it comes from server/narrow.family on the live frame).
Metrics per arm: hit@K (every gold group matched in the top-K titles + snippets), p50 / p90 wall latency, est. $ per
question (Telem's price table; exa est.), failure rate (error or zero sources), engines used.

Spend guard: --max-usd (default 2.5) stops the run before the next call would exceed it; TELEM_PROCESS_USD is the
second fence. Needs TELEM_API_KEY in the environment (never printed).

  set -a; . ./.env; set +a; .venv/bin/python eval/telem_ab.py --out runs/telem_ab.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import lanes, telem  # noqa: E402

ARMS = {"tavily": {"mode": "fixed", "providers": ["tavily"]},
        "auto": {"mode": "auto", "providers": []},
        "family": {"mode": "family"}}


def hit(expect: list[str], docs, k: int) -> bool:
    text = " ".join(f"{d.title} {d.text}" for d in docs[:k]).lower()
    return all(re.search(g.lower(), text) for g in expect)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


async def one(q: dict, arm: str, k: int) -> dict:
    hints = {"family": q.get("family", "")}
    p = telem.plan(text=q["text"], final=True, hints=hints)
    p.update({kk: v for kk, v in ARMS[arm].items()})
    if arm == "family":
        p["providers"] = list(telem.FAMILY_PROVIDERS.get(q.get("family", "")) or telem.FAMILY_PROVIDERS[""])
    p.pop("published_after", None)          # same window for every arm: no date floor
    meter: dict = {}
    t0 = time.perf_counter()
    row = {"id": q["id"], "arm": arm, "family": q.get("family"), "kind": q.get("kind")}
    try:
        docs = await telem.search(q["text"], p, lanes.Ledger(budget=0), meter, final=True)
        tm = meter.get("telem") or {}
        row.update(ok=bool(docs), n=len(docs), hit=hit(q["expect"], docs, k), usd=tm.get("usd") or 0.0,
                   engines=[r["provider"] for r in tm.get("runs") or [] if r.get("ok")],
                   failed_engines=[r["provider"] for r in tm.get("runs") or [] if not r.get("ok")],
                   engine_ms={r["provider"]: r.get("ms") for r in tm.get("runs") or []},
                   routing=tm.get("routing"), interaction_id=tm.get("interaction_id"),
                   top=[{"url": d.url, "provider": d.meta.get("provider"), "agree": d.meta.get("agree")}
                        for d in docs[:k]])
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, n=0, hit=False, usd=0.0, error=f"{type(e).__name__}: {e}"[:160])
    row["ms"] = round((time.perf_counter() - t0) * 1000)
    return row


def summarize(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        if not r:
            continue
        ms = [x["ms"] for x in r if x["ok"]]
        fams = {}
        for x in r:
            f = fams.setdefault(x["kind"], [0, 0])
            f[0] += x["hit"]
            f[1] += 1
        eng = {}
        for x in r:
            for e in x.get("engines") or []:
                eng[e] = eng.get(e, 0) + 1
        out[arm] = {"n": len(r), "hit": sum(x["hit"] for x in r), "hit_rate": round(sum(x["hit"] for x in r) / len(r), 3),
                    "p50_ms": pct(ms, 0.5), "p90_ms": pct(ms, 0.9),
                    "usd_per_q_est": round(sum(x["usd"] for x in r) / len(r), 5),
                    "usd_total_est": round(sum(x["usd"] for x in r), 4),
                    "fail_rate": round(sum(1 for x in r if not x["ok"]) / len(r), 3),
                    "engine_failures": sum(len(x.get("failed_engines") or []) for x in r),
                    "by_kind": {k: f"{a}/{b}" for k, (a, b) in fams.items()}, "engines": eng,
                    "mean_ms": round(statistics.mean(ms)) if ms else None}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval/telem/questions.jsonl"))
    ap.add_argument("--arms", default="tavily,auto,family")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=2.5)
    ap.add_argument("--out", default=str(ROOT / "runs/telem_ab.json"))
    a = ap.parse_args()
    if not telem.TELEM_API_KEY:
        sys.exit("TELEM_API_KEY is not set (add it to .env)")
    qs = [json.loads(l) for l in open(a.questions)]
    if a.limit:
        qs = qs[:a.limit]
    arms = [x for x in a.arms.split(",") if x in ARMS]
    rows, spent = [], 0.0
    for q in qs:
        for arm in arms:                    # interleaved: time-of-day drift hits every arm alike
            if spent + 0.02 > a.max_usd:
                print(f"stopping: est. spend ${spent:.3f} near --max-usd {a.max_usd}")
                break
            r = await one(q, arm, a.k)
            spent += r["usd"]
            rows.append(r)
            print(f"{q['id']:>6} {arm:7} hit={int(r['hit'])} {r['ms']:5d}ms ${r['usd']:.4f} "
                  f"{','.join(r.get('engines') or []) or r.get('error', '')}", flush=True)
    s = summarize(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "k": a.k, "summary": s,
                                       "spent_usd_est": round(spent, 4), "rows": rows}, indent=1))
    print(json.dumps(s, indent=1))
    print(f"est. spend ${spent:.4f}")
    await lanes.client().aclose()


if __name__ == "__main__":
    asyncio.run(main())
