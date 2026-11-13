"""Router lane latency, cold (every cache cleared before each call): the router's plan for every eval / showcase
question, then the cheap lane it picked, timed. Text frames (eval/replay_frames.py), no audio.

    .venv/bin/python eval/router_latency.py [--reps 2] [--out runs/router_latency.json]

Prints p50 / p90 per vertical and per source, and which questions came back strong (web skipped) vs fell back.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from replay_frames import frame_for  # noqa: E402
from server import embed, knowledge, lanes, router  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]) if xs else None


async def main_async(a):
    await embed.arun(embed.encode_queries, ["warm up"])
    knowledge.leads_index()
    await router.warm()
    rows = []
    qs = []
    for p in ("eval/questions.jsonl", "eval/showcase/questions.jsonl"):
        qs += [json.loads(x) for x in (ROOT / p).read_text().splitlines() if x.strip()]
    for rep in range(a.reps):
        for q in qs:
            f = frame_for(q.get("arc_text") or q["text"])
            f.route = "web"
            pl = router.plan(f)
            if pl.lane not in ("card", "facts"):
                if rep == 0:
                    rows.append({"id": q["id"], "lane": pl.lane, "vertical": pl.vertical, "why": pl.why})
                continue
            router.reset_caches()
            m: dict = {}

            async def web():
                r = lanes.WebResult()
                r.backend = "none"
                return r
            t = time.perf_counter()
            res = await router.route(f.query(), f, final=True, meter=m, web_fn=web)
            ms = (time.perf_counter() - t) * 1000
            rows.append({"id": q["id"], "rep": rep, "lane": pl.lane, "vertical": pl.vertical, "ms": round(ms),
                         "tier": m.get("tier"), "src": res[0].source if len(res) else None,
                         "strong": m.get("tier") in ("card", "facts"), "error": m.get("route_error"),
                         "fallback": m.get("route_fallback")})
    timed = [r for r in rows if "ms" in r]
    out = {"by_vertical": {}, "by_source": {}, "rows": rows}
    for key, field in (("by_vertical", "vertical"), ("by_source", "src")):
        for k in sorted({str(r[field]) for r in timed}):
            g = [r["ms"] for r in timed if str(r[field]) == k]
            out[key][k] = {"n": len(g), "p50_ms": pct(g, 50), "p90_ms": pct(g, 90),
                           "mean_ms": round(statistics.mean(g))}
    out["strong_rate"] = round(sum(r["strong"] for r in timed) / len(timed), 3) if timed else None
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        if r.get("rep", 0) == 0 and r["lane"] in ("card", "facts") and not r.get("strong"):
            print("fell back to web:", r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
