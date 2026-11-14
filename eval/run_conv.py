"""Multi-turn A/B: each conversation is fed as one live stream (turn, gap, turn, ...) through the configured ear +
Session, in real time. Scores every turn (follow-ups need conversation memory) and reports follow-up latency.

  WEB_BACKEND=wikipedia .venv/bin/python eval/run_conv.py                    # both modes, eval/conversations.jsonl
  ... --modes headstart --repeat 2 --tag after --events                      # dump per-turn event timelines too
  ... --ids conv01,conv02                                                   # a subset
Writes runs/conv_<tag>.json (rows) and, with --events, runs/conv_<tag>_events.json."""
import argparse, asyncio, json, os, statistics, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--no-memory", action="store_true")
ap.add_argument("--modes", default="baseline,headstart")
ap.add_argument("--convs", default=str(ROOT / "eval" / "conversations.jsonl"))
ap.add_argument("--ids", default="")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--gap", type=float, default=2.5, help="seconds of silence between turns")
ap.add_argument("--tag", default="")
ap.add_argument("--events", action="store_true")
args = ap.parse_args()
from server import ear, embed, lanes, answer, searchcache, config, router   # noqa
from server.session import Session            # noqa

KEEP_EV = {"spec", "spec_skip", "endpoint", "sources", "metric", "memory", "spec_hold", "prewarm", "frame", "extended", "director", "eot_score", "words"}


def ok(ans, keys):
    a = (ans or "").lower()
    return all(any(k2 in a for k2 in k.split("|")) for k in keys)


async def run(conv, mode):
    ev = []
    t0 = time.perf_counter()

    def sink(e):
        if args.events and e.get("t") in KEEP_EV:
            e = dict(e)
            if e["t"] == "frame":
                e = {k: e.get(k) for k in ("t", "q", "ms", "query", "carried", "final", "n", "memory")}
            ev.append(e)

    s = Session(sink, mode=mode, t0=t0)
    if args.no_memory:
        s.carry_pool = {}
    e = ear.make_ear([s])
    parts = []
    for f in conv["files"]:
        w, _ = sf.read(ROOT / "eval" / "audio" / f, dtype="float32")
        parts += [w, np.zeros(int(16000 * args.gap), np.float32)]
    a = np.concatenate(parts)
    for i in range(0, len(a), 1600):
        while time.perf_counter() < t0 + (i + 1600) / 16000:
            await asyncio.sleep(0.003)
        if args.no_memory:
            s.carry = {}; s.carry_pool = {}
            if hasattr(s, "mem") and s.mem is not None:
                s.mem.clear()
        await e.feed(a[i:i + 1600])
    await e.flush()
    if s.final_task:
        try: await s.final_task
        except Exception: pass
    # a final answer may still be streaming: wait for the queue to drain (bounded)
    t_end = time.perf_counter() + 8
    while (s.busy or s.pending_finals) and time.perf_counter() < t_end:
        await asyncio.sleep(0.05)
    s.close()
    return s.history, ev


async def main():
    ear.warmup(); embed.warmup(); await lanes.warm_web(); await answer.warm()
    convs = [json.loads(x) for x in Path(args.convs).read_text().splitlines() if x.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        convs = [c for c in convs if c["id"] in want]
    res, events = [], {}
    for rep in range(args.repeat):
        for c in convs:
            for mode in args.modes.split(","):
                searchcache.reset()                      # A/B-honest: no cross-run cached web results
                router.reset_caches()                    # ... and cold card / fact lookups per run
                h, ev = await run(c, mode)
                h = [m for m in h if not m.get("skipped")]
                row = {"id": c["id"], "mode": mode, "rep": rep, "n_answers": len(h), "n_turns": len(c["turns"]),
                       "turns": []}
                for i, m in enumerate(h[:len(c["turns"])]):
                    keys = c["expect"][i] if i < len(c["expect"]) else []
                    row["turns"].append({
                        "i": i + 1, "q": m.get("question"), "query": m.get("query"), "ready_ms": m.get("ready_ms"),
                        "path": m.get("path"), "paid": m.get("paid_calls"), "web_calls": m.get("web_calls"),
                        "memory": m.get("memory"), "correct": ok(m.get("answer"), keys) if keys else None,
                        "answer": (m.get("answer") or "")[:160]})
                res.append(row)
                if args.events:
                    events[f'{c["id"]}/{mode}/{rep}'] = ev
                fu = " | ".join(f'{t["i"]}:{t["query"]!r} {t["ready_ms"]}ms {t["path"]} ok={t["correct"]}'
                                for t in row["turns"])
                print(f'{c["id"]} {mode:9s} r{rep} answers={len(h)}/{len(c["turns"])} {fu}', flush=True)
    tag = args.tag or ("nomem" if args.no_memory else "")
    out = ROOT / "runs" / (f"conv_{tag}.json" if tag else "conv.json")
    out.write_text(json.dumps(res, indent=1))
    if args.events:
        out.with_name(out.stem + "_events.json").write_text(json.dumps(events, indent=1))
    # summary: follow-up turns (2..n) per mode
    print("\nfollow-up turns (2..n):")
    for mode in args.modes.split(","):
        rows = [r for r in res if r["mode"] == mode]
        fu = [t for r in rows for t in r["turns"][1:]]
        lat = [t["ready_ms"] for t in fu if t.get("ready_ms") is not None]
        sc = [t["correct"] for t in fu if t["correct"] is not None]
        first = [r["turns"][0]["ready_ms"] for r in rows if r["turns"] and r["turns"][0].get("ready_ms") is not None]
        miss = sum(1 for r in rows if r["n_answers"] != r["n_turns"])
        print(f"  {mode:9s} n={len(lat)} p50={statistics.median(lat) if lat else None:.0f} "
              f"mean={statistics.mean(lat) if lat else 0:.0f} p90={sorted(lat)[int(0.9 * (len(lat) - 1))] if lat else None} "
              f"correct={sum(sc)}/{len(sc)} paid/q={statistics.mean([t['paid'] or 0 for t in fu]) if fu else 0:.2f} "
              f"| turn1 p50={statistics.median(first) if first else 0:.0f} | convs with wrong #answers={miss}")
    print("wrote", out)
asyncio.run(main())
