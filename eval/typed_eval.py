"""Typed questions: time from Enter to the first answer token, typed at a realistic speed vs pasted all at once.

Drives a running server (TYPED_INPUT=1) over /ws exactly as the page does: {"t":"start","typed":true} then one
{"t":"type","text"} per keystroke and {"final":true} at Enter. Typed arm: characters at --cps with jitter (default
5 chars/s ~ 60 wpm), so the speculator sees words as they are finished; pasted arm: the whole text in one message,
then Enter at once (nothing to speculate on). Each ask runs on a fresh connection (no conversation memory), the
server's search / router caches are cleared before every ask (POST /api/cache/clear), and the arm order alternates per
question. Run the server with WEB_BACKEND=wikipedia (no Tavily spend) and INDEX=0 (Keystone would remember the first
ask's answer for the second).

  .venv/bin/python eval/typed_eval.py --port 8811 [--n 30] [--cps 5] [--out eval/typed/results.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent


async def ask(port: int, text: str, typed: bool, cps: float, rng: random.Random, modes) -> dict:
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/cache/clear", method="POST"),
                           timeout=10).read()
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
        while json.loads(await ws.recv()).get("t") != "ready":
            pass
        rid = 1
        await ws.send(json.dumps({"t": "start", "typed": True, "modes": modes, "rid": rid}))
        events: list = []

        async def reader():
            async for raw in ws:
                e = json.loads(raw)
                e["_t"] = time.perf_counter()
                events.append(e)
                if e.get("t") == "run_done":
                    return
        rd = asyncio.create_task(reader())
        if typed:
            for k in range(1, len(text) + 1):
                await ws.send(json.dumps({"t": "type", "rid": rid, "text": text[:k]}))
                await asyncio.sleep(max(0.03, rng.gauss(1.0 / cps, 0.35 / cps)))
        else:
            await ws.send(json.dumps({"t": "type", "rid": rid, "text": text}))
        t_enter = time.perf_counter()
        await ws.send(json.dumps({"t": "type", "rid": rid, "text": text, "final": True}))
        await asyncio.wait_for(rd, 60)
    out = {}
    for m in modes:
        first = next((e["_t"] for e in events if e.get("mode") == m and e.get("t") == "answer"), None)
        met = next((e for e in events if e.get("mode") == m and e.get("t") == "metric"), {})
        spec = [e for e in events if e.get("mode") == m and e.get("t") == "spec" and e.get("state") == "fired"
                and not str(e.get("lane", "")).startswith("final")]
        out[m] = {"enter_to_answer_ms": None if first is None else round((first - t_enter) * 1000),
                  "ready_ms": met.get("ready_ms"), "path": met.get("path"), "lane": met.get("lane") or met.get("route"),
                  "spec_fires": met.get("spec_fires"), "spec_used": met.get("spec_used"), "fires_before_enter":
                  sum(1 for e in spec if e["_t"] < t_enter), "web_calls": met.get("web_calls"),
                  "failed": bool(met.get("failed")), "modality": met.get("modality")}
    return out


def p50(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs)) if xs else None


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cps", type=float, default=5.0)
    ap.add_argument("--modes", default="headstart")
    ap.add_argument("--out", default=str(ROOT / "eval/typed/results.json"))
    a = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / "eval/showcase/questions.jsonl").read_text().splitlines() if x.strip()][:a.n]
    rng = random.Random(7)
    modes = a.modes.split(",")
    res = []
    for i, r in enumerate(rows):
        order = [True, False] if i % 2 == 0 else [False, True]
        row = {"id": r["id"], "text": r["text"]}
        for typed in order:
            try:
                row["typed" if typed else "pasted"] = await ask(a.port, r["text"], typed, a.cps, rng, modes)
            except Exception as e:  # noqa: BLE001
                row["typed" if typed else "pasted"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        res.append(row)
        print(r["id"], {k: {m: v.get("enter_to_answer_ms") for m, v in row[k].items() if isinstance(v, dict)}
                        for k in ("typed", "pasted")}, flush=True)
    summ = {}
    for m in modes:
        for arm in ("typed", "pasted"):
            xs = [(x[arm].get(m) or {}).get("enter_to_answer_ms") for x in res if isinstance(x[arm].get(m), dict)]
            rs = [(x[arm].get(m) or {}).get("ready_ms") for x in res if isinstance(x[arm].get(m), dict)]
            fb = [(x[arm].get(m) or {}).get("fires_before_enter") or 0 for x in res if isinstance(x[arm].get(m), dict)]
            summ[f"{m}.{arm}"] = {"n": len([x for x in xs if x is not None]), "enter_to_answer_p50_ms": p50(xs),
                                  "enter_to_answer_p90_ms": pct(xs, 0.9), "server_ready_p50_ms": p50(rs),
                                  "fires_before_enter_mean": round(sum(fb) / max(1, len(fb)), 2)}
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "web_backend": "wikipedia (server env)", "cps": a.cps,
           "modes": modes, "summary": summ, "rows": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
