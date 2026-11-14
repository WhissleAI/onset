"""CALIB_ROUTE off vs on, text path, Wikipedia lane (no Tavily spend), on the calibrator's held-out TEST split.

Off = the eval/calib_data.py row as recorded (today's route, ranking and answer). On = the same, except the pre-answer
confidence (server/calibrate.py, fitted on the train split only) picks an action first, as speculator._calib_pre does:
  high  answer now (row unchanged)
  mid   one free lookup (Wikidata fact + local leads), new docs join the ranking by cosine, re-answer
  low   one more web search of calibrate.alt_query (counted as a paid call; Wikipedia stands in for Tavily) when the
        question's parity budget allows (final_max = 2), else the free lookup; re-answer
Reports accuracy, paid calls / question, answer-model calls, extra lookup ms, per band. Untimed for the answer itself
(the live latency cost is measured by the run_ab replay, PROGRESS.md).

  .venv/bin/python eval/calib_route_eval.py [--rows runs/calib/rows.jsonl] [--out runs/calib/route_eval.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ["WEB_BACKEND"] = "wikipedia"
os.environ["TAVILY_API_KEY"] = ""
os.environ.setdefault("INDEX", "0")
os.environ.setdefault("WEB_CACHE_RECORD", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from calib_fit import doc_of, row_features  # noqa: E402
from replay_frames import frame_for  # noqa: E402
from run_ab import score  # noqa: E402
from server import answer, calibrate, embed, lanes, verify  # noqa: E402

PAID_MAX = 2          # COST_MODE=parity: cost.profile("parity").final_max (1 paid + 1 the endpoint may add)


def merge(qtext, ranked, extra):
    seen = {d.id for d, _ in ranked}
    extra = [d for d in extra if d.id not in seen]
    if not extra:
        return ranked
    qv = embed.encode_queries([qtext])[0]
    dv = embed.encode_docs([d.text[:600] for d in extra], [d.title for d in extra])
    new = [(d, float(v @ qv)) for d, v in zip(extra, dv)]
    card = [r for r in ranked if r[0].meta.get("card")]
    rest = sorted([r for r in ranked if not r[0].meta.get("card")] + new, key=lambda r: -r[1])
    return (card + rest)[:5]


async def one(r, sem, frames):
    x = row_features(r, frames)
    p = calibrate.raw(x, "pre")
    paid_off = int((r.get("route") or {}).get("paid") or 0)
    out = {"text": r["text"], "set": r.get("set"), "conf_pre": p, "band": calibrate.band(p), "correct_off": r["correct"],
           "paid_off": paid_off, "correct_on": r["correct"], "paid_on": paid_off, "act": "", "llm_on": 0}
    act = calibrate.route_action(p, paid_off < PAID_MAX, True)
    if not act:
        return out
    async with sem:
        f = frames.get(r["text"]) or frame_for(r["text"])
        ranked = [(doc_of(d), d.get("rr") or 0.0) for d in r["ranked"]]
        t0 = time.perf_counter()
        extra = []
        if act == "paid":
            aq = calibrate.alt_query(f, {f.query()})
            if aq:
                try:
                    extra = list(await lanes.wikipedia(aq, k=4))
                except Exception:  # noqa: BLE001
                    extra = []
                out["paid_on"] = paid_off + 1
            else:
                act = "free"
        if act == "free":
            try:
                extra = await asyncio.wait_for(calibrate.free_docs(f.resolved_text, f), 3.0)
            except Exception:  # noqa: BLE001
                extra = []
        out.update(act=act, extra=len(extra), extra_ms=round((time.perf_counter() - t0) * 1000))
        new = await embed.arun(merge, f.resolved_text, ranked, extra)
        if [d.id for d, _ in new] == [d.id for d, _ in ranked]:
            return out                                  # nothing new made the top 5: the same answer
        st = answer.Stream(f.resolved_text, [d for d, _ in new], tag="final", hint=f.answer_hint())
        async for _ in st.follow():
            pass
        out["llm_on"] = 1
        out["answer_on"] = st.text()
        out["correct_on"] = score(st.text(), r["expect"])
        v = verify.check(st.text(), [d for d, _ in new], f.resolved_text)
        out["verify_on"] = v["status"]
        return out


def summary(rows):
    n = len(rows)
    s = {"n": n,
         "acc_off": round(sum(bool(r["correct_off"]) for r in rows) / n, 3),
         "acc_on": round(sum(bool(r["correct_on"]) for r in rows) / n, 3),
         "paid_per_q_off": round(sum(r["paid_off"] for r in rows) / n, 3),
         "paid_per_q_on": round(sum(r["paid_on"] for r in rows) / n, 3),
         "extra_answer_calls": sum(r["llm_on"] for r in rows),
         "fixed": sum(1 for r in rows if r["correct_on"] and not r["correct_off"]),
         "broken": sum(1 for r in rows if r["correct_off"] and not r["correct_on"]),
         "acts": {}}
    for r in rows:
        k = r["act"] or "answer now"
        a = s["acts"].setdefault(k, {"n": 0, "acc_off": 0, "acc_on": 0})
        a["n"] += 1
        a["acc_off"] += int(bool(r["correct_off"]))
        a["acc_on"] += int(bool(r["correct_on"]))
    ms = sorted(r["extra_ms"] for r in rows if r.get("extra_ms") is not None)
    s["extra_lookup_ms_p50"] = ms[len(ms) // 2] if ms else None
    return s


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROOT / "runs/calib/rows.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "runs/calib/route_eval.json"))
    ap.add_argument("--low", type=float, default=None, help="override the low band edge (e.g. = high: paid for all)")
    a = ap.parse_args()
    if a.low is not None:
        calibrate.model().setdefault("bands", {})["low"] = a.low
    rows = [json.loads(x) for x in Path(a.rows).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("correct") is not None and not r.get("error") and r.get("split") == a.split]
    await answer.warm()
    await embed.arun(embed.warmup)
    frames: dict = {}
    sem = asyncio.Semaphore(3)
    res = await asyncio.gather(*[one(r, sem, frames) for r in rows])
    out = {"split": a.split, "bands": calibrate.bands(), "summary": summary(res), "rows": res}
    print(json.dumps(out["summary"], indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))


asyncio.run(main())
