"""A small REAL-Tavily labelled slice for the calibrator (its src_top / src_margin / src_age_d features are constant
on the Wikipedia lane, so they were dropped from the fit). Hard cap: 100 Tavily credits in total, across all runs,
enforced by a ledger (runs/train/tavily/ledger.json) checked BEFORE every request; basic depth (1 credit / search).

Questions: synthetic standalone turns with Wikidata gold (synth.py) first, then the 58 set's scored web questions.
Per question: Arc Tagger frame -> one Tavily /search (final plan, basic) -> the live final ranking (EmbeddingGemma,
top 5) -> the real answer model -> verify -> keyword scorer. Rows go to runs/train/calib/label_rows.jsonl in
eval/calib_data.py's format (group "syn-tavily-..." = always TRAIN), so calib_step.py refits with them.

  .venv/bin/python eval/train/tavily_slice.py --max-credits 60      (never more than 100 - already used)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["WEB_BACKEND"] = "tavily"
os.environ.setdefault("INDEX", "0")
os.environ.setdefault("WEB_CACHE_RECORD", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.sys_path()

CAP = 100.0
D = C.TRAIN / "tavily"
LEDGER = D / "ledger.json"


def ledger() -> dict:
    return C.read_json(LEDGER, {"credits_used": 0.0, "searches": 0, "cap": CAP})


def allowed(cost: float, budget: float, led: dict | None = None) -> bool:
    led = led or ledger()
    return led["credits_used"] + cost <= min(CAP, budget) + 1e-9


def charge(credits: float):
    led = ledger()
    led["credits_used"] = round(led["credits_used"] + credits, 3)
    led["searches"] += 1
    led["updated"] = C.now_iso()
    C.write_json(LEDGER, led)


def questions(limit: int) -> list[dict]:
    out, seen = [], set()
    for p in sorted(C.TRAIN.glob("synth/c*/sessions.json")):
        for s in C.read_json(p, []):
            t = s["turns"][0]
            g = t["gold"]
            if g.get("expect") and not g.get("followup") and s["cls"] in ("topic_switch", "elliptical", "self_correction"):
                if t["text"] not in seen:
                    seen.add(t["text"])
                    out.append({"text": t["text"], "expect": g["expect"], "group": f"syn-tavily-{t['tid']}-{p.parent.name}",
                                "set": "tavily:synth", "kind": s["cls"]})
    for line in (C.ROOT / "eval/questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        if q.get("expect") and not q.get("local") and q.get("kind") != "current" and q["text"] not in seen:
            seen.add(q["text"])
            out.append({"text": q["text"], "expect": q["expect"], "group": f"syn-tavily-{q['id']}", "set": "tavily:q58",
                        "kind": q.get("kind", "")})
    return out[:limit]


def doc_json(d) -> dict:
    """eval/calib_data.py's row format for a doc (that script runs on import, so it is not imported)."""
    meta = {k: v for k, v in (d.meta or {}).items() if isinstance(v, (str, int, float, bool, type(None)))}
    return {"id": d.id, "title": d.title, "text": d.text[:1200], "url": d.url, "source": d.source,
            "score": d.score, "meta": meta}


def rank(qtext: str, docs: list) -> list:
    """The live final ranking as eval/calib_data.rank: EmbeddingGemma cosine over the pool, top 5."""
    from server import embed
    if not docs:
        return []
    qv = embed.encode_queries([qtext])[0]
    dv = embed.encode_docs([d.text for d in docs], [d.title for d in docs])
    s = dv @ qv
    order = sorted(range(len(docs)), key=lambda i: (-(1 if docs[i].meta.get("card") else 0), -s[i]))[:5]
    return [(docs[i], float(s[i])) for i in order]


async def one(q: dict, budget: float) -> dict | None:
    from replay_frames import frame_for
    from run_ab import score
    from server import answer, embed, lanes, verify
    f = frame_for(q["text"])
    plan = lanes.web_plan(f.intent, f.text, final=True)
    plan["search_depth"] = "basic"
    cost = lanes.COST.get("basic", 1)
    if not allowed(cost, budget):
        return None
    m: dict = {}
    res = await lanes.tavily(f.query(), plan, meter=m, final=True)
    charge(float(m.get("credits") or cost))
    ranked = await embed.arun(rank, f.resolved_text, list(res))
    if not ranked:
        return {**q, "error": "no docs"}
    st = answer.Stream(f.resolved_text, [d for d, _ in ranked], tag="final", hint=f.answer_hint())
    async for _ in st.follow():
        pass
    text = st.text()
    v = verify.check(text, [d for d, _ in ranked], f.resolved_text)
    return {**q, "route": {"lane": "web", "tier": "web", "paid": 1, "backend": "tavily"}, "answer": text,
            "error": st.error, "verify": {k: v[k] for k in ("status", "n_claims", "found")},
            "ranked": [{**doc_json(d), "rr": round(s, 4)} for d, s in ranked], "correct": score(text, q["expect"]),
            "split": "train", "credits": m.get("credits")}


async def amain(a):
    from server import answer, config, embed
    if not config.TAVILY_API_KEY:
        return {"skip": "no Tavily key"}
    out = C.TRAIN / "calib" / "label_rows.jsonl"
    done = {r["text"] for r in C.read_jsonl(out) if str(r.get("group", "")).startswith("syn-tavily")}
    qs = [q for q in questions(10_000) if q["text"] not in done]
    await answer.warm()
    await embed.arun(embed.warmup)
    rows = []
    for q in qs:
        if not allowed(1, a.max_credits):
            break
        try:
            r = await one(q, a.max_credits)
        except Exception as e:  # noqa: BLE001
            r = {**q, "error": f"{type(e).__name__}: {e}"[:200]}
        if r is None:
            break
        rows.append(r)
        C.append_jsonl(out, [r])
    led = ledger()
    rep = {"rows": len(rows), "correct": sum(1 for r in rows if r.get("correct")), "ledger": led}
    C.write_json(D / "report.json", rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-credits", type=float, default=60.0, help="total credits this slice may reach (<= 100)")
    a = ap.parse_args(argv)
    print(json.dumps(asyncio.run(amain(a)), default=str))


if __name__ == "__main__":
    main()
