"""Labelled answers for the answer calibrator (server/calibrate.py), text path, WEB_BACKEND=wikipedia (no Tavily spend).

For every labelled question (Keystone families with gold / LLM-gold `expect`: canon + two paraphrases, the 58 set's
web questions, the showcase scored set, the Arc-value corrections):
  Arc Tagger frame (eval/replay_frames.frame_for) -> today's router path (router.route(final=True), Keystone off; the
  recorded kdir fixture's docs are reused when it has this exact text, read-only) -> EmbeddingGemma rerank (pool
  cosine, card first, top 5, as the live final ranking) -> the real answer model -> verify.check -> keyword scorer.
Rows carry the raw docs (trimmed), the rerank scores, the route and the answer, so features can be recomputed without
re-running. Resumable: rows already in --out are skipped. Answer-model spend is counted (est. $ from server/meter.py).

  .venv/bin/python eval/calib_data.py --out runs/calib/rows.jsonl [--fixture ../headstart-kdir/runs/index/kdir_fixture.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ["WEB_BACKEND"] = "wikipedia"
os.environ["TAVILY_API_KEY"] = ""
os.environ.setdefault("INDEX", "0")
os.environ.setdefault("WEB_CACHE_RECORD", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from replay_frames import frame_for  # noqa: E402
from run_ab import score  # noqa: E402
from server import answer, embed, lanes, meter, router, verify  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "runs/calib/rows.jsonl"))
ap.add_argument("--families", default=str(ROOT.parent / "headstart/runs/index/families.jsonl"))
ap.add_argument("--fixture", default="")
ap.add_argument("--surfaces", type=int, default=3, help="canon + this many - 1 paraphrases per family")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conc", type=int, default=3)
args = ap.parse_args()


def questions() -> list[dict]:
    out, seen = [], set()

    def add(text, expect, group, src, kind):
        if text and text not in seen and expect:
            seen.add(text)
            out.append({"text": text, "expect": expect, "group": group, "set": src, "kind": kind})
    fp = Path(args.families)
    if fp.exists():
        for line in fp.read_text().splitlines():
            d = json.loads(line)
            if not d.get("expect") or d["kind"] in ("conversation", "part") or d["kind"].startswith("multipart"):
                continue
            surf = [d["canon"]] + list(d.get("paraphrases") or [])
            for t in surf[:args.surfaces]:
                add(t, d["expect"], d["fid"], "families:" + d.get("expect_src", ""), d["kind"])
    for path, name in ((ROOT / "eval/questions.jsonl", "q58"), (ROOT / "eval/arcvalue/showcase_scored.jsonl", "showcase"),
                       (ROOT / "eval/arcvalue/corrections.jsonl", "corrections")):
        for line in path.read_text().splitlines():
            q = json.loads(line)
            if q.get("local") or q.get("kind") == "current":
                continue
            add(q["text"], q.get("expect") or [], q["id"], name, q.get("kind", ""))
    return out


def doc_json(d) -> dict:
    meta = {k: v for k, v in (d.meta or {}).items() if isinstance(v, (str, int, float, bool, type(None)))}
    return {"id": d.id, "title": d.title, "text": d.text[:1200], "url": d.url, "source": d.source,
            "score": d.score, "meta": meta}


def doc_of(j: dict) -> lanes.Doc:
    return lanes.Doc(id=j["id"], title=j.get("title", ""), text=j.get("text", ""), url=j.get("url", ""),
                     source=j.get("source", ""), score=float(j.get("score") or 0.0), meta=dict(j.get("meta") or {}))


async def route_docs(f, fx: dict | None) -> tuple[list, dict]:
    p = router.plan(f)
    if fx is not None and not fx.get("error"):
        return [doc_of(d) for d in fx.get("docs") or []], {"lane": p.lane, "conf": p.conf, "tier": fx.get("tier"),
                                                            "paid": fx.get("paid", 0), "fixture": True}
    m: dict = {}

    async def web_fn():
        return await lanes.web(f.query(), final=True, meter=m, entity=f.entity_query())
    res = []
    for attempt in range(3):
        m.clear()
        try:
            res = await router.route(f.query(), f, final=True, meter=m, web_fn=web_fn)
            break
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2.0 * (attempt + 1))
    return list(res), {"lane": p.lane, "conf": p.conf, "tier": m.get("tier"), "paid": m.get("paid", 0)}


def rank(qtext: str, docs: list) -> list:
    """The live final ranking (speculator.rank + _card_first): EmbeddingGemma cosine over the pool, card first."""
    if not docs:
        return []
    qv = embed.encode_queries([qtext])[0]
    dv = embed.encode_docs([d.text for d in docs], [d.title for d in docs])
    s = dv @ qv
    order = sorted(range(len(docs)), key=lambda i: (-(1 if docs[i].meta.get("card") else 0), -s[i]))[:5]
    return [(docs[i], float(s[i])) for i in order]


SPEND = {"usd": 0.0, "calls": 0}


async def one(q: dict, fixture: dict, sem) -> dict | None:
    async with sem:
        f = frame_for(q["text"])
        f.route = "web"
        docs, route = await route_docs(f, fixture.get(q["text"]))
        ranked = await embed.arun(rank, f.resolved_text, docs)
        if not ranked:
            return {**q, "error": "no docs", "route": route}
        st = answer.Stream(f.resolved_text, [d for d, _ in ranked], tag="final", hint=f.answer_hint())
        async for _ in st.follow():
            pass
        text = st.text()
        try:
            u = meter.fields([st], st, {"paid_calls": 0})
            SPEND["usd"] += float(u.get("usd_llm") or 0.0)
        except Exception:  # noqa: BLE001
            pass
        SPEND["calls"] += 1
        v = verify.check(text, [d for d, _ in ranked], f.resolved_text)
        return {**q, "route": route, "answer": text, "fallback": getattr(st, "fallback", None), "error": st.error,
                "by": getattr(st, "by", None),
                "verify": {k: v[k] for k in ("status", "n_claims", "found")},
                "ranked": [{**doc_json(d), "rr": round(s, 4)} for d, s in ranked],
                "correct": score(text, q["expect"]),
                "split": "test" if zlib.crc32(q["group"].encode()) % 10 < 3 else "train"}


async def main():
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["text"] for x in out.read_text().splitlines() if x.strip()}
    qs = [q for q in questions() if q["text"] not in done]
    if args.limit:
        qs = qs[:args.limit]
    fixture = json.loads(Path(args.fixture).read_text()) if args.fixture and Path(args.fixture).exists() else {}
    print(f"{len(qs)} to do ({len(done)} done), fixture {len(fixture)}", flush=True)
    await answer.warm()
    await embed.arun(embed.warmup)
    sem = asyncio.Semaphore(args.conc)
    t0 = time.time()
    with out.open("a") as fh:
        for i in range(0, len(qs), 30):
            rows = await asyncio.gather(*[one(q, fixture, sem) for q in qs[i:i + 30]], return_exceptions=True)
            for r in rows:
                if isinstance(r, dict):
                    fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{i + len(rows)}/{len(qs)}  {time.time() - t0:.0f}s  answer calls {SPEND['calls']}  "
                  f"est. ${SPEND['usd']:.3f}", flush=True)


asyncio.run(main())
