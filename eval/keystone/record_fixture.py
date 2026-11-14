"""Record what TODAY's retrieval path returns for every distinct question of the workload streams, once, so every
policy replays against the same results (deterministic, no repeated network, no Tavily spend).

For each question text: Arc Tagger frame (word by word, as live; eval/replay_frames.py), the router's plan, then
router.route(final=True) with the paid web lane stood in by Wikipedia (WEB_BACKEND=wikipedia: real network latency).
Free card / Wikidata lanes run for real. Recorded per question: the frame read-out, plan, docs (+ meta), which tier
answered, paid (1 = a Tavily-equivalent call), wall ms. Then EmbeddingGemma vectors for every doc (the reranker's doc
encoding) and every question (query encoding) and every family surface form (for the hash).

    WEB_BACKEND=wikipedia .venv/bin/python eval/keystone/record_fixture.py runs/index/stream.jsonl runs/index/stream_s2.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WEB_BACKEND", "wikipedia")
os.environ.setdefault("WEB_CACHE_FALLBACK", "0")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from replay_frames import frame_for  # noqa: E402
from server import embed, knowledge, lanes, router  # noqa: E402

OUT = ROOT / "runs/index"


def frame_json(f) -> dict:
    return {"words": list(f.words), "intent": f.intent, "intent_p": round(f.intent_p, 3), "coarse": f.coarse,
            "slots": [list(s) for s in f.slots], "content": list(f.content), "phrases": list(f.phrases),
            "ent_types": list(f.ent_types), "num_types": list(f.num_types), "acts": list(f.acts), "when": f.when,
            "query": f.query(), "entity_query": f.entity_query(), "resolved": f.resolved_text}


def doc_json(d) -> dict:
    meta = {}
    for k, v in (d.meta or {}).items():
        try:
            json.dumps(v)
            meta[k] = v
        except TypeError:
            meta[k] = str(v)
    return {"id": d.id, "title": d.title, "text": d.text[:1500], "url": d.url, "source": d.source, "score": d.score,
            "meta": meta}


async def one(text: str, sem: asyncio.Semaphore) -> dict:
    f = frame_for(text)
    f.route = "web"
    p = router.plan(f)
    m: dict = {}
    web_ms = [None]

    async def web_fn():
        t = time.perf_counter()
        try:
            r = await lanes.web(f.query(), final=True, meter=m, entity=f.entity_query())
        finally:
            web_ms[0] = round((time.perf_counter() - t) * 1000)
        return r

    async with sem:
        for attempt in range(4):
            m.clear()
            lanes.reset_web_status()
            t = time.perf_counter()
            err = None
            try:
                res = await router.route(f.query(), f, final=True, meter=m, web_fn=web_fn)
            except Exception as e:  # noqa: BLE001
                res, err = [], f"{type(e).__name__}: {e}"[:160]
            ms = round((time.perf_counter() - t) * 1000)
            if err is None:
                break
            await asyncio.sleep(2.0 * (attempt + 1))          # Wikipedia 429 / timeout: back off, retry
        await asyncio.sleep(0.25)
    return {"text": text, "frame": frame_json(f), "plan": p.to_json(), "tier": m.get("tier"), "paid": m.get("paid", 0),
            "ms": ms, "web_ms": web_ms[0], "route_fallback": m.get("route_fallback"), "error": err,
            "docs": [doc_json(d) for d in res][:8]}


async def main_async(paths):
    texts = []
    for p in paths:
        for x in Path(p).read_text().splitlines():
            if x.strip():
                q = json.loads(x)["query"]
                if q not in texts:
                    texts.append(q)
    fx_path = OUT / "fixture.json"
    old = json.loads(fx_path.read_text()) if fx_path.exists() else {}
    todo = [t for t in texts if t not in old or old[t].get("error")]
    print(f"{len(texts)} texts, {len(todo)} to record", flush=True)
    await embed.arun(embed.encode_queries, ["warm up"])
    knowledge.leads_index()
    await router.warm()
    sem = asyncio.Semaphore(2)
    rows = await asyncio.gather(*[one(t, sem) for t in todo])
    fx = {**old, **{r["text"]: r for r in rows}}
    fx_path.write_text(json.dumps(fx))
    # vectors: every doc (doc encoding, title-prefixed as the reranker does), every question and family surface form
    vec_path = OUT / "fixture_vecs.npz"
    have = dict(np.load(vec_path, allow_pickle=False)) if vec_path.exists() else {}
    dkeys = list(have.get("doc_ids", np.array([], dtype=str)))
    dvec = list(have.get("doc_vecs", np.zeros((0, 768), np.float16)))
    seen = set(dkeys)
    need = []
    for r in fx.values():
        for d in r["docs"]:
            if d["id"] not in seen:
                seen.add(d["id"])
                need.append(d)
    for i in range(0, len(need), 32):
        b = need[i:i + 32]
        v = embed.encode_docs([d["text"][:600] for d in b], [d["title"] for d in b])
        dkeys += [d["id"] for d in b]
        dvec += list(v.astype(np.float16))
    fam = [json.loads(x) for x in (OUT / "families.jsonl").read_text().splitlines() if x.strip()]
    qtexts = list(dict.fromkeys(list(fx) + [s for f in fam for s in [f["canon"], *(f.get("paraphrases") or []),
                                                                     f.get("near_miss") or "", f.get("resolved") or ""]
                                             if s]))
    qkeys = list(have.get("q_texts", np.array([], dtype=str)))
    qvec = list(have.get("q_vecs", np.zeros((0, 768), np.float16)))
    qseen = set(qkeys)
    qneed = [t for t in qtexts if t not in qseen]
    for i in range(0, len(qneed), 64):
        b = qneed[i:i + 64]
        qkeys += b
        qvec += list(embed.encode_queries(b).astype(np.float16))
    np.savez(vec_path, doc_ids=np.array(dkeys), doc_vecs=np.array(dvec, np.float16), q_texts=np.array(qkeys),
             q_vecs=np.array(qvec, np.float16))
    tiers = {}
    for r in fx.values():
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print(f"fixture {len(fx)} questions, tiers {tiers}, paid {sum(r['paid'] for r in fx.values())}, "
          f"docs {len(dkeys)}, question vectors {len(qkeys)}")


if __name__ == "__main__":
    asyncio.run(main_async(sys.argv[1:]))
