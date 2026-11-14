"""Counterfactual labels for the KDIR workload: for every distinct question of the kdir streams (and the old s1 / s2
streams), what EVERY action returns, recorded once (a response cache: re-runs only fetch what is missing).

  route   today's path: Arc frame -> router plan -> router.route(final=True) (record_fixture.one); Keystone off
  paid    the paid web lane alone (WEB_BACKEND=wikipedia stands in for Tavily: real network latency, no Tavily spend)
          -- for card / facts questions, which today's path answers without it
  free    the free lanes alone, for web-tier questions today's path never sends there: the local Wikipedia leads
          index (offline) + the Wikipedia summary of the frame's entity (free REST), or the planned card / Wikidata
          lane when the router has one

Then EmbeddingGemma vectors for every new doc (doc encoding) and question surface (query encoding), as
record_fixture.py does. Outputs runs/index/kdir_fixture.json (old fixture entries are copied in) and
runs/index/kdir_vecs.npz. Saves every 100 questions.

    .venv/bin/python eval/keystone/kdir_record.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ["WEB_BACKEND"] = "wikipedia"
os.environ["TAVILY_API_KEY"] = ""
os.environ["WEB_CACHE_FALLBACK"] = "0"
os.environ["WEB_CACHE_RECORD"] = "0"
os.environ["INDEX"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/keystone"))

import numpy as np  # noqa: E402

import record_fixture as RF  # noqa: E402
from replay_frames import frame_for  # noqa: E402
from server import config, embed, knowledge, lanes, router  # noqa: E402

OUT = ROOT / "runs/index"
FX = OUT / "kdir_fixture.json"
VEC = OUT / "kdir_vecs.npz"
STREAMS = ["kdir_train.jsonl", "kdir_dev.jsonl", "kdir_test.jsonl", "stream.jsonl", "stream_s2.jsonl"]


async def alt(text: str, r: dict, sem) -> dict:
    """The counterfactual lane(s) today's path did not run for this question."""
    f = frame_for(text)
    f.route = "web"
    out = {}
    async with sem:
        if r["tier"] in ("facts", "card"):
            t = time.perf_counter()
            try:
                res = await lanes.web(f.query(), final=True, meter={}, entity=f.entity_query())
                out["alt_paid"] = {"docs": [RF.doc_json(d) for d in res][:8], "ms": round((time.perf_counter() - t) * 1000)}
            except Exception as e:  # noqa: BLE001
                out["alt_paid"] = {"docs": [], "ms": round((time.perf_counter() - t) * 1000),
                               "error": f"{type(e).__name__}: {e}"[:120]}
        else:
            t = time.perf_counter()
            p = router.plan(f)
            docs = []
            try:
                if p.lane in ("card", "facts"):
                    docs = await asyncio.wait_for(router._cheap(p, f, {}), config.ROUTER_LANE_TIMEOUT_S)
                else:
                    ent = f.entity_query()
                    ld = []          # the offline leads index is searched at replay time from the question vector
                    ws = []
                    if ent:
                        try:
                            ws = await asyncio.wait_for(knowledge.wiki_summary(ent), 3.0)
                        except Exception:  # noqa: BLE001
                            ws = []
                    seen = {d.url for d in ws}
                    docs = list(ws) + [d for d in ld if d.url not in seen]
                err = None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:120]
            out["free"] = {"docs": [RF.doc_json(d) for d in docs][:6], "ms": round((time.perf_counter() - t) * 1000),
                           "lane": p.lane, "vertical": p.vertical, "error": err}
        await asyncio.sleep(0.1)
    return out


async def main_async():
    texts = []
    seen = set()
    for s in STREAMS:
        p = OUT / s
        if not p.exists():
            continue
        for x in p.read_text().splitlines():
            if x.strip():
                q = json.loads(x)["query"]
                if q not in seen:
                    seen.add(q)
                    texts.append(q)
    fx = json.loads(FX.read_text()) if FX.exists() else {}
    if not fx and (OUT / "fixture.json").exists():
        fx = json.loads((OUT / "fixture.json").read_text())
    print(f"{len(texts)} texts, {sum(1 for t in texts if t in fx and not fx[t].get('error'))} recorded", flush=True)
    await embed.arun(embed.encode_queries, ["warm up"])
    knowledge.leads_index()
    await router.warm()
    sem = asyncio.Semaphore(3)
    todo = [t for t in texts if t not in fx or fx[t].get("error")]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        rows = await asyncio.gather(*[RF.one(t, sem) for t in chunk])
        for r in rows:
            fx[r["text"]] = r
        FX.write_text(json.dumps(fx))
        print(f"route {i + len(chunk)}/{len(todo)}", flush=True)
    need = [t for t in texts if "alt_paid" not in fx[t] and "free" not in fx[t]]
    for i in range(0, len(need), 100):
        chunk = need[i:i + 100]
        rows = await asyncio.gather(*[alt(t, fx[t], sem) for t in chunk])
        for t, r in zip(chunk, rows):
            fx[t].update(r)
        FX.write_text(json.dumps(fx))
        print(f"alt {i + len(chunk)}/{len(need)}", flush=True)
    vectors(fx)


def vectors(fx):
    have = dict(np.load(VEC, allow_pickle=False)) if VEC.exists() else (
        dict(np.load(OUT / "fixture_vecs.npz", allow_pickle=False)) if (OUT / "fixture_vecs.npz").exists() else {})
    dkeys = list(have.get("doc_ids", np.array([], dtype=str)))
    dvec = list(have.get("doc_vecs", np.zeros((0, 768), np.float16)))
    seen = set(dkeys)
    need = []
    for r in fx.values():
        for d in r["docs"] + (r.get("alt_paid") or {}).get("docs", []) + (r.get("free") or {}).get("docs", []):
            if d["id"] not in seen:
                seen.add(d["id"])
                need.append(d)
    print(f"doc vectors: {len(need)} new", flush=True)
    for i in range(0, len(need), 32):
        b = need[i:i + 32]
        v = embed.encode_docs([d["text"][:600] for d in b], [d["title"] for d in b])
        dkeys += [d["id"] for d in b]
        dvec += list(v.astype(np.float16))
        if i % 3200 == 0:
            print(f"  docs {i}/{len(need)}", flush=True)
    fam = [json.loads(x) for x in (OUT / "kdir_families.jsonl").read_text().splitlines() if x.strip()]
    fam += [json.loads(x) for x in (OUT / "families.jsonl").read_text().splitlines() if x.strip()]
    qtexts = list(dict.fromkeys(list(fx) + [s for f in fam for s in [f["canon"], *(f.get("paraphrases") or []),
                                                                     f.get("near_miss") or "", f.get("resolved") or ""]
                                             if s]))
    qkeys = list(have.get("q_texts", np.array([], dtype=str)))
    qvec = list(have.get("q_vecs", np.zeros((0, 768), np.float16)))
    qseen = set(qkeys)
    qneed = [t for t in qtexts if t not in qseen]
    print(f"question vectors: {len(qneed)} new", flush=True)
    for i in range(0, len(qneed), 64):
        b = qneed[i:i + 64]
        qkeys += b
        qvec += list(embed.encode_queries(b).astype(np.float16))
    np.savez(VEC, doc_ids=np.array(dkeys), doc_vecs=np.array(dvec, np.float16), q_texts=np.array(qkeys),
             q_vecs=np.array(qvec, np.float16))
    tiers = {}
    for r in fx.values():
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print(f"fixture {len(fx)} questions, tiers {tiers}, docs {len(dkeys)}, question vectors {len(qkeys)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--vectors"]:
        vectors(json.loads(FX.read_text()))
    else:
        asyncio.run(main_async())
