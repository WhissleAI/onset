"""Multi-hop questions answered by graph lookups over cached Wikidata 1-hop neighbourhoods (free; no web search).

For each two-hop question of eval/multipart (the planner's "hop" kind: "where was the author of Pride and Prejudice
born"): its typed key born_place(@author_of(pride and prejudice)); fetch the inner subject's neighbourhood
(knowledge.prefetch: entity search + one SPARQL query, the same call memory's pre-warm makes), absorb it into the graph,
follow the inner relation to its value node, fetch + absorb that node's neighbourhood, look the outer relation up.
Correct = the hop value and the final value contain the expected keywords. Timed end to end (network included) and
again warm (graph only).

    .venv/bin/python eval/keystone/hop_eval.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from replay_frames import frame_for  # noqa: E402
from server import config, knowledge  # noqa: E402
from server.keystone import hook  # noqa: E402
from server.keystone.core import Flags, Keystone  # noqa: E402
from server.keystone.relkey import key_of_frame  # noqa: E402


def ok(groups, text):
    low = text.lower()
    return all(any(re.search(re.escape(a.strip().lower()), low) for a in g.split("|")) for g in groups)


async def main():
    config.INDEX, config.INDEX_PERSIST = True, False
    ks = hook.reset(Keystone(Flags(hash=False)))
    rows = []
    for q in [json.loads(x) for x in (ROOT / "eval/multipart/questions.jsonl").read_text().splitlines() if x.strip()]:
        k = key_of_frame(frame_for(q["text"]))
        if k is None or not k.subject.startswith("@"):
            continue
        inner, subj = k.subject[1:-1].split("(", 1)
        t = time.perf_counter()
        await knowledge.prefetch(subj)
        hook.absorb_wikidata(subj, ks)
        node, vals = ks.g.lookup(subj, inner)
        hop = vals[0] if vals else None
        final = []
        if hop and hop.get("node"):
            name = ks.g.nodes[hop["node"]]["name"]
            await knowledge.prefetch(name)
            hook.absorb_wikidata(name, ks)
            _, final = ks.g.lookup(k.subject, k.relation)
        cold_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        _, again = ks.g.lookup(k.subject, k.relation)
        warm_us = (time.perf_counter() - t) * 1e6
        text = " ".join([hop["value"] if hop else ""] + [v["value"] for v in final])
        rows.append({"id": q["id"], "key": k.text(), "hop": hop["value"] if hop else None,
                     "answer": [v["value"] for v in final][:2], "correct": ok(q["expect"][-2:], text) if final else False,
                     "hop_correct": bool(hop) and ok(q["expect"][:1], hop["value"]) if q["id"] != "mp21" else None,
                     "cold_ms": round(cold_ms), "warm_us": round(warm_us, 1)})
        print(rows[-1], flush=True)
    s = {"n": len(rows), "answered": sum(1 for r in rows if r["answer"]), "correct": sum(1 for r in rows if r["correct"]),
         "cold_ms_p50": sorted(r["cold_ms"] for r in rows)[len(rows) // 2] if rows else None,
         "warm_us_p50": sorted(r["warm_us"] for r in rows)[len(rows) // 2] if rows else None}
    print(s)
    (ROOT / "runs/index/hop_eval.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
