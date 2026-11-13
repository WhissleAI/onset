"""Confidence-weighted web queries vs the default query, on the live Tavily lane (eval/narrow_eval.py machinery: same
request builder, same disk cache runs/narrow_cache.json, same blind judge + grade cache runs/narrow_judge.json).

Questions: the 80 real Flux traces (runs/flux_traces.jsonl). Heard text = Flux's final words; per-word fused confidence
from runs/conf_features.jsonl (the final update of the final turn: Flux conf + revision history + tagger span).
Variants of the FINAL query (the one on the critical path):
  base   the default (shaped query + family domains, NARROW_* defaults)
  quote  multi-word entity phrases whose words are all >= QUOTE_MIN confident go in "double quotes"
  drop   content words that are NOT in a typed slot and are < DROP_MAX confident are dropped
  both   quote + drop
Only questions whose query changes are sent / judged (paired). Budget: --max-credits (Tavily), judge calls counted.

  .venv/bin/python eval/confq_eval.py --dry
  .venv/bin/python eval/confq_eval.py --max-credits 80 --judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import narrow_eval as ne  # noqa: E402
from server import confidence, lanes, narrow  # noqa: E402
from server import frame as framemod  # noqa: E402
from server.ear import flux_word  # noqa: E402

QUOTE_MIN, DROP_MAX = 0.9, 0.5


def final_records():
    out = {}
    for line in (ROOT / "runs" / "conf_features.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["turn"] == r["final_turn"] and r["complete"]:
            out[r["id"]] = r                      # the last complete update wins (arrival order in the file)
    return out


def transform(q: str, f, wc: list[float], quote: bool, drop: bool) -> str:
    toks = q.split()
    if drop:
        slot_tok = set()
        for s in f.spans:
            if s.get("family") in ("entity", "number") and s["type"] != "entity.numeral":
                slot_tok |= {framemod.norm(f.words[i]) for i in range(s["start"], min(s["end"], len(f.words)))}
        low = {framemod.norm(f.words[i]) for i in f.content_idx if i < len(wc) and wc[i] < DROP_MAX}
        low -= slot_tok
        kept = [t for t in toks if t not in low]
        if len(kept) >= 2:
            toks = kept
    out = " ".join(toks)
    if quote:
        for s in f.spans:
            if not s["type"].startswith("entity.mention") or s["end"] - s["start"] < 2 or s["end"] > len(wc):
                continue
            if min(wc[s["start"]:s["end"]]) < QUOTE_MIN:
                continue
            ph = " ".join(framemod.norm(w) for w in f.words[s["start"]:s["end"]])
            if ph in out and f'"{ph}"' not in out:
                out = out.replace(ph, f'"{ph}"', 1)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--max-credits", type=float, default=80)
    ap.add_argument("--out", default=str(ROOT / "runs" / "confq_eval.json"))
    a = ap.parse_args()
    ne.set_cfg({})
    recs = final_records()
    traces = {json.loads(x)["id"]: json.loads(x) for x in (ROOT / "runs" / "flux_traces.jsonl").read_text().splitlines()}
    items = []
    for qid, r in recs.items():
        words = r["words"]
        frames = ne.prefix_frames(" ".join(words))
        f = frames[-1]
        wc = confidence.word_confs(r["words"], r["conf"], r["revs"], r["age"], r["spans"])
        q0, body0 = ne.request(f, final=True)
        it = {"id": qid, "heard": " ".join(words), "query": q0, "body": {"base": body0}, "q": {"base": q0},
              "min_conf": round(min(wc), 3), "low_words": [w for w, c in zip(words, wc) if c < DROP_MAX]}
        for name, qq, dd in (("quote", True, False), ("drop", False, True), ("both", True, True)):
            q = transform(q0, f, wc, qq, dd)
            it["q"][name] = q
            it["body"][name] = {**body0, "query": q}
        items.append(it)
    changed = {v: [it for it in items if it["q"][v] != it["q"]["base"]] for v in ("quote", "drop", "both")}
    print(f"{len(items)} questions; changed: " + ", ".join(f"{v} {len(x)}" for v, x in changed.items()))
    for v, xs in changed.items():
        for it in xs[:40]:
            print(f"  {v:5s} {it['id']}: {it['q']['base']!r} -> {it['q'][v]!r}")
    if a.dry:
        return
    cache = json.loads(ne.CACHE.read_text()) if ne.CACHE.exists() else {}
    bodies = [it["body"]["base"] for v in changed for it in changed[v]] + \
             [it["body"][v] for v in changed for it in changed[v]]
    spent, n = await ne.run_calls(bodies, cache, a.max_credits)
    ne.CACHE.write_text(json.dumps(cache))
    print(f"sent {n} Tavily requests, {spent} credits")
    jpath = ROOT / "runs" / "narrow_judge.json"
    grades = json.loads(jpath.read_text()) if jpath.exists() else {}
    calls = {"n": 0}
    if a.judge:
        from server import answer, config

        async def judge(it):
            seen = {}
            for v in ("base", "quote", "drop", "both"):
                for d in ne.docs_of(cache.get(ne.bkey(it["body"][v]))):
                    seen.setdefault(d.url, d)
            g = grades.setdefault(it["id"], {})
            new = [d for u, d in seen.items() if u not in g]
            for i in range(0, len(new), 25):
                chunk = new[i:i + 25]
                txt = "\n".join(f"[{j + 1}] {d.title} ({narrow._host(d.url)}): {d.text[:260]}" for j, d in enumerate(chunk))
                calls["n"] += 1
                r = await answer.client().messages.create(
                    model=config.ANSWER_MODEL, max_tokens=200, system="You are a strict search-quality rater.",
                    **answer.model_opts(config.ANSWER_MODEL),
                    messages=[{"role": "user", "content": ne.JUDGE.format(q=it["heard"], n=len(chunk), items=txt)}])
                out = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                try:
                    arr = json.loads(out[out.index("["):out.rindex("]") + 1])
                except ValueError:
                    arr = []
                if len(arr) == len(chunk):
                    for d, s in zip(chunk, arr):
                        g[d.url] = int(s)
        todo = {it["id"]: it for v in changed for it in changed[v]}
        await asyncio.gather(*[judge(it) for it in todo.values()])
        jpath.write_text(json.dumps(grades, indent=0))
    rep = {"n": len(items), "credits": spent, "judge_calls": calls["n"], "variants": {}}
    for v, xs in changed.items():
        rows = []
        for it in xs:
            g = grades.get(it["id"], {})
            row = {"id": it["id"], "base": it["q"]["base"], v: it["q"][v]}
            for name in ("base", v):
                docs = ne.docs_of(cache.get(ne.bkey(it["body"][name])))
                gr = [g.get(d.url) for d in docs[:5]]
                if docs and all(x is not None for x in gr):
                    gr5 = gr + [0] * (5 - len(gr))
                    row[name + "_m"] = {"p5": sum(x >= 1 for x in gr5) / 5, "g5": sum(gr5) / 5,
                                        "top1": int(gr[0] == 2), "ndcg": ne.ndcg(gr5), "any2": int(2 in gr5)}
            rows.append(row)
        ok = [r for r in rows if "base_m" in r and v + "_m" in r]
        summ = {"n_changed": len(xs), "n_scored": len(ok)}
        for k in ("p5", "g5", "top1", "ndcg", "any2"):
            if ok:
                summ[k] = [round(sum(r["base_m"][k] for r in ok) / len(ok), 3), round(sum(r[v + "_m"][k] for r in ok) / len(ok), 3)]
        rep["variants"][v] = {"summary": summ, "rows": rows}
        print(v, json.dumps(summ))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
