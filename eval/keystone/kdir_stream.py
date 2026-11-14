"""Keystone director (KDIR) workload: a ~11k-question multi-user stream over REAL question pools, in three disjoint
splits (held-out users AND held-out question families):

  train  7,000 questions, 120 users, 60 % of the root families
  dev    2,000 questions,  40 users, 20 %
  test   2,000 questions,  40 users, 20 %

Question pools (all text, no audio, no personal data):
  rich      runs/index/families.jsonl (669 families with paraphrases, near-misses, gold / silver answer keywords)
  director  the director's route-labelled spoken-search questions ($ARC_HANDOFF/data/from_director/route_labels.jsonl,
            ~14.7k; kind / follow-up chains from runs/director_queries). PII lines, "not a request" and commands are
            dropped; texts already in `rich` are dropped; one family per distinct text.
  openmic   addressed questions from the open-mic scenes ($ARC_HANDOFF/.../openmic_utterances.jsonl,
            kind question|addressed; context-dependent follow-ups are left out)

Session / popularity model (make_stream.py's, scaled): Zipf(1.0) popularity over each split's roots, each user has 3
favourite kinds (45 % of their asks), sessions of 2-6 questions, a follow-up next with p 0.6, else a near-miss about
the same entity with p 0.3; 60 % of asks are paraphrases when the family has them; Poisson arrivals with the
per-user rate of the 800-question stream (mean gap 180 s x 30 / users).

Families that are likely to repeat (the top Zipf ranks of each split) and follow-ups get paraphrases / a near-miss /
the resolved follow-up / a live flag from the answer model, ~25 per call (tokens counted, cached in
runs/index/kdir_labels.json so a re-run costs nothing).

    ARC_HANDOFF=... .venv/bin/python eval/keystone/kdir_stream.py [--label-top 800,300,300] [--no-llm]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/keystone"))

OUT = ROOT / "runs/index"
HANDOFF = Path(os.environ.get("ARC_HANDOFF", ROOT.parent / "arc_tagger_handoff"))
SKIP_KINDS = {"not_search", "work_email_calendar", "smart_home_devices", "in_car_navigation", "names_homophones"}
SKIP_STYLES = {"pii", "not_search", "command"}
ZIPF = [1.0]
SPLITS = {"train": (7000, 120, 0.6), "dev": (2000, 40, 0.2), "test": (2000, 40, 0.2)}


def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def split_of(fid: str) -> str:
    h = int(hashlib.md5(fid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.6 else ("dev" if h < 0.8 else "test")


def pool() -> list[dict]:
    fam = load(OUT / "families.jsonl")
    for f in fam:
        f["pool"] = "rich"
    seen = {f["canon"].lower().strip() for f in fam}
    dq = {}
    dqp = ROOT / "runs/director_queries"
    for p in sorted(dqp.glob("part_*.jsonl")):
        for r in load(p):
            dq[r["id"]] = r
    rl = HANDOFF / "data/from_director/route_labels.jsonl"
    rows = load(rl) if rl.exists() else []
    by_text = {}
    for r in rows:
        if r["route"] in ("none", "local") or r.get("style") in SKIP_STYLES or r["split"].endswith("_pii"):
            continue
        base = r["id"].split("~")[0]
        d = dq.get(base, {})
        if d.get("kind") in SKIP_KINDS or d.get("style") in SKIP_STYLES:
            continue
        t = r["text"].strip()
        if not t or t.lower() in seen or t.lower() in by_text or len(t.split()) > 24:
            continue
        by_text[t.lower()] = {"fid": "r" + base, "canon": t, "kind": d.get("kind") or "director", "expect": [],
                              "expect_src": "", "source": "director", "pool": "director", "route": r["route"],
                              "director_id": base, "style": r.get("style", ""),
                              "followup_of_raw": d.get("followup_of")}
    dfam = list(by_text.values())
    ids = {f["director_id"]: f["fid"] for f in dfam}
    for f in dfam:
        par = f.pop("followup_of_raw", None)
        if par and par in ids and f["style"] == "followup":
            f["followup_of"] = ids[par]
    om = HANDOFF / "data/from_director/openmic_utterances.jsonl"
    ofam = []
    if om.exists():
        for u in load(om):
            t = u["text"].strip()
            if u.get("addressed") and u.get("kind") == "question" and t.lower() not in seen and t.lower() not in by_text:
                seen.add(t.lower())
                ofam.append({"fid": f"o{u['scene']}-{u['utt']}", "canon": t, "kind": "openmic", "expect": [],
                             "expect_src": "", "source": "openmic", "pool": "openmic"})
    allf = fam + dfam + ofam
    by = {f["fid"]: f for f in allf}

    def root(f):
        seen_ = set()
        while f.get("followup_of") and f["followup_of"] in by and f["fid"] not in seen_:
            seen_.add(f["fid"])
            f = by[f["followup_of"]]
        return f["fid"]
    for f in allf:
        f["split"] = split_of(root(f))
    return allf


def order_of(fam: list[dict], split: str, seed: int):
    roots = [f for f in fam if f["split"] == split and not f.get("followup_of")]
    rng = random.Random(seed)
    rng.shuffle(roots)
    return roots


# ------------------------------------------------------------------------------------------- LLM labels
PROMPT = """You are building evaluation data for a spoken search engine. For each numbered question below, return a JSON
object with these fields:
- "i": the number
- "paraphrases": 3 different spoken rewordings that have exactly the SAME answer (vary the wording a lot: synonyms,
  word order, possessives, filler like "um" or "hey" sometimes; keep the same entities and constraints)
- "near_miss": 1 question about the SAME main entity whose answer is DIFFERENT (e.g. "who founded nvidia" ->
  "when was nvidia founded"); for a question with no entity, a different question on the same topic
- "resolved": if the question is a follow-up (it is marked FOLLOW-UP with its previous question), the follow-up
  rewritten to be self-contained; else ""
- "live": true if the answer changes within days (news, scores, prices, weather, schedules, opening hours), else false
Reply with ONLY a JSON array, no prose.

Questions:
"""


async def _label(batch, sem, client, model):
    import re
    lines = []
    for j, f in enumerate(batch):
        s = f"{j + 1}. {f['canon']}"
        if f.get("followup_of_text"):
            s += f"   [FOLLOW-UP; previous question: {f['followup_of_text']}]"
        lines.append(s)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.messages.create(model=model, max_tokens=8000,
                                                 messages=[{"role": "user", "content": PROMPT + "\n".join(lines)}])
                txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
                return arr, (r.usage.input_tokens, r.usage.output_tokens)
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, type(e).__name__, str(e)[:120], flush=True)
                await asyncio.sleep(3)
    return [], (0, 0)


async def label_all(todo: list[dict], cache: dict) -> tuple[int, int]:
    from server import answer, config
    client = answer.client()
    need = [f for f in todo if f["fid"] not in cache]
    batches = [need[i:i + 25] for i in range(0, len(need), 25)]
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[_label(b, sem, client, config.ANSWER_MODEL) for b in batches])
    tin = tout = 0
    for b, (arr, (i, o)) in zip(batches, res):
        tin, tout = tin + i, tout + o
        got = {int(x.get("i", 0)): x for x in arr if isinstance(x, dict)}
        for j, f in enumerate(b):
            x = got.get(j + 1)
            if x:
                cache[f["fid"]] = {"paraphrases": [p for p in (x.get("paraphrases") or []) if isinstance(p, str)][:3],
                                   "near_miss": x.get("near_miss") or "", "resolved": x.get("resolved") or "",
                                   "live": bool(x.get("live"))}
    return tin, tout


# ------------------------------------------------------------------------------------------- stream
def make(fam, split, n, n_users, seed, gap_s):
    rng = random.Random(seed)
    by = {f["fid"]: f for f in fam}
    children = {}
    for f in fam:
        if f.get("followup_of") and f["split"] == split:
            children.setdefault(f["followup_of"], []).append(f["fid"])
    order = order_of(fam, split, seed)
    w = [1.0 / (r + 1) ** ZIPF[0] for r in range(len(order))]
    kinds = sorted({f["kind"] for f in order})
    users = []
    for u in range(n_users):
        fav = set(rng.sample(kinds, 3))
        idx = [i for i, f in enumerate(order) if f["kind"] in fav]
        users.append({"u": f"{split[:2]}{u:03d}", "idx": idx})

    def draw(user):
        if user["idx"] and rng.random() < 0.45:
            ii = user["idx"]
            return order[rng.choices(ii, weights=[w[i] for i in ii])[0]]
        return order[rng.choices(range(len(order)), weights=w)[0]]

    def surface(f):
        ps = f.get("paraphrases") or []
        if ps and rng.random() < 0.6:
            return rng.choice(ps), "paraphrase"
        return f["canon"], "canon"

    out, t, sess = [], 0.0, 0
    while len(out) < n:
        user = rng.choice(users)
        sess += 1
        n_q = rng.randint(2, 6)
        f = draw(user)
        for j in range(n_q):
            if len(out) >= n:
                break
            text, form = surface(f)
            qtext = text
            if f.get("followup_of") and form == "canon" and f.get("resolved"):
                qtext = f["resolved"]
            t += rng.expovariate(1.0 / gap_s)
            out.append({"pos": len(out), "t": round(t, 1), "user": user["u"], "session": f"{split[:2]}s{sess}",
                        "turn": j, "fid": f["fid"], "text": text, "query": qtext, "form": form, "kind": f["kind"],
                        "live": bool(f.get("live")), "expect": f.get("expect") or [],
                        "expect_src": f.get("expect_src", ""), "source": f.get("source", ""),
                        "pool": f.get("pool", ""), "director_id": f.get("director_id", ""),
                        "followup": bool(f.get("followup_of")), "split": split})
            kids = children.get(f["fid"])
            r = rng.random()
            if kids and r < 0.6:
                f = by[rng.choice(kids)]
            elif f.get("near_miss") and r < 0.9 and not f["fid"].endswith("~nm"):
                nm = by.get(f["fid"] + "~nm")
                if nm is None:
                    nm = {"fid": f["fid"] + "~nm", "canon": f["near_miss"], "kind": f["kind"], "expect": [],
                          "expect_src": "", "source": "near_miss", "paraphrases": [], "live": f.get("live", False),
                          "pool": f.get("pool", ""), "split": split}
                    by[nm["fid"]] = nm
                f = nm
            else:
                f = draw(user)
    return out


def main(a):
    ZIPF[0] = a.zipf
    fam = pool()
    by = {f["fid"]: f for f in fam}
    for f in fam:
        if f.get("followup_of") and f["followup_of"] in by:
            f["followup_of_text"] = by[f["followup_of"]]["canon"]
    from collections import Counter
    print("pool", len(fam), Counter(f["pool"] for f in fam), Counter(f["split"] for f in fam), flush=True)
    tops = [int(x) for x in a.label_top.split(",")]
    todo = []
    for (split, _), k in zip(SPLITS.items(), tops):
        order = order_of(fam, split, a.seed)
        head = [f for f in order[:k] if not f.get("paraphrases")]
        todo += head
        hs = {f["fid"] for f in head}
        todo += [f for f in fam if f.get("followup_of") in hs and f["split"] == split and not f.get("paraphrases")]
    cp = OUT / "kdir_labels.json"
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    tin = tout = 0
    if not a.no_llm:
        tin, tout = asyncio.run(label_all(todo, cache))
        cp.write_text(json.dumps(cache))
    for f in fam:
        x = cache.get(f["fid"])
        if x and not f.get("paraphrases"):
            f.update(paraphrases=x["paraphrases"], near_miss=x["near_miss"], resolved=x["resolved"], live=x["live"])
    (OUT / "kdir_families.jsonl").write_text("\n".join(json.dumps(f) for f in fam) + "\n")
    rep = {"pool": len(fam), "labelled": len(todo), "tokens_in": tin, "tokens_out": tout, "streams": {}}
    for i, (split, (n, users, _)) in enumerate(SPLITS.items()):
        s = make(fam, split, n, users, a.seed + i, 180.0 * 30 / users)
        Path(OUT / f"kdir_{split}.jsonl").write_text("\n".join(json.dumps(x) for x in s) + "\n")
        fids = [x["fid"] for x in s]
        seenf, rep_n = set(), 0
        for x in fids:
            rep_n += x in seenf
            seenf.add(x)
        st = {"n": len(s), "users": users, "families": len(set(fids)), "repeat_share": round(rep_n / len(s), 3),
              "distinct_texts": len({x["query"] for x in s}), "live": sum(x["live"] for x in s),
              "with_expect": sum(1 for x in s if x["expect"]), "pools": dict(Counter(x["pool"] for x in s)),
              "span_h": round(s[-1]["t"] / 3600, 1)}
        rep["streams"][split] = st
        print(split, st, flush=True)
    (OUT / "kdir_stream_report.json").write_text(json.dumps(rep, indent=1))
    print(f"labels: {len(todo)} families, tokens in {tin} out {tout}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-top", default="800,300,300")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--zipf", type=float, default=0.9)
    main(ap.parse_args())
