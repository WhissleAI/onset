"""A realistic multi-user query stream over the question families (runs/index/families.jsonl).

Real usage has repetition and topic locality; the stream models four sources of it:
  popularity  : Zipf(s) over families (a few questions are asked by many people)
  interests   : each user has 3 favourite kinds; 45 % of their questions come from those (user locality)
  sessions    : 2-6 questions per sitting; after a question, a follow-up family (if one exists) comes next with
                p=0.6, else a RELATED question about the same entity (the family's near-miss: same subject, different
                answer) with p=0.3 (topical locality: the graph's job)
  surface     : each ask is the canonical wording (40 %) or one of 3 paraphrases (60 %): the same answer in other words
Fresh / time-sensitive families (live=true: news, scores, prices, weather, schedules) keep their natural share.
Arrivals: Poisson, mean gap 3 min across all users (800 questions ~ 40 h).

    .venv/bin/python eval/keystone/make_stream.py [--n 800] [--users 30] [--zipf 1.0] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(a):
    fam = [json.loads(x) for x in Path(a.families).read_text().splitlines() if x.strip()]
    rng = random.Random(a.seed)
    by = {f["fid"]: f for f in fam}
    children = {}
    for f in fam:
        if f.get("followup_of"):
            children.setdefault(f["followup_of"], []).append(f["fid"])
    roots = [f for f in fam if not f.get("followup_of")]
    order = roots[:]
    rng.shuffle(order)
    w = [1.0 / (r + 1) ** a.zipf for r in range(len(order))]
    kinds = sorted({f["kind"] for f in roots})
    users = []
    for u in range(a.users):
        fav = set(rng.sample(kinds, 3))
        idx = [i for i, f in enumerate(order) if f["kind"] in fav]
        users.append({"u": f"u{u:02d}", "fav": sorted(fav), "idx": idx})

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
    while len(out) < a.n:
        user = rng.choice(users)
        sess += 1
        n_q = rng.randint(2, 6)
        f = draw(user)
        for j in range(n_q):
            if len(out) >= a.n:
                break
            text, form = surface(f)
            qtext = text
            if f.get("followup_of") and form == "canon" and f.get("resolved"):
                qtext = f["resolved"]          # memory resolves the reference (what MEM_INLINE produces)
            t += rng.expovariate(1.0 / a.gap_s)
            out.append({"pos": len(out), "t": round(t, 1), "user": user["u"], "session": f"s{sess}", "turn": j,
                        "fid": f["fid"], "text": text, "query": qtext, "form": form, "kind": f["kind"],
                        "live": bool(f.get("live")), "expect": f.get("expect") or [],
                        "expect_src": f.get("expect_src", ""), "source": f.get("source", ""),
                        "followup": bool(f.get("followup_of"))})
            # next question in the session
            kids = children.get(f["fid"])
            r = rng.random()
            if kids and r < 0.6:
                f = by[rng.choice(kids)]
            elif f.get("near_miss") and r < 0.9 and not f["fid"].endswith("~nm"):
                nm = by.get(f["fid"] + "~nm")
                if nm is None:
                    nm = {"fid": f["fid"] + "~nm", "canon": f["near_miss"], "kind": f["kind"], "expect": [],
                          "expect_src": "", "source": "near_miss", "paraphrases": [], "live": f.get("live", False)}
                    by[nm["fid"]] = nm
                f = nm
            else:
                f = draw(user)
    Path(a.out).write_text("\n".join(json.dumps(x) for x in out) + "\n")
    fids = [x["fid"] for x in out]
    rep = sum(1 for i, x in enumerate(fids) if x in set(fids[:i]))
    print(f"{len(out)} questions, {len(set(fids))} families, repeats {rep} ({rep / len(out):.0%}), "
          f"distinct texts {len({x['query'] for x in out})}, live {sum(x['live'] for x in out)}, "
          f"with expect {sum(1 for x in out if x['expect'])}, span {t / 3600:.1f} h -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=str(ROOT / "runs/index/families.jsonl"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--users", type=int, default=30)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--gap-s", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "runs/index/stream.jsonl"))
    main(ap.parse_args())
