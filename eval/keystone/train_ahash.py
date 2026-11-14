"""Train + evaluate the learned answer-equivalence hash (server/keystone/ahash.py) against plain embedding NN and the
typed-key rule.

Data (free supervision, no labelling):
  families    runs/index/families.jsonl: canonical question + 3 paraphrases = one answer; each family's near-miss is
              its own answer (a hard negative: same entity, different question). Split by family id: 70 % train /
              30 % test (test families are never seen in training).
  templates   (train only) typed relation questions over entities of the local Wikipedia lead index, 3-5 phrasings per
              (entity, relation): the same (entity, relation) = one answer; same entity + other relation and same
              relation + other entity are the hard negatives. Entities whose kind can't be read off their lead are
              skipped. Template entities are disjoint between train and the template test split.

Evaluation = the cache decision. Memory holds one stored question per answer (the canonical wording; plus every
near-miss as a distractor); queries are the other wordings. A method reuses the stored answer of its best match or
abstains. Reported: precision@1 (reused answer is the right one) vs coverage (share of queries that reuse), for
  emb-nn  cosine of EmbeddingGemma query vectors, threshold sweep
  key     exact typed RelKey match (keystone/relkey.py), complete keys only
  hash    Hamming distance of the learned 64-bit codes, radius sweep (ties: cosine)
  key+hash  key when there is one, else hash

    .venv/bin/python eval/keystone/train_ahash.py [--bits 64] [--q-dim 256] [--steps 1500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from server.keystone import ahash  # noqa: E402
from server.keystone.relkey import key_of  # noqa: E402

OUT = ROOT / "runs/index"
TEMPLATES = {
    "person": {"born_date": ["when was {e} born", "what year was {e} born", "{e}'s birthday, what year",
                             "um when's {e}'s birth date"],
               "born_place": ["where was {e} born", "{e}'s birthplace", "which city was {e} born in",
                              "hey where is {e} from originally"],
               "died_date": ["when did {e} die", "what year did {e} pass away", "{e}, when did they die"],
               "spouse_of": ["who was {e} married to", "who is {e}'s wife or husband", "{e}'s spouse"]},
    "place": {"capital_of": ["what is the capital of {e}", "{e}'s capital city", "um capital of {e}",
                             "which city is the capital of {e}"],
              "population_of": ["what is the population of {e}", "how many people live in {e}",
                                "{e} population", "hey how big is the population of {e}"],
              "language_of": ["what language do they speak in {e}", "official language of {e}",
                              "which language is spoken in {e}"],
              "currency_of": ["what currency does {e} use", "{e}'s currency", "what money do they use in {e}"]},
    "org": {"founded_by": ["who founded {e}", "who started {e}", "{e}'s founder", "who was behind founding {e}"],
            "inception": ["when was {e} founded", "what year did {e} start", "how long ago was {e} founded",
                          "{e}, what year was it established"],
            "hq_of": ["where is {e} headquartered", "{e} headquarters location", "where is {e} based"],
            "ceo_of": ["who is the ceo of {e}", "who runs {e}", "{e}'s chief executive"]},
    "work": {"author_of": ["who wrote {e}", "who is the author of {e}", "{e}, who wrote it"],
             "inception": ["when did {e} come out", "what year was {e} released", "release year of {e}"],
             "director_of": ["who directed {e}", "{e}'s director", "who was the director of {e}"]},
}


def kind_of_lead(text: str) -> str | None:
    t = text[:300].lower()
    if re.search(r"\(born \d|\(\d{1,2} \w+ \d{4} – |\(c?\.? ?\d{3,4} – \d{3,4}\)|\bwas an? [a-z ]*(?:singer|actor|"
                 r"actress|writer|politician|painter|player|scientist|composer|poet|businessman|engineer)", t):
        return "person"
    if re.search(r"\bis a (?:sovereign )?country\b|\bis a city\b|\bis the capital\b|\bis a state\b|\bis a nation\b|"
                 r"\bis a landlocked country\b|\bis an island country\b", t):
        return "place"
    if re.search(r"\b(?:is an?|was an?) [a-z -]*(?:company|corporation|manufacturer|conglomerate|retailer|"
                 r"multinational)\b", t):
        return "org"
    if re.search(r"\bis an? [0-9a-z -]*(?:novel|film|album|play|book|television series)\b", t):
        return "work"
    return None


def template_items(n_per_kind: int, seed: int):
    leads = [json.loads(x) for x in (ROOT / "index/data/knowledge/leads.jsonl").read_text().splitlines() if x.strip()]
    rng = random.Random(seed)
    by_kind: dict = {}
    for d in leads:
        k = kind_of_lead(d["text"])
        if k and 3 <= len(d["title"]) <= 30 and "(" not in d["title"]:
            by_kind.setdefault(k, []).append(d["title"])
    items = []
    for k, ts in by_kind.items():
        rng.shuffle(ts)
        for e in ts[:n_per_kind]:
            for rel, forms in TEMPLATES[k].items():
                for form in forms:
                    items.append({"fam": f"T:{rel}:{e}", "text": form.format(e=e), "ent": e, "rel": rel})
    return items


def split_of(fid: str) -> str:
    return "test" if int(hashlib.md5(fid.encode()).hexdigest(), 16) % 10 < 3 else "train"


def build(a):
    fam = [json.loads(x) for x in (OUT / "families.jsonl").read_text().splitlines() if x.strip()]
    # families with the same answer under different ids (the same question in two eval sets): union them, by an
    # identical surface wording or identical GOLD answer keywords (independent of every method compared here)
    parent = {f["fid"]: f["fid"] for f in fam}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen_text, seen_exp = {}, {}
    for f in fam:
        for t in [f.get("resolved") or f["canon"], *(f.get("paraphrases") or [])]:
            n = " ".join(re.findall(r"[a-z0-9]+", t.lower()))
            if n in seen_text:
                parent[find(f["fid"])] = find(seen_text[n])
            else:
                seen_text[n] = f["fid"]
        if f.get("expect_src") == "gold" and f.get("expect"):
            e = tuple(sorted(f["expect"]))
            if e in seen_exp:
                parent[find(f["fid"])] = find(seen_exp[e])
            else:
                seen_exp[e] = f["fid"]
    rows = []                       # {fam, text, split, role: canon|para|nm, live}
    for f in fam:
        root = find(f["fid"])
        sp = split_of(root)
        orig = f["fid"]
        if orig != root:              # merged into another family: every wording is a query of the root's answer
            for p in [f.get("resolved") or f["canon"], *(f.get("paraphrases") or [])]:
                rows.append({"fam": root, "text": p, "split": sp, "role": "para", "live": f.get("live", False)})
            if f.get("near_miss"):
                rows.append({"fam": orig + "~nm", "text": f["near_miss"], "split": sp, "role": "nm",
                             "live": f.get("live", False)})
            continue
        f = {**f, "fid": root}
        canon = f.get("resolved") or f["canon"]
        rows.append({"fam": f["fid"], "text": canon, "split": sp, "role": "canon", "live": f.get("live", False)})
        for p in f.get("paraphrases") or []:
            rows.append({"fam": f["fid"], "text": p, "split": sp, "role": "para", "live": f.get("live", False)})
        if f.get("near_miss"):
            rows.append({"fam": f["fid"] + "~nm", "text": f["near_miss"], "split": sp, "role": "nm",
                         "live": f.get("live", False)})
    tmpl = template_items(a.n_tmpl, a.seed)
    ents = sorted({t["ent"] for t in tmpl})
    test_ents = {e for e in ents if split_of("E" + e) == "test"}
    for t in tmpl:
        sp = "ttest" if t["ent"] in test_ents else "train"
        rows.append({"fam": t["fam"], "text": t["text"], "split": sp, "role": "tmpl", "live": False})
    return rows


def featurize(rows):
    from replay_frames import frame_for
    from server import embed
    have = {}
    p = OUT / "fixture_vecs.npz"
    if p.exists():
        z = np.load(p)
        have = {t: v for t, v in zip(z["q_texts"], z["q_vecs"])}
    need = [r["text"] for r in rows if r["text"] not in have]
    need = list(dict.fromkeys(need))
    for i in range(0, len(need), 64):
        b = need[i:i + 64]
        for t, v in zip(b, embed.encode_queries(b)):
            have[t] = v
    Q = np.stack([np.asarray(have[r["text"]], np.float32) for r in rows])
    arcs, keys = [], []
    import pickle
    cp = OUT / "ahash_frames.pkl"
    try:
        cache = pickle.loads(cp.read_bytes())
    except (OSError, ValueError, EOFError):
        cache = {}
    n0 = len(cache)
    for r in rows:
        t = r["text"]
        if t not in cache:
            f = frame_for(t)
            k = key_of(f.words, f.slots, f.intent)
            cache[t] = (ahash.arc_of(f.words, f.intent, f.slots, f.acts, r["live"], k), k)
        a_, k_ = cache[t]
        if bool(a_.get("live")) != bool(r["live"]):
            a_ = {**a_, "live": bool(r["live"])}
        arcs.append(a_)
        keys.append(k_)
    if len(cache) != n0:
        cp.write_bytes(pickle.dumps(cache))
    return Q, arcs, keys


def train(X, fams, train_idx, a, nm_of):
    import torch
    torch.manual_seed(a.seed)
    Xt = torch.tensor(X[train_idx])
    fam_t = [fams[i] for i in train_idx]
    by: dict = {}
    for j, f in enumerate(fam_t):
        by.setdefault(f, []).append(j)
    multi = [f for f, js in by.items() if len(js) >= 2]
    W = torch.nn.Parameter(torch.randn(X.shape[1], a.bits) * 0.05)
    opt = torch.optim.Adam([W], lr=a.lr)
    rng = random.Random(a.seed)
    for step in range(a.steps):
        fs = rng.sample(multi, min(a.batch, len(multi)))
        ia, ib, extra = [], [], []
        for f in fs:
            x, y = rng.sample(by[f], 2)
            ia.append(x)
            ib.append(y)
            nm = by.get(nm_of.get(f, ""), [])
            if nm:
                extra.append(rng.choice(nm))
        za = torch.tanh(Xt[ia] @ W)
        zb = torch.tanh(Xt[ib + extra] @ W)
        S = (za @ zb.T) / a.bits / a.tau                          # [B, B + extra]
        target = torch.arange(len(ia))
        loss = torch.nn.functional.cross_entropy(S, target) + \
            torch.nn.functional.cross_entropy(S[:, :len(ia)].T, target)
        quant = ((za.abs() - 1) ** 2).mean() + ((zb.abs() - 1) ** 2).mean()     # push towards +-1 (binary)
        bal = (za.mean(0) ** 2).mean()                                            # bits balanced
        (loss + a.lam_q * quant + a.lam_b * bal).backward()
        opt.step()
        opt.zero_grad()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    return W.detach().numpy()


def evaluate(split_rows, Q, codes, keys, name_split, R=None):
    """Memory: one canonical/stored question per answer + near-misses; queries: the other wordings."""
    store = [i for i in split_rows if split_rows_role[i] in ("canon", "nm") or
             (split_rows_role[i] == "tmpl" and first_of_fam[fams[i]] == i)]
    queries = [i for i in split_rows if split_rows_role[i] in ("para",) or
               (split_rows_role[i] == "tmpl" and first_of_fam[fams[i]] != i)]
    sv = Q[store]
    sc = codes[store]
    res = {"n_store": len(store), "n_query": len(queries)}
    cos_best, cos_fam, ham_best, ham_fam, ham_cos = [], [], [], [], []
    key_fam = []
    skeys = {}
    for i in store:
        if keys[i] is not None and keys[i].complete:
            skeys.setdefault(keys[i].text(), i)
    for q in queries:
        s = sv @ Q[q]
        j = int(np.argmax(s))
        cos_best.append(float(s[j]))
        cos_fam.append(fams[store[j]] == fams[q])
        h = ahash.hamming(sc, codes[q]).astype(np.float32) - 1e-3 * s       # ties broken by cosine
        j2 = int(np.argmin(h))
        ham_best.append(int(round(h[j2] + 1e-3 * s[j2])))
        ham_fam.append(fams[store[j2]] == fams[q])
        ham_cos.append(float(s[j2]))
        k = keys[q]
        if k is not None and k.complete and k.text() in skeys:
            key_fam.append(fams[skeys[k.text()]] == fams[q])
        else:
            key_fam.append(None)
    nq = len(queries)

    def curve(scores, ok, higher=True, grid=None):
        out = []
        for thr in grid:
            sel = [o for s, o in zip(scores, ok) if (s >= thr if higher else s <= thr)]
            out.append({"thr": thr, "coverage": round(len(sel) / nq, 3),
                        "precision": round(sum(sel) / len(sel), 3) if sel else None})
        return out

    res["emb_nn"] = curve(cos_best, cos_fam, True, [0.70, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96])
    res["hash"] = curve(ham_best, ham_fam, False, [0, 1, 2, 3, 4, 6, 8, 10, 12, 16])
    kk = [x for x in key_fam if x is not None]
    res["key"] = {"coverage": round(len(kk) / nq, 3), "precision": round(sum(kk) / len(kk), 3) if kk else None}
    comb = []
    for kf, hb, hf in zip(key_fam, ham_best, ham_fam):
        comb.append((kf, hb, hf))
    res["key+hash"] = []
    for r in [0, 2, 4, 6, 8]:
        sel = [kf if kf is not None else hf for kf, hb, hf in comb if kf is not None or hb <= r]
        res["key+hash"].append({"thr": r, "coverage": round(len(sel) / nq, 3),
                                "precision": round(sum(sel) / len(sel), 3) if sel else None})
    res["top1_any"] = {"emb_nn": round(sum(cos_fam) / nq, 3), "hash": round(sum(ham_fam) / nq, 3)}
    # hash as a candidate filter + cosine verification: within radius r, the best cosine, if >= 0.85
    hv = []
    for q in queries:
        s = sv @ Q[q]
        h = ahash.hamming(sc, codes[q])
        for r in (2, 4, 6, 8):
            m = np.where(h <= r, s, -1.0)
            j = int(np.argmax(m))
            hv.append((r, float(m[j]), fams[store[j]] == fams[q]))
    res["hash_then_cos"] = []
    for r in (2, 4, 6, 8):
        for thr in (0.8, 0.85, 0.9):
            sel = [ok for rr, sc_, ok in hv if rr == r and sc_ >= thr]
            res["hash_then_cos"].append({"r": r, "thr": thr, "coverage": round(len(sel) / nq, 3),
                                         "precision": round(sum(sel) / len(sel), 3) if sel else None})
    if R is not None:
        Rs = R[store]
        lb, lf = [], []
        for q in queries:
            s = Rs @ R[q]
            j = int(np.argmax(s))
            lb.append(float(s[j]))
            lf.append(fams[store[j]] == fams[q])
        res["learned_cos"] = curve(lb, lf, True, [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        res["top1_any"]["learned_cos"] = round(sum(lf) / nq, 3)
    return res


def pair_x(i, j, Q, arcs, keys):
    """Answer-equivalence verifier features for (query i, stored question j): the embedding cosine plus Arc
    agreement (answer type, relation, typed key, subject, entity / number types, freshness)."""
    from server.keystone.verify_eq import pair_features
    return pair_features(float(Q[i] @ Q[j]), arcs[i], arcs[j], keys[i], keys[j])


def verifier_data(idx, Q, arcs, keys, k=3):
    store = [i for i in idx if split_rows_role[i] in ("canon", "nm") or
             (split_rows_role[i] == "tmpl" and first_of_fam[fams[i]] == i)]
    queries = [i for i in idx if split_rows_role[i] == "para" or
               (split_rows_role[i] == "tmpl" and first_of_fam[fams[i]] != i)]
    S = Q[store]
    X, y, grp = [], [], []
    for q in queries:
        s = S @ Q[q]
        for j in np.argsort(-s)[:k]:
            X.append(pair_x(q, store[j], Q, arcs, keys))
            y.append(fams[store[j]] == fams[q])
            grp.append(q)
    return np.array(X), np.array(y, float), np.array(grp), len(queries)


def verifier_eval(w, X, y, grp, nq):
    from server.keystone.voi import fit  # noqa: F401
    p = 1 / (1 + np.exp(-(X @ w)))
    best = {}
    for pi, yi, g in zip(p, y, grp):
        if g not in best or pi > best[g][0]:
            best[g] = (pi, yi)
    out = []
    for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        sel = [yy for pp, yy in best.values() if pp >= thr]
        out.append({"thr": thr, "coverage": round(len(sel) / nq, 3),
                    "precision": round(sum(sel) / len(sel), 3) if sel else None})
    return out


def main(a):
    global split_rows_role, first_of_fam, fams
    t0 = time.time()
    rows = build(a)
    print(f"{len(rows)} rows ({sum(r['split'] == 'train' for r in rows)} train)", flush=True)
    Q, arcs, keys = featurize(rows)
    print(f"featurized in {time.time() - t0:.0f}s", flush=True)
    fams = [r["fam"] for r in rows]
    split_rows_role = [r["role"] for r in rows]
    first_of_fam = {}
    for i, f in enumerate(fams):
        first_of_fam.setdefault(f, i)
    F = np.stack([ahash.arc_features(x) for x in arcs])
    qd = Q[:, : a.q_dim] / np.linalg.norm(Q[:, : a.q_dim], axis=1, keepdims=True)
    X = np.concatenate([qd, a.w_arc * F], axis=1).astype(np.float32)
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    mean = X[tr].mean(0)
    Xc = X - mean
    nm_of = {f: f + "~nm" for f in set(fams)}
    W = train(Xc, fams, tr, a, nm_of)
    codes = np.packbits((Xc @ W) > 0, axis=1)
    R = np.tanh(Xc @ W)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    report = {"args": vars(a), "rows": len(rows), "train_rows": len(tr)}
    for sp in ("test", "ttest", "train"):
        idx = [i for i, r in enumerate(rows) if r["split"] == sp]
        report[sp] = evaluate(idx, Q, codes, keys, sp, R)
    # ablation: hash without Arc features (embedding only) on the family test split
    if a.ablate:
        X0 = np.concatenate([qd, 0 * F], axis=1).astype(np.float32)
        m0 = X0[tr].mean(0)
        W0 = train(X0 - m0, fams, tr, a, nm_of)
        c0 = np.packbits(((X0 - m0) @ W0) > 0, axis=1)
        report["test_no_arc"] = evaluate([i for i, r in enumerate(rows) if r["split"] == "test"], Q, c0, keys, "test")
    # learned answer-equivalence VERIFIER (embedding NN candidates + Arc agreement), the alternative to the hash
    from server.keystone import verify_eq
    from server.keystone.voi import fit
    Xv, yv, gv, _ = verifier_data([i for i in tr if rows[i]["split"] == "train"], Q, arcs, keys)
    wv = fit(Xv, yv, l2=1.0)
    report["verifier_w"] = dict(zip(verify_eq.NAMES, map(float, wv)))
    for sp in ("test", "ttest"):
        Xt, yt, gt, nq = verifier_data([i for i, r in enumerate(rows) if r["split"] == sp], Q, arcs, keys)
        report[sp]["verifier"] = verifier_eval(wv, Xt, yt, gt, nq)
    # hard negatives: a near-miss question (same entity, DIFFERENT answer) asked against a memory holding the
    # canonical questions only: any reuse is a wrong answer. False-accept rate per method / threshold.
    te = [i for i, r in enumerate(rows) if r["split"] == "test"]
    st = [i for i in te if split_rows_role[i] == "canon"]
    nmq = [i for i in te if split_rows_role[i] == "nm"]
    Sv, Sc = Q[st], codes[st]
    fa = {"n": len(nmq), "emb_nn": {}, "verifier": {}, "hash": {}, "key": 0}
    skey = {keys[i].text() for i in st if keys[i] is not None and keys[i].complete}
    cosb, pb, hb = [], [], []
    for qi in nmq:
        sc_ = Sv @ Q[qi]
        cosb.append(float(sc_.max()))
        hb.append(int(ahash.hamming(Sc, codes[qi]).min()))
        top = np.argsort(-sc_)[:3]
        pb.append(max(1 / (1 + np.exp(-np.dot(wv, pair_x(qi, st[j], Q, arcs, keys)))) for j in top))
        fa["key"] += int(keys[qi] is not None and keys[qi].complete and keys[qi].text() in skey)
    for t_ in (0.85, 0.88, 0.9, 0.92, 0.94):
        fa["emb_nn"][t_] = round(sum(c >= t_ for c in cosb) / len(nmq), 3)
    for t_ in (0.3, 0.5, 0.7, 0.8):
        fa["verifier"][t_] = round(sum(p_ >= t_ for p_ in pb) / len(nmq), 3)
    for r_ in (2, 4, 6):
        fa["hash"][r_] = round(sum(h <= r_ for h in hb) / len(nmq), 3)
    fa["key"] = round(fa["key"] / len(nmq), 3)
    report["test"]["near_miss_false_accept"] = fa
    print(" near-miss false accepts", fa)
    verify_eq.MODEL_PATH.write_text(json.dumps({"w": list(map(float, wv)), "names": verify_eq.NAMES}))
    print(" verifier", report["test"]["verifier"])
    np.savez(ahash.MODEL_PATH, W=W.astype(np.float16), q_dim=a.q_dim, bits=a.bits, w_arc=a.w_arc,
             mean=mean.astype(np.float16))
    (OUT / "ahash_eval.json").write_text(json.dumps(report, indent=1))
    t = report["test"]
    print("TEST families:", json.dumps({k: t[k] for k in ("key", "top1_any")}))
    print(" emb_nn", t["emb_nn"])
    print(" hash  ", t["hash"])
    print(" k+h   ", t["key+hash"])
    print(" h->cos", t["hash_then_cos"])
    print(" learned", t["learned_cos"], t["top1_any"])
    if "test_no_arc" in report:
        print(" hash(no arc)", report["test_no_arc"]["hash"], report["test_no_arc"]["top1_any"])
    tt = report["ttest"]
    print("TEMPLATE test:", tt["key"], tt["top1_any"], "\n hash", tt["hash"][:6], "\n emb", tt["emb_nn"][:6])
    print(f"model -> {ahash.MODEL_PATH} ({ahash.MODEL_PATH.stat().st_size // 1024} KB); {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--q-dim", type=int, default=256)
    ap.add_argument("--w-arc", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lam-q", type=float, default=0.1)
    ap.add_argument("--lam-b", type=float, default=0.1)
    ap.add_argument("--n-tmpl", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", action="store_true")
    main(ap.parse_args())
